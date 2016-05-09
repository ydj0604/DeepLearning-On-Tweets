import cPickle
import numpy as np
import theano
import theano.tensor as T
import warnings
from collections import OrderedDict
import pandas as pd
import time
from collections import defaultdict
import csv
import re
warnings.filterwarnings("ignore")


# different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)

def Tanh(x):
    y = T.tanh(x)
    return(y)

def Iden(x):
    y = x
    return(y)


class ConvNetModel():
    def __init__(self, x_len, classifier_params, conv_layer_params, word_embeddings):
        self.img_h = x_len
        self.img_w = len(word_embeddings[0])
        self.filter_hs = [3, 4, 5]
        self.filter_w = len(word_embeddings[0])
        self.feature_maps = 100
        rng = np.random.RandomState(3435)

        filter_shapes = []
        pool_sizes = []
        for filter_h in self.filter_hs:
            filter_shapes.append((self.feature_maps, 1, filter_h, self.filter_w))
            pool_sizes.append((self.img_h-filter_h+1, self.img_w-self.filter_w+1))

        #define model architecture
        self.index = T.lscalar()
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        self.Words = theano.shared(value=word_embeddings, name="Words")

        layer0_input = self.Words[T.cast(self.x.flatten(),dtype="int32")].reshape((self.x.shape[0],1,self.x.shape[1],self.Words.shape[1]))
        self.conv_layers = []
        layer1_inputs = []
        for i in xrange(len(self.filter_hs)):
            filter_shape = filter_shapes[i]
            pool_size = pool_sizes[i]
            conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(None, 1, self.img_h, self.img_w),
                                    filter_shape=filter_shape, poolsize=pool_size, non_linear="relu", W=None, b=None)

            layer1_input = conv_layer.output.flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        layer1_input = T.concatenate(layer1_inputs,1)

        # load trained params
        self.classifier = MLPDropout(rng, input=layer1_input, layer_sizes=[300, 3], activations=[Iden], dropout_rates=[0.5],
                                W=None, b=None)

        # params
        self.params = self.classifier.params
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params
        self.params += [self.Words]
        self.cost = self.classifier.negative_log_likelihood(self.y)
        self.dropout_cost = self.classifier.dropout_negative_log_likelihood(self.y)
        self.grad_updates = sgd_updates_adadelta(self.params, self.dropout_cost, 0.95, 1e-6, 9.0)

    def train(self, data, batch_size, num_epochs):
        x, y = shared_dataset((data[:, :-1], data[:, -1]))
        n_train_batches = int(len(data) / batch_size)
        if len(data) % batch_size > 0:
            n_train_batches += 1

        print '... training'

        train_model = theano.function([self.index], self.cost, updates=self.grad_updates,
                                      givens={
                                          self.x: x[self.index*batch_size:(self.index+1)*batch_size],
                                          self.y: y[self.index*batch_size:(self.index+1)*batch_size]},
                                      allow_input_downcast=True)
        test_model = theano.function([self.index], self.classifier.errors(self.y),
                                     givens={
                                         self.x: x[self.index * batch_size: (self.index + 1) * batch_size],
                                         self.y: y[self.index * batch_size: (self.index + 1) * batch_size]},
                                     allow_input_downcast=True)

        # zero vec
        zero_vec_tensor = T.vector()
        zero_vec = np.zeros(self.img_w)
        set_zero = theano.function([zero_vec_tensor], updates=[(self.Words, T.set_subtensor(
                self.Words[0,:], zero_vec_tensor))], allow_input_downcast=True)

        epoch = 0
        while (epoch < num_epochs):
            start_time = time.time()
            epoch += 1
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
                # print 'cost: %.2f' % cost_epoch

            train_losses = [test_model(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            print 'epoch: %i, training time: %.2f secs, train perf: %.2f %%'\
                  % (epoch, time.time()-start_time, train_perf * 100.)

    def predict(self, data):
        test_pred_layers = []
        test_size = int(data.shape[0])
        test_layer0_input = self.Words[T.cast(self.x.flatten(),dtype="int32")].reshape((test_size,1,self.img_h,self.Words.shape[1]))
        for conv_layer in self.conv_layers:
            test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
            test_pred_layers.append(test_layer0_output.flatten(2))
        test_layer1_input = T.concatenate(test_pred_layers, 1)
        test_y_pred = self.classifier.predict(test_layer1_input)
        test_predict_all = theano.function([self.x], test_y_pred, allow_input_downcast=True)
        predictions = test_predict_all(data)
        return predictions


def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def get_idx_from_sent(sent, word_idx_map, max_l, filter_h):
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if len(x) == max_l+2*pad:
            print sent
            break
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, max_l, filter_h):
    data = []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
        sent.append(rev["y"])
        data.append(sent)
    data = np.array(data, dtype="int")
    return data


def clean_tweet(tweet):
    # reduction
    tweet = re.sub(r"!!![!]*", "!!!", tweet)
    tweet = re.sub(r"\?\?\?[\?]*", "???", tweet)

    # separate
    tweet = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", tweet)
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)

    # URL normalization
    tweet = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
                    "<url>", tweet)

    # Twitter ID normalization
    tweet = re.sub(r"@([A-Za-z0-9_]{1,15})", "<twitter-id>", tweet)

    # Positive emoticon normalization
    tweet = re.sub(r":\)", " <pos-emo>", tweet)
    tweet = re.sub(r":-\)", " <pos-emo>", tweet)
    tweet = re.sub(r":]", " <pos-emo>", tweet)
    tweet = re.sub(r"=]", " <pos-emo>", tweet)

    # Negative emoticon normalization
    tweet = re.sub(r":\(", " <neg-emo>", tweet)
    tweet = re.sub(r":-\(", " <neg-emo>", tweet)
    tweet = re.sub(r":\[", " <neg-emo>", tweet)
    tweet = re.sub(r"=\[", " <neg-emo>", tweet)
    return tweet.strip().lower()


def all_freshmen_tweets_pred(file, model, word_idx_map, sent_len_max, filter_h):
    print "make predictions for all freshmen tweets..."
    tweets = []
    with open(file, "rb") as f_in:
        with open("tweet/all_preds.csv", "wb") as f_out:
            rdr = csv.reader(f_in)
            writer = csv.writer(f_out)
            hdr = True
            for row in rdr:
                if hdr:  # skip header
                    hdr = False
                    row.append("Sentiment (0=pos 1=neutral 2=negative)")
                    writer.writerow(row)
                    continue
                tweet_processed = get_idx_from_sent(clean_tweet(row[3]), word_idx_map, sent_len_max, filter_h)
                input = np.array([tweet_processed], dtype="int")
                pred = model.predict(input)[0]
                row.append(str(pred))
                writer.writerow(row)


if __name__=="__main__":
    execfile("conv_net_classes.py")

    print "load data..."
    x = cPickle.load(open("tweet.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    sent_len_max = np.max(pd.DataFrame(revs)["num_words"])
    train_data = make_idx_data_cv(revs, word_idx_map, sent_len_max, 5)

    model = ConvNetModel(len(train_data[0])-1, [0.0, 0.0], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], W2)
    model.train(train_data, 50, 0)
    all_freshmen_tweets_pred("tweet/justin_data.csv", model, word_idx_map, sent_len_max, 5)

    # # preds = model.predict(np.array([get_idx_from_sent("I hate him", word_idx_map, sent_len_max, 5),
    # #                                 get_idx_from_sent("I love him", word_idx_map, sent_len_max, 5)], dtype="int"))

