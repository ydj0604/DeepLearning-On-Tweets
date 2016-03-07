import tensorflow as tf
import argparse
import os
import numpy as np
import datetime

from jin_rnn import JinRNN
import data_helpers


def main():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state (if pre-trained embedding is not used)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='rnn',
                       help='rnn or lstm')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--evaluate_every', type=int, default=100,
                       help='development test frequency')
    parser.add_argument('--save_every', type=int, default=500,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--num_folds', type=int, default=10,
                       help='the number of folds to be used')
    parser.add_argument('--l2_limit', type=float, default=3.0,
                        help='L2 norm limit')

    # misc parameters
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    parser.add_argument('--allow_soft_placement', type=int, default=1,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=int, default=0,
                        help='Log placement of ops on devices')

    args = parser.parse_args()

    # report parameters
    print "\nParameters:"
    for arg in args.__dict__:
        print("{}={}".format(arg.upper(), args.__dict__[arg]))
    print ""

    # start training
    train(args)


def split_into_k_folds(x, y, k=10):
    # shuffle first
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x = x[shuffle_indices]
    y = y[shuffle_indices]

    # split into folds
    fold_size = len(y) / 10
    x_folds = []
    y_folds = []
    for i in range(k):
        if i == k-1:
            fold_x = x[i*fold_size:]
            fold_y = y[i*fold_size:]
        else:
            fold_x = x[i*fold_size: (i+1)*fold_size]
            fold_y = y[i*fold_size: (i+1)*fold_size]
        x_folds.append(fold_x)
        y_folds.append(fold_y)
    return x_folds, y_folds


def train(args):
    # load data
    print "Loading data ..."
    x, y, vocab, vocab_inv, emb_vocab, num_classes = data_helpers.load_twitter_rnn()

    # split into k folds
    x_folds, y_folds = split_into_k_folds(x, y, args.num_folds)

    # fill out missing arg values
    args.vocab_size = len(vocab)
    args.seq_length = len(x[0])
    args.num_classes = num_classes
    if emb_vocab is not None:
        args.emb_vocab = emb_vocab
        args.rnn_size = len(emb_vocab[emb_vocab.keys()[0]][1])
    else:
        args.emb_vocab = None

    # report
    print "Vocabulary Size: {:d}".format(len(vocab))
    print "Total/fold size: {:d}/{:d}".format(len(y), len(x_folds[0]))
    print "Sequence Length: {:d}".format(len(y[0]))

    # initialize a rnn model
    model = JinRNN(args)

    # define output directory
    time_str = datetime.datetime.now().isoformat()
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))

    # prepare saver
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.all_variables())

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():

        # final results
        test_token_acc_list = []
        test_sentence_acc_list = []

        # for each fold
        for i in range(args.num_folds):
            # initialize
            tf.initialize_all_variables().run()

            # use ith fold as a testset, and the rest as a trainset
            x_test, y_test = x_folds[i], y_folds[i]
            x_train, y_train = np.array([]), np.array([])
            for j in range(args.num_folds):
                if j != i:
                    x_train = x_folds[j] if len(x_train) == 0 else np.concatenate((x_train, x_folds[j]), axis=0)
                    y_train = y_folds[j] if len(y_train) == 0 else np.concatenate((y_train, y_folds[j]), axis=0)

            print "Fold #{} Train/Test Size: {}/{}".format(i, len(y_train), len(y_test))

            # generate batches
            train_batches = data_helpers.batch_iter(x_train, y_train, args.batch_size, args.num_epochs)
            test_batches = data_helpers.batch_iter(x_test, y_test, args.batch_size, args.num_epochs)

            # current fold results
            curr_best_sentene_acc = 0.0
            curr_best_token_acc = 0.0

            # for each batch
            for x_train_batch, y_train_batch in train_batches:
                # obtain start time
                time_str = datetime.datetime.now().isoformat()

                # train
                feed = {model.inputs: x_train_batch,
                        model.targets: y_train_batch,
                        model.dropout_keep_prob: args.dropout_keep_prob}
                current_step, train_loss, _ = sess.run([model.global_step, model.cost, model.train_op], feed)
                sess.run(model.weight_clipping_op, feed)  # rescale weight
                # print "{}: step {}, loss {:g}".format(time_str, current_step, train_loss)

                # evaluate with test set
                if current_step % args.evaluate_every == 0:
                    print "\nEvaluation"
                    sum_accuracy = 0.0
                    sum_accuracy_sentence = 0.0
                    num_batches = 0

                    for x_test_batch, y_test_batch in test_batches:
                        feed = {model.inputs: x_test_batch,
                                model.targets: y_test_batch,
                                model.dropout_keep_prob: 1.0}
                        current_step, accuracy, accuracy_sentence, predictions_sentence, loss = sess.run(
                                [model.global_step, model.accuracy, model.accuracy_sentence, model.predictions_sentence,
                                 model.cost],
                                feed)

                        # for i in range(len(y_dev_batch)):
                        #     curr_sentence = x_dev_batch[i]
                        #     curr_target_codes = y_dev_batch[i]
                        #     curr_predicted_codes = predictions_sentence[i]
                        #
                        #     # to see if the model predicts some difficult examples correctly
                        #     if ((1 in list(curr_predicted_codes) or 2 in list(curr_predicted_codes))
                        #         and list(curr_predicted_codes) == list(curr_target_codes)):
                        #         print ' '.join([vocab_inv[e] for e in curr_sentence])
                        #         print curr_target_codes
                        #         print curr_predicted_codes

                        # print "{}: step {}, token-accuracy {:g}, sentence-accuracy {:g}"\
                        #     .format(time_str, current_step, accuracy, accuracy_sentence)

                        sum_accuracy += accuracy
                        sum_accuracy_sentence += accuracy_sentence
                        num_batches += 1

                    print "{}: step {}, token-accuracy {:g}, sentence-accuracy {:g}, loss {:g}\n".format(
                            time_str, current_step, sum_accuracy/num_batches, sum_accuracy_sentence/num_batches, loss)

                    # set the best result for the current fold
                    curr_best_sentene_acc = max(curr_best_sentene_acc, sum_accuracy_sentence/num_batches)
                    curr_best_token_acc = max(curr_best_token_acc, sum_accuracy/num_batches)

                # save the model
                if current_step % args.save_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print "Saved model checkpoint to {}\n".format(path)

            print "-------------------------------------------------------------"
            print "Fold #{} RESULTS: token-accuracy {:g}, sentence-accuracy {:g}"\
                .format(i, curr_best_token_acc, curr_best_sentene_acc)
            print "-------------------------------------------------------------"

            # add to the results list
            test_sentence_acc_list.append(curr_best_sentene_acc)
            test_token_acc_list.append(curr_best_token_acc)

        print "=========================================================="
        print "FINAL RESULTS: token-accuracy {:g}, sentence-accuracy {:g}"\
            .format(np.mean(test_token_acc_list), np.mean(test_sentence_acc_list))
        print "=========================================================="


if __name__ == '__main__':
    main()
