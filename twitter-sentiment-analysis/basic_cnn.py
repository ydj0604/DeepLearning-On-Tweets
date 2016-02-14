import tensorflow as tf
import numpy as np


class BasicCNN(object):
    def __init__(self, args):
        # prepare
        self.input_x = tf.placeholder(tf.int32, [None, args.seq_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, args.num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # keep track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # embedding layer
        with tf.name_scope("embedding"):
            if args.use_pretrained_embedding:
                # make an embedding matrix
                embedding_size = len(args.vocabulary_embedding[args.vocabulary_embedding.keys()[0]][1])
                embedding_mat = np.random.rand(args.vocab_size, embedding_size)
                for word, (idx, embedding_vec) in args.vocabulary_embedding.iteritems():
                    embedding_mat[idx] = embedding_vec

                # make a word embedding variable
                W = tf.Variable(
                    tf.convert_to_tensor(embedding_mat, dtype=tf.float32),
                    name="W")

            else:  # use random embeddings
                embedding_size = 256  # use 256 as default
                W = tf.Variable(
                    tf.random_uniform([args.vocab_size, 256], -1.0, 1.0),
                    name="W")

            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # convolution + max-pool layer
        pooled_outputs = []
        for i, filter_size in enumerate(args.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # convolution
                filter_shape = [filter_size, embedding_size, 1, args.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[args.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # max-pooling
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, args.seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # combine pooled features
        num_filters_total = args.num_filters * len(args.filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # dropout layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # output layer (fully-connected layer included)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, args.num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[args.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # calculate accuracy
        with tf.name_scope("accuracy"):
            self.targets = tf.argmax(self.input_y, 1)
            correct_predictions = tf.equal(self.predictions, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # calculate loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
            self.loss = tf.reduce_mean(losses) + args.l2_reg_lambda * l2_loss

        # train and update
        with tf.name_scope("update"):
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            self.grads_and_vars = optimizer.compute_gradients(self.loss)
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.train_op = optimizer.apply_gradients(self.grads_and_vars, global_step=self.global_step)