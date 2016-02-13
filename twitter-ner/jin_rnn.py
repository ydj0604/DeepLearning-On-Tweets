import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class jin_rnn(object):
    def __init__(self, args):

        cell_fn = rnn_cell.BasicRNNCell
        cell = cell_fn(args.rnn_size)
        cell = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            W = tf.get_variable("W", [args.rnn_size, args.num_classes])
            b = tf.get_variable("b", [args.num_classes])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("word_embedding", [args.vocab_size, args.rnn_size])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        outputs, states = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.rnn_size])
        self.logits = tf.nn.xw_plus_b(output, W, b)
        self.probs = tf.nn.softmax(self.logits)

        # calculate token-level accuracy
        self.predictions = tf.cast(tf.argmax(self.logits, 1), tf.int32)
        self.reshaped_targets = tf.reshape(self.targets, [-1])
        correct_predictions = tf.equal(self.predictions, self.reshaped_targets)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

        # calculate sentence-level accuracy
        self.predictions_sentence = tf.reshape(self.predictions, [-1, args.seq_length]) # batch_size * seq_length
        correct_predictions_sentence_tokens = tf.equal(self.predictions_sentence, self.targets)  # batch_size X seq_length
        multiply_mat = tf.constant(1, shape=[args.seq_length, 1])
        sentence_accuracy_mat = tf.matmul(tf.cast(correct_predictions_sentence_tokens, tf.int32), multiply_mat)  # batch_size X 1
        correct_predictions_sentence = \
            tf.equal(sentence_accuracy_mat, tf.constant(args.seq_length, shape=[args.batch_size, 1]))  # batch_size X 1
        self.accuracy_sentence = tf.reduce_mean(tf.cast(correct_predictions_sentence, "float"))

        # calculate loss
        self.loss = seq2seq.sequence_loss_by_example(
                [self.logits],  # TODO: should I use a list of 2D tensors ?
                [self.reshaped_targets],  # TODO: correct ???
                [tf.ones([args.batch_size * args.seq_length])],
                args.num_classes
        )
        self.cost = tf.reduce_sum(self.loss) / args.batch_size / args.seq_length

        # update
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)  # TODO: correct ???
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)