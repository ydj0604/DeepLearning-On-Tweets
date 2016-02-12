import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class jin_rnn(object):
    def __init__(self, args, infer):

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
        self.loss = seq2seq.sequence_loss_by_example(
                [self.logits],  # TODO: should I use a list of 2D tensors ?
                [tf.reshape(self.targets, [-1])],  # TODO: correct ???
                [tf.ones([args.batch_size * args.seq_length])],
                args.num_classes
        )
        self.cost = tf.reduce_sum(self.loss) / args.batch_size / args.seq_length
        self.final_state = states[-1]
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)  # TODO: correct ???
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
