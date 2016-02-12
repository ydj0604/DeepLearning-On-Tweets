import tensorflow as tf
import argparse
import os
import numpy as np
import datetime

from jin_rnn import jin_rnn
import data_helpers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_size', type=int, default=128,
                       help='size of RNN hidden state = embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')
    parser.add_argument('--evaluate_every', type=int, default=200,
                       help='development test frequency')
    parser.add_argument('--save_every', type=int, default=500,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                       help='decay rate for rmsprop')
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    args = parser.parse_args()
    train(args)


def train(args):
    # load data
    [x, y, vocab, vocab_inv, num_classes] = data_helpers.load_twitter_rnn()

    # shuffle and split data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_size = len(y) * args.dev_ratio
    x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
    y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]

    # fill out missing arg values
    args.vocab_size = len(vocab)
    args.seq_length = len(x[0])
    args.num_classes = num_classes

    # initialize a rnn model
    model = jin_rnn(args, False)

    # generate batches from data
    batches = data_helpers.batch_iter(x_train, y_train, args.batch_size, args.num_epochs)

    # train
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        tf.initialize_all_variables().run()
        for x_batch, y_batch in batches:
            # measure start time
            time_str = datetime.datetime.now().isoformat()

            # train
            feed ={
                model.input_data: x_batch,
                model.targets: y_batch
            }
            current_step, train_loss, state, _ = sess.run([model.global_step, model.cost, model.final_state, model.train_op], feed)
            print("{}: step {}, loss {:g}".format(time_str, current_step, train_loss))

            # evaluate with dev set
            if current_step % args.evaluate_every == 0:
                print("\nEvaluation:")
            if current_step % args.save_every == 0:
                out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))
                checkpoint_dir = os.path.join(out_dir, 'checkpoints')
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print "Saved model checkpoint to {}\n".format(path)


if __name__ == '__main__':
    main()
