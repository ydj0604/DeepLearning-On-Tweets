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
                       help='size of RNN hidden state = embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=50,
                       help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=200,
                       help='number of epochs')
    parser.add_argument('--evaluate_every', type=int, default=50,
                       help='development test frequency')
    parser.add_argument('--save_every', type=int, default=500,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='learning rate')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                       help='decay rate for rmsprop')

    # misc parameters
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    parser.add_argument('--allow_soft_placement', type=int, default=1,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=int, default=0,
                        help='Log placement of ops on devices')

    args = parser.parse_args()

    # report parameters
    print("\nParameters:")
    for arg in args.__dict__:
        print("{}={}".format(arg.upper(), args.__dict__[arg]))
    print("")

    # start training
    train(args)


def train(args):
    # load data
    print("Loading data ...")
    [x, y, vocab, vocab_inv, num_classes] = data_helpers.load_twitter_rnn()

    # shuffle and split data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    dev_size = int(len(y) * args.dev_ratio) / 50 * 50
    x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
    y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]

    # fill out missing arg values
    args.vocab_size = len(vocab)
    args.seq_length = len(x[0])
    args.num_classes = num_classes

    # report
    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    print("Sequence Length: {:d}".format(len(y_train[0])))

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

    # generate batches from data
    batches = data_helpers.batch_iter(x_train, y_train, args.batch_size, args.num_epochs)

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        # initialize
        tf.initialize_all_variables().run()

        for x_batch, y_batch in batches:
            # obtain start time
            time_str = datetime.datetime.now().isoformat()

            # train
            feed = {model.inputs: x_batch, model.targets: y_batch}
            current_step, train_loss, _ = sess.run([model.global_step, model.cost, model.train_op], feed)
            # print("{}: step {}, loss {:g}".format(time_str, current_step, train_loss))

            # evaluate with dev set
            if current_step % args.evaluate_every == 0:
                print("\nEvaluation")
                dev_batches = data_helpers.batch_iter(x_dev, y_dev, args.batch_size, 1)
                sum_accuracy = 0.0
                sum_accuracy_sentence = 0.0
                num_batches = 0

                for x_dev_batch, y_dev_batch in dev_batches:
                    feed = {model.inputs: x_dev_batch, model.targets: y_dev_batch}
                    current_step, accuracy, accuracy_sentence, predictions_sentence, loss = sess.run(
                            [model.global_step, model.accuracy, model.accuracy_sentence, model.predictions_sentence,
                             model.cost],
                            feed)

                    for i in range(len(y_dev_batch)):
                        curr_sentence = x_dev_batch[i]
                        curr_target_codes = y_dev_batch[i]
                        curr_predicted_codes = predictions_sentence[i]

                        # to see if the model predicts some difficult examples correctly
                        if ((1 in list(curr_predicted_codes) or 2 in list(curr_predicted_codes))
                            and list(curr_predicted_codes) == list(curr_target_codes)):
                            print ' '.join([vocab_inv[e] for e in curr_sentence])
                            print curr_target_codes
                            print curr_predicted_codes

                    sum_accuracy += accuracy
                    sum_accuracy_sentence += accuracy_sentence
                    num_batches += 1

                print("{}: step {}, token-accuracy {:g}, sentence-accuracy {:g}, loss {:g}\n".format(
                        time_str, current_step, sum_accuracy/num_batches, sum_accuracy_sentence/num_batches, loss))

            # save the model
            if current_step % args.save_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print "Saved model checkpoint to {}\n".format(path)


if __name__ == '__main__':
    main()
