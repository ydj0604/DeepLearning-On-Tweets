import tensorflow as tf
import argparse
import os
from datetime import datetime
import data_helpers
from basic_cnn import BasicCNN
from deep_cnn import DeepCNN
from logger import Logger
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                        help='Comma-separated filter sizes')
    parser.add_argument('--num_filters', type=int, default=128,
                        help='Number of filters per filter size (default: 256)')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability (default: 0.5)')
    parser.add_argument('--l2_reg_lambda', type=float, default=3.0,
                        help='L2 regularizaion lambda (default: 3.0)')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--evaluate_every', type=int, default=200,
                        help='Evaluate model on dev set after this many steps (default: 200)')
    parser.add_argument('--checkpoint_every', type=int, default=400,
                        help='Save model after this many steps (default: 400)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--use_pretrained_embedding', type=int, default=1,
                        help='Use pre-trained word embeddings')
    parser.add_argument('--dev_ratio', type=float, default=0.1,
                       help='decay rate for rmsprop')

    # misc parameters
    parser.add_argument('--allow_soft_placement', type=int, default=1,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=int, default=0,
                        help='Log placement of ops on devices')
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    parser.add_argument('--train', type=int, default=1,
                       help='train from scratch')
    parser.add_argument('--model', type=str, default='basic',
                       help='which model to run')
    parser.add_argument('--data', type=str, default='amazon',
                       help='which data to run')

    args = parser.parse_args()

    # start training
    initiate(args)


def initiate(args):
    # define output directory
    time_str = datetime.now().strftime('%H-%M-%b-%d-%Y')
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # initiate logger
    log_file_path = os.path.join(out_dir, 'log')
    logger = Logger(log_file_path)

    # report parameters
    logger.write("\nParameters:")
    for arg in args.__dict__:
        logger.write("{}={}".format(arg.upper(), args.__dict__[arg]))
    logger.write("")

    # load data
    logger.write("Loading data...")
    x_all, y_all, vocabulary, vocabulary_inv, vocabulary_embedding = data_helpers.load_data(args.use_pretrained_embedding)

    # shuffle and split data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y_all)))
    x_shuffled = x_all[shuffle_indices]
    y_shuffled = y_all[shuffle_indices]
    dev_size = int(len(y_all) * args.dev_ratio)
    x_train, x_dev = x_shuffled[:-dev_size], x_shuffled[-dev_size:]
    y_train, y_dev = y_shuffled[:-dev_size], y_shuffled[-dev_size:]

    # report
    logger.write("Vocabulary Size: {:d}".format(len(vocabulary)))
    logger.write("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # fill out missing arg values
    args.seq_length = x_train.shape[1]
    args.num_classes = y_train.shape[1]  # len(y[0])
    args.vocab_size = len(vocabulary)
    args.filter_sizes = map(int, args.filter_sizes.split(","))
    args.vocabulary_embedding = vocabulary_embedding

    # initialize a model
    if args.model == 'basic':
        model = BasicCNN(args)
    elif args.model == 'deep':
        model = DeepCNN(args)
    else:
        logger.write("Invalid model")
        sys.exit()

    # for train summary
    grad_summaries = []
    for g, v in model.grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.merge_summary(grad_summaries)

    loss_summary = tf.scalar_summary("loss", model.loss)
    acc_summary = tf.scalar_summary("accuracy", model.accuracy)

    train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")

    # prepare saver
    checkpoint_dir = os.path.join(out_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    saver = tf.train.Saver(tf.all_variables())

    # generate batches
    batches = data_helpers.batch_iter(x_train, y_train, args.batch_size, args.num_epochs)

    # define train / test methods
    def train_model(x, y, dropout_prob, writer, log=False):
        feed_dict = {
          model.input_x: x,
          model.input_y: y,
          model.dropout_keep_prob: dropout_prob
        }
        _, step, loss, accuracy, summaries = sess.run(
            [model.train_op, model.global_step, model.loss, model.accuracy, train_summary_op],
            feed_dict)
        writer.add_summary(summaries, step)
        if log:
            time_str = datetime.now().isoformat()
            logger.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step-1, loss, accuracy))

    def test_model(x, y):
        logger.write("\nEvaluate:")
        feed_dict = {
          model.input_x: x,
          model.input_y: y,
          model.dropout_keep_prob: 1.0
        }
        step, loss, accuracy, predictions, targets = sess.run(
                [model.global_step, model.loss, model.accuracy, model.predictions, model.targets],
                feed_dict)
        time_str = datetime.now().isoformat()
        logger.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        logger.write("")

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement, log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        # initialize
        tf.initialize_all_variables().run()
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
        current_step = 0

        if args.train:  # train the model from scratch
            for x_batch, y_batch in batches:
                # train
                train_model(x_batch, y_batch, args.dropout_keep_prob, train_summary_writer,
                            current_step % (args.evaluate_every/4) == 0)
                current_step = tf.train.global_step(sess, model.global_step)

                # evaluate with dev set
                if current_step % args.evaluate_every == 0:
                    test_model(x_dev, y_dev)

                # save model
                if current_step % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.write("Saved model checkpoint to {}\n".format(path))

        else:  # load the model
            logger.write("Loading the model...")


if __name__ == '__main__':
    main()
