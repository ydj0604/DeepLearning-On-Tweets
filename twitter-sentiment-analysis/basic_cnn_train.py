import tensorflow as tf
import argparse
import os
import datetime
import data_helpers
from basic_cnn import BasicCNN
from deep_cnn import DeepCNN
import sys


def main():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                        help='Comma-separated filter sizes')
    parser.add_argument('--num_filters', type=int, default=128,
                        help='Number of filters per filter size (default: 256)')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability (default: 0.5)')
    parser.add_argument('--l2_reg_lambda', type=float, default=5.0,
                        help='L2 regularizaion lambda (default: 0.0)')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--evaluate_every', type=int, default=100,
                        help='Evaluate model on dev set after this many steps (default: 200)')
    parser.add_argument('--checkpoint_every', type=int, default=100,
                        help='Save model after this many steps (default: 500)')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate for optimizer (default: 1e-4)')
    parser.add_argument('--use_pretrained_embedding', type=int, default=1,
                        help='Use pre-trained word embeddings')

    # misc parameters
    parser.add_argument('--allow_soft_placement', type=int, default=1,
                        help='Allow device soft device placement')
    parser.add_argument('--log_device_placement', type=int, default=0,
                        help='Log placement of ops on devices')
    parser.add_argument('--save_dir', type=str, default='runs',
                       help='directory to store checkpointed models')
    parser.add_argument('--train', type=int, default=1,
                       help='train from scratch')
    parser.add_argument('--model', type=str, default='deep',
                       help='which model to run')
    parser.add_argument('--data', type=str, default='semeval',
                       help='which data to run')

    args = parser.parse_args()

    # report parameters
    print("\nParameters:")
    for arg in args.__dict__:
        print("{}={}".format(arg.upper(), args.__dict__[arg]))
    print("")

    # start training
    initiate(args)


def initiate(args):
    # load data
    print("Loading data...")
    x_train, y_train, x_dev, y_dev, vocabulary, vocabulary_inv, vocabulary_embedding = \
        data_helpers.load_data_semeval_only(args.use_pretrained_embedding) if args.data == 'semeval' \
        else data_helpers.load_data(args.use_pretrained_embedding)
    num_classes = len(y_train[0])

    # report
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    # fill out missing arg values
    args.seq_length = x_train.shape[1]
    args.vocab_size = len(vocabulary)
    args.filter_sizes = map(int, args.filter_sizes.split(","))
    args.vocabulary_embedding = vocabulary_embedding
    args.num_classes = num_classes

    # initialize a model
    if args.model == 'deep':
        model = DeepCNN(args)
    elif args.model == 'basic':
        model = BasicCNN(args)
    else:
        print("Invalid model")
        sys.exit()

    # define output directory
    time_str = datetime.datetime.now().isoformat()
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))

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
    def train_model(x, y, dropout_prob, writer):
        feed_dict = {
          model.input_x: x,
          model.input_y: y,
          model.dropout_keep_prob: dropout_prob
        }
        _, step, loss, accuracy, summaries = sess.run(
            [model.train_op, model.global_step, model.loss, model.accuracy, train_summary_op],
            feed_dict)
        writer.add_summary(summaries, step)
        # time_str = datetime.datetime.now().isoformat()
        # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def test_model(x, y):
        print("\nEvaluate:")
        feed_dict = {
          model.input_x: x,
          model.input_y: y,
          model.dropout_keep_prob: 1.0
        }
        step, loss, accuracy, predictions, targets = sess.run(
                [model.global_step, model.loss, model.accuracy, model.predictions, model.targets],
                feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        print("")

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        # initialize
        tf.initialize_all_variables().run()
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)

        if args.train:  # train the model from scratch
            for x_batch, y_batch in batches:
                # train
                train_model(x_batch, y_batch, args.dropout_keep_prob, train_summary_writer)
                current_step = tf.train.global_step(sess, model.global_step)

                # evaluate with dev set
                if current_step % args.evaluate_every == 0:
                    test_model(x_dev, y_dev)

                # save model
                if current_step % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

        else:  # load the model
            print 'Loading the model...'


if __name__ == '__main__':
    main()
