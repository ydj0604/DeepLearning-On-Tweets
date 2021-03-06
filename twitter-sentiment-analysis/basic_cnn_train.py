import tensorflow as tf
import argparse
import os
import datetime
import data_helpers
from basic_cnn import BasicCNN
from deep_cnn import DeepCNN
from logger import Logger
import sys


def main():
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--filter_sizes', type=str, default='3,4,5',
                        help='Comma-separated filter sizes')
    parser.add_argument('--num_filters', type=int, default=100,
                        help='Number of filters per filter size')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5,
                        help='Dropout keep probability (default: 0.5)')
    parser.add_argument('--l2_reg_lambda', type=float, default=0.5,
                        help='L2 regularizaion lambda (default: 0.5)')
    parser.add_argument('--l2_limit', type=float, default=3.0,
                        help='L2 norm limit for rescaling')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size (default: 64)')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
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
    parser.add_argument('--model', type=str, default='basic',
                       help='which model to run')
    parser.add_argument('--data', type=str, default='gameforum',
                       help='which data to run')

    args = parser.parse_args()

    # start training
    initiate(args)


def initiate(args):
    # define output directory
    time_str = datetime.datetime.now().isoformat()
    out_dir = os.path.abspath(os.path.join(os.path.curdir, args.save_dir, time_str))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # initiate logger
    log_file_path = os.path.join(out_dir, 'log')
    logger = Logger(log_file_path)
    analysis_file_path = os.path.join(out_dir, 'analysis')
    analysis_logger = Logger(analysis_file_path)

    # report parameters
    logger.write("\nParameters:")
    for arg in args.__dict__:
        logger.write("{}={}".format(arg.upper(), args.__dict__[arg]))
    logger.write("")

    # load data
    logger.write("Loading data...")
    if args.data == 'gameforum':
        x_train, y_train, x_dev, y_dev, vocabulary, vocabulary_inv, vocabulary_embedding = data_helpers.load_data_gameforum_only(args.use_pretrained_embedding);
    elif args.data == 'semeval':
        x_train, y_train, x_dev, y_dev, vocabulary, vocabulary_inv, vocabulary_embedding = data_helpers.load_data_semeval_only(args.use_pretrained_embedding)
    else:
        x_train, y_train, x_dev, y_dev, vocabulary, vocabulary_inv, vocabulary_embedding = data_helpers.load_data(args.use_pretrained_embedding)
    num_classes = len(y_train[0])

    # report
    logger.write("Vocabulary Size: {:d}".format(len(vocabulary)))
    logger.write("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

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
        sess.run(model.weight_rescaling_op)  # l2 norm rescaling
        writer.add_summary(summaries, step)
        if log:
            time_str = datetime.datetime.now().isoformat()
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
        time_str = datetime.datetime.now().isoformat()
        logger.write("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        logger.write("")
        return accuracy, predictions, targets

    # start a session
    sess_conf = tf.ConfigProto(
            allow_soft_placement=args.allow_soft_placement,
            log_device_placement=args.log_device_placement)
    sess = tf.Session(config=sess_conf)
    with sess.as_default():
        # initialize
        tf.initialize_all_variables().run()
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
        current_step = 0

        if args.train:  # train the model from scratch
            best_test_accuracy = 0.0
            for x_batch, y_batch in batches:
                # train
                train_model(x_batch, y_batch, args.dropout_keep_prob, train_summary_writer,
                            current_step % (args.evaluate_every/4) == 0)
                current_step = tf.train.global_step(sess, model.global_step)

                # evaluate with dev set
                if current_step % args.evaluate_every == 0:
                    accuracy, predictions, targets = test_model(x_dev, y_dev)

                    # Conduct analysis if the current model is the best so far
                    if accuracy > best_test_accuracy:
                        best_test_accuracy = accuracy
                        analysis_logger.write("Analysis at {}: acc={}".format(current_step, accuracy), begin=True)
                        analysis_logger.write("Tweet\tPred\tTrue (0=Positive, 1=Neutral, 2=Negative)")
                        for i in range(len(x_dev)):
                            tweet_idx = x_dev[i]
                            prediction, true_label = predictions[i], targets[i]
                            try:
                                tweet = " ".join([vocabulary_inv[word_idx] for word_idx in tweet_idx if word_idx != 0])
                                analysis_logger.write("{}\t{}\t{}".format(tweet, prediction, true_label))
                            except UnicodeEncodeError:
                                analysis_logger.write("{}\t{}\t{}".format("ENCODING ERROR", prediction, true_label))
                        analysis_logger.write("\n")

                # save model
                if current_step % args.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    logger.write("Saved model checkpoint to {}\n".format(path))

        else:  # load the model
            logger.write("Loading the model...")


if __name__ == '__main__':
    main()
