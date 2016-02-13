import numpy as np
import itertools
from collections import Counter
import sys

# TODO
# handle @
# handle #
# handle upper-case
# handle url
# handle emoticons


def load_twitter_rnn():
    # dataset = list(open("./input/twitter/training_set_05.tsv"))
    dataset = list(open("./input/twitter/training_set_100.tsv"))
    dataset.extend(list(open("./input/alan-ritter.txt")))
    # dataset.extend(list(open("./input/mark-dredze-train.txt")))
    # dataset.extend(list(open("./input/mark-dredze-test.txt")))

    tokenized_tweets = []
    labels = []
    curr_tweet = []
    curr_label = []

    for line in dataset:
        if line == '\n':
            if len(curr_tweet) > 0:
                tokenized_tweets.append(curr_tweet)
                labels.append(curr_label)
                curr_tweet = []
                curr_label = []
            continue
        curr_token = line.split('\t')[0]
        curr_code = line.split('\t')[1][0].upper() # IOB
        curr_tweet.append(curr_token)
        curr_label.append(curr_code)

    if len(curr_tweet) > 0:
        tokenized_tweets.append(curr_tweet)
        labels.append(curr_label)

    # pad tweets
    max_len = max([len(tweet) for tweet in tokenized_tweets])
    for i in range(len(tokenized_tweets)):
        tweet = tokenized_tweets[i]
        num_padding = max_len - len(tweet)
        new_tweet = tweet + ['<PAD/>'] * num_padding
        tokenized_tweets[i] = new_tweet
        labels[i] = labels[i] + ['O'] * num_padding

    # convert each token label into a vector
    def convert_code_to_vec(code):
        if code == 'O':
            return 0
        elif code == 'B':
            return 1
        elif code == 'I':
            return 2
        else:
            print 'Wrong label'
            sys.exit()
    labels = [[convert_code_to_vec(curr_code) for curr_code in label] for label in labels]

    # build vocab
    word_counts = Counter(itertools.chain(*tokenized_tweets))
    vocab_inv = [x[0] for x in word_counts.most_common()]
    vocab = {x: i for i, x in enumerate(vocab_inv)}

    # construct input
    x = np.array([[vocab[word] for word in tweet] for tweet in tokenized_tweets])
    y = np.array(labels)

    num_classes = 3

    return [x, y, vocab, vocab_inv, num_classes]


def batch_iter(x, y, batch_size, num_epochs):
    data_size = len(y)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = x[shuffle_indices]
        shuffled_y = y[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            if end_index <= data_size:
                yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]
