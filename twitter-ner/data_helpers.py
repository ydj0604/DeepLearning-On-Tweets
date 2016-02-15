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


# convert each token label into a vector
def convert_code_to_vec(code):
    if len(code) == 0:
        print 'No label'
        return -1
    elif code[0] == 'O':
        return 0
    elif code[0] == 'B':
        return 1
    elif code[0] == 'I':
        return 2
    else:
        print 'Wrong label: ' + code
        return -1


def load_twitter_rnn():
    # dataset = list(open("./input/twitter/training_set_05.tsv"))
    dataset = list(open("./input/twitter/training_set_100.tsv"))
    dataset.extend(list(open("./input/alan-ritter.txt")))
    dataset.extend(list(open("./input/mark-dredze-train.txt")))
    dataset.extend(list(open("./input/mark-dredze-test.txt")))

    tokenized_tweets = []
    labels = []
    curr_tweet = []
    curr_label = []
    vocab = {}
    max_len = 0

    for line in dataset:
        line = line.strip()
        if line == '\n' or line == '':
            if len(curr_tweet) > 0:
                # one tweet is complete
                tokenized_tweets.append(curr_tweet)
                labels.append(curr_label)
                max_len = max(max_len, len(curr_tweet))
                curr_tweet = []
                curr_label = []
            continue

        # token
        curr_token = line.split('\t')[0]
        if curr_token == '':
            continue

        # code = token label
        curr_code = convert_code_to_vec(line.split('\t')[1].upper())
        if curr_code == -1:
            continue

        # put the token in vocab, convert it into its idx in vocab
        if curr_token in vocab:
            curr_token_idx = vocab[curr_token]
        else:
            curr_token_idx = len(vocab)
            vocab[curr_token] = curr_token_idx

        # append the token and the code
        curr_tweet.append(curr_token_idx)
        curr_label.append(curr_code)

    if len(curr_tweet) > 0:
        max_len = max(max_len, len(curr_tweet))
        tokenized_tweets.append(curr_tweet)
        labels.append(curr_label)

    # pad tweets to make them the same length
    vocab['<PAD/>'] = len(vocab)
    for i in range(len(tokenized_tweets)):
        tweet = tokenized_tweets[i]
        num_padding = max_len - len(tweet)
        new_tweet = tweet + [vocab['<PAD/>']] * num_padding
        tokenized_tweets[i] = new_tweet
        labels[i] = labels[i] + [0] * num_padding

    # construct vocab_inv
    vocab_inv = [None] * len(vocab)
    for token, idx in vocab.iteritems():
        vocab_inv[idx] = token

    # construct input
    x = np.array(tokenized_tweets)
    y = np.array(labels)

    # I, O, B encoding
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
