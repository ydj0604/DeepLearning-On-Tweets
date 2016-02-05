import numpy as np
import re
import itertools
from collections import Counter
from nltk.tokenize import TweetTokenizer


def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    # string = re.sub(r",", " , ", string)
    # string = re.sub(r"\.", " . ", string)
    # string = re.sub(r"\(", " \( ", string)
    # string = re.sub(r"\)", " \) ", string)
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\?", " \? ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels_semeval():
    # load the entire semeval dataset
    dataset = list(open("./input/2013-dev"))
    dataset.extend(list(open("./input/2013-train")))
    dataset.extend(list(open("./input/2016-train")))

    # filter out netural ones
    dataset = [entry for entry in dataset
               if entry.split('\t')[2] == 'positive' or entry.split('\t')[2] == 'negative']

    # generate x
    tk = TweetTokenizer(reduce_len=True)
    x_text = [entry.split('\t')[3] for entry in dataset]
    x_text = [clean_str(tweet) for tweet in x_text]
    x_text = [tk.tokenize(tweet) for tweet in x_text]

    # generate y
    y = [entry.split('\t')[2] for entry in dataset]
    for idx, label in enumerate(y):
        if label == 'positive':
            y[idx] = [0, 1]
        elif label == 'negative':
            y[idx] = [1, 0]
        else:
            print 'neither pos nor neg'
    return [x_text, y]


def load_data_and_labels_sam():
    # load
    dataset = list(open("./input/2780_freshmen_tweets.csv"))
    dataset = dataset[1:]
    dataset = [entry for entry in dataset
               if entry.split(',')[4] == '1' or entry.split(',')[4] == '3']

    # generate x
    tk = TweetTokenizer(reduce_len=True)
    x_text = [entry.split(',')[3] for entry in dataset]
    x_text = [clean_str(tweet) for tweet in x_text]
    x_text = [tk.tokenize(tweet) for tweet in x_text]

    # generate y
    y = [entry.split(',')[4] for entry in dataset]
    for idx, label in enumerate(y):
        if label == '1':
            y[idx] = [0, 1]
        elif label == '3':
            y[idx] = [1, 0]
        else:
            print 'neither pos nor neg'
    return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    # load train set from Semeval
    sentences, labels = load_data_and_labels_semeval()
    train_size = len(labels)

    # add dev set from Sam
    sentences_dev, labels_dev = load_data_and_labels_sam()
    sentences.extend(sentences_dev)
    labels.extend(labels_dev)

    # preprocess
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, train_size]

def batch_iter(data, batch_size, num_epochs):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]