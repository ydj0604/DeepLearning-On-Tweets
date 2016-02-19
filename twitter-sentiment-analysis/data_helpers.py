import numpy as np
import re
import itertools
from collections import Counter
from nltk.tokenize import TweetTokenizer
import csv
import os.path


def clean_str(string):
    # separates helping verbs
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)

    # URL normalization
    string = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
                    "<url>", string)

    # Twitter ID normalization
    string = re.sub(r"@([A-Za-z0-9_]{1,15})", "<twitter-id>", string)

    # Positive emoticon normalization
    string = re.sub(r":\)", " <pos-emo>", string)
    string = re.sub(r":-\)", " <pos-emo>", string)
    string = re.sub(r":]", " <pos-emo>", string)
    string = re.sub(r"=]", " <pos-emo>", string)

    # Negative emoticon normalization
    string = re.sub(r":\(", " <neg-emo>", string)
    string = re.sub(r":-\(", " <neg-emo>", string)
    string = re.sub(r":\[", " <neg-emo>", string)
    string = re.sub(r"=\[", " <neg-emo>", string)

    return string.strip().lower()


def load_data_and_labels_semeval():
    # load the entire semeval dataset
    old_dataset = list(open("./input/2013-dev"))
    old_dataset.extend(list(open("./input/2013-devtest")))
    old_dataset.extend(list(open("./input/2013-train")))
    old_dataset.extend(list(open("./input/2014-devtest")))

    new_dataset = list(open("./input/2016-train"))
    new_dataset.extend(list(open("./input/2016-dev")))
    new_dataset.extend(list(open("./input/2016-devtest")))

    # filter out invalid tweets from new dataset
    new_dataset = [entry for entry in new_dataset if entry.split('\t')[2] != 'Not Available\n']

    # generate x from old
    tk = TweetTokenizer(reduce_len=True) # handles punctuations
    x_text = [entry.split('\t')[3] for entry in old_dataset]
    x_text = [clean_str(tweet) for tweet in x_text]
    x_text = [tk.tokenize(tweet) for tweet in x_text]

    # generate x from new
    x_text_new = [entry.split('\t')[2] for entry in new_dataset]
    x_text_new = [clean_str(tweet) for tweet in x_text_new]
    x_text_new = [tk.tokenize(tweet) for tweet in x_text_new]

    # concat x and x_new
    x_text.extend(x_text_new)

    # generate y from old
    y = [entry.split('\t')[2] for entry in old_dataset]
    for idx, label in enumerate(y):
        if label == 'positive':
            y[idx] = [1, 0, 0]
        elif label == 'neutral':
            y[idx] = [0, 1, 0]
        elif label == 'negative':
            y[idx] = [0, 0, 1]
        else:
            print 'wrong label in semeval: ' + label

    # generate y from new
    y_new = [entry.split('\t')[1] for entry in new_dataset]
    for idx, label in enumerate(y_new):
        if label == 'positive':
            y_new[idx] = [1, 0, 0]
        elif label == 'neutral':
            y_new[idx] = [0, 1, 0]
        elif label == 'negative':
            y_new[idx] = [0, 0, 1]
        else:
            print 'wrong label in semeval: ' + label

    # concat y and y_new
    y.extend(y_new)

    return [x_text, y]


def load_data_and_labels_sam():
    # load
    with open("./input/2780_freshmen_tweets.csv", 'rU') as f:
        rdr = csv.reader(f)
        dataset = list(rdr)[1:]  # remove header

    # filter out tweets with unknown sentiment
    dataset = [entry for entry in dataset if entry[4] != '0']

    # generate x
    tk = TweetTokenizer(reduce_len=True)
    x_text = [entry[3] for entry in dataset]
    x_text = [clean_str(tweet) for tweet in x_text]
    x_text = [tk.tokenize(tweet) for tweet in x_text]

    # generate y
    y = [entry[4] for entry in dataset]
    for idx, label in enumerate(y):
        if label == '1': # positive
            y[idx] = [1, 0, 0]
        elif label == '2': # neutral
            y[idx] = [0, 1, 0]
        elif label == '3': # negative
            y[idx] = [0, 0, 1]
        else:
            print 'wrong label in sam: ' + label

    return [x_text, y]


def pad_sentences(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_vocab_embedding(vocab):
    # define files
    original_embedding_file = "../twitter-vectors-256-skip.txt"
    reduced_embedding_file = "twitter-sentiment-dictionary-embedding.txt"

    # try to load the reduced embedding file, or load the original
    print("Loading pre-trained embeddings...")
    if os.path.isfile(reduced_embedding_file):
        embedding_file = open(reduced_embedding_file)
    else:
        if os.path.isfile(original_embedding_file):
            embedding_file = open(original_embedding_file)
        else:
            return None

    # read the first line to get info
    first_line = embedding_file.readline()
    dict_size = int(first_line.split(' ')[0])
    embedding_dim = int(first_line.split(' ')[1])

    # build a new vocab that contains embeddings
    new_vocab = {}
    for i in range(dict_size):
        curr_line = embedding_file.readline()
        curr_word = curr_line.split(' ')[0]
        if curr_word in vocab:
            embedding_vec = curr_line.split(' ')[1:1+embedding_dim]
            try:
                embedding_vec = [float(e) for e in embedding_vec]
            except ValueError:
                continue
            new_vocab[curr_word] = (vocab[curr_word], embedding_vec)

    for word, idx in vocab.iteritems():
        if word not in new_vocab:
            new_vocab[word] = (idx, [0.0] * embedding_dim)

    # save reduced embedding file
    if not os.path.isfile("twitter-sentiment-dictionary-embedding.txt"):
        save_file = open("twitter-sentiment-dictionary-embedding.txt", 'w')
        save_file.write("{} {}\n".format(len(new_vocab), embedding_dim))
        for word, (idx, embedding_vec) in new_vocab.iteritems():
            save_file.write("{} {}\n".format(word.encode('utf-8'), ' '.join([str(e) for e in embedding_vec])))

    return new_vocab


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(use_pretrained_embedding):
    # load train set from Semeval
    tweets_train, labels_train = load_data_and_labels_semeval()

    # load dev set from Sam's
    tweets_dev, labels_dev = load_data_and_labels_sam()

    # pad train and dev tweets
    max_seq_len = max([len(tweet) for tweet in tweets_train])
    max_seq_len = max([max_seq_len] + [len(tweet) for tweet in tweets_dev])
    tweets_padded_train = pad_sentences(tweets_train, max_seq_len)
    tweets_padded_dev = pad_sentences(tweets_dev, max_seq_len)

    # build vocab
    tweets_padded_total = tweets_padded_train + tweets_padded_dev
    vocabulary, vocabulary_inv = build_vocab(tweets_padded_total)
    vocabulary_embedding = build_vocab_embedding(vocabulary) if use_pretrained_embedding else None

    # prepare input
    x_train, y_train = build_input_data(tweets_padded_train, labels_train, vocabulary)
    x_dev, y_dev = build_input_data(tweets_padded_dev, labels_dev, vocabulary)
    return [x_train, y_train, x_dev, y_dev, vocabulary, vocabulary_inv, vocabulary_embedding]


def batch_iter(x, y, batch_size, num_epochs):
    data_size = len(y)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_x = x[shuffle_indices]
        shuffled_y = y[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_x[start_index:end_index], shuffled_y[start_index:end_index]