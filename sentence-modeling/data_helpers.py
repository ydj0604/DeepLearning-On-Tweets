import numpy as np
import itertools
from collections import Counter
from nltk.tokenize import TweetTokenizer
import csv
import os.path


def load_data_amazon():
    # load
    with open("./data/amazon-fine-foods/Reviews_reduced.csv", 'rU') as f:
        dataset = list(csv.reader(f))[1:]  # remove header

    # generate x
    tokenizer = TweetTokenizer(reduce_len=True)
    x_text = [line[9].strip() for line in dataset]
    x_text = [tokenizer.tokenize(review) for review in x_text]

    # generate y
    y = [line[6] for line in dataset]
    for idx, label in enumerate(y):
        if label == '1':
            y[idx] = [1, 0, 0, 0, 0]
        elif label == '2':
            y[idx] = [0, 1, 0, 0, 0]
        elif label == '3':
            y[idx] = [0, 0, 1, 0, 0]
        elif label == '4':
            y[idx] = [0, 0, 0, 1, 0]
        elif label == '5':
            y[idx] = [0, 0, 0, 0, 1]
        else:
            print 'wrong label in amazon: ' + label

    return [x_text, y]


def build_vocab_embedding(vocab, binary=True):
    # define files
    original_embedding_file = "./data/GoogleNews-vectors-negative300.bin"
    reduced_embedding_file = "./data/GoogleNews-vectors-negative300-reduced.txt"
    use_reduced = False

    # try to load the reduced embedding file, or load the original
    if os.path.isfile(reduced_embedding_file):
        embedding_file = open(reduced_embedding_file)
        use_reduced = True
    else:
        embedding_file = open(original_embedding_file)

    if not use_reduced and binary:
        header = embedding_file.readline()
        dict_size, embedding_dim = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * embedding_dim

        new_vocab = {}
        for line in xrange(dict_size):
            curr_word = []
            while True:
                ch = embedding_file.read(1)
                if ch == ' ':
                    curr_word = ''.join(curr_word)
                    break
                if ch != '\n':
                    curr_word.append(ch)
            embedding_vec = list(np.fromstring(embedding_file.read(binary_len), dtype='float32'))
            if curr_word in vocab:
                new_vocab[curr_word] = (vocab[curr_word], embedding_vec)
    else:
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
    if not os.path.isfile(reduced_embedding_file):
        save_file = open(reduced_embedding_file, 'w')
        save_file.write("{} {}\n".format(len(new_vocab), embedding_dim))
        for word, (idx, embedding_vec) in new_vocab.iteritems():
            save_file.write("{} {}\n".format(word.encode('utf-8'), ' '.join([str(e) for e in embedding_vec])))

    return new_vocab


def make_sentences_equal_len(sentences, sequence_length, padding_word="<PAD/>"):
    padded_sentences = []
    for i in range(len(sentences)):
        if sequence_length > len(sentences[i]):
            num_padding = sequence_length - len(sentences[i])
            new_sentence = sentences[i] + [padding_word] * num_padding
        else:
            new_sentence = sentences[i][:sequence_length]
        padded_sentences.append(new_sentence)

    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data(use_pretrained_embedding):
    # load train set from Semeval
    reviews, ratings = load_data_amazon()

    # pad train and dev tweets
    # max_seq_len = max([len(review_tokenized) for review_tokenized in reviews])
    max_seq_len = 300
    reviews_equal_len = make_sentences_equal_len(reviews, max_seq_len)

    # build vocab
    vocabulary, vocabulary_inv = build_vocab(reviews_equal_len)
    vocabulary_embedding = build_vocab_embedding(vocabulary) if use_pretrained_embedding else None

    # prepare input
    x, y = build_input_data(reviews_equal_len, ratings, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, vocabulary_embedding]


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