import numpy as np
import cPickle
from collections import defaultdict
import re
import pandas as pd
import csv


def build_data_tweet(cv):
    tweet_data = []
    vocab = defaultdict(float)
    with open("tweet/2780_freshmen_tweets.csv", "rb") as f:
        rdr = csv.reader(f)
        for row in rdr:
            tweet = clean_tweet(row[3])
            sentiment = row[4]
            if sentiment != '1' and sentiment != '2' and sentiment != '3':
                continue
            words = set(tweet.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": int(sentiment)-1,
                     "text": tweet,
                     "num_words": len(tweet.split()),
                     "split": np.random.randint(0, cv)}
            tweet_data.append(datum)

    with open("tweet/more_tweets.csv", "rb") as f:
        rdr = csv.reader(f)
        for row in rdr:
            tweet = clean_tweet(row[0])
            sentiment = row[1]
            if sentiment != '1' and sentiment != '2' and sentiment != '3':
                continue
            words = set(tweet.split())
            for word in words:
                vocab[word] += 1
            datum = {"y": int(sentiment)-1,
                     "text": tweet,
                     "num_words": len(tweet.split()),
                     "split": np.random.randint(0, cv)}
            tweet_data.append(datum)

    return tweet_data, vocab

def get_W(word_vecs, k=400):
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=400):
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)


def clean_tweet(tweet):
    # separate
    tweet = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", tweet)
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    tweet = re.sub(r"\s{2,}", " ", tweet)

    # URL normalization
    tweet = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
                    "<url>", tweet)

    # Twitter ID normalization
    tweet = re.sub(r"@([A-Za-z0-9_]{1,15})", "<twitter-id>", tweet)

    # Positive emoticon normalization
    tweet = re.sub(r":\)", " <pos-emo>", tweet)
    tweet = re.sub(r":-\)", " <pos-emo>", tweet)
    tweet = re.sub(r":]", " <pos-emo>", tweet)
    tweet = re.sub(r"=]", " <pos-emo>", tweet)

    # Negative emoticon normalization
    tweet = re.sub(r":\(", " <neg-emo>", tweet)
    tweet = re.sub(r":-\(", " <neg-emo>", tweet)
    tweet = re.sub(r":\[", " <neg-emo>", tweet)
    tweet = re.sub(r"=\[", " <neg-emo>", tweet)
    return tweet.strip().lower()


if __name__=="__main__":
    w2v_file = "../word2vec_twitter_model.bin"
    print "loading data...",
    revs, vocab = build_data_tweet(cv=10)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of tweets: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("tweet.p", "wb"))
    print "dataset created!"
