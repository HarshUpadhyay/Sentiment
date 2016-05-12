import nltk
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train")
parser.add_argument("--tweet", help="tweet")
args = parser.parse_args()


def find_ones(a):
    b = np.asarray(a)
    c = np.where(b == 1)
    return (len(c[0]))


def file_len(fname):
    with open(fname) as f:
        for item, l in enumerate(f):
            pass
    return item + 1


f = file_len(args.train) / 2


def get_words_in_tweets(tweets):
    all_words = []
    for (words, sentiment) in tweets:
        all_words.extend(words)
    return all_words


def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features


tweets = []
for i, line in enumerate(open(args.train)):
    if i < f:
        words = re.split(' ', line)
        sent = words[-1].rstrip(' \n')
        wds = words[0:len(words) - 1]
        f = [e.lower() for e in wds if len(e) >= 3]
        tweets.append((f, sent))

word_features = get_word_features(get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
n_pos = []
for line in open(args.tweet):
    out = classifier.classify(extract_features(line.split()))
    if out == "positive":
        n_pos.append(1)
    else:
        n_pos.append(-1)

ff = file_len(args.tweet)
print("total nos= ", ff)
print(find_ones(n_pos))

# starting second round

f = file_len(args.train) / 2

tweets = []
for i, line in enumerate(open(args.train)):
    if i > f:
        words = re.split(' ', line)
        sent = words[-1].rstrip(' \n')
        wds = words[0:len(words) - 1]
        ff = [e.lower() for e in wds if len(e) >= 3]
        tweets.append((ff, sent))

word_features = get_word_features(get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
n_pos = []
for line in open(args.tweet):
    out = classifier.classify(extract_features(line.split()))
    if out == "positive":
        n_pos.append(1)
    else:
        n_pos.append(-1)

ff = file_len(args.tweet)
print("total nos= ", ff)
print(find_ones(n_pos))
f = file_len(args.train) / 2
tweets = []
for i, line in enumerate(open(args.train)):
    words = re.split(' ', line)
    sent = words[-1].rstrip(' \n')
    wds = words[0:len(words) - 1]
    ff = [e.lower() for e in wds if len(e) >= 3]
    tweets.append((ff, sent))

word_features = get_word_features(get_words_in_tweets(tweets))
training_set = nltk.classify.apply_features(extract_features, tweets)
classifier = nltk.NaiveBayesClassifier.train(training_set)
n_pos = []
for line in open(args.tweet):
    out = classifier.classify(extract_features(line.split()))
    if out == "positive":
        n_pos.append(1)
    else:
        n_pos.append(-1)

ff = file_len(args.tweet)
print("total nos= ", ff)
print(find_ones(n_pos))
