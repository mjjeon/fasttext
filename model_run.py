import pickle
from dataset import *
import numpy as np
from model import *
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction import FeatureHasher
from collections import defaultdict
import re
from scipy.sparse import csr_matrix
def load_dataset(path):
    print("loading")
    dataset = pickle.load(open(path, "rb"))
    return dataset


def define_bow(dataset):
    print("bag of words")
    words = set()
    for x in dataset:
        print (x)
        words.update([y for y in tokens(x)])
    return list(words)

def string_vectorizer(sentence, bow):
    sentence_onehot_vector = [[0 if bow_word != sentence_word else 1 for bow_word in bow]
                  for sentence_word in sentence[0].split()]

    sentence_onehot_vector_sum = np.sum(sentence_onehot_vector, axis=0)
    sentence_onehot_vector_sum = sentence_onehot_vector_sum.reshape(len(bow), -1)

    return sentence_onehot_vector_sum.T

# def parse_dataset(dataset, bow):
#     feature_size = len(bow)
#     sample_size = len(dataset.train)
#     parsed_sentences = np.zeros((feature_size, sample_size))
#     labels = list()
#     class_dict = dict()
#     last_class_index = 0
#
#     for i in range(sample_size):
#         parsed_sentences[:, i] = string_vectorizer(dataset.train[i], bow)
#         if dataset.train[i][1] in class_dict:
#             labels.append(class_dict[dataset.train[i][1]])
#         else:
#             class_dict[dataset.train[i][1]] = last_class_index
#             labels.append(last_class_index)
#             last_class_index += 1
#
#     return parsed_sentences, convert_to_one_hot(labels).T
def tokens(doc):
    """Extract tokens from doc.

    This uses a simple regex to break strings into tokens. For a more
    principled approach, see CountVectorizer or TfidfVectorizer.
    """
    return (tok.lower() for tok in re.findall(r"\w+", doc))


def token_freqs(doc):
    """Extract a dict mapping tokens from doc to their frequencies."""
    freq = defaultdict(int)
    for tok in tokens(doc):
        freq[tok] += 1
    return freq

def parse_dataset(dataset, hasher, kth, batch_size):

    # sample_size = len(dataset.train)
    a = hasher.transform(tokens(d[0]) for d in dataset.train[kth*batch_size:(kth+1)*batch_size])
    sample_size = a.shape[0]
    labels = list()
    class_dict = dict()
    last_class_index = 0

    for i in range(sample_size):
        i = kth*batch_size + i
        if dataset.train[i][1] in class_dict:
            labels.append(class_dict[dataset.train[i][1]])
        else:
            class_dict[dataset.train[i][1]] = last_class_index
            labels.append(last_class_index)
            last_class_index += 1
    return csr_matrix.todense(a).T, convert_to_one_hot(labels).T


def convert_to_one_hot(Y):
    n_values = np.max(Y) + 1
    return np.eye(n_values)[Y]

def read_data_batch(filename, batch_size = 8):

    X, Y = load_dataset(filename)
    # bow = define_bow(X)
    batch_X, batch_Y = tf.train.batch([X, Y], batch_size = batch_size)
    return batch_X, batch_Y

if __name__ == "__main__":

    dataset_save_path = "./dataset/AG/dataset.pickle"
    model_name = "./model/AG/model"
    batch_size = 8

    dataset = load_dataset(dataset_save_path)
    hashing_trick = dataset.bow_size
    feature_size = hashing_trick
    h_size = 10
    class_size = dataset.train_label.shape[0]
    print("feature size", hashing_trick, "h size", h_size, "class size", class_size)
    m = model(feature_size, h_size, class_size, dataset)
    loss = m.train(model_name, batch_size=batch_size, epochs=5, learning_rate=0.001)
    print(loss)