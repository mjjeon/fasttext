
import pickle
import random
from collections import defaultdict
import re
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from scipy.sparse import find
import csv
import argparse

class Dataset(object):
    def __init__(self, dataset_path, dataset_save_path, ngram):
        self.split_ratio = [30000, 1900]
        # self.split_ratio = [30, 19]
        self.topk_topic = 4
        self.dataset_path = dataset_path
        self.dataset_save_path = dataset_save_path
        self.ngram = ngram

    def get_bow(self):
        self.bow_size = len(self.hasher.vocabulary_)
        return self.hasher.vocabulary_

    def process(self):
        self.hasher = CountVectorizer(ngram_range=(1, self.ngram))
        self.train_text, self.test_text = self.sampling_dataset()

        self.train_idx_version, self.train_label = self.parse_dataset(self.train_text, self.hasher)
        self.test_idx_version, self.test_label = self.parse_dataset(self.test_text, self.hasher)
        self.get_bow()


    def sampling_dataset(self):
        filenames = ["train.csv", "test.csv"]
        train_text = []
        test_text = []
        for filename in filenames:
            with open(self.dataset_path+filename, "r") as f:
                reader = csv.reader(f, quotechar = '"')

                for row in reader:
                    if len(row) != 3:
                        continue
                    sample = (" ".join([row[1], row[2]]), int(row[0].strip().replace("\"","")))
                    if "train" in filename:
                        train_text.append(sample)
                    else:
                        test_text.append(sample)

        self.hasher.fit([x[0] for x in train_text])
        print ("Training sample size {}, Test sample size {}".format(len(train_text), len(test_text)))
        return train_text, test_text

    def convert_to_one_hot(self, Y):
        n_values = np.max(Y) + 1
        return np.eye(n_values)[Y]


    def parse_dataset(self, samples, hasher):
        samples_idx_version = []
        self.labels = list()
        self.class_dict = dict()
        self.last_class_index = 0

        for sample in samples:
            a = hasher.transform([sample[0]])
            a = a/a.max()

            samples_idx_version.append(a)
            if sample[1] in self.class_dict:
                self.labels.append(self.class_dict[sample[1]])
            else:
                self.class_dict[sample[1]] = self.last_class_index
                self.labels.append(self.last_class_index)
                self.last_class_index += 1

        return samples_idx_version, self.convert_to_one_hot(self.labels).T

    def todense(self, samples):
        densed_samples = []
        for sample in samples:
            densed_samples.append(csr_matrix.todense(sample))
        densed_samples_np =  np.asarray(densed_samples).reshape(len(samples),-1)
        return densed_samples_np

    def save(self):
        pickle.dump(self, open(self.dataset_save_path, "wb"))

if __name__ == "__main__":
    # dataset_path = "./dataset/AG/newsSpace_sample"
    # parser = argparse.ArgumentParser(description = "")
    # parser.add_argument("--dataset_path", )
    dataset_path = "./dataset/AG/ag_news_csv/"
    dataset_save_path = "./dataset/AG/dataset.pickle"
    ngram = 2
    dataset = Dataset(dataset_path, dataset_save_path, ngram)
    dataset.process()
    print("bag of word size", dataset.bow_size)

