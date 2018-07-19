
import pickle
import random
from collections import defaultdict
import re
from scipy.sparse import csr_matrix, coo_matrix
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from scipy.sparse import find


class Dataset(object):
    def __init__(self, dataset_path):
        self.split_ratio = [30000, 1900]
        # self.split_ratio = [30, 19]
        self.topk_topic = 4

        # self.train_idxversion = self.text2idx(self.train_text)
        # self.test_idxversion = self.text2idx(self.test_text)

    def get_bow(self):
        # print("bag of words")
        # words = set()
        # for x in self.train_text:
        #     words.update([y for y in self.tokens(x[0])])
        self.bow_size = len(self.hasher.vocabulary_)
        print(self.hasher.vocabulary_)
        # return list(words)
        return self.hasher.vocabulary_

    def process(self, n_gram):
        self.hasher = CountVectorizer(ngram_range=(1, n_gram))
        self.samples_by_category = self.sampling_dataset(dataset_path)
        self.train_text, self.test_text = self.split_dataset(self.samples_by_category)

        # self.hasher = HashingVectorizer(norm='l1', ngram_range=(1, n_gram), n_features = self.bow_size)

        self.train_idx_version, self.train_label = self.parse_dataset(self.train_text, self.hasher)
        self.test_idx_version, self.test_label = self.parse_dataset(self.test_text, self.hasher)
        self.get_bow()
        # print(self.todense(self.train_idx_version))
        # print(self.train_label)


    def sampling_dataset(self, dataset_path):
        samples_by_category = dict()
        self.category_to_idx = dict()
        with open(dataset_path, "r", errors = 'surrogateescape') as f:
            rows = f.read().split("\\N")
            for row in rows:
                if row != "":
                    try:
                        title = row.split("\t")[2]
                        description = row.split("\t")[5]
                        category = row.split("\t")[4]
                    except:
                        continue

                    sample = (" ".join([title, description]), category)
                    if category in samples_by_category:
                        samples_by_category[category].append(sample)
                    else:
                        samples_by_category[category] = [sample]

        return samples_by_category

    def split_dataset(self, samples_by_category):
        train_text = []
        test_text = []

        selected_topics = []
        i = 0
        for k in sorted(samples_by_category, key=lambda k: len(samples_by_category[k]), reverse=True):
            if i < self.topk_topic:
                selected_topics.append(k)
            else:
                break
            i += 1
        print("Selected topics {}".format(selected_topics))
        for selected_topic in selected_topics:
            samples = samples_by_category[selected_topic]
            random.shuffle(samples)
            train_text.extend(samples[:self.split_ratio[0]])
            test_text.extend(samples[self.split_ratio[0]:self.split_ratio[0]+self.split_ratio[1]])

        self.hasher.fit([x[0] for x in train_text])
        # random.seed(9001)
        # random.shuffle(train_text)
        # random.seed(9001)
        # random.shuffle(test_text)
        print ("Training sample size {}, Test sample size {}".format(len(train_text), len(test_text)))
        return train_text, test_text

    # def text2idx(self, samples):
    #     bow = self.get_bow()
    #     samples_to_index = list()
    #     for sample in samples:
    #         # for w in self.tokens(sample[0]):
    #         #     print (w,)
    #
    #         a = [bow.index(w) if w in bow else len(bow)+1 for w in self.tokens(sample[0])]
    #         samples_to_index.append(a)
    #     return samples_to_index
    #         # print(a)



    # def tokens(self, doc):
    #     """Extract tokens from doc.
    #
    #     This uses a simple regex to break strings into tokens. For a more
    #     principled approach, see CountVectorizer or TfidfVectorizer.
    #     """
    #     # return (tok.lower() for tok in re.findall(r"\w+", doc))
    #
    #     return (tok.lower() for tok in doc.split(" "))

    def convert_to_one_hot(self, Y):
        n_values = np.max(Y) + 1
        return np.eye(n_values)[Y]


    # def parse_dataset(self, samples, hasher):
    #     samples_idx_version = []
    #     self.labels = list()
    #     self.class_dict = dict()
    #     self.last_class_index = 0
    #
    #     for sample in samples:
    #         a = hasher.transform([sample[0]])
    #         samples_idx_version.append(find(a.tocsr())[1].tolist())
    #         if sample[1] in self.class_dict:
    #             self.labels.append(self.class_dict[sample[1]])
    #         else:
    #             self.class_dict[sample[1]] = self.last_class_index
    #             self.labels.append(self.last_class_index)
    #             self.last_class_index += 1
    #     return samples_idx_version, self.convert_to_one_hot(self.labels).T


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
            # print([csr_matrix.todense(sample).tolist()])
            densed_samples.append(csr_matrix.todense(sample))
        densed_samples_np =  np.asarray(densed_samples).reshape(len(samples),-1)
        # print(densed_samples_np.shape)
        return densed_samples_np

if __name__ == "__main__":
    # dataset_path = "./dataset/AG/newsSpace_sample"
    dataset_path = "./dataset/AG/newsSpace"
    dataset_save_path = "./dataset/AG/dataset.pickle"
    dataset = Dataset(dataset_path)
    dataset.process(n_gram=1)
    print("bag of word size", dataset.bow_size)
    # np.savetxt("tmp.csv",dataset.todense(dataset.train_idx_version), delimiter=",")
    pickle.dump(dataset, open(dataset_save_path, "wb"))


    # dataset = pickle.load(open(dataset_save_path))
    # print(dataset)