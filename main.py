import argparse
from dataset_newversion import *

DATASET_PATH = "./dataset/AG/ag_news_csv/"
DATASET_SAVE_PATH = "./dataset/AG/dataset.pickle"
NGRAM = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--dataset-save-path", type=str, default=DATASET_SAVE_PATH)
    parser.add_argument("--ngram", type=int, default=NGRAM)

    args = parser.parse_args()

    dataset = Dataset(args.)

if __name__ == "__main__":




