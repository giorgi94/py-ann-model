from os.path import join, dirname
from os import listdir
import json
import pickle
import random


from server import main as run_server

from ann_words.word2vec import get_word_list

from ann_words.naive_bayes_classifier import NaiveBayesClassification, test


base_dir = "nbc_data"


def train():
    with open(join(base_dir, "texts/ambebi_comments_2.json"), "r") as fp:
        data = json.load(fp)
        data = [(get_word_list(d["text"]), d["label"]) for d in data]

    nbc = NaiveBayesClassification()
    nbc.load(join(base_dir, "trained.json"))

    total = len(data)
    print(total)

    for i, (words, label) in enumerate(data):
        print(i + 1, "/", total)
        nbc.parse_text(words, "+" if label else "-")

    nbc.train()
    nbc.dump(join(base_dir, "trained.json"))


def main():
    nbc = NaiveBayesClassification()
    nbc.load(join(base_dir, "trained.json"))

    text = """
    ფსიქოპატს გავს...
    """

    print(nbc.classify(text))


if __name__ == "__main__":
    # train()

    # main()

    run_server()
