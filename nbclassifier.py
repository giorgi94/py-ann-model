from os.path import join, dirname
from os import listdir
import json
import pickle
import random

from ann_words.word2vec import get_word_list, remove_dublicates, clean_stopwords

from ann_words.naive_bayes_classifier import NaiveBayesClassification, test


base_dir = "nbc_data/texts"
weights_dir = "nbc_data/weights"


def train():
    with open(join(base_dir, "data.json"), "r") as fp:
        data = json.load(fp)

    nbc = NaiveBayesClassification()
    nbc.load("trained.json")

    total = len(data)

    for i, (words, label) in enumerate(data):
        print(i + 1, "/", total)
        nbc.parse_text(words, "+" if label else "-")

    nbc.train()
    nbc.dump("trained.json")


def main():
    nbc = NaiveBayesClassification()
    nbc.load("trained.json")

    text = """
    ეკა გეთანხმები ასეთი პატრონები უნდა დაისაჯაონ აუცილებლად , მაგრამ რა ღა დროს როცა ადამიანი დაშავდება მერე??? ამიტომ უასაბელოდ და ნამორდნიკის გარეშე ძაღლები არ უნდა გამოყავდეთ გარეთ
    """

    print(nbc.classify(text))


if __name__ == "__main__":
    main()
    # train()


"""



def get_data():
    files = [join(base_dir, f) for f in listdir(base_dir) if f.endswith(".json")]

    data = []

    for p in files:
        with open(p) as fp:
            dt = json.load(fp)
            data.extend([(clean_stopwords(t), c) for t, c in dt])

    return data


def train(i):
    data = get_data()

    nbc = NaiveBayesClassification()

    sample = random.sample(data, 8000)

    nbc.train(sample)

    nbc.dump(join(weights_dir, f"model_{i}.json"))


def load_models():
    files = [join(weights_dir, f) for f in listdir(weights_dir) if f.endswith(".json")]
    models = [NaiveBayesClassification() for _ in range(len(files))]
    for f, m in zip(files, models):
        m.load(f)

    return models


def classify(models, text):
    p, q = 1, 1
    for m in models:
        a, b = m.classify(text)
        p *= a
        q *= b

    return int(p > q)


def main():
    models = load_models()

    with open("samples.json", "r") as fp:
        samples = json.load(fp)

    total = len(samples)

    correct = 0

    for text, s in samples:
        score = classify(models, text)

        a = int(score > 1)
        b = int(s)

        if a == b:
            correct += 1

    print(round(100 * correct / total, 2))


if __name__ == "__main__":
    # train(5)
    main()
"""
