import json
import pickle
import random
import re

from ann_words.models import WordSimilarity

from ann_words.word2vec import distance, remove_dublicates


def const_to_word_list(content):
    return re.findall(r"[ა-ჰ]+", content)


def dump_data(name, data):
    with open(f"dist/{name}.pkl", "wb") as fp:
        pickle.dump(data, fp)


def load_data(name):
    with open(f"dist/{name}.pkl", "rb") as fp:
        return pickle.load(fp)


def collect_data():
    with open("dist/comments.txt", "r") as fp:
        data = eval(fp.read())

    data = [(const_to_word_list(c), o) for c, o in data]
    data = [(c, o) for c, o in data if c]

    dump_data("ambebi_comments", data)


def collect_words():
    data = load_data("ambebi_comments")
    WORDS = []

    for c, _ in data:
        WORDS.extend([remove_dublicates(i) for i in c])

    WORDS = list(set(WORDS))

    dump_data("ambebi_words", WORDS)


def create_psudo_cluster_words():
    words = load_data("ambebi_words")

    clusters = []

    for w in words:
        print(len(words))
        clu = None

        if len(w) > 4:
            clu = w[:4]
        else:
            continue

        clus = [w for w in words if clu in w]
        for c in clus:
            words.remove(c)
        clusters.append((clu, set(clus)))

    print(len(clusters))

    dump_data("word_pseudo_cluster", clusters)


def order_words(words):
    ordered = []

    i = -1

    for w in words:
        i += 1
        for v in words[i + 1 :]:
            ordered.append((w, v))

    return ordered


def cluster_words():
    N: WordSimilarity = WordSimilarity.create_model(load="dist/word_model.pkl")

    pseudo_clusters = load_data("word_pseudo_cluster")
    pseudo_clusters = random.sample(pseudo_clusters, 10)

    pairs = []

    for _, words in pseudo_clusters:
        o = order_words(list(words))
        random.shuffle(o)
        pairs.extend(o[:10])

    random.shuffle(pairs)

    print(f"0 - no, 1 - yes; n: {len(pairs)}\n")

    f = open("dist/data.txt", "a")

    for i, (a, b) in enumerate(pairs):
        try:
            predict = N.predict(a, b)

            s = input(f"{i+1}) {a} is {b} ({predict}%): ")

            if s == "0" or s == "1":
                f.write(f"{a}, {b}, {s};\n")
                N.correction(a, b, s == "1")

        except KeyboardInterrupt:
            print()
            break

    f.close()


if __name__ == "__main__":
    # collect_data()
    # collect_words()
    # create_psudo_cluster_words()

    cluster_words()
