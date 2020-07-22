import re
import pickle
import random
import json

from ann.word2vec import remove_dublicates, distance


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

    for (a, b) in pairs:
        try:
            s = input(f"{a} is {b}: ")

            if s == "0" or s == "1":
                f.write(f"{a}, {b}, {s};\n")
        except KeyboardInterrupt:
            print()
            break

    f.close()


if __name__ == "__main__":
    # collect_data()
    # collect_words()
    # create_psudo_cluster_words()

    cluster_words()
