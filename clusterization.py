import pickle
import random


from word_model import WordSimilarity


WORDS = []


def read_words():
    global WORDS
    with open("textclassifier/data/words.pkl", "rb") as fp:
        WORDS = pickle.load(fp)


def generate_psudo_clusters():
    words = WORDS.copy()

    # words = words[: len(words) // 2 + 1]
    words = words[len(words) // 2 :]

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

    print("Clusters:")
    print(len(clusters))

    with open("dist/clusters_2.pkl", "wb") as fp:
        pickle.dump(clusters, fp)

    # words = [w for w in WORDS if "საზო" in w]

    # for w in words:
    #     print(w)


def main():
    with open("dist/clusters.pkl", "rb") as fp:
        clusters = pickle.load(fp)

    print(clusters[0])
    print(clusters[1])
    print(clusters[2])


if __name__ == "__main__":
    read_words()
    generate_psudo_clusters()
    # main()
