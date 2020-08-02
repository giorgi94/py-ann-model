import pickle
import random

from os.path import abspath, dirname, join


from ann_words.models import WordSimilarity


WORDS = []

base_dir = dirname(abspath(__file__))
dist_dir = join(base_dir, "dist")


def get_data():
    def modify_1(data):
        s = random.sample(data, 50)
        for (a, b, c) in s:
            i = random.randint(0, len(b) - 1)

            if random.randint(0, 5) > 2:
                d = b[0:i] + b[i + 1 :]
            else:
                bc = chr(ord("ა") + random.randint(0, 32))
                d = b[0:i] + bc + b[i + 1 :]

            data.append((a, d, c))

    def modify_2(data):
        words = []

        for (a, b, _) in data:
            words.append(a)
            words.append(b)
        words = [(w, w, True) for w in set(words)]
        data.extend(words)

    def clean(x):
        if len(x) != 3:
            return None
        a, b, c = x
        a = a.strip()
        b = b.strip()
        c = c.strip() == "1"
        return a, b, c

    with open("dist/data.txt", "r") as fp:
        data = [clean(f.split(",")) for f in fp.read().split(";")]
        data = [d for d in data if d is not None]

    modify_1(data)
    modify_2(data)
    return data


def get_data_words():
    def modify(data: list):
        ndata = []

        for (a, b, c) in data:
            i = random.randint(0, len(b) - 1)

            if random.randint(0, 5) > 2:
                d = b[0:i] + b[i + 1 :]
            else:
                bc = chr(ord("ა") + random.randint(0, 32))
                d = b[0:i] + bc + b[i + 1 :]
            ndata.append((a, d, c))

        data.extend(ndata)

    with open("dist/ambebi_words.pkl", "rb") as fp:
        data = pickle.load(fp)
        data = [(w, w, True) for w in data if isinstance(w, str) and len(w) > 2]

    modify(data)
    return data


def main():
    model_path = join(dist_dir, "word_model.pkl")

    N: WordSimilarity = WordSimilarity.create_model(load=model_path)

    def train():

        data = get_data()

        for i in range(500):
            for a, b, c in random.sample(data, 20):
                e = N.correction(a, b, c)
                print(i, e)

        N.save(model_path)

    def check():
        a, b = "კომპიუტერი", "კომპიუტერი"

        print(N.predict(a, b))

    train()
    # check()


if __name__ == "__main__":

    main()

    # data = get_data_words()
    # print(len(data))
