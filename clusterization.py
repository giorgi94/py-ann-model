import pickle
import random


from word_model import WordSimilarity


WORDS = []


def get_data():
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

    return data


def main():
    N: WordSimilarity = WordSimilarity.create_model(load="dist/word_model.pkl")

    def train():

        data = get_data()

        for i in range(500):
            for a, b, c in random.sample(data, 20):
                X = N.get_X(a, b)
                Y = N.get_Y(c)

                N.forward(X)
                print(i, N.backward(X, Y))

        N.save("dist/word_model.pkl")

    def check():
        a, b = "ტელეფონმა", "სატელეფონო"

        X = N.get_X(a, b)

        print(N.validate(X))

    # train()
    check()


if __name__ == "__main__":

    main()

    # get_data()
