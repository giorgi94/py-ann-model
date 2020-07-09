import json

from ann.word2vec import distance


def cluster():
    with open("dist/words.json", "r") as fp:
        words = json.load(fp)
        words = [w for w in words if len(w) > 2]

    Clu = []

    for a in words:
        col = [a]
        words.remove(a)
        for b in words:
            if distance(a, b) < 1:
                col.append(b)
                words.remove(b)
        # print(a)
        Clu.append(col)

    # with open("dist/cluster.json", "w") as fp:
    #     json.dump(Clu, fp, ensure_ascii=False)


if __name__ == "__main__":
    print(distance("ყიდვა", "გაყიდვა"))
