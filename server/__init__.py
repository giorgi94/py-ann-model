import pickle
import random
from flask import Flask, render_template, request

from word_model import WordSimilarity

app = Flask(__name__)


with open("textclassifier/data/words.pkl", "rb") as fp:
    WORDS = pickle.load(fp)


@app.route("/")
def index_view():
    return render_template("index.html")


@app.route("/api/test/", methods=["GET", "POST"])
def test_view():
    if request.method == "GET":
        return {"words": random.sample(WORDS, 2)}

    return {}


@app.context_processor
def get_default_context():

    return {"words": [w for w in WORDS if "საზოგა" in WORDS]}


def main():

    app.run(debug=True)
