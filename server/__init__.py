import pickle
import random
from flask import Flask, render_template, request


from ann_words.word2vec import get_word_list
from ann_words.naive_bayes_classifier import NaiveBayesClassification


app = Flask(__name__)

nbc = NaiveBayesClassification()
nbc.load("nbc_data/trained.json")


@app.route("/")
def index_view():
    return render_template("index.html")


@app.route("/api/test/", methods=["POST"])
def test_view():

    text = request.form.get("comment")

    return {"label": nbc.classify(text)}


@app.context_processor
def get_default_context():

    return {}


def main():

    app.run(host="0.0.0.0", port=3000, debug=True)
