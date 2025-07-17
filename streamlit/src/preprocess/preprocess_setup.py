import spacy
import string


NLP = spacy.load("en_core_web_sm")


def lemmatize_word_list(words: list):
    doc = []
    words = [word.lower().strip() if isinstance(word, str) else word for word in words]
    doc = NLP(" ".join(words))
    doc = [token.lemma_ for token in doc if token.lemma_ not in string.punctuation]
    return doc
