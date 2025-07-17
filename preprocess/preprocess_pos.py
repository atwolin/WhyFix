import os
import json
import spacy
from tqdm import tqdm


def get_pos_list_in_cambridge():
    """Get the list of POS tags from the Cambridge dictionary data."""
    data_folder_path = "../../data/cambridge-parse/vocabV2"
    posList = set()
    for fname in os.listdir(data_folder_path):
        fpath = os.path.join(data_folder_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for vocab in data:
                    for pos_content in data[vocab]["poses"]:
                        # print(pos_content["pos"])
                        if pos_content["pos"][0] != "":
                            posList.add(pos_content["pos"][0],)
                        # break
    print(posList)
    return posList


def get_pos_list_in_spacy():
    """Get the list of POS tags from the spacy library."""
    # nlp = spacy.load("en_core_web_sm")
    # tagger = nlp.get_pipe("tagger")
    # print(tagger.labels)

    # for label in tagger.labels:
    #     print(f"{label}: {spacy.explain(label)}")

    # return tagger.labels

    nlp = spacy.load("en_core_web_sm")
    # with open("/home/nlplab/atwolin/thesis/data/examples-sentences/combined.txt", 'r', encoding='utf-8') as f:
    #     # for line in f:
    #     for line in tqdm(f):
    #         doc = nlp(line.strip())
    #         for token in doc:
    #             posList.add(token.pos_)

    # print(posList)

    data_folder_path = "../../data/cambridge-parse/vocabV2"
    posList = set()
    for fname in tqdm(os.listdir(data_folder_path)):
        fpath = os.path.join(data_folder_path, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for vocab in data:
                    # print(vocab)
                    doc = nlp(vocab)
                    for token in doc:
                        posList.add(token.pos_)
                    # break
    print(posList)
    return posList


if __name__ == "__main__":
    posList_campbridge = get_pos_list_in_cambridge()
    posList_spacy = get_pos_list_in_spacy()


    # posMap_cambridge_to_spacy = {
    #     "adj": "ADJ",
    #     "adjective": "ADJ",
    #     "adjecive": "ADJ",
    #     "adjectives": "ADJ",
    #     "adv": "ADV",
    #     "adverb": "ADV",
    #     "verb": "VERB",
    #     "phrasal verb": "VERB",
    #     "auxiliary verb": "AUX",
    #     "modal verb": "AUX",
    #     "noun": "NOUN",
    #     "noun or exclamation": "NOUN",
    #     "collocation": "X",
    #     "idiom": "X",
    #     "exclamation": "INTJ",
    #     "interjection": "INTJ",
    #     "prefix": "X",
    #     "suffix": "X",
    #     "combining form": "X",
    #     "abbreviation": "X",
    #     "short form": "X",
    #     "number": "NUM",
    #     "ordinal number": "NUM",
    #     "preposition": "ADP",
    #     "determiner": "DET",
    #     "predeterminer": "DET",
    #     "conjunction": "CCONJ",
    #     "pronoun": "PRON",
    #     "symbol": "SYM"
    # }

    # posMap_spacy_to_cambridge = {
    #     "ADJ": ["adj", "adjective", "adjecive", "adjectives"],
    #     "ADV": ["adv", "adverb"],
    #     "VERB": ["verb", "phrasal verb"],
    #     "AUX": ["auxiliary verb", "modal verb"],
    #     "NOUN": ["noun", "noun or exclamation"],
    #     "X": [
    #         "collocation",
    #         "idiom",
    #         "prefix",
    #         "suffix",
    #         "combining form",
    #         "abbreviation",
    #         "short form"
    #     ],
    #     "INTJ": ["exclamation", "interjection"],
    #     "NUM": ["number", "ordinal number"],
    #     "ADP": ["preposition"],
    #     "DET": ["determiner", "predeterminer"],
    #     "CCONJ": ["conjunction"],
    #     "PRON": ["pronoun"],
    #     "SYM": ["symbol"]
    # }

    posMap_cambridge_to_spacy = {
        "adj": "ADJ",
        "adjective": "ADJ",
        "adjecive": "ADJ",
        "adjectives": "ADJ",
        "adv": "ADV",
        "adverb": "ADV",
        "verb": "VERB",
        "phrasal verb": "VERB",
        "auxiliary verb": "AUX",
        "modal verb": "AUX",
        "noun": "NOUN",
        "noun or exclamation": "NOUN",
        "collocation": "X",
        "idiom": "X",
        "exclamation": "INTJ",
        "interjection": "INTJ",
        "prefix": "X",
        "suffix": "X",
        "combining form": "X",
        "abbreviation": "X",
        "short form": "X",
        "number": "NUM",
        "ordinal number": "NUM",
        "preposition": "ADP",
        "determiner": "DET",
        "predeterminer": "DET",
        "conjunction": "CCONJ",
        "pronoun": "PRON",
        "symbol": "SYM"
    }

    posMap_spacy_to_cambridge = {
        "ADJ": ["adj", "adjective", "adjecive", "adjectives"],
        "ADV": ["adv", "adverb"],
        "VERB": ["verb", "phrasal verb"],
        "AUX": ["auxiliary verb", "modal verb"],
        "NOUN": ["noun", "noun or exclamation"],
        "X": [
            "collocation",
            "idiom",
            "prefix",
            "suffix",
            "combining form",
            "abbreviation",
            "short form"
        ],
        "INTJ": ["exclamation", "interjection"],
        "NUM": ["number", "ordinal number"],
        "ADP": ["preposition"],
        "DET": ["determiner", "predeterminer"],
        "CCONJ": ["conjunction"],
        "PRON": ["pronoun"],
        "SYM": ["symbol"]
    }

    print(len(posList_campbridge), len(posMap_cambridge_to_spacy))
    print(len(posList_spacy), len(posMap_spacy_to_cambridge))

    os.makedirs("../data/pos", exist_ok=True)
    with open("../data/pos/cambridge_to_spacy.v2.json", 'w', encoding='utf-8') as f:
        json.dump(posMap_cambridge_to_spacy, f, indent=4)

    with open("../data/pos/spacy_to_cambridge.v2.json", 'w', encoding='utf-8') as f:
        json.dump(posMap_spacy_to_cambridge, f, indent=4)
