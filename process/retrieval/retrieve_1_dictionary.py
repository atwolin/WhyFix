import os
# from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
# from torch.utils.data import DataLoader
import spacy
import json
import pandas as pd

# from ..utils.helper_functions import pretify_print

data_folder_path = "/home/nlplab/atwolin/thesis/data/example-sentences/combined.v2.txt"
model = CrossEncoder("cross-encoder/stsb-distilroberta-base", device="cuda:0")
nlp = spacy.load("en_core_web_sm")

posMap_spacy2cambridge = dict()
with open("/home/nlplab/atwolin/thesis/data/pos/spacy_to_cambridge.json", 'r', encoding='utf-8') as f:
    posMap_spacy2cambridge = json.load(f)


def lemmatize(target_word):
    target_word = target_word.lower().strip() if isinstance(target_word, str) else target_word
    doc = nlp(target_word)
    lemmar_word = doc[0].lemma_
    return lemmar_word


def get_sentences_containing_target_word(target_word):
    """Get all sentences containing the target word from the data folder."""
    sentences = []
    with open(data_folder_path, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if target_word == word.strip():
                    # print(f"get_sentences_containing_target_word(): Found {target_word} in: {line.strip()}")
                    if " = " in line:
                        sentences.extend([sent.strip() for sent in line.split(" = ")])
                    else:
                        sentences.append(line.strip())
                    continue
    sentences = list(set(sentences))
    # print(f"\n>>>>> Found {len(sentences)} sentences containing target_word '{target_word}' <<<<<\n{'=' * 50}")
    return sentences


def get_top_similar_sentences(sentence, target_word, n=5):
    """Find the top n similar sentences to the given sentence."""
    corpus = get_sentences_containing_target_word(target_word)

    if len(corpus) == 0:
        return ["nan"]

    ranks = model.rank(sentence, corpus)

    # Print the top 5 similar sentences
    cnt = 0
    for rank in ranks:
        if cnt >= 5:
            break
        print(f"{rank['score']:.2f}, {corpus[rank['corpus_id']]} ")
        cnt += 1

    top_n_sentences = [corpus[rank['corpus_id']] for rank in ranks[:n]]
    # print(f"\n>>>>> Finished getting top similar sentences containing target_word '{target_word}'<<<<<\n{'=' * 50}")
    return top_n_sentences


def get_infomation_in_cambridge_json(sentence, target_word):
    doc = nlp(sentence)
    pos_space = ""

    # Get the POS of the target word in spacy
    for token in doc:
        if token.text == target_word:
            pos_space = token.pos_
            # print(token.text, token.pos_, token.dep_)
            break
    # print(f"{target_word} in spacy: {pos_space}")

    # Find the definition of the target word in Cambridge
    if pos_space != "":
        data_path = ""
        if target_word[0].isalpha():
            data_path = f"/home/nlplab/atwolin/thesis/data/cambridge-parse/vocab/cambridge.{(target_word[0]).lower()}.json"
        else:
            data_path = "/home/nlplab/atwolin/thesis/data/cambridge-parse/vocab/cambridge.0-9.json"

        # print(data_path)
        # /home/nlplab/atwolin/thesis/data/cambridge-parse/vocab/cambridge._.json

        if os.path.isfile(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for pos_content in data[target_word]["poses"]:
                    pos_cambridge = pos_content["pos"][0]

                    # print(pos_cambridge)
                    # print(posMap_spacy2cambridge[pos_space])

                    if pos_cambridge in posMap_spacy2cambridge[pos_space]:
                        for big_sense_content in pos_content["big_sense"]:
                            # print(big_sense_content)
                            for sense in big_sense_content["senses"]:
                                # print(sense)
                                # print(f"{target_word}'s level: {sense['level']}")
                                # print(f"{target_word}'s definition: {sense['eng_def']}")

                                return pos_cambridge, sense['level'], sense['eng_def']

    return None


def get_infomation_in_cambridge_csv(essay_sentence, target_word):
    data_path = "/home/nlplab/atwolin/thesis/data/cambridge-parse/cambridge_parse.words.v2.csv"
    df = pd.read_csv(data_path)

    if isinstance(target_word, float):
        return target_word, "nan", "nan", "nan"
    if isinstance(essay_sentence, float):
        return target_word, "nan", "nan", "nan"

    # Lemmatize
    # doc = nlp(target_word)
    # lemmar_word = doc[0].lemma_
    lemmar_word = lemmatize(target_word)
    # print(f"Word: {target_word}, {doc[0].lemma}, {doc[0].lemma_}")

    # Find the most relevant example sentences
    word_senses = df[df["word"] == lemmar_word]
    # print(f"get_infomation_in_cambridge_csv(): Found word_senses in cmb: {word_senses}\n")
    if word_senses.empty:
        return lemmar_word, "nan", "nan", "nan"

    # Get all sentences in the target word's definitions
    # corpus = word_senses["eng_sentence"].dropna().tolist()
    corpus = [s for s in word_senses["eng_sentence"].dropna().tolist() if isinstance(s, str)]

    ranks = model.rank(essay_sentence, corpus)
    top_one_sentence = corpus[ranks[0]['corpus_id']]
    # print(f"Found top_one_sentence in cmb: {top_one_sentence}")

    # Get information of the target word in Cambridge dictionary
    target_row = df[df["eng_sentence"] == top_one_sentence]
    if target_row.empty:
        return lemmar_word, "nan", "nan", "nan"
    # print(f"Found target_row in cmb: {target_row['word']}, {target_row['eng_sentence']}")

    pos = target_row["pos_1"].values[0] if isinstance(target_row["pos_2"].values[0], float) else str(target_row["pos_1"].values[0]) + str(target_row["pos_2"].values[0])
    level = "unknown" if isinstance(target_row["level"].values[0], float) else target_row["level"].values[0]
    return lemmar_word, pos, level, target_row["dict_eng_def"].values[0]


def check_target_word_in_akl(target_word):
    akl_path = "/home/nlplab/atwolin/thesis/data/akl/akl_list.csv"
    if os.path.isfile(akl_path):
        df = pd.read_csv(akl_path)
        if target_word in df["word"].values:
            return True
    return False


def check_target_word_in_awl(target_word):
    awl_path = "/home/nlplab/atwolin/thesis/data/awl/awl_individual.csv"
    if os.path.isfile(awl_path):
        df = pd.read_csv(awl_path)
        if target_word in df["word"].values:
            return True
    return False


if __name__ == "__main__":
    # query_revised = "I think there are two reasons for it.  First, the computers are too widespread."
    # targetWord_revised = "widespread"
    # topSimilarSentences_revised = get_top_similar_sentences(query_revised, targetWord_revised)
    # info_revised = get_infomation_in_cambridge_csv(query_revised, targetWord_revised)

    # query_original = "The study showed overspending was more prevalent among women than men."
    # targetWord_original = "prevalent"
    # topSimilarSentences_original = get_top_similar_sentences(query_original, targetWord_original)
    # info_original = get_infomation_in_cambridge_csv(query_original, targetWord_original)

    query = "If I did the same thing every day, I would be dull."
    target_word = "dull"
    topSimilarSentences_revised = get_top_similar_sentences(query, target_word)
    info_revised = get_infomation_in_cambridge_csv(query, target_word)

    from pprint import pprint
    pprint(topSimilarSentences_revised)
    print(info_revised)
    # pprint(topSimilarSentences_original)
    # print(info_original)
