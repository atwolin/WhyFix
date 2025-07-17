import os
import pandas as pd
from tqdm import tqdm

from sentence_transformers.cross_encoder import CrossEncoder

from preprocess.preprocess_setup import NLP
from utils.files_io import (
    FilePath,
)


MODEL = CrossEncoder("cross-encoder/stsb-distilroberta-base", device="cuda:0")
PATHS = FilePath()


class INFO_ROW:
    word_learner: str
    lemmaWord_learner: any
    pos_learner: any
    level_learner: any
    definition_learner: any
    examples_learner: any
    in_akl_learner: bool

    word_editor: str
    lemmaWord_editor: any
    pos_editor: any
    level_editor: any
    definition_editor: any
    examples_editor: any
    in_akl_editor: bool


def lemmatize(target_word):
    """Lemmatize the target word and return the first lemmatized word."""
    target_word = target_word.lower().strip() if isinstance(target_word, str) else target_word
    doc = NLP(target_word)
    lemmar_word = doc[0].lemma_
    return lemmar_word


def get_sentences_containing_target_word(target_word):
    """Get all sentences containing the target word from the data folder."""
    sentences = []
    with open(PATHS.filePath_dictionary_expample_sentences, 'r', encoding='utf-8') as f:
        for line in f:
            for word in line.split():
                if target_word == word.strip():
                    if " = " in line:
                        sentences.extend([sent.strip() for sent in line.split(" = ")])
                    else:
                        sentences.append(line.strip())
                    continue
    sentences = list(set(sentences))
    return sentences


def get_top_similar_sentences(sentence, target_word, n=5):
    """Find the top n similar sentences to the given sentence."""
    corpus = get_sentences_containing_target_word(target_word)

    if len(corpus) == 0:
        return ["nan"]

    ranks = MODEL.rank(sentence, corpus)
    top_n_sentences = [corpus[rank['corpus_id']] for rank in ranks[:n]]
    return top_n_sentences


def get_infomation_in_cambridge_csv(essay_sentence, target_word):
    data_path = PATHS.filePath_dictionary_cambridge
    df = pd.read_csv(data_path)

    if isinstance(target_word, float):
        return target_word, "nan", "nan", "nan"
    if isinstance(essay_sentence, float):
        return target_word, "nan", "nan", "nan"

    # Lemmatize
    lemmar_word = lemmatize(target_word)

    # Find the most relevant example sentences
    word_senses = df[df["word"] == lemmar_word]
    if word_senses.empty:
        return lemmar_word, "nan", "nan", "nan"

    # Get all sentences in the target word's definitions
    corpus = [s for s in word_senses["eng_sentence"].dropna().tolist() if isinstance(s, str)]

    ranks = MODEL.rank(essay_sentence, corpus)
    top_one_sentence = corpus[ranks[0]['corpus_id']]

    # Get information of the target word in Cambridge dictionary
    target_row = df[df["eng_sentence"] == top_one_sentence]
    if target_row.empty:
        return lemmar_word, "nan", "nan", "nan"

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


def retrieve_cambridge_data(row):
    learnerSentence = row["learner_sentence"].lower() if isinstance(row["learner_sentence"], str) else row["learner_sentence"]
    learnerWord = row["learner_word"].lower() if isinstance(row["learner_word"], str) else row["learner_word"]
    editorSentence = row["editor_sentence"].lower() if isinstance(row["editor_sentence"], str) else row["editor_sentence"]
    editorWord = row["editor_word"].lower() if isinstance(row["editor_word"], str) else row["editor_word"]

    topSimilarSentences_learner = get_top_similar_sentences(learnerSentence, learnerWord, 2)
    lemmaWord_learner, pos_learner, level_learner, engDef_learner = get_infomation_in_cambridge_csv(learnerSentence, learnerWord)
    inAkl_learner = check_target_word_in_akl(learnerWord)

    topSimilarSentences_editor = get_top_similar_sentences(editorSentence, editorWord, 2)
    lemmaWord_editor, pos_editor, level_editor, engDef_editor = get_infomation_in_cambridge_csv(editorSentence, editorWord)
    inAkl_editor = check_target_word_in_akl(editorWord)

    learnerInfo = {
        "learner_words": learnerWord,
        "word": lemmaWord_learner,
        "pos": pos_learner,
        "level": level_learner,
        "definition": engDef_learner,
        "examples": topSimilarSentences_learner,
        "in_akl": inAkl_learner
    }
    editorInfo = {
        "editor_word": editorWord,
        "word": lemmaWord_editor,
        "pos": pos_editor,
        "level": level_editor,
        "definition": engDef_editor,
        "examples": topSimilarSentences_editor,
        "in_akl": inAkl_editor
    }
    InfoList = [
        learnerWord, lemmaWord_learner, pos_learner, level_learner, engDef_learner, topSimilarSentences_learner, inAkl_learner,
        editorWord, lemmaWord_editor, pos_editor, level_editor, engDef_editor, topSimilarSentences_editor, inAkl_editor
    ]
    return learnerInfo, editorInfo, InfoList


def concat_cambridge_data(path_input, path_output):
    df_data = pd.read_csv(path_input, encoding='utf-8')
    cambridge_info = []
    for index, row in tqdm(df_data.iterrows(), total=len(df_data), colour='green', ncols=100):
        learnerInfo, editorInfo, infoList = retrieve_cambridge_data(row)
        cambridge_info.append(infoList)

    df_data = pd.concat([df_data, pd.DataFrame(cambridge_info, columns=INFO_ROW.__annotations__.keys())], axis=1)
    df_data.to_csv(path_output, index=False, encoding='utf-8')
    print(f"Cambridge data retrieved and saved to {path_output}")


def concat_cambridge_data_runtime():
    path_input = PATHS.filePath_runtime_sentences
    path_output = PATHS.filePath_runtime_dictInfo

    df_data = pd.read_csv(path_input, encoding='utf-8')
    cambridge_info = []
    for index, row in tqdm(df_data.iterrows(), total=len(df_data), colour='green', ncols=100):
        learnerInfo, editorInfo, infoList = retrieve_cambridge_data(row)
        cambridge_info.append(infoList)

    df_data = pd.concat([df_data, pd.DataFrame(cambridge_info, columns=INFO_ROW.__annotations__.keys())], axis=1)
    df_data.to_csv(path_output, index=False, encoding='utf-8')
    print(f"Cambridge data retrieved and saved to {path_output}")

    return learnerInfo, editorInfo
