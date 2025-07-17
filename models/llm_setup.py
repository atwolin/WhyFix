import os
import regex as re
from dotenv import load_dotenv
from datetime import datetime
import string
import pandas as pd
import json
import yaml
import spacy
import time

from typing import List
from pydantic import BaseModel, Field
# from tooldantic import ToolBaseModel, OpenAiResponseFormatGenerator

from openai import OpenAI

from process.retrieval.retrieve_1_dictionary import (
    get_top_similar_sentences,
    get_infomation_in_cambridge_csv,
    check_target_word_in_akl,
    )

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

client = OpenAI()
nlp = spacy.load("en_core_web_sm")


class FilePath():
    def __init__(self, embedding_size):
        # CLC FEC dataset
        self.folderPath_thesis = "/home/nlplab/atwolin/thesis"

        self.folderPath_fce = os.path.join(self.folderPath_thesis, "data/fce-released-dataset/data")
        self.filePath_fce_sentences = os.path.join(self.folderPath_fce, "fce_sentences.csv")
        self.filePath_fce_sentences_withIndex = os.path.join(self.folderPath_fce, "fce_sentences_with_index.csv")
        # --- information files ---
        self.filePath_fce_dictInfo = os.path.join(self.folderPath_fce, "fce_sentences_dictionary.csv")
        self.filePath_fce_dictInfo_filtered = os.path.join(self.folderPath_fce, "fce_sentences_dictionary_filtered.csv")
        self.filePath_fce_withCollocation = os.path.join(self.folderPath_fce, f"fce_sentences_with_collocation_{embedding_size}.json")
        # self.filePath_fce_allInfo = os.path.join(self.folderPath_fce, "fce_sentences_information.csv")  # dictionary + L2 knowledge
        # --- sample files ---
        self.filePath_fce_sample = os.path.join(self.folderPath_fce, "fce_sample.csv")
        self.filePath_fce_sample_filtered = os.path.join(self.folderPath_fce, "fce_sample_filtered.csv")
        self.filePath_fce_sample_withCollocation = os.path.join(self.folderPath_fce, f"fce_sample_with_collocation_{embedding_size}.json")

        # Longman Dictionary of Common Errors
        self.folderPath_longman = os.path.join(self.folderPath_thesis, "data/longman")
        self.filePath_longman_sentences = os.path.join(self.folderPath_longman, "longman_sentences.csv")
        self.filePath_longman_sentences_withIndex = os.path.join(self.folderPath_longman, "longman_sentences_with_index.csv")
        self.filePath_longman_withCollocation = os.path.join(self.folderPath_longman, f"longman_sentences_with_collocation_{embedding_size}.json")
        # --- information files ---
        self.filePath_longman_dictInfo = os.path.join(self.folderPath_longman, "longman_sentences_dictionary.csv")
        self.filePath_longman_dictInfo_filtered = os.path.join(self.folderPath_longman, "longman_sentences_dictionary_filtered.csv")
        self.filePath_longman_original = os.path.join(self.folderPath_longman, "longman_sentences_dictionary_filtered_groundtruth.csv")
        self.filePath_longman_dictInfo_one_replace = os.path.join(self.folderPath_longman, "longman_sentences_dictionary_one_replace.csv")
        # self.filePath_longman_allInfo = os.path.join(self.folderPath_longman, "longman_sentences_information.csv")
        # --- sample files ---
        self.filePath_longman_sample = os.path.join(self.folderPath_longman, "longman_sample.csv")
        self.filePath_longman_sample_filtered = os.path.join(self.folderPath_longman, "longman_sample_filtered.csv")
        self.filePath_longman_sample_withCollocation = os.path.join(self.folderPath_longman, "longman_sample_with_collocation.json")
        self.filePath_longman_sample_one_replace = os.path.join(self.folderPath_longman, "longman_sample_one_replace.csv")
        self.filePath_longman_sample_one_replace_withCollocation = os.path.join(self.folderPath_longman, f"longman_sample_one_replace_with_collocation_{embedding_size}.json")

        # Test
        self.folderPath_test = os.path.join(self.folderPath_thesis, "data/system-test")
        self.filePath_test = os.path.join(self.folderPath_test, "test.csv")
        # --- information files ---
        self.filePath_test_dictInfo = os.path.join(self.folderPath_test, "test_dictionary.csv")
        self.filePath_test_withCollocation = os.path.join(self.folderPath_test, f"test_with_collocation_{embedding_size}.json")

        # Collocations
        self.folderPath_collocation_corpus = os.path.join(self.folderPath_thesis, "data/collocation-corpus")
        self.filePath_collocation_corpus = os.path.join(self.folderPath_collocation_corpus, "data.txt")

        self.folderPath_longman_performanceTest = os.path.join(self.folderPath_longman, "performance-test-v2")
        os.makedirs(self.folderPath_longman_performanceTest, exist_ok=True)

        self.folderPath_batch = os.path.join(self.folderPath_thesis, "data/batch")
        self.folderPath_batchFile = os.path.join(self.folderPath_batch, "batchFile")
        self.folderPath_batchResult = os.path.join(self.folderPath_batch, "batchResult")

        # Structured data
        self.folderPath_structured_data = os.path.join(self.folderPath_thesis, "data/results/structured_data")

        # Application folder
        self.folderPath_app = os.path.join(self.folderPath_thesis, "code/web/backend/app/data")


class ExplanationSchema(BaseModel):
    explanation_en: str = Field(description="The comprehensive explanation of the grammar correction in English. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(description="The Traditional Chinese (Taiwan) translation of the English explanation.")

    example_en: List[str] = Field(description="Do NOT contain markers([--] and {++}). A list of exactly two English example sentences: the first for the original learner's word, the second for the editor's recommended word, showcasing their general usage.")
    example_zh_tw: List[str] = Field(description="The Traditional Chinese (Taiwan) translation of the two sentences in 'example_en'.")


class SimpleRagSchema(BaseModel):
    class Explanation(BaseModel):
        majorPremise: str
        minorPremise: str
        # conclusion: str

    explanation_en: Explanation
    # explanation_zh_tw: Explanation
    example_en: List[str]
    example_zh_tw: List[str]


class SimpleRagExtractSchema(BaseModel):
    explanation_en: str
    explanation_zh_tw: str


class KnowledgeMatchSchema(BaseModel):
    matched_items: List[str]


class CollocationSchema(BaseModel):
    explanation_en: str = Field(description="The comprehensive explanation of the grammar correction in English. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(
        description="The Traditional Chinese (Taiwan) translation of the English explanation in 'explanation_en'. If 'explanation_en' is empty or translation fails, this field should be an empty string (\"\")."
    )

    example_en: List[str] = Field(description="Do NOT contain markers([--] and {++}). A list of exactly two English example sentences: the first for the original learner's word, the second for the editor's recommended word, showcasing their general usage.")
    example_zh_tw: List[str] = Field(
        description="The Traditional Chinese (Taiwan) translation of the two sentences in 'example_en', maintaining a list of two strings. If 'example_en' is not a list of two valid sentences or if translation of either sentence fails, this field should be an empty list ([])."
    )

    corresponding_collocation_en: List[str] = Field(
        description="A list containing exactly one string. This string represents the error leading to the corrected collocation, with the part-of-speech (POS) of the corrected component. Format: '[error_component] -> [correct_component] ([POS]) + [category]', where [POS] is the error_component_pos. Example: 'accomplish -> achieve (VERB) + Goal'. Should be an empty list if source data is empty or no relevant entry is found."
    )
    corresponding_collocation_examples_en: List[str] = Field(
        description="A list of exactly two strings for the most relevant related collocation, representing alternative pivot usage or a self-created example if needed. Each string must be formatted as '[error_component] -> [correct_component] [pivot_phrase_or_concept]'. Example: ['accomplish -> achieve success', 'accomplish -> attain victory']. Should be an empty list if source data is empty or no relevant entry is found."
    )

    # corresponding_collocation_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the string in 'corresponding_collocation_en', maintaining the format including the POS tag (e.g., using '(動詞)' for verbs). Example: ['完成 -> 達成 (動詞) + 一個目標']. If 'corresponding_collocation_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )
    # corresponding_collocation_examples_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the two strings in 'corresponding_collocation_examples_en', maintaining their specified format. Example: ['完成 -> 取得成功', '完成 -> 獲得勝利']. If 'corresponding_collocation_examples_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )

    other_collocations_en: List[str] = Field(
        description="A list of 'component_change_category'-like strings (up to three) selected from the 'other_category_collocations' data. Each string formatted as '[error_component] -> [correct_component] ([POS]) + [Pivot_Category_Name]', where [POS] is the error_component_pos from the source entry. Example: ['accomplish -> make (VERB) + Decision', 'accomplish -> bring (VERB) + Change', ...]. Should be an empty list if source data is empty."
    )
    other_collocations_examples_en: List[str] = Field(
        description="A list of strings (up to three). Each string represents a corrected alternative collocation from the 'other_category_collocations' data, formatted similarly to strings in `corresponding_collocation_examples_en`: `'[error_component] -> [correct_component] [pivot_phrase_or_concept]'`. The [pivot_phrase_or_concept] should be derived from the [Pivot_Category_Name] of the source entry. Example: ['accomplish -> make decision', 'accomplish -> bring change', 'accomplish -> carry out plan']. Should be an empty list if source data is empty."
    )

    # other_collocations_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the strings in 'other_collocations_en', maintaining the format including the POS tag (e.g., using '(動詞)' for verbs). Example: ['完成 -> 做出 (動詞) + 決定', ...]. If 'other_collocations_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )
    # other_collocations_examples_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the strings in `other_collocations_examples_en`, maintaining their specified format. Example: ['完成 -> 做出決定', '完成 -> 帶來改變', '完成 -> 執行計畫']. If `other_collocations_examples_en` is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )


class GeneralExplanationSchema(BaseModel):
    explanation_en: str = Field(description="A comprehensive explanation in English of the key differences between the original and alternative sentences. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(description="The Traditional Chinese (Taiwan) translation of the English explanation provided in 'explanation_en'.")

    example_en: List[str] = Field(
        description="A list of exactly two English example sentences. The first demonstrates typical usage of the specific word or phrase from the original sentence (identified by '[-...-]'). The second demonstrates typical usage of the corresponding word or phrase from the alternative sentence (identified by '{{+... +}}')."
    )
    example_zh_tw: List[str] = Field(
        description="The Traditional Chinese (Taiwan) translation of the two English example sentences provided in 'example_en'."
    )


class GeneralCollocationSchema(BaseModel):
    explanation_en: str = Field(description="A comprehensive explanation in English of the key differences between the original and alternative sentences. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(description="The Traditional Chinese (Taiwan) translation of the English explanation provided in 'explanation_en'.")

    example_en: List[str] = Field(
        description="A list of exactly two English example sentences. The first demonstrates typical usage of the specific word or phrase from the original sentence (identified by '[-...-]'). The second demonstrates typical usage of the corresponding word or phrase from the alternative sentence (identified by '{{+... +}}')."
    )
    example_zh_tw: List[str] = Field(
        description="The Traditional Chinese (Taiwan) translation of the two English example sentences provided in 'example_en'."
    )

    corresponding_collocation_en: List[str] = Field(
        description="A list containing exactly one string. This string represents the error leading to the corrected collocation, with the part-of-speech (POS) of the corrected component. Format: '[error_component] -> [correct_component] ([POS]) + [category]', where [POS] is the error_component_pos. Example: 'accomplish -> achieve (VERB) + Goal'. Should be an empty list if source data is empty or no relevant entry is found."
    )
    corresponding_collocation_examples_en: List[str] = Field(
        description="A list of exactly two strings for the most relevant related collocation, representing alternative pivot usage or a self-created example if needed. Each string must be formatted as '[error_component] -> [correct_component] [pivot_phrase_or_concept]'. Example: ['accomplish -> achieve success', 'accomplish -> attain victory']. Should be an empty list if source data is empty or no relevant entry is found."
    )

    # corresponding_collocation_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the string in 'corresponding_collocation_en', maintaining the format including the POS tag (e.g., using '(動詞)' for verbs). Example: ['完成 -> 達成 (動詞) + 一個目標']. If 'corresponding_collocation_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )
    # corresponding_collocation_examples_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the two strings in 'corresponding_collocation_examples_en', maintaining their specified format. Example: ['完成 -> 取得成功', '完成 -> 獲得勝利']. If 'corresponding_collocation_examples_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )

    other_collocations_en: List[str] = Field(
        description="A list of 'component_change_category'-like strings (up to three) selected from the 'other_category_collocations' data. Each string formatted as '[error_component] -> [correct_component] ([POS]) + [Pivot_Category_Name]', where [POS] is the error_component_pos from the source entry. Example: ['accomplish -> make (VERB) + Decision', 'accomplish -> bring (VERB) + Change', ...]. Should be an empty list if source data is empty."
    )
    other_collocations_examples_en: List[str] = Field(
        description="A list of strings (up to three). Each string represents a corrected alternative collocation from the 'other_category_collocations' data, formatted similarly to strings in `corresponding_collocation_examples_en`: `'[error_component] -> [correct_component] [pivot_phrase_or_concept]'`. The [pivot_phrase_or_concept] should be derived from the [Pivot_Category_Name] of the source entry. Example: ['accomplish -> make decision', 'accomplish -> bring change', 'accomplish -> carry out plan']. Should be an empty list if source data is empty."
    )

    # other_collocations_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the strings in 'other_collocations_en', maintaining the format including the POS tag (e.g., using '(動詞)' for verbs). Example: ['完成 -> 做出 (動詞) + 決定', ...]. If 'other_collocations_en' is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )
    # other_collocations_examples_zh_tw: List[str] = Field(
    #     description="The Traditional Chinese (Taiwan) translation of the strings in `other_collocations_examples_en`, maintaining their specified format. Example: ['完成 -> 做出決定', '完成 -> 帶來改變', '完成 -> 執行計畫']. If `other_collocations_examples_en` is empty or its content cannot be properly translated and formatted, this field should be an empty list ([])."
    # )


def load_experiment():
    experiment = {}
    with open("/home/nlplab/atwolin/thesis/code/model/experiment.yaml", "r") as f:
        experiment = yaml.load(f, Loader=yaml.FullLoader)
    return experiment


experimentDoc = load_experiment()


def pretty_print(text):
    for idx, word in enumerate(text.split()):
        if idx % 15 == 0:
            print()
        print(word, end=' ')


def get_placeholders(prompt):
    formatter = string.Formatter()
    placeholders = {field_name: "" for _, field_name, _, _ in formatter.parse(prompt) if field_name}
    print(f"Placeholders in the prompt: {json.dumps(placeholders, indent=2)}")
    print('=' * 50)
    return placeholders


def get_support_material(row):
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


########################################
#      chat functions
########################################
def chat_response(prompt, run=True, hasSystem=False, systemPrompt="", modelType="gpt-4.1-nano", temp=0.0, formatSchema="", effort="medium"):
    if modelType.startswith('o'):
        print(f"========================\n{temp}, {modelType}, {effort}\n{prompt}\n========================")
    else:
        print(f"========================\n{temp}, {modelType}\n{prompt}\n========================")
    print(f"Schema: {json.dumps(formatSchema.schema(), indent=4)}")
    if not run:
        return ""

    print(f"{'=' * 50}\nStarting to get response\n{'=' * 50}")
    response = None
    if modelType.startswith('o'):
        response = client.responses.parse(
            model="o3-mini",
            input=prompt,
            reasoning={
                "effort": effort
            },
            text_format=formatSchema,
        )
    elif hasSystem:
        response = client.responses.parse(
            model=modelType,
            temperature=temp,
            instructions=systemPrompt,
            input=prompt,
            text_format=formatSchema,
        )
    else:
        response = client.responses.parse(
            model=modelType,
            temperature=temp,
            input=prompt,
            text_format=formatSchema,
        )
    return response


def chat_completion(prompt, run=True, hasSystem=True, systemPrompt="", modelType="gpt-4o-mini", returnedNum=1, temp=0.0):
    print(f"========================\n{temp}, {modelType}\n{prompt}\n========================")
    if not run:
        return ""

    completion = None
    if hasSystem:
        completion = client.chat.completions.create(
            model=modelType,
            n=returnedNum,
            temperature=temp,
            response_format={
                "type": "json_object"
            },
            messages=[
                {"role": "system", "content": systemPrompt},
                {"role": "user", "content": prompt}
            ]
        )
    else:
        completion = client.chat.completions.create(
            model=modelType,
            n=returnedNum,
            temperature=temp,
            response_format={
                "type": "json_object"
            },
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    return completion


########################################
#      collocation functions
########################################
def get_example_dataframe(keyword, df):
    df_example = df[
        df.apply(lambda x: (
            isinstance(x['learner_word'], str) and keyword in x['learner_word'].lower()) or (
            isinstance(x['editor_word'], str) and keyword in x['editor_word'].lower()), axis=1)
    ]
    return df_example


def get_collocated_words(df_data):
    def re_word(row):
        sentence = row['learner_sentence'].lower().split()
        # print(f"Sentence length: {len(sentence)}")
        for idx, word in enumerate(sentence):
            # print(f"idx: {idx}, word: {word}, target: {row['learner_word']}")
            cur_word = re.sub(r'(\p{P})', r'', word)
            if cur_word == row['learner_word'] and idx + 1 < len(sentence):
                # print(re.sub(r'(\p{P})', r'', sentence[idx + 1]))
                return re.sub(r'(\p{P})', r'', sentence[idx + 1])
        return "none"

    collocated_words = df_data.apply(re_word, axis=1)
    collocated_words = collocated_words.tolist()
    return collocated_words


def lemmatize_word_list(words: list):
    doc = []
    # for word in words:
    #     word = word.lower().strip() if isinstance(word, str) else word
    #     lemmar_word = nlp(word)
    #     lemmared_word = [token.lemma_ for token in lemmar_word if token.lemma_ not in string.punctuation]
    #     print(lemmared_word, type(lemmared_word))
    #     doc.append(lemmar_word)
    words = [word.lower().strip() if isinstance(word, str) else word for word in words]
    doc = nlp(" ".join(words))
    doc = [token.lemma_ for token in doc if token.lemma_ not in string.punctuation]
    return doc


def get_related_collocations(df_collocation, df_data):
    def find_collocation(row):
        entry_words = [lemmatize_word_list(row['learner_word'])]
        mask = (df_collocation['entry'].isin(entry_words))
        df_target = df_collocation.loc[mask]
        collocations = df_target.apply(lambda x: str(x['original_collocation']) + " -> " + str(x['edited_collocation']), axis=1).tolist()
        collocations = ','.join(collocations)
        return collocations

    collocations_list = []
    collocations_list = df_data.apply(find_collocation, axis=1)

    return collocations_list


########################################
#      methods functions
########################################
def format_prompt_baseline(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    sentences = row["learner_sentence"]
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    if not isV2:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "learnerWord": row['learner_sentence'],
            "editorWord": row['editor_sentence'],

            "output_len": experimentDoc['output_len'][output_len]
        }
    else:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "edited_sentence": row['editor_sentence'],
            "formatted_sentences": row['formatted_sentence'],

            "output_len": experimentDoc['output_len'][output_len]
        }
    return prompt.format(**p)


def format_prompt_ragDictionary(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    sentences = row["learner_sentence"]
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    if not isV2:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "learnerWord": row['learner_sentence'],
            "editorWord": row['editor_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "lemmaWord_learner": row['lemmaWord_learner'],
            "learnerWord_pos": row['pos_learner'],
            "learnerWord_level": row['level_learner'],
            "learnerWord_definition": row['definition_learner'],
            "learnerWord_examples": row['examples_learner'],
            "learnerWord_in_akl": row['in_akl_learner'],
            "lemmaWord_editor": row['lemmaWord_editor'],
            "editorWord_pos": row['pos_editor'],
            "editorWord_level": row['level_editor'],
            "editorWord_definition": row['definition_editor'],
            "editorWord_examples": row['examples_editor'],
            "editorWord_in_akl": row['in_akl_editor'],
        }
    else:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "edited_sentence": row['editor_sentence'],
            "formatted_sentences": row['formatted_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "lemmaWord_learner": row['lemmaWord_learner'],
            "learnerWord_pos": row['pos_learner'],
            "learnerWord_level": row['level_learner'],
            "learnerWord_definition": row['definition_learner'],
            "learnerWord_examples": row['examples_learner'],
            "learnerWord_in_akl": row['in_akl_learner'],
            "lemmaWord_editor": row['lemmaWord_editor'],
            "editorWord_pos": row['pos_editor'],
            "editorWord_level": row['level_editor'],
            "editorWord_definition": row['definition_editor'],
            "editorWord_examples": row['examples_editor'],
            "editorWord_in_akl": row['in_akl_editor'],
        }
    return prompt.format(**p)


def format_prompt_ragL2_causes(row, prompt, sentence_type='t', role="linguist"):
    sentences = row['formatted_sentence']
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    p = {
        "role": experimentDoc['role'][role],
        "sentences": sentences,
        "causes": experimentDoc['l2_knowledge']['causes']
    }
    return prompt.format(**p)


def format_prompt_ragL2_academicWriting(row, prompt, sentence_type='t', role="linguist"):
    sentences = row['formatted_sentence']
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    p = {
        "role": experimentDoc['role'][role],
        "sentences": sentences,
        "academic_writing": experimentDoc['l2_knowledge']['academic_writing']
    }
    return prompt.format(**p)


def format_prompt_ragL2_explanation(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    sentences = row["learner_sentence"]
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    if not isV2:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "learnerWord": row['learner_sentence'],
            "editorWord": row['editor_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "causes_of_lexical_errors": row['causes'],
            "features_of_academic_writing": row['academic_writing']
        }
    else:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "edited_sentence": row['editor_sentence'],
            "formatted_sentences": row['formatted_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "causes_of_lexical_errors": row[f'causes_{output_len}'],
            "features_of_academic_writing": row[f'academic_writing_{output_len}']
        }
    return prompt.format(**p)


def format_prompt_ragCollocation(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    sentences = row["learner_sentence"]
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    if not isV2:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "learnerWord": row['learner_sentence'],
            "editorWord": row['editor_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "related_collocations": row['collocations'],
        }
    else:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "edited_sentence": row['editor_sentence'],
            "formatted_sentences": row['formatted_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "related_collocations": row['collocations'],
            "other_category_collocations": '\n'.join([str(item) for item in row['other_categories_formatted_json']])
        }
    return prompt.format(**p)


def format_prompt_ragMix(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    sentences = row["learner_sentence"]
    if sentence_type == "tf":
        sentences += ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else sentences
    if not isV2:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "learnerWord": row['learner_sentence'],
            "editorWord": row['editor_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "related_collocations": row['collocations'],
        }
    else:
        p = {
            "role": experimentDoc['role'][role],

            "sentence": sentences,
            "edited_sentence": row['editor_sentence'],
            "formatted_sentences": row['formatted_sentence'],

            "output_len": experimentDoc['output_len'][output_len],
            "lemmaWord_learner": row['lemmaWord_learner'],
            "learnerWord_pos": row['pos_learner'],
            "learnerWord_level": row['level_learner'],
            "learnerWord_definition": row['definition_learner'],
            "learnerWord_examples": row['examples_learner'],
            "learnerWord_in_akl": row['in_akl_learner'],
            "lemmaWord_editor": row['lemmaWord_editor'],
            "editorWord_pos": row['pos_editor'],
            "editorWord_level": row['level_editor'],
            "editorWord_definition": row['definition_editor'],
            "editorWord_examples": row['examples_editor'],
            "editorWord_in_akl": row['in_akl_editor'],
            "causes_of_lexical_errors": row[f'causes_{output_len}'],
            "features_of_academic_writing": row[f'academic_writing_{output_len}'],
            "related_collocations": row['collocations'],
            "other_category_collocations": '\n'.join([str(item) for item in row['other_categories_formatted_json']])
        }
    return prompt.format(**p)


def performance_test(experiment_name, prompt, role, output_path, schema, run=False, modelType="gpt-4.1-nano", temp=0.0, effort="medium"):
    print(f"{experiment_name}, save to {output_path}")
    timings = {}

    # GPT-4.1-nano
    start = time.perf_counter()
    response_4_1_nano = chat_response(prompt, run=run, hasSystem=False, modelType="gpt-4.1-nano", temp=temp, formatSchema=schema)
    timings["gpt-4.1-nano"] = time.perf_counter() - start

    # GPT-4.1-mini
    start = time.perf_counter()
    response_4_1_mini = chat_response(prompt, run=run, hasSystem=False, modelType="gpt-4.1-mini", temp=temp, formatSchema=schema)
    timings["gpt-4.1-mini"] = time.perf_counter() - start

    # GPT-4.1
    start = time.perf_counter()
    response_4_1 = chat_response(prompt, run=run, hasSystem=False, modelType="gpt-4.1", temp=temp, formatSchema=schema)
    timings["gpt-4.1"] = time.perf_counter() - start

    # GPT-4o-mini
    start = time.perf_counter()
    response_4o_mini = chat_response(prompt, run=run, hasSystem=False, modelType="gpt-4o-mini", temp=temp, formatSchema=schema)
    timings["gpt-4o-mini"] = time.perf_counter() - start

    # GPT-4o
    start = time.perf_counter()
    response_4o = chat_response(prompt, run=run, hasSystem=False, modelType="gpt-4o", temp=temp, formatSchema=schema)
    timings["gpt-4o"] = time.perf_counter() - start

    # o4-mini-medium
    start = time.perf_counter()
    response_o4_mini_medium = chat_response(prompt, run=run, hasSystem=False, modelType="o4-mini", temp=temp, formatSchema=schema, effort="medium")
    timings["o4-mini-medium"] = time.perf_counter() - start

    # o4-mini-high
    start = time.perf_counter()
    response_o4_mini_high = chat_response(prompt, run=run, hasSystem=False, modelType="o4-mini", temp=temp, formatSchema=schema, effort="high")
    timings["o4-mini-high"] = time.perf_counter() - start

    # o3-mini-medium
    start = time.perf_counter()
    response_o3_mini_medium = chat_response(prompt, run=run, hasSystem=False, modelType="o3-mini", temp=temp, formatSchema=schema, effort="medium")
    timings["o3-mini-medium"] = time.perf_counter() - start

    # o3-mini-high
    start = time.perf_counter()
    response_o3_mini_high = chat_response(prompt, run=run, hasSystem=False, modelType="o3-mini", temp=temp, formatSchema=schema, effort="high")
    timings["o3-mini-high"] = time.perf_counter() - start

    print(f"Finished getting responses for {experiment_name}")

    if run:
        print(f"Saving results to {output_path}")
        results = {
            "experiment_name": experiment_name,
            "role": role,
            "prompt": prompt,

            "timings": timings,

            "outputs": {
                "gpt-4.1-nano": to_serializable(response_4_1_nano.output_parsed),
                "gpt-4.1-mini": to_serializable(response_4_1_mini.output_parsed),
                "gpt-4.1": to_serializable(response_4_1.output_parsed),
                "gpt-4o-mini": to_serializable(response_4o_mini.output_parsed),
                "gpt-4o": to_serializable(response_4o.output_parsed),

                "o4-mini-medium": to_serializable(response_o4_mini_medium.output_parsed),
                "o4-mini-high": to_serializable(response_o4_mini_high.output_parsed),
                "o3-mini-medium": to_serializable(response_o3_mini_medium.output_parsed),
                "o3-mini-high": to_serializable(response_o3_mini_high.output_parsed),
            },

            "responses": {
                "gpt-4.1-nano": to_serializable(response_4_1_nano),
                "gpt-4.1-mini": to_serializable(response_4_1_mini),
                "gpt-4.1": to_serializable(response_4_1),
                "gpt-4o-mini": to_serializable(response_4o_mini),
                "gpt-4o": to_serializable(response_4o),

                "o4-mini-medium": to_serializable(response_o4_mini_medium),
                "o4-mini-high": to_serializable(response_o4_mini_high),
                "o3-mini-medium": to_serializable(response_o3_mini_medium),
                "o3-mini-high": to_serializable(response_o3_mini_high),
            }
        }

        # Save results to JSONL file
        with open(output_path, 'w') as f:
            f.write(json.dumps(results) + '\n')
        print(f"Results saved to {output_path}")


########################################
#      save files functions
########################################
def to_serializable(obj):
    if hasattr(obj, "__dict__"):
        # Recursively convert __dict__ values
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):  # fallback for Pydantic v1
        return obj.dict()
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def store_output_with_corresponding_prompt(learner_word, sentence_type, prompt_type, message, completion, modelType, temp):
    if completion == "":
        return

    os.makedirs(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}", exist_ok=True)

    cnt = 0
    for response in completion.choices:
        print(response.message.content, cnt)
        cnt += 1

    current_time = datetime.now().strftime('%m%d.%H.%M.%S')
    with open(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}/{prompt_type}.output.completion.{sentence_type}.{modelType}.temp_{temp}.{current_time}.txt", "a") as f:
        f.write(f"temp: {temp}\n")
        for response in completion.choices:
            f.write(response.message.content + "\n")

    json_output = {"prompt_type": prompt_type,
                   "sentence_type": sentence_type,
                   "modelType": modelType,
                   "temp": temp,
                   "prompt": message,
                   "reply": completion.to_dict()}
    with open(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}/{prompt_type}.output.fullInfo.{sentence_type}.{modelType}.temp_{temp}.{current_time}.json", "w") as f:
        json.dump(json_output, f)
    print(">>>>> Finished storing output <<<<<")


def store_completion_in_csv(learner_word, sentence_type, prompt_type, query, completion, modelType, temp):
    if completion == "":
        return
    now = datetime.now().strftime('%m%d.%H.%M.%S')
    os.makedirs(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}", exist_ok=True)

    # Store responses
    responses = pd.DataFrame(columns=["prompt_type", "sentence_type", "modelType", "temp", "prompt", "response"])
    data = {}
    for response in completion.choices:
        data["prompt_type"] = prompt_type
        data["sentence_type"] = sentence_type
        data["modelType"] = modelType
        data["temp"] = temp
        data["prompt"] = query
        data["response"] = response.message.content
        responses = responses._append(data, ignore_index=True)
    responses.to_csv(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}/{prompt_type}.response.{now}.csv", index=False)
    print(">>>>> Finished storing output in csv <<<<<")

    # Store completion
    completion_data = {
        "prompt_type": prompt_type,
        "sentence_type": sentence_type,
        "modelType": modelType,
        "temp": temp,
        "prompt": query,
        "completion": completion.to_dict()
    }
    completion_df = pd.DataFrame([completion_data])
    completion_df.to_csv(f"/home/nlplab/atwolin/thesis/code/model/result/{learner_word}/{prompt_type}.fullInfo.{now}.csv", index=False)
    print(">>>>> Finished storing completion in csv <<<<<")


########################################
#      read files functions
########################################
