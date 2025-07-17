import os
import string
import json
import pandas as pd
import time
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

from utils.files_io import (
    FilePath,
    load_method_config,
    to_serializable,
)


PATHS = FilePath()
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
client = OpenAI()

experimentDoc = load_method_config("llm")


class ExplanationSchema(BaseModel):
    explanation_en: str = Field(description="The comprehensive explanation of the grammar correction in English. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(description="The Traditional Chinese (Taiwan) translation of the English explanation.")

    example_en: List[str] = Field(description="A list of exactly two English example sentences: the first for the original learner's word, the second for the editor's recommended word, showcasing their general usage.")
    example_zh_tw: List[str] = Field(description="The Traditional Chinese (Taiwan) translation of the two sentences in 'example_en'.")


class KnowledgeMatchSchema(BaseModel):
    matched_items: List[str]


class CollocationSchema(BaseModel):
    explanation_en: str = Field(description="The comprehensive explanation of the grammar correction in English. Strictly follow the word limit.")
    explanation_zh_tw: str = Field(
        description="The Traditional Chinese (Taiwan) translation of the English explanation in 'explanation_en'. If 'explanation_en' is empty or translation fails, this field should be an empty string (\"\")."
    )

    example_en: List[str] = Field(description="A list of exactly two English example sentences: the first for the original learner's word, the second for the editor's recommended word, showcasing their general usage.")
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


def get_placeholders(prompt):
    formatter = string.Formatter()
    placeholders = {field_name: "" for _, field_name, _, _ in formatter.parse(prompt) if field_name}
    print(f"Placeholders in the prompt: {json.dumps(placeholders, indent=2)}")
    print('=' * 50)
    return placeholders


########################################
#      Chat API
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


########################################
#      format prompts
########################################
def format_prompt_baseline(row, prompt, sentence_type='t', output_len='fifty', role="linguist", isV2=True):
    print(f"In format_prompt_baseline(): {row}")
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
    print(f"In format_prompt_ragL2_causes(): {row}")
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
            "causes_of_lexical_errors": row['causes'],
            "features_of_academic_writing": row['academic_writing']
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
            "causes_of_lexical_errors": row['causes'],
            "features_of_academic_writing": row['academic_writing'],
            "related_collocations": row['collocations'],
            "other_category_collocations": '\n'.join([str(item) for item in row['other_categories_formatted_json']])
        }
    return prompt.format(**p)


########################################
#      Performance Test
########################################
def performance_test(experiment_name, prompt, role, output_path, run=False, modelType="gpt-4.1-nano", schema=ExplanationSchema, temp=0.0, effort="medium"):
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
