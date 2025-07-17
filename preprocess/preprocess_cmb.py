import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import json
from tqdm import tqdm

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


class ROW:
    word: str
    pos_1: str
    pos_2: str
    level: str
    headword: str
    guideword: str
    dict_eng_def: str
    dict_cn_def: str
    eng_sentence: str
    cn_sentence: str
    # in_context_eng_def: str
    # in_context_cn_def: str


CSV_FORMAT = list[ROW]

###########################################################
# #  Old version of Cambridge dictionary preprocessing  # #
###########################################################

# Load json format dictionary
# def load_dictionary(dictionary_name, type):
#     data = None
#     with open(f"./data/{dictionary_name}/{dictionary_name}.{type}.json") as f:
#         data = json.load(f)
#     return data


# def add_word_in_examples(dictionary):
#     """
#     Get examples from the dictionary (json format)
#     input:
#         dictionary: the dictionary we want to get examples from
#     output:
#         examples: a list of all examples in the dictionary
#     """
#     for word in dictionary.keys():
#         for pos in dictionary[word]:
#             for big_sense_list in dictionary[word][pos]:
#                 for big_sense in big_sense_list["big_sense"]:
#                     for sense in big_sense["sense"]:
#                         for content in sense["examples"]:
#                             example = content["en"]
#                             content["en"] = [word, example]
#                             print(content["en"])


# def get_a_pair(dictionary):
#     for word in dictionary.keys():
#         for pos in dictionary[word]:
#             for big_sense_list in dictionary[word][pos]:
#                 for big_sense in big_sense_list["big_sense"]:
#                     for sense in big_sense["sense"]:
#                         for content in sense["examples"]:
#                             en_example = content["en"]
#                             cn_example = content["ch"]
#                             print(word + "(" + pos + ")", en_example, cn_example)
#                             return word, pos, en_example, cn_example


# def save_rows(dictionary_name, dictionary):
#     rows = []
#     for word in dictionary.keys():
#         for pos in dictionary[word]:
#             for big_sense_list in dictionary[word][pos]:
#                 for big_sense in big_sense_list["big_sense"]:
#                     for sense in big_sense["sense"]:
#                         one_pair = []
#                         for content in sense["examples"]:
#                             en_example = content["en"]
#                             cn_example = content["ch"]
#                             if pos == "":
#                                 pos = " "
#                             one_pair.append([word, pos, en_example, cn_example])
#                         if one_pair:
#                             rows.extend(one_pair)
#     with open(f"./data/{dictionary_name}/{dictionary_name}.rows.txt", "w") as f:
#         for pair in rows:
#             f.write(f"{pair[0]}\t{pair[1]}\t{pair[2]}\t{pair[3]}\n")


# def read_rows(dictionary_name, num_lines):
#     cnt = 0
#     with open(f"./data/{dictionary_name}/{dictionary_name}.rows.txt", "r") as f:
#         lines = f.readlines()
#         for line in lines:
#             if cnt < num_lines:
#                 yield line
#                 cnt += 1
#             else:
#                 break


# def get_chinese_word(keyword, pos, en_sentence, cn_sentence):
#     input_text = f"""You are provided with an English sentence that includes a specific keyword and its part of speech: {keyword} ({pos}): {en_sentence}. Additionally, there is a corresponding Chinese sentence: {cn_sentence}.

# Your task is to identify and extract the exact Chinese word from the Chinese sentence that matches the meaning of the English keyword in this context.

# Please respond with only the Chinese word followed by its part of speech in parentheses. No additional explanation or reasoning is needed.

# - If the part of speech is not provided, use '無' (which means 'none').

# Example:
# - Keyword and Part of Speech:** 'bank' (noun)
# - English Sentence: 'I went to the bank to withdraw money.'
# - Chinese Sentence: '我到銀行提款。'
# - Correct Answer: 銀行（名詞）

# Your Answer:"""
#     completion = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[
#             {
#                 "role": "system",
#                 "content": input_text,
#             },
#         ],
#     )

#     return completion.choices[0].message


###########################################################
# #  New version of Cambridge dictionary preprocessing  # #
###########################################################
def convert_to_csv(filename):
    data = None
    with open(f"{filename}") as f:
        data = json.load(f)
    # print(f"filename: {filename}")

    rows: CSV_FORMAT = []
    for word in tqdm(data, desc="Word", colour="green"):
        # print(f"data[word]: {data[word].keys()}")
        for pos_group in data[word]["poses"]:
            # print(f"pos_group: {pos_group.keys()}")
            # one_pos = pos_group["pos"][0]
            # print(pos_group["pos"])

            for big_sense_group in pos_group["big_sense"]:
                # print(f"big_sense_group: {big_sense_group.keys()}")

                for sense_group in big_sense_group["senses"]:
                    # print(f"sense_group: {sense_group.keys()}")
                    # eng_def = sense_group["eng_def"]
                    # cn_def = sense_group["cn_def"]
                    # print(sense_group)
                    for sentences in sense_group["examples"]:
                        # print(f"sentences: {sentences.keys()}")

                        # one_en_sentence = sentences["eng_examp"]
                        # one_cn_sentence = sentences["cn_examp"]
                        # rows.append({"word": word, "pos": one_pos,
                        #               "eng_def": sense_group["eng_def"], "cn_def": sense_group["cn_def"],
                        #               "eng_sentence": sentences["eng_examp"], "cn_sentence": sentences["cn_examp"],
                        #               "in_context_eng_def": "", "in_context_cn_def": ""})
                        rows.append({
                            "word": word.lower(), "pos_1": pos_group["pos"][0].lower(), "pos_2": pos_group["pos"][1].lower(),
                            "level": sense_group["level"],
                            "headword": pos_group["headword"].lower(), "guideword": big_sense_group["guideword"].lower(),
                            "dict_eng_def": sense_group["eng_def"].lower(), "dict_cn_def": sense_group["cn_def"].lower(),
                            "eng_sentence": sentences["eng_examp"].lower(), "cn_sentence": sentences["cn_examp"].lower()
                            })
                        # pprint(rows[-1])

                for phrase_group in big_sense_group["phrases"]:
                    # print(f"phrase_group: {phrase_group.keys()}")
                    # eng_def = phrase_group["eng_def"]
                    # cn_def = phrase_group["cn_def"]
                    # print(phrase_group)
                    for content in phrase_group["phrase_definitions"]:
                        # print(f"content: {content.keys()}")
                        for sentences in content["examples"]:
                            # print(f"sentences: {sentences.keys()}")

                            # one_en_sentence = sentences["eng_examp"]
                            # one_cn_sentence = sentences["cn_examp"]
                            rows.append({
                                "word": phrase_group["term"].lower(), "pos_1": pos_group["pos"][0].lower(), "pos_2": pos_group["pos"][1].lower(),
                                "level": sense_group["level"].lower(),
                                "headword": pos_group["headword"].lower(), "guideword": big_sense_group["guideword"].lower(),
                                "dict_eng_def": sense_group["eng_def"].lower(), "dict_cn_def": sense_group["cn_def"].lower(),
                                "eng_sentence": sentences["eng_examp"].lower(), "cn_sentence": sentences["cn_examp"].lower()
                                })

                            # pprint({"word": phrase_group["term"], "pos_1": pos_group["pos"][0], "pos_2": pos_group["pos"][1],
                            #               "headword": pos_group["headword"], "guideword": big_sense_group["guideword"],
                            #               "dict_eng_def": sense_group["eng_def"], "dict_cn_def": sense_group["cn_def"],
                            #               "eng_sentence": sentences["eng_examp"], "cn_sentence": sentences["cn_examp"]
                            #               })
                            # pprint(rows[-1])
                            # break
            # break
    return rows


def get_example_sentences(filepath):
    """
    Get example sentences from the Cambridge dictionary
    :param filepath: the path of the file
    """
    filepath_output = "/home/nlplab/atwolin/thesis/data/example-sentences/examples.cambridge.v2.txt"
    content = set()

    df = pd.read_csv(filepath)
    for index, row in df.iterrows():
        content.add(str(row['eng_sentence']))

    print(f"Extracted {len(content)} English sentences")
    with open(filepath_output, "w") as f:
        for line in content:
            f.write(line + "\n")
    print(f"Finish storing sentences to {filepath_output}")


if __name__ == "__main__":
    # Match example sentences with definitions and other information
    # path_dic = "../../data/cambridge-parse/vocabV2"
    # dataset = pd.DataFrame(columns=ROW.__annotations__.keys())
    # for file in tqdm(os.listdir(path_dic), desc="Category"):
    #     data = convert_to_csv(os.path.join(path_dic, file))
    #     dataset = dataset._append(data, ignore_index=True)
        # break
    # print(dataset.head())
    # print(dataset.info())
    # dataset = dataset.drop_duplicates()  # Check duplicated examples
    filepath_words = "../../data/cambridge-parse/cambridge_parse.words.v2.csv"
    # dataset.to_csv(filepath_words, index=False)

    # Extract example sentences from the dictionary
    get_example_sentences(filepath_words)
