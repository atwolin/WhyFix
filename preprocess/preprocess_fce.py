#!/usr/bin/python
# -*- coding: utf-8 -*-

#############################################################################
# To extract `R`(`Replace`), `FF`(`False Frend`), `CL`(`CoLlocation error`),
# `ID`(`IDiom error`), and `L`(`inappropriate register/Label`) error types
# from the FCE dataset.
#############################################################################

import os
import re
import copy
from bs4 import BeautifulSoup as bs
import spacy
from tqdm import tqdm
import pandas as pd
import json

from .preprocess_general import DATA_ROW, clean_df, retrieve_cambridge_data

datafolder = "/home/nlplab/atwolin/thesis/data/fce-released-dataset/dataset"
outputFolder = "/home/nlplab/atwolin/thesis/data/fce-released-dataset/data"
nlp = spacy.load("en_core_web_trf")


def extract_sentence_per_coded_answer(learner_answer):
    """
    Extract sentences included the following error types:
    `R`(`Replace`), `FF`(`False Frend`), `CL`(`CoLlocation error`),
    `ID`(`IDiom error`), and `L`(`inappropriate register/Label`)

    :param learner_answer: dict of one coded_answer with metadata
    :return: list of data with error types, coded_answer, and other metadata
    """
    data = []
    data_to_print = []

    sentInDataset = learner_answer["coded_answer"]
    # print(f"@sentInDataset{' ' * (20 - len('@sentInDataset'))}: {sentInDataset}")

    # Get the edited-only essay
    sentInDataset_editor = [copy.deepcopy(s) for s in sentInDataset.find_all('p')]
    for sents in sentInDataset_editor:
        for iTag in sents.find_all('i'):
            iTag.decompose()
    # print(f"@sentInDataset_editor{' ' * (20 - len('@sentInDataset_editor'))}: {len(sentInDataset_editor)}, {sentInDataset_editor}")

    # sentInDataset_editor_text = [s.text for s in sentInDataset_editor]
    # print(f"@sentInDataset_editor_text{' ' * (20 - len('@sentInDataset_editor_text'))}: {len(sentInDataset_editor_text)}, {type(sentInDataset_editor_text)}, {json.dumps(sentInDataset_editor_text, indent=20)}")

    # Split the essay into sentences
    sentInDataset_editor_split = []
    for sents in sentInDataset_editor:
        sents_split = [s.text for s in nlp(sents.text).sents]
        sentInDataset_editor_split.extend(sents_split)
    # print(f"@sentInDataset_editor_split{' ' * (20 - len('@sentInDataset_editor_split'))}: {len(sentInDataset_editor_split)}, {type(sentInDataset_editor_split)}, {json.dumps(sentInDataset_editor_split, indent=20)}")

    # Extract sentences from the coded_answer
    for sents_pTag in sentInDataset.find_all('p'):
        # print(f"@sents_pTag{' ' * (20 - len('@sents_pTag'))}: {sents_pTag}")

        # Collect target error types and their learner's words
        errorTypePairs = []  # error type, learner's word, editor's word
        # errorTypes_occur = []
        errorType_cur = ""   # note[0]
        word_leanerCur = ""  # learner_word
        word_editorCur = ""  # editor_word
        for nsTag in sents_pTag.find_all('NS'):
            errorType_cur = nsTag["type"]
            # print(f"@errorType_cur{' ' * (20 - len('@errorType_cur'))}: {errorType_cur}")
            # print(f"@nsTag{' ' * (20 - len('@nsTag'))}: {nsTag}")
            # errorTypes_occur.append(errorType_cur)
            if errorType_cur.startswith(("CL", "ID", "L", "SA")) or re.match(r"^R[^ACDQTP][JNVY]?", errorType_cur) or re.match(r"^FF[^ACDQT][JNVY]?", errorType_cur):
                # print(f"@nsTag{' ' * (20 - len('@nsTag'))}: {nsTag}")
                sents_learner = copy.deepcopy(sents_pTag)  # learners-only sentences
                sents_format = copy.deepcopy(sents_pTag)   # formatted sentences
                # sents_editor = copy.deepcopy(sents_pTag)   # edited-only sentences

                nsTag_copy = copy.deepcopy(nsTag)
                for nsTag_ in nsTag_copy.find_all('NS'):
                    nsTag_.decompose()
                word_leanerCur = nsTag_copy.find('i')
                if str(word_leanerCur) == '<i/>' or str(word_leanerCur) == '<i> </i>':
                    print(f"@nsTag{' ' * (20 - len('@nsTag'))}: {nsTag}")
                    print(f"@nsTag_find{' ' * (20 - len('@nsTag_find'))}: {nsTag.find('NS')}")
                    word_leanerCur = nsTag.find('NS').find_all('c')[-1]
                word_editorCur = nsTag_copy.find('c')

                # If no in/correct word pair, skip
                if word_leanerCur is None or word_editorCur is None:
                    continue

                errorTypePairs.append([errorType_cur, word_leanerCur.text, word_editorCur.text])
                print(f"@errorTypePair{' ' * (20 - len('@errorTypePair'))}: {errorType_cur}, {word_leanerCur}, {word_leanerCur.text}, {word_editorCur}, {word_editorCur.text}")

                # icPair_count = str(nsTag).count(str(word_leanerCur) + str(word_editorCur))
                # print(f"@icPair_count{' ' * (20 - len('@icPair_count'))}: {icPair_count}")

                # Check if correct word occurs more than once
                replaceCnt = sents_learner.text.count(str(word_editorCur))
                if replaceCnt > 1:
                    # print(f"@replaceCnt{' ' * (20 - len('@replaceCnt'))}: {replaceCnt}")
                    ValueError(f"@replaceCnt{' ' * (20 - len('@replaceCnt'))}: {replaceCnt}")
                    # sents_learner = sents_learner.replace(str(word_editorCur), str(word_leanerCur), 1)

                # Get formatted sentences
                # sents_format = str(sents_format).replace(str(nsTag), f"{{+{word_editorCur.text}+}}[-{word_leanerCur.text}-]")
                sents_format = str(sents_format).replace(str(nsTag), f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}")
                sents_format = bs(sents_format, "xml")
                for iTag in sents_format.find_all('i'):
                    iTag.decompose()
                sents_format = sents_format.text
                # print(f"@sents_format{' ' * (20 - len('@sents_format'))}: {sents_format}")
                sents_format_split = [s.text for s in nlp(sents_format).sents]
                # print(f"@sents_format_split{' ' * (20 - len('@sents_format_split'))}: {json.dumps(sents_format_split, indent=20)}")

                # Get learner, preceding and following sentences
                learner_sentence = ""
                editor_sentence = ""
                formatted_sentence = ""
                preceding_sentence = ""
                following_sentence = ""
                # sents_format_split = sents_format.split()
                print(f"@sents_format_split{' ' * (20 - len('@sents_format_split'))}: {len(sents_format_split)}, {type(sents_format_split)}, {sents_format_split}")

                for idx, sent in enumerate(sents_format_split):
                    for idx_word, word in enumerate(sent.split()):
                        # print(f"@word{' ' * (20 - len('@word'))}: {idx}, {word}")
                        # if "{+" in word:
                        if f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}" == word or \
                           f"{word_editorCur.text.strip().split()[-1]}{word_leanerCur.text.strip().split()[0]}" == word:
                            # word_found = word
                            # idx_word_found = idx_word
                            # while ("+}" not in sent.split()[idx_word_found] and idx_word_found < len(sent.split()) - 1):
                            #     word_found += " " + sent.split()[idx_word_found + 1]
                            #     idx_word_found += 1
                            print(f"found word: {word}")
                            print(f"found sent: {sent}")
                            sent_found = sent.replace(f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}", word_editorCur.text)
                            # print(f"@sent_found{' ' * (20 - len('@sent_found'))}: {sent_found}")

                            for idx_inDataset, sent_inDataset in enumerate(sentInDataset_editor_split):
                                if sent_found in sent_inDataset or sent_inDataset in sent_found:
                                    # if word_editorCur.text in sent_inDataset:
                                    editor_sentence = sentInDataset_editor_split[idx_inDataset]
                                    learner_sentence = editor_sentence.replace(word_editorCur.text, word_leanerCur.text)
                                    formatted_sentence = editor_sentence.replace(word_editorCur.text, f"{{+{word_editorCur.text}+}}[-{word_leanerCur.text}-]")

                                    preceding_sentence = sentInDataset_editor_split[idx_inDataset - 1] if idx_inDataset > 0 else ""
                                    following_sentence = sentInDataset_editor_split[idx_inDataset + 1] if idx_inDataset < len(sentInDataset_editor_split) - 1 else ""

                            # print(f"@editor_sentence{' ' * (20 - len('@editor_sentence'))}: {editor_sentence}")
                            # print(f"@learner_sentence{' ' * (20 - len('@learner_sentence'))}: {learner_sentence}")
                            # print(f"@formatted_sentence{' ' * (20 - len('@formatted_sentence'))}: {formatted_sentence}")

                            # print(f"@preceding_sentence{' ' * (20 - len('@preceding_sentence'))}: {preceding_sentence}")
                            # print(f"@following_sentence{' ' * (20 - len('@following_sentence'))}: {following_sentence}")

                            if learner_sentence == "":
                                print("*****************************************************")
                                print(f"@sentInDataset{' ' * (20 - len('@sentInDataset'))}: {sentInDataset}")
                                print(f"@sentInDataset_editor_split{' ' * (20 - len('@sentInDataset_editor_split'))}: {sentInDataset_editor_split}")

                                print(f"@nsTag{' ' * (20 - len('@nsTag'))}: {nsTag}")
                                print(f"@errorTypePair{' ' * (20 - len('@errorTypePair'))}: {errorType_cur}, {word_leanerCur}, {word_leanerCur.text}, {word_editorCur}, {word_editorCur.text}")

                                print(f"@sents_format_split{' ' * (20 - len('@sents_format_split'))}: {sents_format_split}")
                                print(f"@sent{' ' * (20 - len('@sent'))}: {sent}")
                                print(f"@sent_found{' ' * (20 - len('@sent_found'))}: {sent_found}")
                                print(f"@learner_sentence{' ' * (20 - len('@learner_sentence'))}: {learner_sentence}")
                                print("*****************************************************")

                                # raise ValueError(f"learner_sentence{' ' * (20 - len('learner_sentence'))} == {len(learner_sentence)}")

                            # if preceding_sentence == "" or following_sentence == "":
                            #     print("============================")
                            #     print(f"@sentInDataset_editor_split{' ' * (20 - len('@sentInDataset_editor_split'))}: {sentInDataset_editor_split}")
                            #     print(f"@formatted_sentence{' ' * (20 - len('@formatted_sentence'))}: {formatted_sentence}")
                            #     print(f"@preceding_sentence{' ' * (20 - len('@preceding_sentence'))}: {preceding_sentence}")
                            #     print(f"@following_sentence{' ' * (20 - len('@following_sentence'))}: {following_sentence}")
                            #     print("============================")

                            data.append([
                                word_leanerCur.text, word_editorCur.text,
                                learner_sentence, editor_sentence, formatted_sentence,
                                str(sents_pTag),
                                preceding_sentence, following_sentence,
                                sentInDataset, "fce-dataset", [errorType_cur, learner_answer["foldername"], learner_answer["filename"]]
                            ])

                            data_to_print.append([
                                word_leanerCur.text, word_editorCur.text,
                                learner_sentence, editor_sentence, formatted_sentence,
                                str(sents_pTag),
                                preceding_sentence, following_sentence,
                                str(sentInDataset), "fce-dataset", [errorType_cur, learner_answer["foldername"], learner_answer["filename"]]
                            ])
                            print(f"@data{' ' * (20 - len('@data'))}: {json.dumps(data_to_print[-1], indent=20)}")
                            continue

        # print(f"@errorTypes_occur{' ' * (20 - len('@errorTypes_occur'))}: {errorTypes_occur}")
        # print(f"@errorTypePairs{' ' * (20 - len('@errorTypePairs'))}: {errorTypePairs}")
        # print(f"@new sents_pTag{' ' * (20 - len('@new sents_pTag'))}: {sents_pTag.text}")
    return data


def extract_coded_answer_per_file(filepath):
    """
    Extract coded_answer from each xml file in the CLC FCE dataset
    """
    soup = bs(open(filepath, "r", encoding="utf-8"), "xml")

    # Extract metadata
    filename = os.path.basename(filepath)
    foldername = os.path.dirname(filepath)
    sortkey = soup.find("head")["sortkey"]
    language = soup.find("language").text
    age = soup.find("age").text if soup.find("age") else ""
    score = soup.find("score").text

    learner_answer = []
    for answer in soup.find_all(name=re.compile(r"^answer\d+$")):
        question_num = answer.find("question_number").text
        exam_score = answer.find("exam_score").text if answer.find("exam_score") else ""
        coded_answer = answer.find("coded_answer")  # keep the xml format

        learner_answer.append({
            "coded_answer": coded_answer,
            "question_number": question_num,
            "exam_score": exam_score,
            "language": language,
            "age": age,
            "score": score,
            "sortkey": sortkey,
            "filename": filename,
            "foldername": foldername,
        })
    return learner_answer


def extract_target_sentence_and_info():
    sent_dataset = pd.DataFrame(columns=DATA_ROW.__annotations__.keys())
    essay_dataset = pd.DataFrame(columns=[
        "coded_answer", "question_number", "exam_score",
        "language", "age", "score", "sortkey", "filename", "foldername"
    ])
    data = []
    for subfolder in tqdm(os.listdir(datafolder), desc="folder", colour="green", ncols=100):
        if subfolder.startswith("."):
            continue
        subfolder_path = os.path.join(datafolder, subfolder)
        # print(f"Current subfolder: {subfolder_path} ")
        if os.path.isdir(subfolder_path):
            for filename in tqdm(os.listdir(subfolder_path), desc="files", ncols=100):
                # print(f"Current file: {filename} ")
                if filename.startswith("."):
                    continue
                filepath = os.path.join(subfolder_path, filename)

                coded_answers = extract_coded_answer_per_file(filepath)
                # print(f"Extracted {len(coded_answers)} coded answers")
                essay_dataset = essay_dataset._append(coded_answers, ignore_index=True)

                for coded_answer in coded_answers:
                    cnt = len(data)
                    data.extend(extract_sentence_per_coded_answer(coded_answer))
                    print(f"Extracted {len(data) - cnt} sentences per essay")
                    # sent_dataset = sent_dataset._append(data, ignore_index=True)

        #     break
        # break

    for row in data:
        sent_dataset = sent_dataset._append(dict(zip(DATA_ROW.__annotations__.keys(), row)), ignore_index=True)

    sent_dataset = clean_df(sent_dataset)
    sent_dataset.to_csv(os.path.join(outputFolder, "fce_sentences.csv"), index=False, encoding='utf-8')
    sent_dataset.to_csv(os.path.join(outputFolder, "fce_sentences_with_index.csv"), index=True, index_label='index')
    essay_dataset.to_csv(os.path.join(outputFolder, "fce_essays.csv"), index=False, encoding='utf-8')
    print(f"Extracted {len(essay_dataset)} essays")
    print(f"Extracted {len(sent_dataset)} sentences")


if __name__ == "__main__":
    filePath_fce_sentences_withIndex = os.path.join(outputFolder, "fce_sentences_with_index.csv")
    filePath_fce_dictInfo = os.path.join(outputFolder, "fce_sentences_dictionary.csv")

    extract_target_sentence_and_info()
    retrieve_cambridge_data(filePath_fce_sentences_withIndex, filePath_fce_dictInfo)
