import os
import regex as re
import copy
import json
import errant
from tqdm import tqdm
import pandas as pd
from bs4 import BeautifulSoup as bs

from preprocess.preprocess_setup import (
    NLP,
)
from retrieval.dictionary_info import (
    concat_cambridge_data,
)
from utils.files_io import (
    FilePath,
)

PATHS = FilePath()
SAMPLING_SIZE = 30
SEED = 1126


class DATA_ROW:
    learner_word: str
    editor_word: str

    learner_sentence: str
    editor_sentence: str
    formatted_sentence: str

    sentence_in_dataset: str

    preceding_sentence: str
    following_sentence: str

    text_in_dataset: str
    data_source: str
    note: any


def clean_df(df):
    df = df.drop_duplicates(subset=["learner_sentence", "sentence_in_dataset"])
    df = df.dropna(subset=['learner_sentence'])
    df = df.reset_index(drop=True)
    return df


def fce_extract_coded_answer_per_file(filepath):
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


def fce_extract_sentence_per_coded_answer(learner_answer):
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

    # Get the edited-only essay
    sentInDataset_editor = [copy.deepcopy(s) for s in sentInDataset.find_all('p')]
    for sents in sentInDataset_editor:
        for iTag in sents.find_all('i'):
            iTag.decompose()

    # Split the essay into sentences
    sentInDataset_editor_split = []
    for sents in sentInDataset_editor:
        sents_split = [s.text for s in NLP(sents.text).sents]
        sentInDataset_editor_split.extend(sents_split)

    # Extract sentences from the coded_answer
    for sents_pTag in sentInDataset.find_all('p'):

        # Collect target error types and their learner's words
        errorTypePairs = []  # error type, learner's word, editor's word
        errorType_cur = ""   # note[0]
        word_leanerCur = ""  # learner_word
        word_editorCur = ""  # editor_word
        for nsTag in sents_pTag.find_all('NS'):
            errorType_cur = nsTag["type"]
            if errorType_cur.startswith(("CL", "ID", "L", "SA")) or re.match(r"^R[^ACDQTP][JNVY]?", errorType_cur) or re.match(r"^FF[^ACDQT][JNVY]?", errorType_cur):
                # print(f"@nsTag{' ' * (20 - len('@nsTag'))}: {nsTag}")
                sents_learner = copy.deepcopy(sents_pTag)  # learners-only sentences
                sents_format = copy.deepcopy(sents_pTag)   # formatted sentences

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

                # Check if correct word occurs more than once
                replaceCnt = sents_learner.text.count(str(word_editorCur))
                if replaceCnt > 1:
                    # print(f"@replaceCnt{' ' * (20 - len('@replaceCnt'))}: {replaceCnt}")
                    ValueError(f"@replaceCnt{' ' * (20 - len('@replaceCnt'))}: {replaceCnt}")

                # Get formatted sentences
                sents_format = str(sents_format).replace(str(nsTag), f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}")
                sents_format = bs(sents_format, "xml")
                for iTag in sents_format.find_all('i'):
                    iTag.decompose()
                sents_format = sents_format.text
                sents_format_split = [s.text for s in NLP(sents_format).sents]

                # Get learner, preceding and following sentences
                learner_sentence = ""
                editor_sentence = ""
                formatted_sentence = ""
                preceding_sentence = ""
                following_sentence = ""
                print(f"@sents_format_split{' ' * (20 - len('@sents_format_split'))}: {len(sents_format_split)}, {type(sents_format_split)}, {sents_format_split}")

                for idx, sent in enumerate(sents_format_split):
                    for idx_word, word in enumerate(sent.split()):
                        #
                        if f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}" == word or \
                           f"{word_editorCur.text.strip().split()[-1]}{word_leanerCur.text.strip().split()[0]}" == word:
                            print(f"found word: {word}")
                            print(f"found sent: {sent}")
                            sent_found = sent.replace(f"{word_editorCur.text.strip()}{word_leanerCur.text.strip()}", word_editorCur.text)

                            for idx_inDataset, sent_inDataset in enumerate(sentInDataset_editor_split):
                                if sent_found in sent_inDataset or sent_inDataset in sent_found:
                                    editor_sentence = sentInDataset_editor_split[idx_inDataset]
                                    learner_sentence = editor_sentence.replace(word_editorCur.text, word_leanerCur.text)
                                    formatted_sentence = editor_sentence.replace(word_editorCur.text, f"[-{word_leanerCur.text}-]{{+{word_editorCur.text}+}}")

                                    preceding_sentence = sentInDataset_editor_split[idx_inDataset - 1] if idx_inDataset > 0 else ""
                                    following_sentence = sentInDataset_editor_split[idx_inDataset + 1] if idx_inDataset < len(sentInDataset_editor_split) - 1 else ""

                            if learner_sentence == "" or learner_sentence == " ":
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
    return data


def fce_extract_target_sentence_and_info():
    sent_dataset = pd.DataFrame(columns=DATA_ROW.__annotations__.keys())
    essay_dataset = pd.DataFrame(columns=[
        "coded_answer", "question_number", "exam_score",
        "language", "age", "score", "sortkey", "filename", "foldername"
    ])
    data = []
    for subfolder in tqdm(os.listdir(PATHS.folderPath_fce_raw_data), desc="folder", colour="green", ncols=100):
        if subfolder.startswith("."):
            continue
        subfolder_path = os.path.join(PATHS.folderPath_fce_raw_data, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in tqdm(os.listdir(subfolder_path), desc="files", ncols=100):
                if filename.startswith("."):
                    continue
                filepath = os.path.join(subfolder_path, filename)

                coded_answers = fce_extract_coded_answer_per_file(filepath)
                essay_dataset = essay_dataset._append(coded_answers, ignore_index=True)

                for coded_answer in coded_answers:
                    cnt = len(data)
                    data.extend(fce_extract_sentence_per_coded_answer(coded_answer))
                    print(f"Extracted {len(data) - cnt} sentences per essay")

    for row in data:
        sent_dataset = sent_dataset._append(dict(zip(DATA_ROW.__annotations__.keys(), row)), ignore_index=True)

    essay_dataset.to_csv(PATHS.filePath_fce_essays, index=False, encoding='utf-8')

    sent_dataset = clean_df(sent_dataset)
    sent_dataset.to_csv(PATHS.filePath_fce_sentences, index=False, encoding='utf-8')
    sent_dataset.to_csv(PATHS.filePath_fce_sentences_withIndex, index=True, index_label='index')

    print(f"Extracted {len(essay_dataset)} essays")
    print(f"Extracted {len(sent_dataset)} sentences")


def errant_format_to_m2(original_sentence, corrected_sentence, annotator=None, write_to_file=False, filepath=None):
    """
    Uses ERRANT to process original and corrected sentences and outputs M2 format.
    Args:
        original_sentence (str): 原始的、可能包含錯誤的句子。
        corrected_sentence (str): 更正後的句子。
        annotator (errant.Annotator, optional): 預先載入的 ERRANT annotator。
        write_to_file (bool, optional): 是否將 M2 結果寫入檔案。預設為 False。
        filepath (str, optional): M2 檔案的儲存路徑。如果 write_to_file 為 True，則需要此參數。
    Returns:
        list: 一個包含 M2 格式字串的列表 (第一行為 S-line，後續為 A-lines)。
    """
    if annotator is None:
        annotator = errant.load('en')

    # annotator.parse() 返回 spaCy Doc 對象
    orig_doc = annotator.parse(original_sentence)
    cor_doc = annotator.parse(corrected_sentence)
    edits = annotator.annotate(orig_doc, cor_doc)  # annotator.annotate() 結合了對齊 (align)、合併 (merge) 和分類 (classify)

    m2_output = []
    # S-line 使用原始句子文本，ERRANT 的 tokenization 在 parse 時完成，索引用於這些 token。
    # _generate_runtime_data_from_m2 中 S = lines[0].strip().split(' ')
    # 這裡的 original_sentence 應與 ERRANT 內部 parse 後得到的 S-line tokenization 一致或兼容。
    # 為了安全起見，S-line 最好是 ERRANT tokens join 的結果，但通常直接用 original_sentence 也可以。
    # ERRANT M2 spec: "S This is a sentence ."
    s_line_tokens = [tok.text for tok in orig_doc]
    # m2_output.append(f"S {original_sentence}")
    m2_output.append(f"S {' '.join(s_line_tokens)}")

    # 5. 遍歷所有編輯並格式化為 A-lines
    for edit in edits:
        # edit.o_start: 原始句子中編輯的起始 token 索引 (0-indexed)
        # edit.o_end: 原始句子中編輯的結束 token 索引 (不包含)
        # edit.type: ERRANT 分類的錯誤類型
        # edit.c_str: 更正後的字串

        # 如果是刪除操作 (c_str 為空)，M2 格式中用 "-NONE-" 表示更正字串
        correction_str = edit.c_str if edit.c_str else "-NONE-"

        # M2 格式的固定欄位
        status = "REQUIRED"
        comment = "-NONE-"
        annotator_id = "0"
        a_line = f"A {edit.o_start} {edit.o_end}|||{edit.type}|||{correction_str}|||{status}|||{comment}|||{annotator_id}"
        m2_output.append(a_line)

    if write_to_file:
        if filepath is None:
            # Fallback to the path defined in FilePath if none is provided
            filepath = PATHS.filePath_runtime_m2
            if filepath is None:  # Further fallback if PATHS.filePath_runtime_m2 is also None
                raise ValueError("Filepath for M2 output is not specified and default is not available.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure directory exists
        with open(filepath, 'w', encoding='utf-8') as f:
            for line in m2_output:
                f.write(line + '\n')
        print(f"Saved M2 format to {filepath}")

    return m2_output


def errant_combine_sentences(sentence_list):
    """將包含 'S' 標記和 token 的列表組合成句子字串。"""
    sentence = ' '.join(word for word in sentence_list[1:] if word != '')
    sentence = re.sub(r'\s+(?=(?!\[-)\p{P})', '', sentence)
    sentence = sentence.replace("  ", " ")
    sentence = sentence.replace("   ", " ")
    return sentence

def remove_punctuation(sentence):
    """Remove punctuation from a sentence."""
    return re.sub(r'\p{P}+', '', sentence)


def longman_extract_target_sentence_and_info(dataset, input_file, output_file):
    file = open(input_file, 'r', encoding='utf-8')
    results = file.read()
    file.close()

    output_list = []
    learner_word = ""
    editor_word = ""

    learner_sentence = ""
    editor_sentence = ""
    formatted_sentence = ""

    sentence_in_dataset = ""

    preceding_sentence = ""
    following_sentence = ""

    text_in_dataset = ""
    data_source = dataset
    note = ""

    for lines in tqdm(results.split('\n\n'), colour='green', ncols=100):
        lines = lines.split('\n')
        print(f"{'=' * 50}\nlines: {lines}")

        entry, S, As = lines[0].strip(), lines[1].strip().split(' '), lines[2:]

        # Target sentence format
        target_error_types = ["OTHER", "VERB", "NOUN", "ADJ", "ADV"]
        error_types = [A.split('|||')[1].split(':', 1)[1] for A in As]

        if not any(error_type in target_error_types for error_type in error_types):
            print(f"error_types: {error_types} not in target_error_types, skip this sentence")
            continue

        learner_word = ""
        editor_word = ""
        learner_sentence = copy.deepcopy(S)
        editor_sentence = copy.deepcopy(S)
        formatted_sentence = copy.deepcopy(S)
        sentence_in_dataset = copy.deepcopy(S)
        note = ""

        # Start formatting
        for A in As:
            location, error_type, correction, note1, note2, editor = A.split('|||')

            pfrom, pto = int(location.split(' ')[1]), int(location.split(' ')[2])

            if ':' in error_type:
                sub_error_type, error_tag = error_type.split(':', 1)
            else:
                error_type, error_tag = 'none', 'none'

            if note == "":
                note = entry + '|||' + error_type + ',' + note1 + ',' + note2 + ',' + editor
            else:
                note += '|||' + error_type + ',' + note1 + ',' + note2 + ',' + editor

            # Replacement
            if sub_error_type == 'R':
                if pto - pfrom <= 1:
                    sentence_in_dataset[pto] = '[-' + sentence_in_dataset[pto] + '-]{+' + correction + '+}///' + f'{error_type}///'
                    editor_sentence[pto] = correction
                    formatted_sentence[pto] = '[-' + formatted_sentence[pto] + '-]{+' + correction + '+}'
                else:
                    sentence_in_dataset[pfrom + 1] = ' [-' + sentence_in_dataset[pfrom + 1]
                    formatted_sentence[pfrom + 1] = ' [-' + formatted_sentence[pfrom + 1]
                    sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]{+' + correction + '+}///' + f'{error_type}///'
                    formatted_sentence[pto] = formatted_sentence[pto] + '-]{+' + correction + '+}'

                    for idx in range(pfrom + 1, pto):
                        editor_sentence[idx] = ""
                    editor_sentence[pto] = correction
            # Missing
            elif sub_error_type == 'M':
                sentence_in_dataset[pto] = sentence_in_dataset[pto] + ' {+' + correction + '+}///' + f'{error_type}///'
                formatted_sentence[pto] = formatted_sentence[pto] + ' {+' + correction + '+}'

                editor_sentence[pto] = editor_sentence[pto] + correction
            # Unnecessary
            elif sub_error_type == 'U':
                if pto - pfrom <= 1:
                    sentence_in_dataset[pto] = ' [-' + sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                    formatted_sentence[pto] = ' [-' + formatted_sentence[pto] + '-]'

                    editor_sentence[pto] = ""
                else:
                    sentence_in_dataset[pfrom + 1] = ' [-' + sentence_in_dataset[pfrom + 1]
                    formatted_sentence[pfrom + 1] = ' [-' + formatted_sentence[pfrom + 1]
                    sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                    formatted_sentence[pto] = formatted_sentence[pto] + '-]'

                    for idx in range(pfrom + 1, pto):
                        editor_sentence[idx] = ""

                if correction != '':
                    sentence_in_dataset[pto] = sentence_in_dataset[pto] + '{+' + correction + '+}///' + f'{error_type}///'
                    formatted_sentence[pto] = formatted_sentence[pto] + '{+' + correction + '+}'

                    editor_sentence[pto] = correction

        if error_tag in target_error_types and learner_word == "":
            learner_word = learner_sentence[pfrom: pto + 1]
            learner_word = errant_combine_sentences(learner_word)
        if error_tag in target_error_types and editor_word == "":
            editor_word = correction

        if learner_word == "":
            learner_word = entry
        learner_sentence = errant_combine_sentences(learner_sentence)
        editor_sentence = errant_combine_sentences(editor_sentence)
        formatted_sentence = errant_combine_sentences(formatted_sentence)
        text_in_dataset = sentence_in_dataset = errant_combine_sentences(sentence_in_dataset)

        output_list.append([
            learner_word,
            editor_word,

            learner_sentence,
            editor_sentence,
            formatted_sentence,

            sentence_in_dataset,

            preceding_sentence,
            following_sentence,

            text_in_dataset,
            data_source,
            note
        ])

    sentence_dataset = pd.DataFrame(columns=DATA_ROW.__annotations__.keys())
    for i, row in enumerate(output_list):
        sentence_dataset.loc[i] = row

    sentence_dataset = clean_df(sentence_dataset)
    sentence_dataset.to_csv(output_file, index=False, encoding='utf-8')
    sentence_dataset.to_csv(output_file.replace('.csv', '_withIndex.csv'), index=True, index_label='index', encoding='utf-8')


def _generate_runtime_data_from_m2(m2_lines_list):
    if not m2_lines_list:
        return ""  # 或許返回 S-line 的原始句子部分

    output_list = []
    learner_word = ""
    editor_word = ""

    learner_sentence = ""
    editor_sentence = ""
    formatted_sentence = ""

    sentence_in_dataset = ""

    preceding_sentence = ""
    following_sentence = ""

    text_in_dataset = ""
    data_source = 'runtime'
    note = ""

    S, As = m2_lines_list[0].strip().split(' '), m2_lines_list[1:]

    if not m2_lines_list or not m2_lines_list[0].startswith("S "):
        # 應由 errant_format_to_m2 保證
        print("Warning: M2 data does not start with S-line. Using raw content.")
        return {  # Return empty structure on error or no input
            "learner_word": "", "editor_word": "",
            "learner_sentence": "" if not m2_lines_list else m2_lines_list[0][2:],
            "editor_sentence": "" if not m2_lines_list else m2_lines_list[0][2:],
            "formatted_sentence": "" if not m2_lines_list else m2_lines_list[0][2:],
            "data_source": "runtime_input", "note": "Error: Invalid M2 input"
        }

    # Target sentence format
    # target_error_types = ["OTHER", "VERB", "NOUN", "ADJ", "ADV"]
    # error_types = [A.split('|||')[1].split(':', 1)[1] for A in As]

    # if not any(error_type in target_error_types for error_type in error_types):
    #     print(f"error_types: {error_types} not in target_error_types, skip this sentence")
    #     return ""

    learner_word = ""
    editor_word = ""
    learner_sentence = copy.deepcopy(S)
    editor_sentence = copy.deepcopy(S)
    formatted_sentence = copy.deepcopy(S)
    sentence_in_dataset = copy.deepcopy(S)
    note = ""

    # Start formatting
    for A in As:
        location, error_type, correction, note1, note2, editor = A.split('|||')

        pfrom, pto = int(location.split(' ')[1]), int(location.split(' ')[2])

        if ':' in error_type:
            sub_error_type, error_tag = error_type.split(':', 1)
        else:
            error_type, error_tag = 'none', 'none'

        if note == "":
            note = error_type + ',' + note1 + ',' + note2 + ',' + editor
        else:
            note += '|||' + error_type + ',' + note1 + ',' + note2 + ',' + editor

        # Replacement
        if sub_error_type == 'R':
            if pto - pfrom <= 1:
                sentence_in_dataset[pto] = '[-' + sentence_in_dataset[pto] + '-]{+' + correction + '+}///' + f'{error_type}///'
                editor_sentence[pto] = correction
                formatted_sentence[pto] = '[-' + formatted_sentence[pto] + '-]{+' + correction + '+}'
            else:
                sentence_in_dataset[pfrom + 1] = ' [-' + sentence_in_dataset[pfrom + 1]
                formatted_sentence[pfrom + 1] = ' [-' + formatted_sentence[pfrom + 1]
                sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]{+' + correction + '+}///' + f'{error_type}///'
                formatted_sentence[pto] = formatted_sentence[pto] + '-]{+' + correction + '+}'

                for idx in range(pfrom + 1, pto):
                    editor_sentence[idx] = ""
                editor_sentence[pto] = correction
        # Missing
        elif sub_error_type == 'M':
            sentence_in_dataset[pto] = sentence_in_dataset[pto] + ' {+' + correction + '+}///' + f'{error_type}///'
            formatted_sentence[pto] = formatted_sentence[pto] + ' {+' + correction + '+}'

            editor_sentence[pto] = editor_sentence[pto] + correction
        # Unnecessary
        elif sub_error_type == 'U':
            if pto - pfrom <= 1:
                sentence_in_dataset[pto] = ' [-' + sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                formatted_sentence[pto] = ' [-' + formatted_sentence[pto] + '-]'

                editor_sentence[pto] = ""
            else:
                sentence_in_dataset[pfrom + 1] = ' [-' + sentence_in_dataset[pfrom + 1]
                formatted_sentence[pfrom + 1] = ' [-' + formatted_sentence[pfrom + 1]
                sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                formatted_sentence[pto] = formatted_sentence[pto] + '-]'

                for idx in range(pfrom + 1, pto):
                    editor_sentence[idx] = ""

            if correction != '':
                sentence_in_dataset[pto] = sentence_in_dataset[pto] + '{+' + correction + '+}///' + f'{error_type}///'
                formatted_sentence[pto] = formatted_sentence[pto] + '{+' + correction + '+}'

                editor_sentence[pto] = correction

    print(f"@learner_sentence: {learner_sentence[pfrom + 1: pto + 1][0]}")
    if learner_word == "":
        learner_word = learner_sentence[pfrom + 1: pto + 1][0]
        learner_word = remove_punctuation(learner_word)
    if editor_word == "":
        editor_word = remove_punctuation(correction)

    # if learner_word == "":
    #     learner_word = entry
    learner_sentence = errant_combine_sentences(learner_sentence)
    editor_sentence = errant_combine_sentences(editor_sentence)
    formatted_sentence = errant_combine_sentences(formatted_sentence)
    text_in_dataset = sentence_in_dataset = errant_combine_sentences(sentence_in_dataset)

    output_list.append([
        learner_word,
        editor_word,

        learner_sentence,
        editor_sentence,
        formatted_sentence,

        sentence_in_dataset,

        preceding_sentence,
        following_sentence,

        text_in_dataset,
        data_source,
        note
    ])

    sentence_dataset = pd.DataFrame(columns=DATA_ROW.__annotations__.keys())
    for i, row in enumerate(output_list):
        sentence_dataset.loc[i] = row

    sentence_dataset = clean_df(sentence_dataset)
    sentence_dataset.to_csv(PATHS.filePath_runtime_sentences, index=False, encoding='utf-8')
    sentence_dataset.to_csv(PATHS.filePath_runtime_sentences_withIndex, index=True, index_label='index', encoding='utf-8')
    print(f"Saved runtime sentences to {PATHS.filePath_runtime_sentences}")
    print(f"Saved runtime sentences with index to {PATHS.filePath_runtime_sentences_withIndex}")
    return sentence_dataset


def format_fce_to_bracket():
    fce_extract_target_sentence_and_info()
    concat_cambridge_data(PATHS.filePath_fce_sentences_withIndex, PATHS.filePath_fce_dictInfo)


def format_longman_to_bracket():
    longman_extract_target_sentence_and_info("longman_dictionary_of_common_errors", PATHS.filePath_longman_errant_m2, PATHS.filePath_longman_sentences)
    concat_cambridge_data(PATHS.filePath_longman_sentences_withIndex, PATHS.filePath_longman_dictInfo)


def format_runtime_to_bracket(original_sentence, corrected_sentence, annotator=None):
    # 1. 獲取 M2 格式 (在記憶體中)
    #    如果仍需寫入檔案，可在此處控制 write_to_file 和 filepath 參數
    m2_lines = errant_format_to_m2(original_sentence, corrected_sentence, annotator, write_to_file=True, filepath=PATHS.filePath_runtime_m2)
    # m2_lines = errant_format_to_m2(original_sentence, corrected_sentence, annotator, write_to_file=False)

    if not m2_lines:  # 可能原始句和目標句相同
        return original_sentence

    sentence_df = _generate_runtime_data_from_m2(m2_lines)
    return sentence_df


if __name__ == "__main__":
    format_fce_to_bracket()
    format_longman_to_bracket()
