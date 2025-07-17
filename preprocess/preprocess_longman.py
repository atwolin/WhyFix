import os
import regex as re
import copy
from tqdm import tqdm
import pandas as pd

from .preprocess_general import DATA_ROW, clean_df, retrieve_cambridge_data


folderPath = "/home/nlplab/atwolin/thesis/data/longman/"
filePath_m2 = os.path.join(folderPath, "m2_with_problem_word_copy.txt")
filePath_longman_sentences = os.path.join(folderPath, "longman_sentences.csv")
filePath_longman_sentences_withIndex = os.path.join(folderPath, "longman_sentences_with_index.csv")
filePath_longman_dictInfo = os.path.join(folderPath, "longman_sentences_dictionary.csv")


def combine_sentences(sentence_list):
    sentence = ' '.join(word for word in sentence_list[1:] if word != '')
    sentence = re.sub(r'\s+(?=(?!\{\+|\[\-)[\p{P}])', '', sentence)
    sentence = sentence.replace("  ", " ")
    return sentence


def extract_target_sentence_and_info(path_input):
    # file = open('result_m2.m2', encoding='utf-8')
    # file = open("/home/nlplab/atwolin/thesis/data/longman/m2_with_problem_word_copy.txt", encoding='utf-8')
    file = open(path_input, encoding='utf-8')
    results = file.read()
    results = results.replace("  ", " ")  # double space
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
    data_source = "longman-dictionary-of-common-errors"
    note = ""

    # cnt = 0

    for lines in tqdm(results.split('\n\n'), colour='green', ncols=100):
        lines = lines.split('\n')
        print(f"{'=' * 50}\nlines: {lines}")

        # S, As = lines[0].split(' '), lines[1:]
        entry, S, As = lines[0].strip(), lines[1].strip().split(' '), lines[2:]
        # print(f"{'=' * 50}entry: {entry}{'=' * 50}\nS: {S}\nAs: {As}")

        # Target sentence format
        target_error_types = ["OTHER", "VERB", "NOUN", "ADJ", "ADV"]
        error_types = [A.split('|||')[1].split(':', 1)[1] for A in As]

        # print(f"error_types: {error_types}")

        if not any(error_type in target_error_types for error_type in error_types):
            print(f"error_types: {error_types} not in target_error_types, skip this sentence")
            continue

        # learner_word = entry
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
            # correction = re.sub(' ', '_', correction)

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
                    # S[pto] = '[-' + S[pto] + '//' + error_tag + '-]{+' + correction + '//' + error_tag + '+}'
                    sentence_in_dataset[pto] = '{+' + correction + '+}[-' + sentence_in_dataset[pto] + f'-]///{error_type}///'
                    editor_sentence[pto] = correction
                    formatted_sentence[pto] = '{+' + correction + '+}[-' + formatted_sentence[pto] + '-]'
                else:
                    # for index in range(pfrom + 1, pto):
                    #     S[index] = S[index] + '_'
                    # S[pto] = S[pto] + '//' + error_tag + '-]{+' + correction + '//' + error_tag + '+}'
                    sentence_in_dataset[pfrom + 1] = '{+' + correction + '+}[-' + sentence_in_dataset[pfrom + 1]
                    formatted_sentence[pfrom + 1] = '{+' + correction + '+}[-' + formatted_sentence[pfrom + 1]

                    sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                    formatted_sentence[pto] = formatted_sentence[pto] + '-]'

                    for idx in range(pfrom + 1, pto):
                        editor_sentence[idx] = ""
                    editor_sentence[pto] = correction

                if error_tag in target_error_types and learner_word == "":
                    learner_word = learner_sentence[pfrom: pto + 1]
                    learner_word = combine_sentences(learner_word)
                if error_tag in target_error_types and editor_word == "":
                    editor_word = correction

            # Missing
            elif sub_error_type == 'M':
                # S[pto] = S[pto] + ' {+' + correction + '//' + error_tag + '+}'
                sentence_in_dataset[pto] = sentence_in_dataset[pto] + ' {+' + correction + '+}///' + f'{error_type}///'
                formatted_sentence[pto] = formatted_sentence[pto] + ' {+' + correction + '+}'

                editor_sentence[pto] = editor_sentence[pto] + ' ' + correction

                if error_tag in target_error_types and editor_word == "":
                    editor_word = correction

            # Unnecessary
            elif sub_error_type == 'U':
                if pto - pfrom <= 1:
                    # S[pto] = '[-' + S[pto] + '//' + error_tag + '-]'
                    sentence_in_dataset[pto] = ' [-' + sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                    formatted_sentence[pto] = ' [-' + formatted_sentence[pto] + '-]'

                    editor_sentence[pto] = ""
                else:
                    # for index in range(pfrom+1, pto):
                    #     S[index] = S[index]+'_'
                    sentence_in_dataset[pfrom + 1] = ' [-' + sentence_in_dataset[pfrom + 1]
                    formatted_sentence[pfrom + 1] = ' [-' + formatted_sentence[pfrom + 1]
                    # S[pto] = S[pto] + '//' + error_tag + '-]'
                    sentence_in_dataset[pto] = sentence_in_dataset[pto] + '-]///' + f'{error_type}///'
                    formatted_sentence[pto] = formatted_sentence[pto] + '-]'

                    for idx in range(pfrom + 1, pto):
                        editor_sentence[idx] = ""

                # if correction != '':
                #     # S[pto] = S[pto] + '{+' + correction + '//' + error_tag + '+}'
                #     sentence_in_dataset[pto] = sentence_in_dataset[pto] + '{+' + correction + '+}///' + f'{error_type}///'
                #     formatted_sentence[pto] = formatted_sentence[pto] + '{+' + correction + '+}'

                #     editor_sentence[pto] = correction

                if error_tag in target_error_types and learner_word == "":
                    learner_word = learner_sentence[pfrom: pto + 1]
                    learner_word = combine_sentences(learner_word)
                if error_tag in target_error_types and editor_word == "":
                    editor_word = correction

        if learner_word == "":
            learner_word = entry
        # S = ' '.join(S[1:])
        # # S = S.replace("_ ", "_")
        # S = S.replace("  ", " ")
        # output_list.append(S)
        learner_sentence = combine_sentences(learner_sentence)
        editor_sentence = combine_sentences(editor_sentence)
        formatted_sentence = combine_sentences(formatted_sentence)
        text_in_dataset = sentence_in_dataset = combine_sentences(sentence_in_dataset)

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

        # print(f"learner_word: {learner_word}")
        # print(f"editor_word: {editor_word}")
        # print(f"learner_sentence: {learner_sentence}")
        # print(f"editor_sentence: {editor_sentence}")
        # print(f"formatted_sentence: {formatted_sentence}")
        # print(f"sentence_in_dataset: {sentence_in_dataset}")

        # cnt += 1
        # if cnt > 20:
        #     break

    # with open('result_mixed.txt', 'w', encoding='utf-8') as f:
    #     for line in output_list:
    #         f.write("%s\n" % line)

    sentence_dataset = pd.DataFrame(columns=DATA_ROW.__annotations__.keys())
    for i, row in enumerate(output_list):
        sentence_dataset.loc[i] = row
    # sentence_dataset.to_csv('/home/nlplab/atwolin/thesis/data/longman/longman_sentences.csv', index=False, encoding='utf-8')

    sentence_dataset = clean_df(sentence_dataset)
    sentence_dataset.to_csv(filePath_longman_sentences, index=False, encoding='utf-8')
    sentence_dataset.to_csv(filePath_longman_sentences_withIndex, index=True, index_label='index', encoding='utf-8')


if __name__ == '__main__':
    extract_target_sentence_and_info(filePath_m2)
    retrieve_cambridge_data(filePath_longman_sentences_withIndex, filePath_longman_dictInfo)
