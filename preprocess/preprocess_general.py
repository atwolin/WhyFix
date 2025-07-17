import pandas as pd
from tqdm import tqdm

from model.llm_setup import (
    get_support_material,
)

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


def clean_df(df):
    df = df.drop_duplicates(subset=["learner_sentence", "sentence_in_dataset"])
    df = df.dropna(subset=['learner_sentence'])
    df = df.reset_index(drop=True)
    return df


def retrieve_cambridge_data(path_input, path_output):
    df_data = pd.read_csv(path_input, encoding='utf-8')
    cambridge_info = []
    for index, row in tqdm(df_data.iterrows(), total=len(df_data), colour='green', ncols=100):
        learnerInfo, editorInfo, infoList = get_support_material(row)
        cambridge_info.append(infoList)
        # print(f"\ncambridge_info: {cambridge_info}")
        # break

    df_data = pd.concat([df_data, pd.DataFrame(cambridge_info, columns=INFO_ROW.__annotations__.keys())], axis=1)

    # path_output_duplicated = path_output.replace('.csv', '_duplicated.csv')
    # df_data.to_csv(path_output_duplicated, index=False, encoding='utf-8')

    # Convert all lists to strings to avoid errors with drop_duplicates
    # for col in df_data.columns:
    #     df_data[col] = df_data[col].apply(lambda x: str(x) if isinstance(x, list) else x)

    df_data.to_csv(path_output, index=False, encoding='utf-8')
    print(f"Cambridge data retrieved and saved to {path_output}")
