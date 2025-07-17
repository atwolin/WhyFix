import os
import json
import pandas as pd

def extract_english_sentences(output_file):
    """
    Extract English sentences from the Cambridge JSON files and save them to a text file.
    """
    vocab_dir = "../../data/cambridge-parse/vocabV2"
    eng_examples = []
    for fname in os.listdir(vocab_dir):
        fpath = os.path.join(vocab_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                # To get english examples
                data = json.load(f)
                for vocab in data:
                    for pos in data[vocab]["poses"]:
                        for big_sense in pos["big_sense"]:
                            for sense in big_sense["senses"]:
                                for example in sense["examples"]:
                                    # out.write(example["eng_examp"] + '\n')
                                    eng_examples.append(example["eng_examp"])
    # Remove duplicates
    eng_examples = list(set(eng_examples))

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("\n".join(eng_examples))
    print(f"Extracted {len(eng_examples)} English sentences to {output_file}")


def combine_data():
    """
    Combine all text files in the /data/example-sentences folder into a single file.
    """
    data_folder_path = "../../data/example-sentences"
    combined_file_path = os.path.join(data_folder_path, "combined.v2")

    df_out = pd.DataFrame()
    for fname in os.listdir(data_folder_path):
        if "cambridge.v2" in fname or "mw" in fname:
            print(f"Current file: {fname}")
            df = pd.read_fwf(os.path.join(data_folder_path, fname), header=None)
            print(df.iloc[:, 0].count(), pd.isna(df.iloc[:, 0]).sum(), df.duplicated().sum())
            df_sentences = df.iloc[:, 0].dropna().tolist()
            df_out = pd.concat([df_out, pd.DataFrame(df_sentences)], ignore_index=True)
    print(f"duplicates: {df_out.duplicated().sum()}")
    df_out = df_out.drop_duplicates()
    df_out.to_csv(f"{combined_file_path}.csv", index=False, header=False)
    df_out.to_csv(f"{combined_file_path}.txt", index=False, header=False)


if __name__ == "__main__":
    output_file = "../../data/cambridge-parse/cambridge_eng_sentences.v2.txt"
    # extract_english_sentences(output_file)

    combine_data()
