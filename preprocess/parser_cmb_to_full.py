import os
import json

WORD_LINK = "https://dictionary.cambridge.org//dictionary/english-chinese-traditional/"


if __name__ == "__main__":
    # Load all vocabulary links
    VocabIdx_Dict = dict()
    with open("../data/cambridge-parse/cambridge_parse.all_word_links.json", "r") as f:
        VocabIdx_Dict = json.load(f)

    path = "../data/cambridge-html"
    dirs = os.listdir(path)
    download = dict()
    for dir in dirs:
        download_sublist = list()
        for file in (os.listdir(os.path.join(path, dir))):
            with open(os.path.join(path, dir, file), "r") as f:
                content = f.read()
                if content == "None":
                    word = file.split(".")[0]
                    download_sublist.append(word)

        print(len(download_sublist), download_sublist)
        download[dir] = download_sublist
