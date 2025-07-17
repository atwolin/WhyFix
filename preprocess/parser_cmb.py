#!/usr/bin/env python
# coding: utf-8

import requests
import os
import re
import sys
from typing import Optional, TypedDict, Literal
import asyncio
# from requests_html import AsyncHTMLSession
from bs4 import BeautifulSoup
import json


IDX_PREFIX = "https://dictionary.cambridge.org/browse/english-chinese-traditional/"
WORD_PREFIX = "https://dictionary.cambridge.org/"
PARSED_WORD_PREFIX = "https://dictionary.cambridge.org//dictionary/english-chinese-traditional/"
IDX = str
WORD = str


# Structure of RESULT_FORMAT
#

#
# ```python
# RESULT_FORMAT = dict[VocabField]
# ```
#
# - **VocabField** (TypedDict)
#     - **link**: str
#     - **poses**: list[PosField]
#
#         - **PosField** (TypedDict)
#             - **pos**: str
#             - **headword**: str
#             - **big_sense**: list[GuidewordField]
#
#                 - **GuidewordField** (TypedDict)
#                     - **guideword**: str
#                     - **senses**: list[DefBlock]
#                     - **phrases**: list[PhraseBlock]
#                     - **extra_examples**: list[ExtraExamplesField]
#
#                         - **DefBlock** (TypedDict)
#                             - **level**: str
#                             - **eng_def**: str
#                             - **cn_def**: str
#                             - **examples**: list[ExampField]
#
#                                 - **ExampField** (TypedDict)
#                                     - **eng_examp**: str
#                                     - **cn_examp**: str
#
#                         - **PhraseBlock** (TypedDict)
#                             - **term**: str
#                             - **phrase_definitions**: list[PhraseDefBlock]
#
#                                 - **PhraseDefBlock** (TypedDict)
#                                     - **level**: str
#                                     - **eng_def**: str
#                                     - **cn_def**: str
#                                     - **examples**: list[ExampField]
#
#                         - **ExtraExamplesField** (TypedDict)
#                             - **examples**: list[str]
class SECTIONField(TypedDict):
    word: WORD
    link: str


class IdxField(TypedDict):
    link: str
    section: list[SECTIONField]


class ExampField(TypedDict):
    eng_examp: str
    cn_examp: str


class PhraseDefBlock(TypedDict):
    """

    in class="phrase-body dphrase_b"
    (in class="pr phrase-block dphrase-block ")
    """
    level: str
    eng_def: str
    cn_def: str
    examples: list[ExampField]


class PhraseBlock(TypedDict):
    """
    Different phrases for each headword

    in class="pr phrase-block dphrase-block lmb-25" or
       class="pr phrase-block dphrase-block "

    (in class="sense-body dsense_b", which is in class="pr dsense ")

    term: class="phrase-head dphrase_h"
    definitions: class="phrase-body dphrase_b"
    """
    term: str
    phrase_definitions: list[PhraseDefBlock]


class DefBlock(TypedDict):
    """
    Different senses for each headword

    in class="def-block ddef_block "
    """
    level: str
    eng_def: str
    cn_def: str
    examples: list[ExampField]
    # gcs: str


class ExtraExamplesField(TypedDict):
    """
    Extra examples for each sense

    for each class="eg dexamp hax"
    in class="daccord"
    """
    examples: list[str]


class GuidewordField(TypedDict):
    """
    Different headwords (e.g. go(ATTEMPT), go(OPPORTUNITY))

    For normal words:
        for each class="pr dsense "
        (in class="pr entry-body__el")

        guideword: class="guideword dsense_gw"
        senses: class="def-block ddef_block "                    (in class="sense-body dsense_b")
        phrases: class="pr phrase-block dphrase-block lmb-25"    (in class="sense-body dsense_b")
        extra_examples: class="daccord"                          (in class="sense-body dsense_b")

    For idioms:
        for each class="pr dsense dsense-noh"
        (in class="idiom-body didiom-body")

        guideword: none
        senses: class="def-block ddef_block "

    For phrases:
        for each class="phrase-di-body dphrase-di-body"
        (in class="phrase-di-block dphrase-di-block")

        guideword: none
        senses: class="def-block ddef_block "
    """
    guideword: str
    senses: list[DefBlock]
    phrases: list[PhraseBlock]
    extra_examples: list[ExtraExamplesField]


class PosField(TypedDict):
    """
    Different POS (e.g. noun, verb, adj)


    For normal words:
        for each class="pr entry-body__el"
        (in class="entry")

        pos: class="pos-header dpos-h"
        headword: class starts with "headword"
        big_sense: class="pr dsense "


    For idioms:
        for each class="idiom-body didiom-body"
        (in class="pr idiom-block")

        pos: class="pos dpos"
        headword: class starts with "headword"
        big_sense: class="pr dsense dsense-noh"

    For phrases:
        only one class="phrase-di-body dphrase-di-body"
        (in class="phrase-di-block dphrase-di-block")

        pos: class="pos dpos"                                (not in the class="phrase-di-body dphrase-di-body", in the class="phrase-di-block dphrase-di-block")
        headword: class starts with "headword"               (not in the class="phrase-di-body dphrase-di-body", in the class="phrase-di-block dphrase-di-block")
        big_sense: class="phrase-di-body dphrase-di-body"
    """
    pos: str
    headword: str
    big_sense: list[GuidewordField]


class VocabField(TypedDict):
    """
    Different vocabularies

    For normal words:
        each class="entry"
    For idioms:
        for each class="pr idiom-block"
    For phrases:
        for each class="phrase-di-block dphrase-di-block"
    """
    link: str
    poses: list[PosField]


IDX_FORMAT = dict[IDX, IdxField]
RESULT_FORMAT = dict[VocabField]


# Get all vocabulary index links from Cambridge Dictionary [index](https://dictionary.cambridge.org/browse/english-chinese-traditional/)
# async def fetch_and_parse(url: str) -> Optional[BeautifulSoup]:
def fetch_and_parse(url: str) -> Optional[BeautifulSoup]:
    """
    Fetches the HTML content from the given URL and parses it with BeautifulSoup.

    Args:
        url (str): The URL to fetch the HTML from.

    Returns:
        BeautifulSoup: Parsed BeautifulSoup object of the page.
    """
    try:
        # Send an HTTP GET request to the URL
        response = requests.get(url, headers={'User-Agent': "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0"})
        # response.raise_for_status()  # Raise an exception for HTTP errors

        # Use requests-html to render the JavaScript content
        # session = AsyncHTMLSession()
        # response = await session.get(url)
        # await session.close()

        # await response.html.arender()

        # Parse the response content with BeautifulSoup
        soup = BeautifulSoup(response.content, "html.parser")
        # soup = BeautifulSoup(response.html.raw_html, "html.parser")
        return soup

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while fetching the URL: {e}")
        return None


def extract_section(soup: BeautifulSoup) -> None:
    """
    Extracts the word and link for each section for each letter in the index page.

    Args:
        soup (BeautifulSoup): The parsed BeautifulSoup object of the index page.
    """
    # Iterate through each letter link
    for soup_letter in soup.find_all(class_="hbtn hbtn-tab-b tc-d tb bw"):
        letter = soup_letter.text.strip()
        letter_link = soup_letter['href']
        IDX_DICT[letter] = {
            "link": letter_link,
            "section": []
        }
        soup_section = fetch_and_parse(letter_link)

        # Find all section links for the letter
        for section in soup_section.find_all("a", class_="hlh32 hdb dil tcbd"):
            section_dict = {
                "word": section.text.strip(),
                "link": section['href']
            }
            IDX_DICT[letter]["section"].append(section_dict)


# Get all vocabulary links from index
def get_all_vocab_links(section: list) -> list[str]:
    """
    Gets all the vocabulary links from the index dictionary.

    Args:
        soup (BeautifulSoup): The parsed BeautifulSoup object of one key from IDX_DICT.

    Returns:
        list[str]: List of all the vocabulary links.
    """
    vocab_links = []
    for idx in section:
        # soup = await fetch_and_parse(idx["link"])
        soup = fetch_and_parse(idx["link"])
        for one_link in soup.find_all(class_="hlh32 han"):
            vocab_links.append(WORD_PREFIX + one_link.find("a")['href'])

    return vocab_links


# Get all definitions and phrases for one word
def extract_vocab_info(soup: BeautifulSoup, link: str, Vocab_Dict: dict) -> None:
    """
    Extracts vocabulary information from the given BeautifulSoup object and populates the Vocab_Dict.

    Args:
        soup (BeautifulSoup): The parsed BeautifulSoup object of the vocabulary page.
    """
    def extract_examples(example_soup):
        examples = []
        for example in example_soup.find_all(class_="examp"):
            eng_examp = example.find(class_="eg").text.strip()
            cn_examp = example.find(class_="trans").text.strip() if example.find(class_="trans") else ""

            examples.append(ExampField(eng_examp=eng_examp, cn_examp=cn_examp))
        return examples

    def extract_definitions(def_soup):
        definitions = []
        for definition in def_soup.find_all(class_="def-block ddef_block"):
            if definition.get("data-wl-senseid", "").endswith("panel"):
                continue
            # level = definition.find(class_="epp-xref").text.strip()
            level = definition.select("span[class^=epp-xref]")[-1].text.strip() if definition.select("span[class^=epp-xref]") else ""
            eng_def = definition.find(class_="def ddef_d db").text.strip()
            # cn_def = definition.find(class_="trans dtrans dtrans-se").text.strip()
            cn_def = definition.find(class_="trans dtrans dtrans-se break-cj").text.strip() if definition.find(class_="trans dtrans dtrans-se break-cj") else ""
            examples = extract_examples(definition)
            definitions.append(DefBlock(level=level, eng_def=eng_def, cn_def=cn_def, examples=examples))
        return definitions

    def extract_phrases(phrase_soup):
        phrase = []
        for phrase_block in phrase_soup.find_all(class_="pr phrase-block dphrase-block lmb-25"):
            # term = phrase_block.find(class_="phrase-head dphrase_h").text.strip()
            term = phrase_block.find(class_="phrase-title dphrase-title").text.strip()
            phrase_definitions = []
            # for phrase_def in phrase_block.find_all(class_="phrase-body dphrase_b"):
            for phrase_def in phrase_block.find_all(class_="def-block ddef_block"):

                level = phrase_def.find(class_="epp-xref").text.strip() if phrase_def.find(class_="epp-xref") else ""
                eng_def = phrase_def.find(class_="def ddef_d db").text.strip()
                cn_def = phrase_def.find(class_="trans dtrans dtrans-se break-cj").text.strip() if phrase_def.find(class_="trans dtrans dtrans-se break-cj") else ""
                examples = extract_examples(phrase_def)
                phrase_definitions.append(PhraseDefBlock(level=level, eng_def=eng_def, cn_def=cn_def, examples=examples))
            phrase.append(PhraseBlock(term=term, phrase_definitions=phrase_definitions))

        for phrase_block in phrase_soup.find_all(class_="pr phrase-block dphrase-block"):
            # term = phrase_block.find(class_="phrase-head dphrase_h").text.strip()
            term = phrase_block.find(class_="phrase-title dphrase-title").text.strip()
            phrase_definitions = []
            # for phrase_def in phrase_block.find_all(class_="phrase-body dphrase_b"):
            for phrase_def in phrase_block.find_all(class_="def-block ddef_block"):

                level = phrase_def.find(class_="epp-xref").text.strip() if phrase_def.find(class_="epp-xref") else ""
                eng_def = phrase_def.find(class_="def ddef_d db").text.strip()
                cn_def = phrase_def.find(class_="trans dtrans dtrans-se break-cj").text.strip() if phrase_def.find(class_="trans dtrans dtrans-se break-cj") else ""
                examples = extract_examples(phrase_def)
                phrase_definitions.append(PhraseDefBlock(level=level, eng_def=eng_def, cn_def=cn_def, examples=examples))
            phrase.append(PhraseBlock(term=term, phrase_definitions=phrase_definitions))
        return phrase

    def extract_extra_examples(extra_example_soup):
        extra_examples = []
        for extra_example in extra_example_soup.find_all(class_="eg dexamp hax"):
            extra_examples.append(extra_example.text.strip())
        return extra_examples

    def extract_guidewords(guideword_soup):
        guidewords = []
        # For normal words
        for guideword in guideword_soup.find_all(class_="pr dsense"):
            guideword_text = guideword.find(class_="guideword dsense_gw").text.strip().strip("()")
            definitions = extract_definitions(guideword)
            phrases = extract_phrases(guideword)
            extra_example = extract_extra_examples(guideword)
            guidewords.append(GuidewordField(guideword=guideword_text, senses=definitions, phrases=phrases, extra_examples=extra_example))

        # For idioms
        for guideword in guideword_soup.find_all(class_="pr dsense dsense-noh"):
            definitions = extract_definitions(guideword)
            phrases = extract_phrases(guideword)
            guidewords.append(GuidewordField(guideword="", senses=definitions, phrases=phrases, extra_examples=[]))

        # For phrases
        for guideword in guideword_soup.find_all(class_="phrase-di-body dphrase-di-body"):
            definitions = extract_definitions(guideword)
            phrases = extract_phrases(guideword)
            guidewords.append(GuidewordField(guideword="", senses=definitions, phrases=phrases, extra_examples=[]))
        return guidewords

    def extract_pos(pos_soup):
        pos_list = []
        # For normal words
        for pos in pos_soup.find_all(class_="entry-body__el"):
            pos_text = []
            # pos_text = pos.find(class_="posgram dpos-g hdib lmr-5").text.strip() if pos.find(class_="posgram dpos-g hdib lmr-5") else ""
            pos_text.append(pos.find(class_="pos dpos").text.strip() if pos.find(class_="pos dpos") else "")
            pos_text.append(pos.find(class_="gram dgram").text.strip() if pos.find(class_="gram dgram") else "")
            headword = pos.find(class_="headword").text.strip()
            big_sense = extract_guidewords(pos)
            pos_list.append(PosField(pos=pos_text, headword=headword, big_sense=big_sense))

        # For idioms
        # for pos in pos_soup.find_all(class_="idiom-body didiom-body"):
        for pos in pos_soup.find_all(class_="idiom-block"):
            # print(pos)
            pos_text = []
            # pos_text = pos.find(class_="pos dpos").text.strip()
            pos_text.append(pos.find(class_="pos dpos").text.strip() if pos.find(class_="pos dpos") else "")
            pos_text.append(pos.find(class_="gram dgram").text.strip() if pos.find(class_="gram dgram") else "")

            headword = pos.find(class_="headword").text.strip()
            big_sense = extract_guidewords(pos)
            pos_list.append(PosField(pos=pos_text, headword=headword, big_sense=big_sense))

        # For phrases
        for pos in pos_soup.find_all(class_="phrase-di-body dphrase-di-body"):
            pos_text = []
            # pos_text = pos_soup.find(class_="pos dpos").text.strip()
            pos_text.append(pos.find(class_="pos dpos").text.strip() if pos.find(class_="pos dpos") else "")
            pos_text.append(pos.find(class_="gram dgram").text.strip() if pos.find(class_="gram dgram") else "")

            headword = pos_soup.find(class_="headword").text.strip()
            big_sense = extract_guidewords(pos_soup)
            pos_list.append(PosField(pos=pos_text, headword=headword, big_sense=big_sense))
        return pos_list

    # soup = await fetch_and_parse(link)
    vocab = link.split("/")[-1]
    # Words
    for entry in soup.find_all(class_="entry"):
        pos_list = extract_pos(entry)
        # Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        if vocab not in Vocab_Dict:
            Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        else:
            Vocab_Dict[vocab]["poses"].extend(pos_list)

    # Idioms
    for idiom in soup.find_all(class_="pr idiom-block"):
        pos_list = extract_pos(idiom)
        # Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        if vocab not in Vocab_Dict:
            Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        else:
            Vocab_Dict[vocab]["poses"].extend(pos_list)

    # Phrases
    for phrase in soup.find_all(class_="phrase-di-block dphrase-di-block"):
        pos_list = extract_pos(phrase)
        # Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        if vocab not in Vocab_Dict:
            Vocab_Dict[vocab] = VocabField(link=link, poses=pos_list)
        else:
            Vocab_Dict[vocab]["poses"].extend(pos_list)


# def get_infomation_in_cambridge(query, target_word):
    # doc = nlp(query)
    # pos_space = ""

    # # Get the POS of the target word in spacy
    # for token in doc:
    #     if token.text == target_word:
    #         pos_space = token.pos_
    #         # print(token.text, token.pos_, token.dep_)
    #         break
    # print(f"{target_word} in spacy: {pos_space}")

    # # Find the definition of the target word in Cambridge
    # if pos_space != "":
    #     data_path = ""
    #     if target_word[0].isalpha():
    #         data_path = f"../data/cambridge-parse/vocab/cambridge.{(target_word[0]).lower()}.json"
    #     else:
    #         data_path = "../data/cambridge-parse/vocab/cambridge.0-9.json"

    #     print(data_path)
    #     # ../data/cambridge-parse/vocab/cambridge._.json

    #     if os.path.isfile(data_path):
    #         with open(data_path, 'r', encoding='utf-8') as f:
    #             data = json.load(f)
    #             for pos_content in data[target_word]["poses"]:
    #                 pos_cambridge = pos_content["pos"][0]

    #                 print(pos_cambridge)
    #                 print(posMap_spacy2cambridge[pos_space])

    #                 if pos_cambridge in posMap_spacy2cambridge[pos_space]:
    #                     for senses in pos_content["big_sense"]:
    #                         for sense in senses["senses"]:
    #                             print(f"{target_word}: {sense['eng_def']}")


async def main():
    from tqdm import tqdm
    import pickle
    # from pprint import pprint

    # Load all vocabulary links
    VocabIdx_Dict = dict()
    with open("../data/cambridge-parse/cambridge_parse.all_word_links.json", "r") as f:
        VocabIdx_Dict = json.load(f)

    # Get all vocabularies
    for category, links in tqdm(VocabIdx_Dict.items(), colour="green", desc="Category"):
        if category == "0–9":
            continue
        SubVocab_Dict: RESULT_FORMAT = {}
        os.makedirs(f"../data/cambridge-pickle/{category}", exist_ok=True)

        for idx in tqdm(range(len(links)), desc=f"Vocabulary-{category}"):
            with open(f"../data/cambridge-pickle/{category}/{links[idx].split('/')[-1]}.pickle", "wb") as f:
                pickle.dump(links[idx], f)

            # print(f"Fetching {links[idx].split('/')[-1]}")
            soup = await fetch_and_parse(links[idx])
            extract_vocab_info(soup, links[idx], SubVocab_Dict)

            if idx % 15 == 0:
                await asyncio.sleep(5)

        os.makedirs(f"../data/cambridge-parse/vocabV2", exist_ok=True)
        with open(f"../data/cambridge-parse/vocabV2/cambridge.{category}.json", "w") as f:
            json.dump(SubVocab_Dict, f, ensure_ascii=False, indent=4)

        # break

    # SubVocab_Dict: RESULT_FORMAT = {}
    # url = "https://dictionary.cambridge.org//dictionary/english-chinese-traditional/20-30-etc-years-senior"
    # soup = await fetch_and_parse(url)
    # extract_vocab_info(soup, url, SubVocab_Dict)
    # pprint(SubVocab_Dict)

    # print("\nDone")


if __name__ == "__main__":
    # asyncio.run(main())
    from tqdm import tqdm
    # import time
    # from pprint import pprint

    # Load all vocabulary links
    VocabIdx_Dict = dict()
    with open("/home/nlplab/atwolin/thesis/data/cambridge-parse/cambridge_parse.all_word_links.json", "r") as f:
        VocabIdx_Dict = json.load(f)

    # Get and save all vocabulary content
    # for category, links in tqdm(VocabIdx_Dict.items(), colour="green", desc="Category", ncols=100):
    #     if category == sys.argv[2]:
    #         break

    #     if category == "0–9" or category == "a" or category != sys.argv[1]:
    #         continue

    #     # print(f"Category: {category}")

    #     os.makedirs(f"../data/cambridge-html/{category}", exist_ok=True)

    #     for idx in tqdm(range(len(links)), desc=f"Vocabulary-{category}", ncols=100):
    #         soup = fetch_and_parse(links[idx])

    #         with open(f"../data/cambridge-html/{category}/{links[idx].split('/')[-1]}.html", "w") as f:
    #             f.write(str(soup))

    #         if idx != 0 and idx % 100 == 0:
    #             time.sleep(50)

    #         if idx != 0 and idx % 1000 == 0:
    #             time.sleep(300)

    # Convert all HTML files to JSON
    path = "/home/nlplab/atwolin/thesis/data/cambridge-html"
    dirs = os.listdir(path)
    download = dict()
    os.makedirs("../data/cambridge-parse/vocabV2", exist_ok=True)
    for dir in tqdm(dirs, colour="green", desc="Category", ncols=100):
        SubVocab_Dict: RESULT_FORMAT = {}
        for file in tqdm(os.listdir(os.path.join(path, dir)), desc=f"Vocabulary-{dir}", ncols=100):
            word = file.split(".")[0]

            with open(os.path.join(path, dir, file), "r") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                extract_vocab_info(soup, f"{PARSED_WORD_PREFIX}{word}", SubVocab_Dict)

        with open(f"/home/nlplab/atwolin/thesis/data/cambridge-parse/vocabV2/cambridge.{dir}.json", "w") as f:
            json.dump(SubVocab_Dict, f, ensure_ascii=False, indent=4)
