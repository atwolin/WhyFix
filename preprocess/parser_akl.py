import requests
import os
from bs4 import BeautifulSoup
import pandas as pd

url = "https://www.eapfoundation.com/vocab/academic/akl/?utm_source=chatgpt.com"


def get_akl_lisk():
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    akl_list = []
    for row in soup.find(class_="offset").find_all("tr"):
        num, word, pos, gsl, awl, avl = row.find_all("td")
        print(num.text, word.text, pos.text, gsl.text, awl.text, avl.text)

        akl_list.append([word.text, pos.text])

    return akl_list


def save_to_csv(akl_list):
    df = pd.DataFrame(akl_list, columns=["word", "pos"])
    os.makedirs("../data/akl", exist_ok=True)
    df.to_csv("../data/akl/akl_list.csv", index=False)


if __name__ == "__main__":
    akl = get_akl_lisk()
    save_to_csv(akl[1:])
    print(akl[:5])
