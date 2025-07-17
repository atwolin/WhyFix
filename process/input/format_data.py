# python -m process.input.format_data
import os
from tqdm import tqdm
import json

from ..utils.helper_functions import read_pdf_to_string


def create_documents():
    folderPath_allData = "/home/nlplab/atwolin/thesis/data/L2-knowledge/all"
    files = os.listdir(folderPath_allData)

    dataset = []
    print("Creating documents")
    for file in tqdm(files):
        data = {}
        filePath = os.path.join(folderPath_allData, file)
        data['title'] = ' '.join(' '.join(file.split('.')[:-1]).split('_')[1:])
        data['text'] = read_pdf_to_string(filePath)
        dataset.append(data)
        # break
    with open('/home/nlplab/atwolin/thesis/data/L2-knowledge/l2_knowledge.json', 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Saved {len(dataset)} documents to l2_knowledge.json")


def create_examples(input, output):
    data = []
    with open(input, 'r') as f:
        for line in f:
            data.append(line.strip())

    examples = []
    print("Creating examples")
    for i, item in tqdm(enumerate(data)):
        if not item.startswith('Question'):
            continue
        qa_pair = {}
        qa_pair['question'] = ' '.join(item.split(':')[1:]).strip()
        qa_pair['answer'] = ' '.join(data[i + 1].split(':')[1:]).strip()
        examples.append(qa_pair)

    with open(f'/home/nlplab/atwolin/thesis/data/L2-knowledge/{output}.json', 'w') as f:
        json.dump(examples, f, indent=4)
    print(f"Saved {len(examples)} examples to {output}.json")


def main():
    create_documents()
    filePath_cuases = "/home/nlplab/atwolin/thesis/data/L2-knowledge/qa_examples_raw_causes.txt"
    outputFile_cuases = "qa_examples_causes"
    filePath_academic_writing = "/home/nlplab/atwolin/thesis/data/L2-knowledge/qa_examples_raw_academic_writing.txt"
    outputFile_academic_writing = "qa_examples_academic_writing"
    create_examples(filePath_cuases, outputFile_cuases)
    create_examples(filePath_academic_writing, outputFile_academic_writing)


if __name__ == "__main__":
    main()
