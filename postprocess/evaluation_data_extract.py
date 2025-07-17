import json
import pandas as pd
from collections import defaultdict


def convert_data_to_source_blocks():
    # 1. Load the JSONL data
    jsonl_data = []
    with open('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_0622-1458.jsonl', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('//'):  # Skip comments and empty lines
                jsonl_data.append(json.loads(line))

    # 2. Load Longman CSV data
    longman_data = {}
    df = pd.read_csv('/home/nlplab/atwolin/thesis/data/longman/longman_sentences_dictionary_filtered_groundtruth.csv')
    for _, row in df.iterrows():
        idx = row.get('index')
        if pd.notna(idx):
            idx = int(idx)
            explanation = row.get('longman_explantion', '')
            see_note = row.get('longman_see_note_explantion', '')

            # Combine explanations if both exist
            combined_explanation = explanation if pd.notna(explanation) else ''
            if pd.notna(see_note) and see_note.strip():
                if combined_explanation:
                    combined_explanation += "\n\n"
                combined_explanation += see_note

            longman_data[idx] = combined_explanation

    # 3. Group data by dataset and index
    unique_combinations = set()
    for item in jsonl_data:
        dataset = item.get('dataset', '')
        data_index = item.get('data_index', -1)
        if dataset and data_index >= 0:
            unique_combinations.add((dataset, data_index))

    # 4. Create source_data_blocks
    source_data_blocks = []
    # target_index_longman = [
    #     53, 78, 130, 191, 210, 345, 348, 495, 570, 594, 909, 946, 960, 1115,
    #     1257, 1331, 1358, 1409, 1446, 1476, 1512, 1524, 1551, 1568,
    #     1576, 1698, 1737]
    target_index_longman = [
          53,     78,  345,  348,  495,  570,  594,   946, 960, 1115,
          1257, 1331, 1358, 1409, 1476, 1512, 1524, 1568, 1576, 1698]
    # target_index_fce = [
    #     9, 353, 380, 468, 678, 845, 1113, 1553, 1793, 2190, 2425, 2486,
    #     3045, 3071, 3240, 3415, 3541, 3791, 3929, 4085, 4501, 4818, 4981, 5149,
    #     5197, 5491, 5877]
    target_index_fce = [
        9, 380, 678, 845, 1113, 1793, 2190, 2486,  3045, 3071,
        3240, 3415, 3541, 4501, 4818, 4981, 5149,  5197, 5491, 5877
    ]
    for dataset, index in sorted(unique_combinations):
        if dataset == 'longman' and index not in target_index_longman:
            continue
        if dataset == 'fce' and index not in target_index_fce:
            continue
        print(f"Processing dataset: {dataset}, index: {index}")
        # Filter items for this dataset and index
        relevant_items = [item for item in jsonl_data if item.get('dataset') == dataset and item.get('data_index') == index]

        # Get a representative formatted sentence
        formatted_sentence = next((item.get('formatted_sentence', '') for item in relevant_items), '')

        # Initialize random_data with empty entries for each method
        random_data = []
        for method_name in ['baseline', 'dictionary', 'collocation', 'metalinguistic', 'mix']:
            random_data.append({
                'header': method_name,
                'content_en': '',
                'content_zh': ''
            })

        # Add original entry based on dataset
        if dataset == 'longman':
            random_data.append({
                'header': 'original',
                'content_en': longman_data.get(index, ''),
                'content_zh': ''
            })
        else:
            random_data.append({
                'header': 'original',
                'content_en': '',
                'content_zh': ''
            })

        # Initialize static_data with empty entries
        static_data = {
            'example_sentences': {
                'header': 'illustrative example:\nexample sentences',
                'content_en': '',
                'content_zh': ''
            },
            'example_collocations': {
                'header': 'illustrative example:\nexample collocations',
                'content_en': '',
                'content_zh': ''
            },
            'output_80': {
                'header': 'output length = 80',
                'content_en': '',
                'content_zh': ''
            },
            'input_add_one': {
                'header': 'input sentence:\nadd one subsequent sentence',
                'content_en': '',
                'content_zh': ''
            },
            'output_50': {
                'header': 'output length = 50:\ninput sentence only',
                'content_en': '',
                'content_zh': ''
            }
        }

        # Populate random_data and static_data from relevant_items
        for item in relevant_items:
            method = item.get('method', '')
            input_type = item.get('input_type', '')
            output_length = item.get('output_length', '')

            # Skip items without content_payload
            if 'content_payload' not in item:
                continue

            content_en = item['content_payload'].get('explanation_en', '')
            content_zh_tw = item['content_payload'].get('explanation_zh_tw', '')

            # Update random_data for standard methods
            for entry in random_data:
                if (
                    output_length == ' in 50 words or less' and
                    input_type == 'tartget sentence only' and
                    entry['header'] == method
                ):
                    entry['content_en'] = content_en
                    entry['content_zh'] = content_zh_tw

            # Handle example sentences and collocations
            if method == 'dictionary' and 'example_en' in item['content_payload'] and item['content_payload']['example_en']:
                static_data['example_sentences']['content_en'] = str(item['content_payload']['example_en'])
                if 'example_zh_tw' in item['content_payload']:
                    static_data['example_sentences']['content_zh'] = str(item['content_payload']['example_zh_tw'])

            if method == 'collocation' and 'corresponding_collocation_en' in item['content_payload']:
                static_data['example_collocations']['content_en'] = str(item['content_payload']['corresponding_collocation_en'])

            if method == 'dictionary':
                # Handle different output lengths and input types
                if '80' in output_length:
                    if input_type == 'tartget sentence only':
                        static_data['output_80']['content_en'] = content_en
                        static_data['output_80']['content_zh'] = content_zh_tw
                elif output_length == ' in 50 words or less':
                    if input_type == 'tartget sentence only':
                        static_data['output_50']['content_en'] = content_en
                        static_data['output_50']['content_zh'] = content_zh_tw
                    elif input_type == 'target and following sentences':
                        static_data['input_add_one']['content_en'] = content_en
                        static_data['input_add_one']['content_zh'] = content_zh_tw

        # Create the block
        block = {
            'fixed_data': {
                'dataset': dataset,
                'index': index,
                'formatted sentence': formatted_sentence
            },
            'random_data': random_data,
            'static_data': static_data
        }
        source_data_blocks.append(block)

    return source_data_blocks


if __name__ == "__main__":
    source_data_blocks = convert_data_to_source_blocks()

    # Print the first block as an example
    print("First block example:")
    import pprint
    pprint.pprint(source_data_blocks[0])

    print(f"\nTotal blocks generated: {len(source_data_blocks)}")

    # Optionally, save the data to a Python file
    with open('/home/nlplab/atwolin/thesis/code/postprocess/output/source_data_blocks_0622-1458.py', 'w') as f:
        f.write("source_data_blocks = ")
        f.write(repr(source_data_blocks))

    with open('/home/nlplab/atwolin/thesis/data/results/structured_data/source_data_blocks_0622-1458.json', 'w') as f:
        json.dump(source_data_blocks, f, indent=4, ensure_ascii=False)
