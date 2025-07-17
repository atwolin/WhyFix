import warnings
import pandas as pd
import numpy as np
import json
import nltk
from tqdm import tqdm

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction
from ckip_transformers.nlp import CkipWordSegmenter

# Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

# 過濾掉 RoBERTa 模型初始化警告
warnings.filterwarnings("ignore", message="Some weights of .* were not initialized")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")
# 過濾掉 BLEU 計算時的 n-gram 重疊警告
warnings.filterwarnings("ignore", message="The hypothesis contains 0 counts of.*")


BATCH_DATA = {}
ws_driver = CkipWordSegmenter(model="bert-base", device=1)


def load_longman_ground_truth():
    """
    Load the Longman ground truth data from a JSON file.
    """
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
    return longman_data


def load_structured_data(file_path):
    """
    Load structured data from a jsonl file.
    """
    structured_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line)
                    method = data.get('method')
                    idx = data.get('data_index')
                    content = data.get('content_payload')
                    if idx is not None and content:
                        if method not in structured_data:
                            structured_data[method] = {}
                        structured_data[method][idx] = {
                            'explanation_en': content.get('explanation_en', ''),
                            'explanation_zh_tw': content.get('explanation_zh_tw', ''),
                            'example_en': content.get('example_en', []),
                            'example_zh_tw': content.get('example_zh_tw', []),
                            'corresponding_collocation_en': content.get('corresponding_collocation_en', []),
                            'corresponding_collocation_examples_en': content.get('corresponding_collocation_examples_en', []),
                            'other_collocations_en': content.get('other_collocations_en', []),
                            'other_collocations_examples_en': content.get('other_collocations_examples_en', [])
                        }

                except json.JSONDecodeError as e:
                    print(f"\n{'=' * 50}\nError processing line: {line.strip()}\n{'=' * 50}")
                    raise e
    return structured_data


def load_fce_and_longman_data():
    """
    Load FCE and Longman data from structured files.
    """
    fce_t_fifty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_fce_t_fifty_0622-1458-fix.jsonl')
    fce_tf_fifty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_fce_tf_fifty_0622-1458-fix.jsonl')
    fce_t_eighty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_fce_t_eighty_0622-1458-fix.jsonl')
    fce_tf_eighty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_fce_tf_eighty_0622-1458-fix.jsonl')

    longman_ground_truth = load_longman_ground_truth()
    lg_fifty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_longman_t_fifty_0622-1458-fix.jsonl')
    lg_eighty = load_structured_data('/home/nlplab/atwolin/thesis/data/results/structured_data/structured_api_data_longman_t_eighty_0622-1458-fix.jsonl')

    global BATCH_DATA
    BATCH_DATA = {
        'fce_t_fifty': fce_t_fifty,
        'fce_tf_fifty': fce_tf_fifty,
        'fce_t_eighty': fce_t_eighty,
        'fce_tf_eighty': fce_tf_eighty,
        'longman_ground_truth': longman_ground_truth,
        'lg_fifty': lg_fifty,
        'lg_eighty': lg_eighty
    }
    return


def calculate_metrics(references, candidates):
    """
    Calculate BLEU, METEOR, ROUGE, and BERTScore for paired references and candidates.
    Uses BLEU smoothing methods to handle sparse n-gram matches.
    """
    # Initialize ROUGE scorer
    rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Initialize SmoothingFunction
    smoothing = SmoothingFunction()

    # Initialize metric containers
    bleu_scores = []
    bleu_smoothed1_scores = []
    bleu_smoothed2_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Process each reference-candidate pair
    for ref, cand in tqdm(zip(references, candidates), total=len(references), desc="Calculating metrics"):
        if not ref or not cand:
            continue

        # Tokenize for BLEU and METEOR
        ref_tokens = nltk.word_tokenize(ref.lower())
        # cand_tokens = ws_driver(cand.lower())
        if cand.strip():
            ws_result = ws_driver(cand.lower(), show_progress=False)
            # 檢查結果類型並扁平化如果需要
            if isinstance(ws_result[0], list):
                # 如果是嵌套列表，扁平化它
                cand_tokens = [token for sent in ws_result for token in sent]
            else:
                # 如果已經是扁平列表，直接使用
                cand_tokens = ws_result
        else:
            cand_tokens = []

        # Calculate BLEU (standard)
        bleu = sentence_bleu([ref_tokens], cand_tokens)
        bleu_scores.append(bleu)

        # Calculate BLEU with method1 (adding 1 to both numerator and denominator)
        bleu_smoothed1 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing.method1)
        bleu_smoothed1_scores.append(bleu_smoothed1)

        # Calculate BLEU with method2 (adds 1 only when n-gram precision is 0)
        bleu_smoothed2 = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing.method2)
        bleu_smoothed2_scores.append(bleu_smoothed2)

        # Calculate METEOR
        meteor = meteor_score([ref_tokens], cand_tokens)
        meteor_scores.append(meteor)

        # Calculate ROUGE scores
        rouge_scores = rouge_scorer_instance.score(ref, cand)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    # Calculate BERTScore
    if candidates and references:
        P, R, F1 = bert_score(candidates, references, lang='en')
        bert_scores = F1.tolist()
        avg_bert = np.mean(bert_scores) if bert_scores else 0
    else:
        avg_bert = 0

    # Compute averages
    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    avg_bleu_smoothed1 = np.mean(bleu_smoothed1_scores) if bleu_smoothed1_scores else 0
    avg_bleu_smoothed2 = np.mean(bleu_smoothed2_scores) if bleu_smoothed2_scores else 0
    avg_meteor = np.mean(meteor_scores) if meteor_scores else 0
    avg_rouge1 = np.mean(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = np.mean(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = np.mean(rougeL_scores) if rougeL_scores else 0

    return {
        'BLEU': avg_bleu,
        'BLEU-Smoothed1': avg_bleu_smoothed1,
        'BLEU-Smoothed2': avg_bleu_smoothed2,
        'METEOR': avg_meteor,
        'ROUGE-1': avg_rouge1,
        'ROUGE-2': avg_rouge2,
        'ROUGE-L': avg_rougeL,
        'BERTScore': avg_bert
    }


# ---- System Compare ----
# longman ground truth vs. (1) lg_fifty (2) lg_eighty
def longman_vs_one_length_systems(longman_data, system_data):
    """
    Compare Longman ground truth with system outputs.
    """
    results = {}
    for method_name, method_content in system_data.items():
        references = [longman_data.get(idx, '') for idx in sorted(longman_data.keys())]
        candidates = [method_content.get(idx, '').get('explanation_en', '') for idx in sorted(method_content.keys())]

        metrics = calculate_metrics(references, candidates)
        results[method_name] = metrics

    return results


def compare_longman_with_systems():
    print(f"\n{'=' * 50}\nComparing Longman ground truth with system outputs\n{'=' * 50}\n")
    longman_data = BATCH_DATA['longman_ground_truth']
    systems_data = {
        'lg_fifty': BATCH_DATA['lg_fifty'],
        'lg_eighty': BATCH_DATA['lg_eighty'],
    }

    results = {}
    for system_name, system_content in systems_data.items():
        results[system_name] = longman_vs_one_length_systems(longman_data, system_content)
    return results


# ---- Ablation ----
# 1. input sentence:
#    1) fce_t_fifty vs. fce_tf_fifty
#    2) fce_t_eighty vs. fce_tf_eighty
def input_sentence_comparison(target_sentence_data, tf_sentence_data):
    """
    Compare input sentences between two systems:
    target sentence pair: only one error-and-original sentence, which contains a single error.
    target and one following sentence pair: two sentences, one with an error and the subsequent sentence.
    """
    results = {}
    method_names = sorted(set(target_sentence_data.keys()) & set(tf_sentence_data.keys()))

    # 添加進度條來顯示方法的處理進度
    for method_name in tqdm(method_names, desc="Comparing methods"):
        target_method = target_sentence_data[method_name]
        tf_method = tf_sentence_data[method_name]

        # Get common indices
        common_indices = set(target_method.keys()) & set(tf_method.keys())

        # Extract paired sentences
        en_references = []
        en_candidates = []
        zh_tw_references = []
        zh_tw_candidates = []

        for idx in tqdm(sorted(common_indices), desc=f"Processing {method_name}", leave=False):
            target_en = target_method[idx].get('explanation_en', '')
            tf_en = tf_method[idx].get('explanation_en', '')
            if target_en and tf_en:
                en_references.append(target_en)
                en_candidates.append(tf_en)

            target_zh = target_method[idx].get('explanation_zh_tw', '')
            tf_zh = tf_method[idx].get('explanation_zh_tw', '')
            if target_zh and tf_zh:
                zh_tw_references.append(target_zh)
                zh_tw_candidates.append(tf_zh)

        # Calculate metrics
        metrics_en = calculate_metrics(en_references, en_candidates)
        metrics_zh_tw = calculate_metrics(zh_tw_references, zh_tw_candidates)

        results[method_name] = {
            'en': metrics_en,
            'zh_tw': metrics_zh_tw,
            'samples_en': len(en_references),
            'samples_zh_tw': len(zh_tw_references)
        }

    return results


def compare_fce_t_vs_tf():
    """ Compare FCE target sentences with FCE TF sentences."""
    # Compare input sentences
    print(f"\n{'=' * 50}\nComparing FCE target sentences with FCE TF sentences\n{'=' * 50}\n")
    return {
        'fce_t_fifty_vs_fce_tf_fifty': input_sentence_comparison(
            BATCH_DATA['fce_t_fifty'], BATCH_DATA['fce_tf_fifty']
        ),
        'fce_t_eighty_vs_fce_tf_eighty': input_sentence_comparison(
            BATCH_DATA['fce_t_eighty'], BATCH_DATA['fce_tf_eighty']
        )
    }


# 2. example:
#    1) #example_en, #example_zh: (1) lg_fifty (2) lg_eighty (3) fce_t_fifty (4) fce_tf_fifty (5) fce_t_eighty (6) fce_tf_eithty
#    2) #collocation_en, #collocation_zh: (1) lg_fifty (2) lg_eighty (3) fce_t_fifty (4) fce_tf_fifty (5) fce_t_eighty (6) fce_tf_eithty
def enhanced_examples_statistics(system_data):
    """
    Enhanced statistics for examples and collocations in the system data.
    Counts both occurrences and token lengths using NLTK tokenization.
    """
    results = {}
    field_types = [
        'example_en',
        'example_zh_tw',
        'corresponding_collocation_en',
        'corresponding_collocation_examples_en',
        'other_collocations_en',
        'other_collocations_examples_en'
    ]

    for method_name, method_content in system_data.items():
        stats = {}

        for field_type in field_types:
            # Initialize statistics for this field
            field_stats = {
                'count': 0,                   # Total number of items
                'instances': 0,               # Number of data points that have this field
                'token_counts': [],           # List of token counts for each item
                'avg_tokens': 0,              # Average tokens per item
                'std_tokens': 0,              # Standard deviation of token counts
                'min_tokens': 0,              # Minimum tokens in an item
                'max_tokens': 0,              # Maximum tokens in an item
                'items_with_expected_count': 0  # Items with expected count (e.g., 2 for example_en)
            }

            # Expected counts for validation
            expected_counts = {
                'example_en': 2,
                'example_zh_tw': 2,
                'corresponding_collocation_en': 1,
                'corresponding_collocation_examples_en': 2,
                'other_collocations_en': 3,  # Up to 3
                'other_collocations_examples_en': 3  # Up to 3
            }

            all_items = []  # Collect all items for this field

            # Process each data point
            for idx, item in method_content.items():
                items = item.get(field_type, [])

                if items:
                    field_stats['instances'] += 1
                    if len(items) == expected_counts.get(field_type, 0):
                        field_stats['items_with_expected_count'] += 1

                field_stats['count'] += len(items)
                all_items.extend(items)

                # Calculate token length for each item
                for text in items:
                    if text is None or text == '':
                        continue
                    if not isinstance(text, str):
                        text = str(text)
                    if field_type == 'example_zh_tw':
                        # Use CkipWordSegmenter for Traditional Chinese
                        if text.strip():  # Make sure text is not empty or just whitespace
                            tokens = ws_driver(text, show_progress=False)
                        else:
                            # tokens = []
                            continue  # Skip empty strings
                    else:
                        tokens = nltk.word_tokenize(text)
                    field_stats['token_counts'].append(len(tokens))

            # Calculate statistics if we have any items
            if field_stats['token_counts']:
                field_stats['avg_tokens'] = np.mean(field_stats['token_counts'])
                field_stats['std_tokens'] = np.std(field_stats['token_counts'])
                field_stats['min_tokens'] = min(field_stats['token_counts']) if field_stats['token_counts'] else 0
                field_stats['max_tokens'] = max(field_stats['token_counts']) if field_stats['token_counts'] else 0

            stats[field_type] = field_stats

        results[method_name] = stats

    return results


def count_fce_and_longman_examples():
    print(f"\n{'=' * 50}\nCounting examples and collocations in FCE and Longman systems\n{'=' * 50}\n")
    results = {}
    for system_name in tqdm(BATCH_DATA.keys(), desc="Processing systems"):
        if system_name == 'longman_ground_truth':
            continue
        results[system_name] = enhanced_examples_statistics(BATCH_DATA[system_name])
    return results


# 3. output length:
#    1) length statistics: (1) lg_fifty (2) lg_eighty (3) fce_t_fifty (4) fce_tf_fifty (5) fce_t_eighty (6) fce_tf_eithty
#    2) lg_eighty vs. lg_fifty
#    3) fce_t_eighty vs. fce_t_fifty
#    4) fce_tf_eighty vs. fce_tf_fifty
def length_statistics(system_data):
    """
    Calculate length statistics for a single system's data.
    Focuses on explanation texts and their token lengths.

    Args:
        system_data (dict): A dictionary containing system data

    Returns:
        dict: Statistics for each field type including token counts and averages
    """
    results = {}
    # Fields to analyze
    field_types = [
        'explanation_en',
        'explanation_zh_tw',
        # 'example_en',
        # 'example_zh_tw',
        # 'corresponding_collocation_en',
        # 'corresponding_collocation_examples_en',
        # 'other_collocations_en',
        # 'other_collocations_examples_en'
    ]

    for method_name, method_content in system_data.items():
        method_stats = {}

        for field in field_types:
            # Initialize stats for this field
            field_stats = {
                'token_counts': [],         # List of token counts for each item
                'instance_count': 0,        # Number of instances containing this field
                'total_tokens': 0,          # Total tokens across all instances
                'avg_tokens': 0,            # Average tokens per item
                'std_tokens': 0,            # Standard deviation of tokens
                'min_tokens': 0,            # Minimum tokens in an item
                'max_tokens': 0             # Maximum tokens in an item
            }

            # Process each data point
            for idx, item in method_content.items():
                # Handle explanation fields (single strings)
                if field in ['explanation_en', 'explanation_zh_tw']:
                    text = item.get(field, '')
                    if text:
                        if field == 'explanation_en':
                            tokens = nltk.word_tokenize(text)
                        else:
                            tokens = ws_driver(text, show_progress=False)
                        token_count = len(tokens)
                        field_stats['token_counts'].append(token_count)
                        field_stats['total_tokens'] += token_count
                        field_stats['instance_count'] += 1

                # Handle array fields
                else:
                    items = item.get(field, [])
                    if items:
                        field_stats['instance_count'] += 1

                        for text in items:
                            tokens = nltk.word_tokenize(text)
                            token_count = len(tokens)
                            field_stats['token_counts'].append(token_count)
                            field_stats['total_tokens'] += token_count

            # Calculate summary statistics if we have any data
            if field_stats['token_counts']:
                field_stats['avg_tokens'] = np.mean(field_stats['token_counts'])
                field_stats['std_tokens'] = np.std(field_stats['token_counts'])
                field_stats['min_tokens'] = min(field_stats['token_counts'])
                field_stats['max_tokens'] = max(field_stats['token_counts'])

            method_stats[field] = field_stats

        results[method_name] = method_stats

    return results


def count_fce_and_longman_explanations():
    print(f"\n{'=' * 50}\nCounting explanations in FCE and Longman systems\n{'=' * 50}\n")
    results = {}
    for system_name in tqdm(BATCH_DATA.keys(), desc="Processing systems"):
        if system_name != 'longman_ground_truth':
            results[system_name] = length_statistics(BATCH_DATA[system_name])
        else:
            longman_data = {}
            longman_data['longman_ground_truth'] = {}
            for idx, explanation in BATCH_DATA[system_name].items():
                if isinstance(explanation, str):
                    longman_data[system_name][idx] = {
                        'explanation_en': explanation,
                        'explanation_zh_tw': ''
                    }
            results[system_name] = length_statistics(longman_data)
            # break
    return results


def compare_length_explanations(system_data_fifty, system_data_eighty):
    """
    Compare explanations between fifty and eighty word limit systems,
    treating eighty word limit as reference and fifty word limit as candidate.
    Calculate metrics for both English and Traditional Chinese explanations.
    """
    results = {}
    for fifty_method_name, eighty_method_name in zip(sorted(system_data_fifty.keys()), sorted(system_data_eighty.keys())):
        fifty_content = system_data_fifty[fifty_method_name]
        eighty_content = system_data_eighty[eighty_method_name]

        # Prepare references and candidates for English explanations
        en_references = []
        en_candidates = []
        for idx in sorted(eighty_content.keys()):
            eighty_explanation = eighty_content.get(idx, {}).get('explanation_en', '')
            fifty_explanation = fifty_content.get(idx, {}).get('explanation_en', '')
            if eighty_explanation and fifty_explanation:
                en_references.append(eighty_explanation)
                en_candidates.append(fifty_explanation)

        # Prepare references and candidates for Traditional Chinese explanations
        zh_tw_references = []
        zh_tw_candidates = []
        for idx in sorted(eighty_content.keys()):
            eighty_explanation = eighty_content.get(idx, {}).get('explanation_zh_tw', '')
            fifty_explanation = fifty_content.get(idx, {}).get('explanation_zh_tw', '')
            if eighty_explanation and fifty_explanation:
                zh_tw_references.append(eighty_explanation)
                zh_tw_candidates.append(fifty_explanation)

        # Calculate metrics
        en_metrics = calculate_metrics(en_references, en_candidates)
        zh_tw_metrics = calculate_metrics(zh_tw_references, zh_tw_candidates)

        results[fifty_method_name] = {
            'explanation_en': en_metrics,
            'explanation_zh_tw': zh_tw_metrics,
            'metrics_en': en_metrics,
            'metrics_zh_tw': zh_tw_metrics
        }
    return results


def compare_fce_and_longman_pair_explanations():
    """
    Compare explanations between FCE and Longman systems with different length constraints.
    """
    print(f"\n{'=' * 50}\nComparing FCE and Longman explanations for different length constraints\n{'=' * 50}\n")
    # Compare FCE systems
    fce_t_fifty_vs_eighty = compare_length_explanations(
        BATCH_DATA['fce_t_fifty'], BATCH_DATA['fce_t_eighty']
    )
    fce_tf_fifty_vs_eighty = compare_length_explanations(
        BATCH_DATA['fce_tf_fifty'], BATCH_DATA['fce_tf_eighty']
    )

    # Compare Longman systems
    longman_fifty_vs_eighty = compare_length_explanations(
        BATCH_DATA['lg_fifty'], BATCH_DATA['lg_eighty']
    )
    return {
        'fce_t_fifty_vs_eighty': fce_t_fifty_vs_eighty,
        'fce_tf_fifty_vs_eighty': fce_tf_fifty_vs_eighty,
        'longman_fifty_vs_eighty': longman_fifty_vs_eighty
    }


def main():
    """
    Main function to load data and run all comparisons.
    """
    load_fce_and_longman_data()
    with open('./postprocess/output/automatic_metrics_data.json', 'w') as f:
        json.dump(BATCH_DATA, f, indent=4)

    results = {}
    results['system_compare'] = compare_longman_with_systems()
    results['ablation'] = {
        'input_sentence': compare_fce_t_vs_tf(),
        'enhanced_examples': count_fce_and_longman_examples(),
        'length_explanations': count_fce_and_longman_explanations(),
        'pair_explanations': compare_fce_and_longman_pair_explanations()
    }
    return results


if __name__ == "__main__":
    results = main()
    print(f"\n{'=' * 50}\nAutomatic metrics completed.\n{'=' * 50}\n")
    # print(results)

    # Save results to a file
    print(f"\n{'=' * 50}\nSaving results to output/automatic_metrics_results.json\n{'=' * 50}\n")
    with open('./postprocess/output/automatic_metrics_results.json', 'w') as f:
        json.dump(results, f, indent=4)
