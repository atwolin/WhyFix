import os
import sys
import pandas as pd
import json

from .llm_setup import (
    load_experiment,
    FilePath,
    format_prompt_ragL2_explanation,
    format_prompt_ragMix,
)
from .batch_api import (
    create_batch_response_format_tasks,
    perform_batch_workflow,
)

# Load experiment configuration
experimentDoc = load_experiment()

# Define file paths and parameters
LONGMAN_SAMPLE_TYPE = None
EMBEDDING_SIZE = None
ROLE = 'linguist'
DATE = ""
if len(sys.argv) > 3:
    LONGMAN_SAMPLE_TYPE = sys.argv[1]
    EMBEDDING_SIZE = sys.argv[2]
    DATE = sys.argv[3]
else:
    raise ValueError("Please provide the date for batch results as a command line argument.")

PATHS = FilePath(EMBEDDING_SIZE)


# Function to create prompts and perform batch processing
def add_batch_results_to_df(df, output_length, folder_batch_result, column_name):
    """
    customID: f"{datasetType}_{idx}_L2_fourOneNano_zero_{sentenceType}_linguist_{outputLength}"
    idx: df.index
    """
    # 1. 讀取 batchResult.jsonl
    # path_batch_result = PATHS.folderPath_batchResult + folder_batch_result + "batchResult.json"
    path_batch_result = os.path.join(PATHS.folderPath_batchResult, folder_batch_result, "batchResult.jsonl")
    results = {}
    with open(path_batch_result, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):  # 跳過空行或註解
                continue
            obj = json.loads(line)
            custom_id = obj['custom_id']
            # 2. 解析 index
            # custom_id 格式: {datasetType}_{idx}_L2_fourOneNano_zero_{sentenceType}_linguist_{outputLength}
            idx = int(custom_id.split('_')[2])
            cuurent_output_length = custom_id.split('_')[-1]  # 取得 outputLength

            # 3. 取得你要的內容（假設是 matched）
            matched = []
            try:
                if cuurent_output_length != output_length:
                    continue  # 如果 outputLength 不匹配，則跳過
                matched = json.loads(obj['response']['body']['output'][0]['content'][0]['text'])['matched']
                # print(f"Processing idx: {idx}, matched: {matched}")
            except Exception:
                pass
            results[idx] = matched

    # 4. 寫入 DataFrame
    # print(f"In add_batch_results_to_df(), df index: {df.index}, results keys: {results.keys()}")
    df[column_name] = df.index.map(lambda idx: results.get(idx, []))
    # print(f"In add_batch_results_to_df(), added {len(results)} results to column '{column_name}' for output length '{output_length}'.")
    return df


def update_df_with_knownledge(df, dataset, sentenceType, outputLength, date_batch_result=DATE):
    df_updated = df.copy()
    # print(f"in update_df_with_knownledge, {dataset}")
    df_updated = add_batch_results_to_df(
        df_updated,
        outputLength,
        f"ragL2_causes_{dataset}_{sentenceType}_{outputLength}-{date_batch_result}",
        f'causes_{dataset}_{sentenceType}_{outputLength}'
    )
    df_updated = add_batch_results_to_df(
        df_updated,
        outputLength,
        f"ragL2_academic_writing_{dataset}_{sentenceType}_{outputLength}-{date_batch_result}",
        f'academic_writing_{dataset}_{sentenceType}_{outputLength}'
    )
    # print(f"In update_df_with_knownledge(), {df_updated.columns}")
    return df_updated


ROLE = 'linguist'


def create_batch_tasks_part2(df_examples, datasetType, sentenceType, outputLength):
    # Get data from step 2
    df_examples_updated = update_df_with_knownledge(df_examples, datasetType, sentenceType, outputLength)
    with open(f'./model/output/df_examples_updated_{DATE}.json', 'w') as f:
        df_examples_updated.to_json(f, orient='records', lines=True)

    # RAG from L2 linguist research: get explanations and examples
    prompt_ragL2_explanation = experimentDoc['prompt_v2']['ragL2']

    prompts_ragL2 = df_examples_updated.apply(
        format_prompt_ragL2_explanation, axis=1, prompt=prompt_ragL2_explanation,
        sentence_type=sentenceType, output_len=outputLength, role=ROLE,
    ).tolist()

    tasks_ragL2 = create_batch_response_format_tasks(
        df_examples_updated, prompts_ragL2, datasetType, 'L2', sentenceType, outputLength,
        experimentDoc['output_schema']['explanation_and_examples']
    )

    # # RAG mixed
    # prompt_ragMix = experimentDoc['prompt_v2']['ragMix']
    # prompts_ragMix = df_examples_updated.apply(format_prompt_ragMix, axis=1, prompt=prompt_ragMix, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    # tasks_ragMix = create_batch_response_format_tasks(
    #     df_examples_updated, prompts_ragMix, datasetType, 'Mix', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    # )

    # RAG mixed with examples
    prompt_ragMix_withExamples = experimentDoc['prompt_v2']['ragMix_v2']
    prompts_ragMix_withExamples = df_examples_updated.apply(format_prompt_ragMix, axis=1, prompt=prompt_ragMix_withExamples, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragMix_withExamples = create_batch_response_format_tasks(
        df_examples_updated, prompts_ragMix_withExamples, datasetType, 'Mix2', sentenceType, outputLength, experimentDoc['output_schema']['collocation']
    )
    # return tasks_ragL2 + tasks_ragMix + tasks_ragMix_withExamples
    return tasks_ragL2 + tasks_ragMix_withExamples


if __name__ == "__main__":
    # ---- Test ----
    # df_fce_sample = pd.read_json(
    #     "/home/nlplab/atwolin/thesis/data/fce-released-dataset/data/fce_sample_with_collocation.json"
    # ).loc[0:1]
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # df_fce_sample = df_fce_sample.loc[0:1]  # Apply slicing
    # df_fce_sample = df_fce_sample.set_index('index')
    # datasets = {"fce_sp": df_fce_sample}

    # Completely wet
    # df_test = pd.read_json(PATHS.filePath_test_withCollocation)
    # df_test = df_test.set_index('index')
    # datasets = {"longman_test": df_test}

    # Test two sample from full datasets
    # df_fce = pd.read_json(PATHS.filePath_fce_withCollocation)
    # df_fce = df_fce.loc[0:1].copy()  # Apply slicing
    # df_fce = df_fce.set_index('index')
    # df_longman = pd.read_json(PATHS.filePath_longman_withCollocation)
    # df_longman = df_longman.loc[0:1].copy()  # Apply slicing
    # df_longman = df_longman.set_index('index')
    # datasets = {"lg_fu": df_longman, "fce_fu": df_fce}

    # ---- Main ----
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # if LONGMAN_SAMPLE_TYPE == 'R':
    #     df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_one_replace_withCollocation)
    # else:
    #     df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_withCollocation)
    # df_fce_sample = df_fce_sample.set_index('index')
    # df_longman_sample = df_longman_sample.set_index('index')
    # datasets = {"lg_sp": df_longman_sample, "fce_sp": df_fce_sample}

    # ---- Full ----
    df_fce = pd.read_json(PATHS.filePath_fce_withCollocation)
    df_longman = pd.read_json(PATHS.filePath_longman_withCollocation)
    df_fce = df_fce.set_index('index')
    df_longman = df_longman.set_index('index')
    datasets = {"lg_fu": df_longman, "fce_fu": df_fce}

    sentenceTypes = ["t", "tf"]
    outputLenTypes = ["fifty", "eighty"]
    task_collective_part2 = []
    # Create tasks for each dataset, sentence type, and output length
    for dataset in datasets.keys():
        df_examples = datasets[dataset]
        for sentenceType in sentenceTypes:
            if dataset.startswith("l") and sentenceType == "tf":
                continue
            for outputLength in outputLenTypes:
                print(f"Processing dataset: {dataset}, sentenceType: {sentenceType}, outputLength: {outputLength}")

                # Create tasks
                tasks_part2 = create_batch_tasks_part2(df_examples, dataset, sentenceType, outputLength)
                # task_collective_part2.extend(tasks_part2)
                folderPath_tasks_part2 = perform_batch_workflow(tasks_part2, f"part2_{dataset}_{sentenceType}_{outputLength}", DATE)


    # Perform batch processing
    # folderPath_tasks_part2 = perform_batch_workflow(task_collective_part2, "part2", DATE)
