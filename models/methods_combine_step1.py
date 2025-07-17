# filepath: /home/nlplab/atwolin/thesis/code/model/methods_combine.py
import sys
import pandas as pd
from .llm_setup import (
    load_experiment,
    FilePath,
    format_prompt_baseline,
    format_prompt_ragDictionary,
    format_prompt_ragCollocation,
    format_prompt_ragL2_causes,
    format_prompt_ragL2_academicWriting,
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
def create_batch_tasks_first_three(df_examples, datasetType, sentenceType, outputLength):
    # Baseline
    prompt_baseline = experimentDoc['prompt_v2']['baseline']
    prompts_baseline = df_examples.apply(format_prompt_baseline, axis=1, prompt=prompt_baseline, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_baseline = create_batch_response_format_tasks(
        df_examples, prompts_baseline, datasetType, 'BL', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    )

    # RAG from dictionary information
    prompt_ragDictionary = experimentDoc['prompt_v2']['ragDictionary']
    prompts_ragDictionary = df_examples.apply(format_prompt_ragDictionary, axis=1, prompt=prompt_ragDictionary, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragDictionary = create_batch_response_format_tasks(
        df_examples, prompts_ragDictionary, datasetType, 'Dict', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    )

    # # RAG from collocations
    # prompt_ragCollocation = experimentDoc['prompt_v2']['ragCollocation']
    # prompts_ragCollocation = df_examples.apply(format_prompt_ragCollocation, axis=1, prompt=prompt_ragCollocation, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    # tasks_ragCollocation = create_batch_response_format_tasks(
    #     df_examples, prompts_ragCollocation, datasetType, 'Collo', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    # )

    # RAG from collocations and output example collocations
    prompt_ragCollocation_withExamples = experimentDoc['prompt_v2']['ragCollocation_v2']
    prompts_ragCollocation_withExamples = df_examples.apply(format_prompt_ragCollocation, axis=1, prompt=prompt_ragCollocation_withExamples, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragCollocation_withExamples = create_batch_response_format_tasks(
        df_examples, prompts_ragCollocation_withExamples, datasetType, 'Collo2', sentenceType, outputLength, experimentDoc['output_schema']['collocation']
    )

    # all_tasks = tasks_baseline + tasks_ragDictionary + tasks_ragCollocation + tasks_ragCollocation_withExamples
    all_tasks = tasks_baseline + tasks_ragDictionary + tasks_ragCollocation_withExamples
    return all_tasks


def create_batch_tasks_ragL2_resource(df_examples, datasetType, sentenceType, outputLength):
    # RAG from L2 linguist research: get resource
    prompt_ragL2_stage1 = experimentDoc['prompt']['causes']
    prompt_ragL2_stage2 = experimentDoc['prompt']['academic_writing']

    prompts_ragL2_causes = df_examples.apply(format_prompt_ragL2_causes, axis=1, prompt=prompt_ragL2_stage1, sentence_type=sentenceType, role=ROLE).tolist()
    prompts_ragL2_academicWriting = df_examples.apply(format_prompt_ragL2_academicWriting, axis=1, prompt=prompt_ragL2_stage2, sentence_type=sentenceType, role=ROLE).tolist()

    tasks_ragL2_causes = create_batch_response_format_tasks(
        df_examples, prompts_ragL2_causes, datasetType, 'L2', sentenceType, outputLength, experimentDoc['output_schema']['l2_knowledge']
    )
    tasks_ragL2_academicWriting = create_batch_response_format_tasks(
        df_examples, prompts_ragL2_academicWriting, datasetType, 'L2', sentenceType, outputLength, experimentDoc['output_schema']['l2_knowledge']
    )
    return tasks_ragL2_causes, tasks_ragL2_academicWriting


if __name__ == "__main__":
    # ---- Test ----
    # df_fce_sample = pd.read_json(
    #     "/home/nlplab/atwolin/thesis/data/fce-released-dataset/data/fce_sample_with_collocation.json"
    # ).loc[0:1]
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # df_fce_sample = df_fce_sample.loc[0:1]  # Apply slicing
    # df_fce_sample = df_fce_sample.set_index('index')
    # datasets = {"fce_sp": df_fce_sample}

    # Complete wet
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

    # ---- Sample ----
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
    df_fce = df_fce.set_index('index')
    df_longman = pd.read_json(PATHS.filePath_longman_withCollocation)
    df_longman = df_longman.set_index('index')
    datasets = {"lg_fu": df_longman, "fce_fu": df_fce}

    sentenceTypes = ["t", "tf"]
    outputLenTypes = ["fifty", "eighty"]
    task_collective_first_three = []
    task_collective_ragL2_causes = []
    task_collective_ragL2_academicWriting = []
    # Create tasks for each dataset, sentence type, and output length
    for dataset in datasets.keys():
        df_examples = datasets[dataset]
        for sentenceType in sentenceTypes:
            if dataset.startswith("l") and sentenceType == "tf":
                continue
            for outputLength in outputLenTypes:
                print(f"Processing dataset: {dataset}, sentenceType: {sentenceType}, outputLength: {outputLength}")

                # Create tasks
                tasks_first_three = create_batch_tasks_first_three(df_examples, dataset, sentenceType, outputLength)
                tasks_ragL2_causes, tasks_ragL2_academicWriting = create_batch_tasks_ragL2_resource(df_examples, dataset, sentenceType, outputLength)

                # task_collective_first_three.extend(tasks_first_three)
                # task_collective_ragL2_causes.extend(tasks_ragL2_causes)
                # task_collective_ragL2_academicWriting.extend(tasks_ragL2_academicWriting)

                # Perform batch processing
                folderPath_tasks_first_three = perform_batch_workflow(tasks_first_three, f"first_three_methods_{dataset}_{sentenceType}_{outputLength}", DATE)
                print('=' * 50)
                folderPath_tasks_ragL2_causes = perform_batch_workflow(tasks_ragL2_causes, f"ragL2_causes_{dataset}_{sentenceType}_{outputLength}", DATE)
                print('=' * 50)
                folderPath_tasks_ragL2_academicWriting = perform_batch_workflow(tasks_ragL2_academicWriting, f"ragL2_academic_writing_{dataset}_{sentenceType}_{outputLength}", DATE)

    # Perform batch processing
    # folderPath_tasks_first_three = perform_batch_workflow(task_collective_first_three, "first_three_methods", DATE)
    # print('=' * 50)
    # folderPath_tasks_ragL2_causes = perform_batch_workflow(task_collective_ragL2_causes, "ragL2_causes", DATE)
    # print('=' * 50)
    # folderPath_tasks_ragL2_academicWriting = perform_batch_workflow(task_collective_ragL2_academicWriting, "ragL2_academic_writing", DATE)
