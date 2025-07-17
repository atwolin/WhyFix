
import sys
import os
import time
import pandas as pd

from utils.files_io import (
    FilePath,
    load_method_config,
)
from model.model_setup import (
    format_prompt_baseline,
    format_prompt_ragDictionary,
    format_prompt_ragCollocation,
    format_prompt_ragL2_causes,
    format_prompt_ragL2_academicWriting,
    format_prompt_ragL2_explanation,
    format_prompt_ragMix
)
from model.batch_setup import (
    create_batch_response_format_tasks,
    perform_batch_workflow,
    check_batch_jobs,
    parse_batch_results,
)

from postprocess.combine_data import (
    update_df_with_knownledge,
)

# Load experiment configuration
experimentDoc = load_method_config("llm")

# Define file paths
PATHS = FilePath()
ROLE = 'linguist'


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

    # RAG from collocations
    prompt_ragCollocation = experimentDoc['prompt_v2']['ragCollocation']
    prompts_ragCollocation = df_examples.apply(format_prompt_ragCollocation, axis=1, prompt=prompt_ragCollocation, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragCollocation = create_batch_response_format_tasks(
        df_examples, prompts_ragCollocation, datasetType, 'Collo', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    )
    # RAG from collocations and output example collocations
    prompt_ragCollocation_withExamples = experimentDoc['prompt_v2']['ragCollocation_v2']
    prompts_ragCollocation_withExamples = df_examples.apply(format_prompt_ragCollocation, axis=1, prompt=prompt_ragCollocation_withExamples, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragCollocation_withExamples = create_batch_response_format_tasks(
        df_examples, prompts_ragCollocation_withExamples, datasetType, 'Collo2', sentenceType, outputLength, experimentDoc['output_schema']['collocation']
    )

    all_tasks = tasks_baseline + tasks_ragDictionary + tasks_ragCollocation + tasks_ragCollocation_withExamples
    # all_tasks = tasks_ragCollocation_withExamples
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


def create_batch_tasks_part2(df_examples, datasetType, sentenceType, outputLength):

    # Get data from step 2
    df_examples_updated = update_df_with_knownledge(df_examples)
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

    # RAG mixed
    prompt_ragMix = experimentDoc['prompt_v2']['ragMix']
    prompts_ragMix = df_examples_updated.apply(format_prompt_ragMix, axis=1, prompt=prompt_ragMix, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragMix = create_batch_response_format_tasks(
        df_examples_updated, prompts_ragMix, datasetType, 'Mix', sentenceType, outputLength, experimentDoc['output_schema']['explanation_and_examples']
    )

    # RAG mixed with examples
    prompt_ragMix_withExamples = experimentDoc['prompt_v2']['ragMix_v2']
    prompts_ragMix_withExamples = df_examples_updated.apply(format_prompt_ragMix, axis=1, prompt=prompt_ragMix_withExamples, sentence_type=sentenceType, output_len=outputLength, role=ROLE).tolist()
    tasks_ragMix_withExamples = create_batch_response_format_tasks(
        df_examples_updated, prompts_ragMix_withExamples, datasetType, 'Mix2', sentenceType, outputLength, experimentDoc['output_schema']['collocation']
    )
    return tasks_ragL2 + tasks_ragMix + tasks_ragMix_withExamples


def batch_flow_stage1():
    # --------------  test -----------------
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # df_fce_sample = df_fce_sample.loc[0:1]  # Apply slicing

    # df_fce_sample = df_fce_sample.set_index('index')
    # datasets = {"fce_sp": df_fce_sample}

    # --------------  production -----------------
    df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_withCollocation)
    df_fce_sample = df_fce_sample.set_index('index')
    df_longman_sample = df_longman_sample.set_index('index')
    datasets = {"lg_sp": df_longman_sample, "fce_sp": df_longman_sample}

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

                task_collective_first_three.extend(tasks_first_three)
                task_collective_ragL2_causes.extend(tasks_ragL2_causes)
                task_collective_ragL2_academicWriting.extend(tasks_ragL2_academicWriting)

    # Perform batch processing
    perform_batch_workflow(task_collective_first_three, "first_three_methods", DATE)
    print('=' * 50)
    perform_batch_workflow(task_collective_ragL2_causes, "ragL2_causes", DATE)
    print('=' * 50)
    perform_batch_workflow(task_collective_ragL2_academicWriting, "ragL2_academic_writing", DATE)


def batch_flow_stage2():
    path_first_three_job = os.path.join(PATHS.folderPath_batchFile, f"first_three_methods-{DATE}")
    path_ragL2_causes_job = os.path.join(PATHS.folderPath_batchFile, f"ragL2_causes-{DATE}")
    path_ragL2_academicWriting_job = os.path.join(PATHS.folderPath_batchFile, f"ragL2_academic_writing-{DATE}")

    # today = datetime.now().strftime('%m%d-%H%M')
    path_first_three_result = f"first_three_methods-{DATE}"
    path_ragL2_causes_result = f"ragL2_causes-{DATE}"
    path_ragL2_academicWriting_result = f"ragL2_academic_writing-{DATE}"

    # Check batch jobs
    jobs_first_three = check_batch_jobs(path_first_three_job)
    jobs_ragL2_causes = check_batch_jobs(path_ragL2_causes_job)
    jobs_ragL2_academicWriting = check_batch_jobs(path_ragL2_academicWriting_job)

    # Parse batch results
    while not parse_batch_results(jobs_first_three, path_first_three_result):
        print("Waiting for batch jobs to complete...")
        sys.stdout.flush()
        time.sleep(600)
    while not parse_batch_results(jobs_ragL2_causes, path_ragL2_causes_result):
        print("Waiting for batch jobs to complete...")
        sys.stdout.flush()
        time.sleep(600)
    while not parse_batch_results(jobs_ragL2_academicWriting, path_ragL2_academicWriting_result):
        print("Waiting for batch jobs to complete...")
        sys.stdout.flush()
        time.sleep(600)

    print("End of batch processing.")


def batch_flow_stage3():
    # --------------  test -----------------
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # df_fce_sample = df_fce_sample.loc[0:1]  # Apply slicing

    # df_fce_sample = df_fce_sample.set_index('index')
    # datasets = {"fce_sp": df_fce_sample}

    # --------------  production -----------------
    df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_withCollocation)
    df_fce_sample = df_fce_sample.set_index('index')
    df_longman_sample = df_longman_sample.set_index('index')
    datasets = {"longmanSample": df_longman_sample, "fceSample": df_fce_sample}

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

                task_collective_part2.extend(tasks_part2)

    # Perform batch processing
    perform_batch_workflow(task_collective_part2, "part2", DATE)


def batch_flow_stage4():
    path_part2_job = os.path.join(PATHS.folderPath_batchFile, f"part2-{DATE}")
    # path_ragMix_job = os.path.join(PATHS.folderPath_batchFile, f"ragMix-{DATE}")

    path_part2_result = f"part2-{DATE}"

    # Check batch jobs
    jobs_part2 = check_batch_jobs(path_part2_job)

    # Parse batch results
    while not parse_batch_results(jobs_part2, path_part2_result):
        # # Check batch jobs
        # check_batch_jobs(path_part2_job)
        print("Waiting for batch jobs to complete...")
        sys.stdout.flush()
        time.sleep(600)

    print("End of batch processing.")


if __name__ == "__main__":
    DATE = ""
    if len(sys.argv) > 1:
        DATE = sys.argv[1]
    else:
        raise ValueError("Please provide the date for batch results as a command line argument.")
    # Stage 1: Create batch tasks and perform batch processing
    batch_flow_stage1()

    # Stage 2: Check batch jobs and parse results
    batch_flow_stage2()

    # Stage 3: Create tasks for RAG L2 explanation and perform batch processing
    batch_flow_stage3()

    # Stage 4: Check batch jobs and parse results for RAG L2 explanation
    batch_flow_stage4()
