import sys
import os
import time
import pandas as pd
import json

from model.llm_setup import (
    FilePath
)
from model.batch_api import (
    perform_batch_workflow,
    check_batch_jobs,
    parse_batch_results,
)


PATHS = FilePath('_')
DATE = None
if len(sys.argv) > 1:
    DATE = sys.argv[1]
else:
    raise ValueError("Please provide the date as a command line argument.")


def perform_filter_batchjob():
    """
    Filter out sentences that contain grammatical errors.
    Lexical errors:
    •Free combinations: Word elements are used in their literal senses and can be freely substituted.
    •Restricted collocations: One word has a specialized or figurative meaning, limiting its possible partners.
    •Figurative idioms: The entire phrase has a metaphorical meaning, but a literal interpretation is also possible.
    •Pure idioms: The combination has a single meaning that cannot be derived from its individual words.

    Args:
        df (pd.DataFrame): The DataFrame containing sentences.

    Returns:
        pd.DataFrame: A DataFrame with sentences that do not contain grammatical errors.
    """
    errorTypes_prompt = (
        "You are an expert in linguistic error analysis. Your task is to classify an error into one of four specific lexical categories or determine it is not lexical.\n\n"
        "You will be given a sentence '{sentences}' where edits are shown in bracketed notation: "
        "the original text is in [- –] and the revised text is in {{+ +}}.\n\n"
        "## Answering Rules ##\n"
        "You must carefully evaluate the error in the original [-text-] against the definitions provided.\n"
        "1. If the error fits the 'Free combinations' category, your output must be exactly: free combinations\n"
        "2. If the error fits the 'Restricted collocations' category, your output must be exactly: restricted collocations\n"
        "3. If the error fits the 'Figurative idioms' category, your output must be exactly: figurative idioms\n"
        "4. If the error fits the 'Pure idioms' category, your output must be exactly: pure idioms\n"
        "5. If the error does NOT fit into any of the four lexical categories (meaning it is a grammatical error), your output must be exactly: no\n\n"
        "## Definitions ##\n\n"
        "### Lexical Errors ###\n"
        "An error is 'lexical' if it is one of the following types:\n"
        "1. **Free combinations**: The components are used in their literal senses and are freely substitutable. (e.g., 'carry a trumpet', 'on top of the table'). The error involves a word choice that is semantically incorrect but doesn't violate a fixed phrase structure.\n"
        "2. **Restricted collocations**: One component (like a verb or preposition) is used in a specialized sense that is only valid with a limited number of other words. (e.g., 'blow a fuse', 'under attack'). The error violates these word partnership rules.\n"
        "3. **Figurative idioms**: The combination has a metaphorical meaning, but a literal interpretation is still possible. (e.g., 'blow your own trumpet'). The error is in using an incorrect word within this semi-fixed phrase.\n"
        "4. **Pure idioms**: The combination is opaque and fixed, with a meaning that cannot be derived from its parts. (e.g., 'blow the gaff', 'under the weather'). The error is in the structure of this fixed, unchangeable phrase.\n\n"
        "### Grammatical Errors ###\n"
        "A 'grammatical' error involves incorrectness in the structure of a sentence. This includes mistakes in tense, subject-verb agreement, word order, articles, and prepositions (when they affect grammatical structure rather than word choice).\n\n"
        "\nYour entire output must be one of the five options listed in the Answering Rules. Do not provide any explanation."
    )
    output_schema = {
        "type": "object",
        "properties": {
            "error_type": {
                "type": "string",
                "description": "Describes the type of lexical error or answers 'no' if the sentence does not contain a lexical error."
            }
        },
        "required": [
            "error_type"
        ],
        "additionalProperties": False
    }

    def create_batch_response_format_tasks(df, datasetType):
        tasks = []
        for idx, row in df.iterrows():
            cid = f"{datasetType}_{idx}_error_types"
            body = {
                "model": 'gpt-4o',
                "temperature": 0.0,
                "input": errorTypes_prompt.format(sentences=row['formatted_sentence']),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": cid,
                        "schema": output_schema
                    }
                }
            }

            task = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/responses",
                "body": body
            }
            tasks.append(task)
        return tasks

    df_fce = pd.read_csv(PATHS.filePath_fce_dictInfo)
    df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo)

    df_fce = df_fce[df_fce['formatted_sentence'].notna()]
    df_longman = df_longman[df_longman['formatted_sentence'].notna()]

    # ---- test ----
    # tasks_test = create_batch_response_format_tasks(df_fce.loc[0:1], 'fce_test')
    # folderPath_batchTest = perform_batch_workflow(tasks_test, "filter_lexical_error_sentences", DATE)
    # print(f"Batch processing for test completed. Results saved in {folderPath_batchTest}")
    # return folderPath_batchTest

    tasks_fce = create_batch_response_format_tasks(df_fce, 'fce')
    tasks_longman = create_batch_response_format_tasks(df_longman, 'longman')
    tasks = tasks_fce + tasks_longman

    folderPath_batchObj = perform_batch_workflow(tasks, "filter_lexical_error_sentences", DATE)
    print(f"Batch processing completed. Results saved in {folderPath_batchObj}")
    return folderPath_batchObj


def get_filter_batchjob(folderPath_batchObj):
    """
    Filter out sentences that contain grammatical errors.
    """
    filename = f"filter_lexical_error_sentences-{DATE}"
    print("Retrieving batch jobs and parsing results for filtering grammatical error sentences...")

    jobs_result = check_batch_jobs(folderPath_batchObj)
    while not parse_batch_results(jobs_result, filename):
        print("Waiting for batch jobs to complete...")
        sys.stdout.flush()
        time.sleep(60)

    print("End of batch processing for filtering grammatical error sentences.")
    return os.path.join(PATHS.folderPath_batchResult, filename)


def filter_grammatical_error_sentences(filePath_batchResult):
    """
    Filter out sentences that contain grammatical errors from the batch result file.
    """
    batch_results = {}
    df_fce = pd.read_csv(PATHS.filePath_fce_dictInfo)
    df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo)
    df_filtered_fce = pd.DataFrame(columns=df_fce.columns)
    df_filtered_longman = pd.DataFrame(columns=df_longman.columns)
    with open(filePath_batchResult, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):  # Skip empty lines or comments
                continue
            obj = json.loads(line)
            custom_id = obj['custom_id']
            datasetType, idx, *_ = custom_id.split('_')
            error_type = None
            try:
                text_content = obj['response']['body']['output'][0]['content'][0]['text']
                if isinstance(text_content, str):
                    if not text_content.strip():
                        print(f"Warning: Empty text_content for custom_id: {custom_id}. Skipping.")
                        continue
                    try:
                        text_dict = json.loads(text_content)
                        error_type = text_dict['error_type']
                    except json.JSONDecodeError:
                        print(f"Warning: Invalid JSON in text_content for custom_id: {custom_id}. Skipping.")
                        continue
                else:
                    error_type = text_content['error_type']
                batch_results[(datasetType, int(idx))] = error_type
            except KeyError:
                print(f"Error processing custom_id: {custom_id}. Skipping this entry.")
                continue

            # Filter out sentences with grammatical errors
            if error_type.lower() == 'no':
                continue
            if datasetType == 'fce':
                df_row = df_fce.iloc[int(idx)].copy()
                df_row['error_type'] = error_type
                df_filtered_fce = pd.concat([df_filtered_fce, df_row.to_frame().T])
            else:
                df_row = df_longman.iloc[int(idx)].copy()
                df_row['error_type'] = error_type
                df_filtered_longman = pd.concat([df_filtered_longman, df_row.to_frame().T])

    df_filtered_fce.to_csv(PATHS.filePath_fce_dictInfo_filtered, index=False, encoding='utf-8')
    df_filtered_longman.to_csv(PATHS.filePath_longman_dictInfo_filtered, index=False, encoding='utf-8')
    print(f"Filtered sentences saved to {PATHS.filePath_fce_dictInfo_filtered} and {PATHS.filePath_longman_dictInfo_filtered}")


if __name__ == "__main__":
    folderPath_batchObj = perform_filter_batchjob()
    folderPath_batchObj = os.path.join(PATHS.folderPath_batchFile, f"filter_lexical_error_sentences-{DATE}")
    filePath_batchResult = get_filter_batchjob(folderPath_batchObj)

    filter_grammatical_error_sentences(os.path.join(PATHS.folderPath_batchResult, f"filter_lexical_error_sentences-{DATE}", 'batchResult.jsonl'))
    print("Grammatical error filtering completed.")
