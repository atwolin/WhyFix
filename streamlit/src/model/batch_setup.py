import os
# -*- coding: utf-8 -*-
import json


from utils.files_io import (
    FilePath,
)

from model.model_setup import (
    client,
)


PATHS = FilePath()


########################################
#      format batch jobs
########################################
def get_one_batch_response_format_task_format_body(query, outputName, schema, modelClass='gpt', modelType='gpt-4.1-nano', temperature=0.0, reasoningEffort=""):
    body = ""
    # print(type(schema['additionalProperties']), schema['additionalProperties'])
    # schema['additionalProperties'] = False
    # print(type(schema['additionalProperties']), schema['additionalProperties'])
    if modelClass == "gpt":
        body = {
            "model": modelType,
            "temperature": temperature,
            "input": query,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": outputName,
                    "schema": schema
                }
            }
        }
    elif modelClass == "reasoning":
        body = {
            "model": modelType,
            # "temperature": temperature,
            "input": query,
            "reasoning": {
                "effort": reasoningEffort,
                # "summary": "detailed"
            },
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": outputName,
                    "schema": schema
                }
            }
        }
    # print(f"Created batch response body, output name: {outputName}")
    return body


def get_one_batch_response_format_task(customID, body):
    """Create batch response format."""
    task = {
        "custom_id": customID,
        "method": "POST",
        "url": "/v1/responses",
        "body": body
    }
    return task


def create_batch_response_format_tasks(df, prompts, datasetType, methodType, sentenceType, outputLength, schema):
    tasks = []
    cnt = 0
    for idx, row in df.iterrows():
        body = get_one_batch_response_format_task_format_body(
            # query=prompts[idx],
            query=prompts[cnt],
            outputName=f"json_result_{datasetType}_{idx}_{methodType}_fON_zero_{sentenceType}_l_{outputLength}",
            schema=schema,
        )

        task = get_one_batch_response_format_task(
            f"{datasetType}_{idx}_{methodType}_fON_zero_{sentenceType}_l_{outputLength}",
            body
        )
        tasks.append(task)
        cnt += 1
    return tasks


########################################
#      batch workflow
########################################
def create_folder(parentFolder, folderName):
    folderPath = os.path.join(parentFolder, folderName)
    os.makedirs(folderPath, exist_ok=True)
    print(f"Folder created: {folderPath}")
    return folderPath


def store_batch_jsonl(parentFolder, filename, data):
    path = os.path.join(parentFolder, filename)
    with open(path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    print(f"File created: {path}")


def upload_batch_tasks(parentFolder, filename):
    openaiFileObjects = []
    for taskFile in os.listdir(parentFolder):
        if "tasks_batch" not in taskFile:
            continue
        print(f"Uploading {taskFile} to batch file")
        filePath = os.path.join(parentFolder, taskFile)
        batchFile = client.files.create(
            file=open(filePath, 'rb'),
            purpose="batch"
        )
        print("Batch file uploaded")
        openaiFileObjects.append(batchFile.to_dict())
    store_batch_jsonl(parentFolder, filename, openaiFileObjects)
    return openaiFileObjects


def get_openai_objects(data):
    openaiFileObjects = []
    for obj in data:
        openaiFileObj = client.files.retrieve(obj['id'])
        openaiFileObjects.append(openaiFileObj)
    return openaiFileObjects


def create_batch_jobs(parentFolder, openaiFileObjects):
    batchJobObjects = []
    for fileObj in openaiFileObjects:
        print(f"Creating batch job for {fileObj.filename}")
        batchJob = client.batches.create(
            input_file_id=fileObj.id,
            endpoint='/v1/responses',
            completion_window='24h',
        )
        batchJobObjects.append(batchJob.to_dict())
    store_batch_jsonl(parentFolder, "batchJobObjects.jsonl", batchJobObjects)
    return batchJobObjects


def perform_batch_workflow(tasks, outputName, date):
    # Create batch jobs and upload tasks
    # today = datetime.now().strftime('%m%d-%H%M')
    folderPath_tasks = create_folder(PATHS.folderPath_batchFile, f"{outputName}-{date}")

    store_batch_jsonl(folderPath_tasks, f"tasks_batch_{outputName}.jsonl", tasks)
    openaiFileObjs = upload_batch_tasks(folderPath_tasks, "openaiFileObjects.jsonl")
    openaiFileObjs = get_openai_objects(openaiFileObjs)

    # batchJobs = create_batch_jobs(folderPath_tasks, openaiFileObjs)
    create_batch_jobs(folderPath_tasks, openaiFileObjs)
    return folderPath_tasks
