import os
import json
import itertools
import pandas as pd

from model.llm_setup import (
    client,
    FilePath,
    load_experiment,
    get_placeholders,
    get_support_material
)

PATHS = FilePath('_')


########################################
#      format functions
########################################
def format_sentence(sentence, token, punctuation):
    for idx, word in enumerate(sentence.split()):
        if word.startswith('[-'):
            learnerWord = '<span style="color:#ee3f4d;">' + token[token.find('[-') + 2:token.find('-]')] + '</span>' + punctuation
            editorWord = '<span style="color:#55bb8a;">' + token[token.find('{+') + 2:token.find('+}')] + '</span>' + punctuation
            print(f"learnerWord: {learnerWord}")
            print(f"editorWord: {editorWord}")

        learnerSent = sentence.replace(f'[-{learnerWord}-]{{+{editorWord}+}}', learnerWord)
        editorSent = sentence.replace(f'[-{learnerWord}-]{{+{editorWord}+}}', editorWord)

    row = {
        "learner_word": learnerWord,
        "editor_word": editorWord,
        "learner_sentence": learnerSent,
        "editor_sentence": editorSent,
        "formatted_sentence": sentence,
    }
    return pd.DataFrame([row])


def get_input_sentences(sentenceType, row):
    """Extract the sentences based on the sentenceType value"""
    sentences = ""
    if sentenceType == "t":
        sentences = row["formatted_sentence"]
    elif sentenceType == "pt":
        sentences = row["preceding_sentence"] + ' ' + row["formatted_sentence"] if pd.notna(row["preceding_sentence"]) else row["learner_sentence"]
    elif sentenceType == "tf":
        sentences = row["formatted_sentence"] + ' ' + row["following_sentence"] if pd.notna(row["following_sentence"]) else row["learner_sentence"]
    elif sentenceType == "ptf":
        if pd.notna(row["preceding_sentence"]):
            sentences = row["preceding_sentence"] + ' ' + row["formatted_sentence"]
        if pd.notna(row["following_sentence"]):
            sentences += ' ' + row["following_sentence"]
    return sentences


def get_format_prompt(experimentType, prompt, role, formatSentences, learnerInfo, editorInfo, outputLen):
    query = ""
    learnerWord = learnerInfo["learner_words"]
    editorWord = editorInfo["editor_word"]

    if experimentType == "baseline":
        query = prompt.format(
            role=role,
            sentence=formatSentences,
            learnerWord=learnerWord,
            editorWord=editorWord,
            output_len=outputLen
        )
    elif experimentType == "simpleRag":
        query = prompt.format(
            role=role,
            sentence=formatSentences,
            learnerWord=learnerWord,
            learnerWord_pos=learnerInfo["pos"],
            learnerWord_level=learnerInfo["level"],
            learnerWord_definition=learnerInfo["definition"],
            learnerWord_examples=learnerInfo["examples"],
            learnerWord_in_akl=learnerInfo["in_akl"],
            editorWord=editorWord,
            editorWord_pos=editorInfo["pos"],
            editorWord_level=editorInfo["level"],
            editorWord_definition=editorInfo["definition"],
            editorWord_examples=editorInfo["examples"],
            editorWord_in_akl=editorInfo["in_akl"],
            output_len=outputLen
        )
    return query


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


########################################
#      create batch functions
########################################
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


def create_folder(parentFolder, folderName):
    folderPath = os.path.join(parentFolder, folderName)
    os.makedirs(folderPath, exist_ok=True)
    print(f"Folder created: {folderPath}")
    return folderPath


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


def store_batch_jsonl(parentFolder, filename, data):
    path = os.path.join(parentFolder, filename)
    with open(path, 'w') as f:
        for row in data:
            f.write(json.dumps(row) + '\n')
    print(f"File created: {path}")


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


########################################
#      read & check batch functions
########################################
def read_batch_jsonl(filePath):
    with open(filePath, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def get_openai_objects(data):
    openaiFileObjects = []
    for obj in data:
        openaiFileObj = client.files.retrieve(obj['id'])
        openaiFileObjects.append(openaiFileObj)
    return openaiFileObjects


def get_batch_job_objects(data):
    batchJobObjects = []
    for obj in data:
        batchJobObj = client.batches.retrieve(obj['id'])
        batchJobObjects.append(batchJobObj)
    return batchJobObjects


def check_batch_job_status(batchJobObjects):
    for batchJobObj in batchJobObjects:
        batchJobStatus = client.batches.retrieve(batchJobObj.id)
        print(f"Status: {batchJobStatus.status}, output file id: {batchJobStatus.output_file_id}, error file id: {batchJobStatus.error_file_id}, request_counts: {batchJobStatus.request_counts}")
        print('=' * 50)


def get_batch_job_output(batchJobObjects):
    batchOutputIds = []
    errorBatchOutputIds = []
    # for batchJob in batchJobs:
    for batchJob in batchJobObjects:
        batchJobReturn = client.batches.retrieve(batchJob.id)
        if batchJobReturn.status == "completed":
            if batchJobReturn.output_file_id:
                batchOutputIds.append(batchJobReturn.output_file_id)
                print(f"Batch job {batchJob.id} completed. output_file_id: {batchJobReturn.output_file_id}")
            if batchJobReturn.error_file_id:
                errorBatchOutputIds.append(batchJobReturn.error_file_id)
                print(f"Batch job {batchJob.id} completed. error_file_id: {batchJobReturn.error_file_id}")
    return batchOutputIds, errorBatchOutputIds


def get_batch_job_output_content(batchOutputIds):
    batchOutputContents = []
    for batchOutputId in batchOutputIds:
        batchOutputContent = client.files.content(batchOutputId).content
        batchOutputContents.append(batchOutputContent)
    return batchOutputContents


def check_batch_jobs(folderPath_tasks):
    openaiFileObjects = read_batch_jsonl(os.path.join(folderPath_tasks, "openaiFileObjects.jsonl"))
    openaiFileObjects = get_openai_objects(openaiFileObjects)
    batchJobObjects = read_batch_jsonl(os.path.join(folderPath_tasks, "batchJobObjects.jsonl"))
    batchJobObjects = get_batch_job_objects(batchJobObjects)

    # Check batch job status and get results
    check_batch_job_status(batchJobObjects)
    return batchJobObjects


def parse_batch_results(batchJobObjects, folername):
    successBatchOutputIds, errorBatchOutputIds = get_batch_job_output(batchJobObjects)
    batchOutputContents = get_batch_job_output_content(successBatchOutputIds)
    errorBatchOutputContents = get_batch_job_output_content(errorBatchOutputIds)

    if (len(errorBatchOutputContents) == 0):
        # Save results
        folderPath_results = create_folder(PATHS.folderPath_batchResult, folername)
        with open(os.path.join(folderPath_results, "batchResult.jsonl"), "a") as f:
            for content in batchOutputContents:
                f.write(content.decode("utf-8"))
                f.write("\n")
                print("Batch processing completed successfully, results saved.")
                return True
    else:
        print("Errors occurred during batch processing:")
        for error in errorBatchOutputContents:
            print(error)
            return False


########################################
#      for experiment functions
########################################
def fce_create_batch_response_files(dataset, df, sampleNum, experimentType):
    """Create batch response files."""
    experiment = load_experiment()
    prompt = experiment["prompt"][experimentType]["content"]
    print(f"Using DATASET NAME={dataset} and DATA={df.count()}")
    print(f"{'=' * 50}\nNumber of sampling: {sampleNum}\nPrompt template:\n{prompt}{'=' * 50}\n")
    get_placeholders(prompt)

    all_tasks = {}
    for idx, row in df.iterrows():
        if idx == sampleNum:
            break
        learnerInfo, editorInfo = get_support_material(row)

        for modelClass in experiment["model_type"].keys():
            if modelClass == "gpt":
                for modelType_key in experiment["model_type"][modelClass].keys():
                    tasks = []
                    if modelType_key not in all_tasks.keys():
                        all_tasks[modelType_key] = []
                    for inputSent, role_key, temp_key, outputLen_key in itertools.product(
                        experiment["input_sent"].keys(),
                        experiment["role"].keys(),
                        experiment["temp"].keys(),
                        experiment["output_len"].keys()
                    ):
                        sentences = get_input_sentences(inputSent, row)
                        query = get_format_prompt(
                            experimentType=experimentType,
                            prompt=prompt,
                            role=experiment['role'][role_key],
                            formatSentences=sentences,
                            learnerInfo=learnerInfo,
                            editorInfo=editorInfo,
                            outputLen=experiment['output_len'][outputLen_key]
                        )

                        body = get_one_batch_response_format_task_format_body(
                            modelClass=modelClass,
                            modelType=experiment["model_type"][modelClass][modelType_key],
                            query=query,
                            outputName=f"json_result_{experimentType}_{idx}_{modelType_key}_{inputSent}_{temp_key}_{role_key}_{outputLen_key}",
                            schema=experiment["prompt"][experimentType]["output_format"],
                            temperature=experiment["temp"][temp_key]
                        )

                        task = get_one_batch_response_format_task(
                            f"{dataset}_{experimentType}_{idx}_{modelType_key}_{temp_key}_{inputSent}_{role_key}_{outputLen_key}",
                            body)
                        tasks.append(task)
                    all_tasks[modelType_key].extend(tasks)

            elif modelClass == "reasoning":
                for modelType_key in experiment["model_type"][modelClass]["model"].keys():
                    tasks = []
                    if modelType_key not in all_tasks.keys():
                        all_tasks[modelType_key] = []
                    # for reasoningEffort, inputSent, role_key, temp_key, outputLen_key in itertools.product(
                    #     experiment["model_type"][modelClass]["effort"].values(),
                    #     experiment["input_sent"].keys(),
                    #     experiment["role"].keys(),
                    #     experiment["temp"].keys(),
                    #     experiment["output_len"].keys()
                    # ):
                    for reasoningEffort, inputSent, role_key, outputLen_key in itertools.product(
                        experiment["model_type"][modelClass]["effort"].values(),
                        experiment["input_sent"].keys(),
                        experiment["role"].keys(),
                        # experiment["temp"].keys(),  // Cannot use temp for reasoning
                        experiment["output_len"].keys()
                    ):
                        sentences = get_input_sentences(inputSent, row)
                        query = get_format_prompt(
                            experimentType=experimentType,
                            prompt=prompt,
                            role=experiment['role'][role_key],
                            formatSentences=sentences,
                            learnerInfo=learnerInfo,
                            editorInfo=editorInfo,
                            outputLen=experiment['output_len'][outputLen_key]
                        )
                        body = get_one_batch_response_format_task_format_body(
                            modelClass=modelClass,
                            modelType=experiment["model_type"][modelClass]["model"][modelType_key],
                            query=query,
                            reasoningEffort=reasoningEffort,
                            # outputName=f"json_result__{experimentType}_{idx}_{modelType_key}_{inputSent}_{temp_key}_{role_key}_{outputLen_key}",
                            # temperature=experiment["temp"][temp_key]
                            outputName=f"json_result__{experimentType}_{idx}_{modelType_key}_{inputSent}_{role_key}_{outputLen_key}",
                            schema=experiment["prompt"][experimentType]["output_format"]
                        )
                        # task = get_one_batch_response_format_task(
                        #     f"{dataset}_{experimentType}_{idx}_{modelType_key}_{temp_key}_{reasoningEffort}_{inputSent}_{role_key}_{outputLen_key}",
                        #     body)
                        task = get_one_batch_response_format_task(
                            f"{dataset}_{experimentType}_{idx}_{modelType_key}_{reasoningEffort}_{inputSent}_{role_key}_{outputLen_key}",
                            body)

                        tasks.append(task)
                    all_tasks[modelType_key].extend(tasks)
    return all_tasks


def longman_create_batch_response_files(dataset, df, sampleNum, experimentType, learnerInfoDF, editorInfoDF):
    """Create batch response files."""
    experiment = load_experiment()
    prompt = experiment["prompt"][experimentType]["content"]
    print(f"Using DATASET NAME={dataset} and DATA={df.count()}")
    print(f"{'=' * 50}\nNumber of sampling: {sampleNum}\nPrompt template:\n{prompt}{'=' * 50}\n")

    all_tasks = {}
    for idx, row in df.iterrows():
        if idx == sampleNum:
            break
        # learnerInfo, editorInfo = get_support_material(row)
        learnerInfo, editorInfo = learnerInfoDF.iloc[idx], editorInfoDF.iloc[idx]
        sentences = row["formatted_sentence"]

        for modelClass in experiment["model_type"].keys():
            if modelClass == "gpt":
                for modelType_key in experiment["model_type"][modelClass].keys():
                    tasks = []
                    if modelType_key not in all_tasks.keys():
                        all_tasks[modelType_key] = []
                    for role_key, temp_key, outputLen_key in itertools.product(
                        experiment["role"].keys(),
                        experiment["temp"].keys(),
                        experiment["output_len"].keys()
                    ):
                        if temp_key != "zero" or role_key != "linguist":
                            continue

                        query = get_format_prompt(
                            experimentType=experimentType,
                            prompt=prompt,
                            role=experiment['role'][role_key],
                            formatSentences=sentences,
                            learnerInfo=learnerInfo,
                            editorInfo=editorInfo,
                            outputLen=experiment['output_len'][outputLen_key]
                        )
                        body = get_one_batch_response_format_task_format_body(
                            modelClass=modelClass,
                            modelType=experiment["model_type"][modelClass][modelType_key],
                            query=query,
                            outputName=f"json_result_{experimentType}_{idx}_{modelType_key}_t_{temp_key}_{role_key}_{outputLen_key}",
                            schema=experiment["prompt"][experimentType]["output_format"],
                            temperature=experiment["temp"][temp_key]
                        )

                        task = get_one_batch_response_format_task(
                            f"{dataset}_{experimentType}_{idx}_{modelType_key}_{temp_key}_t_{role_key}_{outputLen_key}",
                            body)
                        tasks.append(task)
                    all_tasks[modelType_key].extend(tasks)

            # elif modelClass == "reasoning":
            #     for modelType_key in experiment["model_type"][modelClass]["model"].keys():
            #         tasks = []
            #         if modelType_key not in all_tasks.keys():
            #             all_tasks[modelType_key] = []
            #         # for reasoningEffort, role_key, temp_key, outputLen_key in itertools.product(
            #         #     experiment["model_type"][modelClass]["effort"].values(),
            #         #     experiment["role"].keys(),
            #         #     experiment["temp"].keys(),
            #         #     experiment["output_len"].keys()
            #         # ):
            #         for reasoningEffort, role_key, outputLen_key in itertools.product(
            #             experiment["model_type"][modelClass]["effort"].values(),
            #             experiment["role"].keys(),
            #             # experiment["temp"].keys(),  // Cannot use temp for reasoning
            #             experiment["output_len"].keys()
            #         ):

            #             query = get_format_prompt(
            #                 experimentType=experimentType,
            #                 prompt=prompt,
            #                 role=experiment['role'][role_key],
            #                 formatSentences=sentences,
            #                 learnerInfo=learnerInfo,
            #                 editorInfo=editorInfo,
            #                 outputLen=experiment['output_len'][outputLen_key]
            #             )
            #             body = get_one_batch_response_format_task_format_body(
            #                 modelClass=modelClass,
            #                 modelType=experiment["model_type"][modelClass]["model"][modelType_key],
            #                 query=query,
            #                 reasoningEffort=reasoningEffort,
            #                 # outputName=f"json_result__{experimentType}_{idx}_{modelType_key}_t_{temp_key}_{role_key}_{outputLen_key}",
            #                 # temperature=experiment["temp"][temp_key]
            #                 outputName=f"json_result__{experimentType}_{idx}_{modelType_key}_t_{role_key}_{outputLen_key}",
            #                 schema=experiment["prompt"][experimentType]["output_format"]
            #             )
            #             # task = get_one_batch_response_format_task(
            #             #     f"{dataset}_{experimentType}_{idx}_{modelType_key}_{temp_key}_{reasoningEffort}_t_{role_key}_{outputLen_key}",
            #             #     body)
            #             task = get_one_batch_response_format_task(
            #                 f"{dataset}_{experimentType}_{idx}_{modelType_key}_{reasoningEffort}_t_{role_key}_{outputLen_key}",
            #                 body)

            #             tasks.append(task)
            #         all_tasks[modelType_key].extend(tasks)
    return all_tasks
