"""
Retrieve batch jobs and parse results from methods_combine_step1.py
"""
import sys
import os
import time
from .llm_setup import (
    FilePath,
)
from .batch_api import (
    check_batch_jobs,
    parse_batch_results
)

PATHS = FilePath('_')
DATE = ""
if len(sys.argv) > 1:
    DATE = sys.argv[1]
else:
    raise ValueError("Please provide the date for batch jobs as a command line argument.")


if __name__ == "__main__":
    datasets = {"lg_fu": 'df_longman', "fce_fu": 'df_fce'}
    sentenceTypes = ["t", "tf"]
    outputLenTypes = ["fifty", "eighty"]
    for dataset in datasets.keys():
        # df_examples = datasets[dataset]
        for sentenceType in sentenceTypes:
            if dataset.startswith("l") and sentenceType == "tf":
                continue
            for outputLength in outputLenTypes:

                path_first_three_job = os.path.join(PATHS.folderPath_batchFile, f"first_three_methods_{dataset}_{sentenceType}_{outputLength}-{DATE}")
                path_ragL2_causes_job = os.path.join(PATHS.folderPath_batchFile, f"ragL2_causes_{dataset}_{sentenceType}_{outputLength}-{DATE}")
                path_ragL2_academicWriting_job = os.path.join(PATHS.folderPath_batchFile, f"ragL2_academic_writing_{dataset}_{sentenceType}_{outputLength}-{DATE}")

                # today = datetime.now().strftime('%m%d-%H%M')
                path_first_three_result = f"first_three_methods_{dataset}_{sentenceType}_{outputLength}-{DATE}"
                path_ragL2_causes_result = f"ragL2_causes_{dataset}_{sentenceType}_{outputLength}-{DATE}"
                path_ragL2_academicWriting_result = f"ragL2_academic_writing_{dataset}_{sentenceType}_{outputLength}-{DATE}"

                # Check batch jobs
                jobs_first_three = check_batch_jobs(path_first_three_job)
                jobs_ragL2_causes = check_batch_jobs(path_ragL2_causes_job)
                jobs_ragL2_academicWriting = check_batch_jobs(path_ragL2_academicWriting_job)

                # Parse batch results
                print(f"Parsing {dataset}_{sentenceType}_{outputLength}")
                while not parse_batch_results(jobs_first_three, path_first_three_result):
                    print("Waiting for batch jobs to complete...")
                    sys.stdout.flush()
                    time.sleep(60)
                while not parse_batch_results(jobs_ragL2_causes, path_ragL2_causes_result):
                    print("Waiting for batch jobs to complete...")
                    sys.stdout.flush()
                    time.sleep(60)
                while not parse_batch_results(jobs_ragL2_academicWriting, path_ragL2_academicWriting_result):
                    print("Waiting for batch jobs to complete...")
                    sys.stdout.flush()
                    time.sleep(60)

    print("End of batch processing.")
