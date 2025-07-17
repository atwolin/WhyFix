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
                path_part2_job = os.path.join(PATHS.folderPath_batchFile, f"part2_{dataset}_{sentenceType}_{outputLength}-{DATE}")
                # path_ragMix_job = os.path.join(PATHS.folderPath_batchFile, f"ragMix-{DATE}")

                path_part2_result = f"part2_{dataset}_{sentenceType}_{outputLength}-{DATE}"

                # Check batch jobs
                jobs_part2 = check_batch_jobs(path_part2_job)

                # Parse batch results
                print(f"Parsing {dataset}_{sentenceType}_{outputLength}")
                while not parse_batch_results(jobs_part2, path_part2_result):
                    # # Check batch jobs
                    # check_batch_jobs(path_part2_job)
                    print("Waiting for batch jobs to complete...")
                    sys.stdout.flush()
                    time.sleep(60)

    print("End of batch processing.")
