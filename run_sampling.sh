#!/usr/bin/env bash
set -euo pipefail

if [ $# -ge 2 ]; then
  LONGMAN_SAMPLE_TYPE=$1
  EMBEDDING_SIZE=$2
else
  echo "No arguments provided. Please provide the sample type (e.g., 'longman') and embedding size."
  exit 1
fi


# Step 1: Build collocation vector store and perform retrieval
echo "=== STEP 1: Build collocation vector store and perform retrieval ==="
python -m process.retrieval.retrieve_3_collocation gpt true "${LONGMAN_SAMPLE_TYPE}" "${EMBEDDING_SIZE}"

# Step 2: Run batch job submission and subsequent workflow
echo "=== STEP 2: Run batch job submission and subsequent workflow ==="
./run_batch_jobs.sh "${LONGMAN_SAMPLE_TYPE}" "${EMBEDDING_SIZE}"
