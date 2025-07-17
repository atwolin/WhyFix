#!/usr/bin/env bash
set -euo pipefail

if [ $# -ge 3 ]; then
  LONGMAN_SAMPLE_TYPE=$1
  EMBEDDING_SIZE=$2
  DATE=$3
elif [ $# -eq 2 ]; then
  LONGMAN_SAMPLE_TYPE=$1
  EMBEDDING_SIZE=$2
  DATE=$(date +"%m%d-%H%M")
else
  echo "No arguments provided. Please provide the sample type (e.g., 'longman'), embedding size, and date."
  exit 1
fi

# ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=== STEP 1: Submit first three, causes & academic_writing tasks ==="
# python -m model.methods_combine_step1 "${LONGMAN_SAMPLE_TYPE}" "${EMBEDDING_SIZE}" "${DATE}"

echo "=== STEP 2: Wait for first three, causes & academic_writing jobs to finish ==="
# python -m model.methods_combine_step2 "${DATE}"

echo "=== STEP 3: Submit ragL2_explanation and ragMix tasks ==="
# python -m model.methods_combine_step3 "${LONGMAN_SAMPLE_TYPE}" "${EMBEDDING_SIZE}" "${DATE}"

echo "=== STEP 4: Wait for ragL2_explanation and ragMix jobs to finish ==="
python -m model.methods_combine_step4 "${DATE}"

echo "=== STEP 5: Extract & combine batch results ==="
python -m postprocess.extract_batch_result "${LONGMAN_SAMPLE_TYPE}" "${EMBEDDING_SIZE}" "${DATE}"

echo "All steps completed successfully."
