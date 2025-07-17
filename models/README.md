# Models Module

This module contains the machine learning models and batch processing workflows for the WhyFix L2 learner error correction system. It implements various approaches for generating explanations and corrections for lexical errors made by second language learners.

## Overview

The models module provides core functionality for:

1. **Batch Processing**: Scalable processing using OpenAI's Batch API
2. **Multi-stage Workflows**: Complex pipelines combining different approaches
3. **Experiment Management**: Configuration and execution of large-scale experiments
4. **LLM Integration**: Language model setup and API management

## Directory Structure

```
models/
├── README.md                    # This file
├── __init__.py                 # Module initialization
├── experiment.yaml             # Experiment configuration
├── llm_setup.py               # Language model setup and configuration
├── batch_api.py               # OpenAI Batch API integration
└── methods_combine_step*.py   # Multi-step batch processing workflow
```

## Key Components

### Experiment Configuration (`experiment.yaml`)

The system supports comprehensive experiment settings for L2 error correction:

- **Input Types**:

  - `target`: Target sentence only
  - `preceding + target`: Context before + target sentence
  - `target + following`: Target sentence + context after
  - `preceding + target + following`: Full context
- **Model Types**:

  - **GPT models**: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1-nano`
  - **Reasoning models**: `o1-mini`, `o3-mini`
- **Roles**: `none`, `high_school_teacher`, `linguist`
- **Temperature**: `0.0` (deterministic), `0.7` (creative)
- **Output Length**: `30`, `50`, `80`, `no_limit`

### Batch Processing Workflow

Multi-step processing system for large-scale experiments:

#### Step 1 (`methods_combine_step1.py`)

- Submit initial batch jobs for dictionary, causes, and academic writing methods
- Coordinate parallel processing of multiple approaches

#### Step 2 (`methods_combine_step2.py`)

- Monitor and wait for first batch job completion
- Validate intermediate results

#### Step 3 (`methods_combine_step3.py`)

- Submit follow-up jobs for L2 explanations and mixed approaches
- Build on results from previous steps

#### Step 4 (`methods_combine_step4.py`)

- Final job monitoring and result collection
- Coordinate with postprocess module for result extraction

### LLM Setup (`llm_setup.py`)

Manages language model configuration and API integration:

- **API Key Management**: Secure handling of OpenAI credentials
- **Model Selection**: Dynamic model choice based on experiment parameters
- **Rate Limiting**: API call optimization and throttling
- **Error Handling**: Robust error recovery and retry mechanisms

### Batch API Integration (`batch_api.py`)

Specialized interface for OpenAI's Batch API:

- **Job Submission**: Large-scale batch job creation
- **Status Monitoring**: Real-time job progress tracking
- **Result Retrieval**: Efficient batch result downloading
- **Error Recovery**: Handling of failed or incomplete jobs

## Usage Examples

### Running Batch Workflows

```bash
# Complete multi-step processing pipeline
python -m models.methods_combine_step1 longman small 0622-1458
python -m models.methods_combine_step2 0622-1458
python -m models.methods_combine_step3 longman small 0622-1458
python -m models.methods_combine_step4 0622-1458
```

### Parameters

- `SAMPLE_TYPE`: Dataset type (e.g., "longman", "fce")
- `EMBEDDING_SIZE`: Vector embedding types (small, large)
- `DATE`: Experiment timestamp for tracking and organization

## Integration Points

- **Input**: Preprocessed data from `preprocess/` module
- **Processing**: Enhanced context from `process/` module
- **Output**: Results processed by `postprocess/` module
- **Configuration**: Experiment settings via `experiment.yaml`

## Performance Characteristics

- **Batch Processing**: Handles thousands of samples efficiently
- **Parallel Execution**: Multiple methods run concurrently
- **API Optimization**: Efficient use of OpenAI Batch API quotas
- **Scalability**: Designed for large-scale research experiments

## Output Formats

Results are saved in structured formats:

- **Batch API Responses**: JSONL format with metadata
- **Structured Explanations**: Organized by method and parameters
- **Timestamped Results**: Clear experiment tracking and versioning

## Error Handling

Robust error management for production environments:

- **API Failures**: Automatic retry with exponential backoff
- **Partial Results**: Recovery from incomplete batch jobs
- **Validation**: Data integrity checks at each step
- **Logging**: Comprehensive logging for debugging and monitoring

This module serves as the core engine for generating L2 learner error explanations through scalable, API-driven batch processing workflows.
