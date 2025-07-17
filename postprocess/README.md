# Postprocess Module

This module handles the post-processing, evaluation, and analysis of results from WhyFix L2 learner error correction experiments. It extracts batch processing results, computes evaluation metrics, and generates comprehensive analysis reports.

## Overview

The postprocess module provides essential tools for:

1. **Batch Result Extraction**: Processing OpenAI Batch API responses
2. **Automatic Evaluation**: Computing linguistic and semantic metrics
3. **Statistical Analysis**: Generating performance statistics and comparisons
4. **Data Quality Assurance**: Handling errors and data validation
5. **Result Structuring**: Organizing outputs for further analysis

## Directory Structure

```
postprocess/
├── README.md                      # This file
├── __init__.py                   # Module initialization
├── extract_batch_result.py      # Main batch result extraction
├── automatic_metrics.py         # Evaluation metrics computation
├── evaluation_data_extract.py   # Data extraction for evaluation
├── evaluation_statistics.py     # Statistical analysis
└── evaluation_table3.py         # Result table generation
```

## Key Components

### Batch Result Processing

#### Extract Batch Results (`extract_batch_result.py`)

Primary script for processing OpenAI Batch API outputs:

**Core Functionality:**

- Reads batch result files from multiple experiment stages
- Handles JSON parsing errors and data corruption
- Extracts structured explanations and metadata
- Combines results from different methodologies
- Outputs structured JSONL files for analysis

**Usage Pattern:**

```python
# Called from batch processing pipeline
python -m postprocess.extract_batch_result SAMPLE_TYPE EMBEDDING_SIZE DATE
```

**Error Handling:**
The system robustly handles common batch processing issues:

- Unterminated JSON strings
- Delimiter parsing errors
- Malformed response payloads
- Missing or corrupted data fields

### Evaluation Metrics

#### Automatic Metrics (`automatic_metrics.py`)

Implements comprehensive evaluation metrics for L2 error explanations:

**Supported Metrics:**

- **BLEU Scores**: N-gram precision for explanation quality
- **ROUGE Scores**: Recall-oriented evaluation
- **Semantic Similarity**: Embedding-based content similarity
- **Lexical Diversity**: Vocabulary richness measures
- **Coherence Scores**: Text coherence evaluation
- **Factual Accuracy**: Content verification metrics

#### Evaluation Statistics (`evaluation_statistics.py`)

Generates statistical summaries and comparisons:

**Statistical Analysis:**

- Performance comparisons across methods
- Statistical significance testing
- Confidence interval calculations
- Distribution analysis across datasets
- Method effectiveness evaluation

### Data Processing Pipeline

#### Dataset Processing Flow

The module processes multiple dataset configurations:

```
Processing Flow:
├── Longman dataset (lg_fu)
│   ├── Target sentences (t)
│   ├── Output lengths: fifty, eighty
│   └── Multiple methods: BL, Dict, Collo2, L2
├── FCE dataset (fce_fu)
│   ├── Target sentences (t)
│   ├── Output lengths: fifty, eighty
│   └── Same method combinations
└── Combined analysis and comparison
```

#### Result Structure

Processed results are organized as:

```
/data/results/structured_data/
├── structured_api_data_longman_t_fifty_TIMESTAMP.jsonl
├── structured_api_data_longman_t_eighty_TIMESTAMP.jsonl
├── structured_api_data_fce_t_fifty_TIMESTAMP.jsonl
└── structured_api_data_fce_t_eighty_TIMESTAMP.jsonl
```

### Result Analysis

#### Table Generation (`evaluation_table3.py`)

Generates comprehensive analysis tables:

**Table Types:**

- **Performance Metrics**: Core evaluation scores by method
- **Comparative Analysis**: Side-by-side method comparisons
- **Statistical Summaries**: Significance testing results
- **Error Analysis**: Breakdown of processing issues

#### Data Extraction (`evaluation_data_extract.py`)

Specialized data extraction for evaluation:

**Extraction Features:**

- **Method Attribution**: Clear tracking of generation methods
- **Parameter Preservation**: Experimental settings maintenance
- **Quality Scores**: Computed evaluation metrics integration
- **Metadata Handling**: Complete experimental context

## Usage Examples

### Processing Pipeline

```bash
# Extract and process batch results
python -m postprocess.extract_batch_result longman small 0622-1458

# Generate evaluation metrics
python -m postprocess.automatic_metrics

# Create statistical summaries
python -m postprocess.evaluation_statistics

# Generate analysis tables
python -m postprocess.evaluation_table3
```

### Integration with Batch Processing

```python
# Called automatically from models module
INFO: Starting processing for date: 0622-1458
Processing dataset: lg_fu, sentenceType: t, outputLength: fifty
INFO: Total records extracted from batch results: 6710
結構化結果已成功儲存到 structured_api_data_longman_t_fifty_0622-1458.jsonl
```

## Performance Monitoring

### Processing Statistics

Typical processing yields for large-scale experiments:

- **Record Extraction**: 6000+ records per experiment
- **Error Recovery**: Robust handling of corrupted data
- **Quality Validation**: Comprehensive data integrity checks

### Error Recovery System

Advanced error handling ensures:

- **Partial Recovery**: Salvage data from corrupted batches
- **Detailed Logging**: Comprehensive error tracking
- **Graceful Degradation**: Continued processing with missing data
- **Quality Assurance**: Validation of output completeness

## Output Formats

### Structured Data

- **JSONL Format**: Line-delimited JSON for efficient processing
- **Metadata Preservation**: Complete experimental parameters
- **Method Attribution**: Clear tracking of generation approaches
- **Quality Metrics**: Integrated evaluation scores

### Analysis Reports

- **Statistical Summaries**: Method performance across datasets
- **Comparative Tables**: Effectiveness evaluation matrices
- **Error Analysis**: Processing failure identification
- **Quality Reports**: Data integrity and completeness metrics

## Integration Points

- **Input**: Batch API results from `models/` experiments
- **Dependencies**: Evaluation frameworks and metric libraries
- **Output**: Structured data for research analysis and publication
- **Quality Control**: Validation against `preprocess/` module standards

## Quality Assurance

### Data Validation

- **Format Consistency**: Standardized output structures
- **Completeness Checking**: Missing data identification
- **Error Pattern Analysis**: Systematic failure detection
- **Statistical Validation**: Distribution and outlier analysis

### Performance Optimization

- **Memory Efficiency**: Streaming processing for large files
- **Parallel Processing**: Multi-threaded data extraction
- **Caching**: Intelligent result caching for repeated operations
- **Resource Management**: Efficient CPU and memory utilization

This module serves as the critical quality assurance and analysis stage, transforming raw model outputs into reliable, analyzable data for L2 learner error correction research.
