# Preprocess Module

This module handles the preprocessing and preparation of linguistic datasets for the WhyFix L2 learner error correction system. It processes multiple data sources including dictionaries, corpora, and error-annotated datasets to create standardized inputs for downstream processing.

## Overview

The preprocess module provides comprehensive data preparation tools for:

1. **Dictionary Processing**: Cambridge, Macmillan, and Academic Keyword List (AKL) data
2. **Corpus Processing**: Error-annotated datasets (FCE, EAAN) and linguistic corpora
3. **Error Type Analysis**: Classification and preprocessing of learner errors
4. **Data Standardization**: Consistent formatting for system integration
5. **Linguistic Annotation**: POS tagging, parsing, and error categorization

## Directory Structure

```
preprocess/
├── README.md                              # This file
├── __init__.py                           # Module initialization
├── preprocess_error_type.py             # Error type classification
├── preprocess_general.py                # General preprocessing functions
├── preprocess_fce.py                    # FCE dataset processing
├── preprocess_cmb.py                    # Cambridge dictionary processing
├── preprocess_longman.py                # Longman dictionary processing
├── preprocess_collocation.py            # Collocation extraction
├── preprocess_pos.py                    # Part-of-speech processing
├── preprocess_sentence.py               # Sentence-level processing
├── parser_akl.py                        # Academic Keyword List parser
├── parser_cmb.py                        # Cambridge dictionary parser
├── parser_cmb.sh                        # Cambridge parsing script
└── parser_cmb_to_full.py                # Cambridge full processing
```

## Key Components

### Dictionary Knowledge Processing

#### Cambridge Dictionary (`preprocess_cmb.py`, `parser_cmb.py`)
Comprehensive processing of Cambridge Dictionary data:

**Processing Features:**
- **CEFR Level Integration**: A1-C2 proficiency level mapping
- **Multi-definition Support**: Multiple meanings per word
- **Example Sentence Extraction**: Rich contextual examples
- **POS Tag Standardization**: Consistent part-of-speech labeling

**Shell Script Integration:**
```bash
# Automated Cambridge processing
./parser_cmb.sh
```

#### Longman Dictionary (`preprocess_longman.py`)
Processing Longman Dictionary of Common Errors:

**Specialized Features:**
- **Error-focused Processing**: Common learner error patterns
- **Contextual Examples**: Error correction demonstrations
- **Pattern Recognition**: Systematic error identification
- **Integration Ready**: Formatted for RAG system input

#### Academic Keyword List (`parser_akl.py`)
Academic vocabulary processing for educational contexts:

**Vocabulary Categories:**
- **Academic Domains**: Subject-specific terminology
- **Frequency Analysis**: Usage pattern identification
- **Educational Levels**: Appropriate complexity mapping
- **Cross-reference Integration**: Connection with other dictionaries

### Error-Annotated Dataset Processing

#### FCE Dataset (`preprocess_fce.py`)
Cambridge First Certificate in English dataset processing:

**Target Error Categories:**
- **R[^ACDQTP][JNVY]?**: Replacement errors
- **FF[^ACDQT][JNVY]?**: False friend errors
- **CL**: Collocation errors
- **ID**: Idiom errors
- **L**: Inappropriate register/label
- **SA**: American spelling errors

**Linguistic Annotation:**
- **POS Categories**: Comprehensive part-of-speech tagging
- **Error Context**: Surrounding sentence analysis
- **Correction Patterns**: Systematic error-correction mapping

#### Error Type Processing (`preprocess_error_type.py`)
Systematic error classification and analysis:

**Classification Features:**
- **Automatic Detection**: Pattern-based error identification
- **Category Assignment**: Systematic error type labeling
- **Context Preservation**: Surrounding text maintenance
- **Temporal Tracking**: Time-stamped processing for experiments

### Linguistic Processing Components

#### Collocation Processing (`preprocess_collocation.py`)
Advanced collocation extraction and analysis:

**Data Sources:**
- **CLC FCE Dataset**: Learner error patterns
- **CLANG8 Dataset**: Grammatical error corrections
- **Write & Improve**: Educational writing samples

**Processing Features:**
- **N-gram Extraction**: Multi-word pattern identification
- **Frequency Analysis**: Usage pattern quantification
- **Context Preservation**: Surrounding linguistic environment
- **Error Association**: Collocation-error relationship mapping

#### POS Processing (`preprocess_pos.py`)
Part-of-speech tagging and linguistic analysis:

**Capabilities:**
- **Universal POS Mapping**: Standardized tag sets
- **Language-specific Adaptation**: Tailored for L2 contexts
- **Error-context Analysis**: POS patterns in error environments
- **Integration Support**: Compatible with error classification systems

#### Sentence Processing (`preprocess_sentence.py`)
Sentence-level preprocessing and formatting:

**Processing Pipeline:**
- **Boundary Detection**: Accurate sentence segmentation
- **Context Window Extraction**: Preceding/following sentence handling
- **Format Standardization**: Consistent input formatting
- **Model Input Preparation**: Ready for downstream processing

## Data Format Standards

### Standardized Output Format

All preprocessing modules output data in a consistent format:

| Field | Description | Example |
|-------|-------------|---------|
| `learner_word` | Original learner word | "advices" |
| `editor_word` | Corrected word | "advice" |
| `learner_sentence` | Original sentence | "I need some advices." |
| `editor_sentence` | Corrected sentence | "I need some advice." |
| `formatted_sentence` | Model-ready format | "[MASK] need some advice." |
| `data_source` | Source dataset | "fce-dataset" |
| `error_type` | Classification | "countable-uncountable" |
| `context` | Surrounding sentences | {...} |

### Integration Format

Preprocessed data integrates seamlessly with:
- **Models Module**: Direct input for explanation generation
- **Process Module**: Enhanced with retrieval context
- **Evaluation Systems**: Standardized for metric computation

## Usage Examples

### Dictionary Processing

```bash
# Cambridge dictionary processing
python -m preprocess.parser_cmb

# Longman processing
python -m preprocess.preprocess_longman

# Academic keyword processing
python -m preprocess.parser_akl
```

### Dataset Processing

```bash
# FCE dataset preprocessing
python -m preprocess.preprocess_fce

# Error type classification
python -m preprocess.preprocess_error_type

# Collocation extraction
python -m preprocess.preprocess_collocation
```

### Pipeline Integration

```bash
# Complete preprocessing pipeline
python -m preprocess.preprocess_general DATASET_TYPE
```

## Performance Characteristics

### Processing Efficiency
- **Memory Optimization**: Efficient handling of large datasets
- **Streaming Processing**: Memory-efficient file handling
- **Parallel Processing**: Multi-threaded where applicable
- **Incremental Updates**: Support for dataset additions

### Quality Assurance
- **Data Validation**: Format and content verification
- **Error Detection**: Malformed data identification
- **Statistical Validation**: Distribution analysis
- **Consistency Checking**: Cross-dataset validation

## Integration Points

- **Input**: Raw linguistic datasets, dictionaries, and corpora
- **Processing**: Standardized data preparation and formatting
- **Output**: Model-ready datasets for `models/` and `process/` modules
- **Quality Control**: Validation frameworks for data integrity

## Dependencies

### Core Libraries
- **NLTK**: Natural language processing toolkit
- **spaCy**: Advanced NLP processing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical processing

### Specialized Tools
- **ERRANT**: Error annotation toolkit
- **Language Parsers**: Dictionary-specific processing
- **Corpus Tools**: Large dataset handling utilities

## Configuration Management

### Processing Parameters
- **Error Type Filters**: Configurable error categories
- **Data Quality Thresholds**: Minimum quality requirements
- **Output Formats**: Flexible formatting options
- **Integration Settings**: Downstream module compatibility

This module serves as the foundation for all WhyFix processing, ensuring high-quality, standardized linguistic data that enables effective L2 learner error correction and explanation generation.
