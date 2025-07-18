# WhyFix: Correction Explanation System

WhyFix is an advanced system designed to provide explanations for lexical errors made by second language (L2) learners. The system leverages Retrieval-Augmented Generation (RAG) techniques and large language models to deliver contextually appropriate feedback for academic writing.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

## Overview

WhyFix addresses the challenge of providing meaningful explanations for lexical errors in L2 academic writing. The system combines:

- **Retrieval-Augmented Generation (RAG)** for context-aware explanations
- **Multi-stage processing pipelines** for comprehensive error analysis
- **Batch processing capabilities** for large-scale evaluation
- **Interactive web interface** for real-time feedback
- **Comprehensive evaluation metrics** for system assessment

### Key Features

- **Error Detection & Correction**: Identifies and corrects lexical errors in academic writing
- **Contextual Explanations**: Provides detailed explanations for why corrections are necessary
- **Multi-source RAG**: Integrates multiple knowledge sources for enhanced accuracy
- **Scalable Processing**: Supports batch processing for large datasets
- **Interactive Interface**: Streamlit-based web application for user interaction

## System Architecture

```
WhyFix/
├── models/                     # ML models and batch processing
├── process/                    # Core processing pipeline
│   ├── input/                 # Input data handling
│   ├── rag/                   # RAG system implementations
│   ├── retrieval/             # Vector retrieval mechanisms
│   └── utils/                 # Processing utilities
├── preprocess/                # Data preprocessing
├── postprocess/              # Result analysis and metrics
├── streamlit/                # Web interface
├── data/                     # Training and evaluation data
└── scripts/                  # Automation scripts
```

## Installation

### Prerequisites

- Python 3.10+
- Git
- Virtual environment (recommended)

### Setup

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd WhyFix
   ```
2. **Create and activate virtual environment**:

   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```
3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configurations
   ```

## Usage

### Quick Start

1. **Run the sampling workflow**:

   ```bash
   ./run_sampling.sh A small
   ```
2. **Execute batch processing**:

   ```bash
   ./run_batch_jobs.sh A small $(date +"%m%d-%H%M")
   ```
3. **Launch web interface**:

   ```bash
   cd streamlit/src/app
   streamlit run main.py
   ```

### Command Line Interface

#### RAG-based Retrieval

```bash
python -m process.retrieval.retrieve_3_collocation gpt true A small
```

#### Batch Processing

```bash
./run_sampling.sh A small
```

## Data Sources

### Knowledge Base Sources

1. **Academic Writing Guidelines**

   - English Academic Writing for Students and Researchers
   - Cultural Issues in Academic Writing
   - Features of Academic Writing
2. **Dictionary Resources**

   - Cambridge Dictionary (definitions, CEFR levels, examples)
   - Macmillan English Dictionary
   - Academic Keyword List (AKL)
3. **Collocation Resources**

   - Collocation databases

### Evaluation Datasets

1. **CLC FCE Dataset**

   - 4,853 exam scripts from Cambridge FCE
   - Multiple error types: Replace, False Friend, Collocation, etc.
   - Demographic and proficiency information

2. **Longman Dictionary of Common Errors**

   - 1,342 sentence pairs


## Workflow

### 1. Data Preprocessing

- Text normalization and tokenization
- Error annotation parsing
- Context extraction and formatting

### 2. Vector Store Construction

- Embedding generation for knowledge sources
- Vector database indexing
- Similarity threshold optimization

### 3. Retrieval-Augmented Generation

- Query formulation from error context
- Multi-source retrieval (academic writing, collocations, dictionary)
- Context ranking and selection

### 4. Error Correction Pipeline

- Error detection and classification
- Candidate correction generation
- Explanation synthesis

### 5. Evaluation and Metrics

- Automatic evaluation metrics
- Human evaluation protocols
- Performance analysis

## Configuration

### Experiment Configuration (`models/experiment.yaml`)

```yaml
model_settings:
  embedding_model: "text-embedding-3-large"
  llm_model: "gpt-4.1-nano"
  temperature: 0.0
  max_tokens: 40000

retrieval_settings:
  top_k: 5

data_settings:
  sample_type: "longman", "fce"
  embedding_size: text-embedding-3-small
```

### Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=your_openai_api_key

# Elastic Search
ES_USER=your_es_user
ES_PASSWORD=your_es_password
ES_ENDPOINT="localhost:9200"
ES_URL=your_es_url
ES_API_KEY=your_es_api_key
ES_INDEX_NAME="collocation"
```

## Evaluation

### Automatic Metrics

- **BLEU Score**: Translation quality assessment
- **ROUGE Score**: Summary quality evaluation
- **BERTScore**: Semantic similarity measurement
- **Exact Match**: Precision of corrections

<!-- ### Human Evaluation

- **Explanation Quality**: Clarity and usefulness of explanations
- **Correction Accuracy**: Appropriateness of suggested corrections
- **Contextual Relevance**: Suitability for academic writing context -->

### Running Evaluation

```bash
python -m postprocess.automatic_metrics --input results/ --output evaluation/
```

## Directory Structure

```
WhyFix/
├── README.md                   # This file
├── .gitignore                 # Git ignore patterns
├── requirements.txt           # Python dependencies
├── run_batch_jobs.sh          # Batch processing script
├── run_sampling.sh            # Sampling workflow script
├── models/                    # ML models and batch processing
│   ├── __init__.py
│   ├── README.md
│   ├── batch_api.py
│   ├── experiment.yaml
│   ├── llm_setup.py
│   └── methods_combine_step*.py
├── process/                   # Core processing pipeline
│   ├── __init__.py
│   ├── README.md
│   ├── input/
│   ├── rag/
│   ├── retrieval/
│   └── utils/
├── preprocess/               # Data preprocessing
├── postprocess/             # Result analysis
├── streamlit/              # Web interface
└── data/                   # Data storage
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is part of academic research. Please refer to the institution's guidelines for usage and distribution.

<!-- ## Citation

If you use WhyFix in your research, please cite:

```bibtex
@misc{whyfix2025,
  title={WhyFix: L2 Learner Error Correction System with Retrieval-Augmented Generation},
  author={[Author Name]},
  year={2025},
  institution={[Institution Name]}
}
``` -->

<!-- ## Contact

For questions and support, please contact: [Your Email] -->

---

**Last Updated**: July 17, 2025
