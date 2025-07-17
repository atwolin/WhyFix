import os
import json
from  dataclasses import dataclass, field, asdict
from  typing import List, Optional ,Dict, Any
import torch
from datetime import datetime
from langchain_core.documents import Document

# @dataclass
# class Document:
#     content: str
#     doc_id: str
#     socre: float


@dataclass
class QueryResult:
    """" this will store the results for both drag and iterDrag processing"""
    query: str
    documents: List[Document]
    answer: str
    confidence: float
    sub_queries: List[str] = field(default_factory=list)
    intermediate_answers: List[str] = field(default_factory=list)
    effective_context_length: int = 0

@dataclass
class RAGExample:
    """Stores demonstration examples for in-context learning"""
    documents: List[Document]
    query :str
    answer:str
    sub_queries: Optional[List[str]] = None
    intermediate_answers: Optional[List[str]] = None

@dataclass
class RAGConfig:
    """Configuration for RAG processing"""
    num_documents: int = 50  # k in paper
    num_shots: int = 8      # m in paper
    max_iterations: int = 5  # n in paper
    max_doc_length: int = 1024
    max_context_length: int = 1_000_000  # Gemini 1.5 Flash context window
    model_settings: Optional[Dict[str, str]] = None


def _json_default(o):
    # If it's a LangChain Document, turn into a dict
    if isinstance(o, Document) and hasattr(o, "to_dict"):
        return o.to_dict()
    # Fallback for any object with a __dict__
    if hasattr(o, "__dict__"):
        return o.__dict__
    # Last‐ditch fallback
    return str(o)


def prepare_results_payload(
    topic: str,
    rag_config,
    results: List[QueryResult],
    performance: Dict[str, Any],
    results_key: str,
    performance_key: str
) -> Dict[str, Any]:
    """
    Wrap results and performance under one outer key: TOPIC_YYYYMMDD_HHMMSS
    """
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    outer_key = f"{topic}_{timestamp}"
    topic_name = f"{topic}"
    if isinstance(rag_config, RAGConfig):
        rag_config = asdict(rag_config)
    return {
        outer_key: {
            "topic_name": topic_name,
            "rag_config": rag_config,
            results_key: [asdict(r) for r in results],
            performance_key: performance
        }
    }


def save_payload_to_json(payload: Dict[str, Any], filepath: str) -> None:
    """Serialize the wrapped payload into a JSON file."""
    """Serialize the wrapped payload into a JSON file (appending to a list)."""
    data = []
    # 1) Ensure parent dir exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 2) Create the file with an empty list if it's missing
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump([], f)

    # 3) Load current contents (safely)
    with open(filepath, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []

    # 4) Append and write back
    data.append(payload)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4, default=_json_default)
