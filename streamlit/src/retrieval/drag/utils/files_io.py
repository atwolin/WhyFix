import os
import json
import yaml
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import asdict
from langchain_core.documents import Document

from retrieval.drag.utils.data_types import QueryResult, RAGConfig


class FilePathL2():
    def __init__(self, topic, embeddingType):
        # L2 knowledge
        self.folderPath_l2_main = "/home/nlplab/atwolin/thesis/data/L2-knowledge"

        # Configuration file
        self.filePath_method_config = os.path.join("./utils/retrieval_config.yaml")

        # Vector store
        self.folderPath_l2_chroma = os.path.join(self.folderPath_l2_main, "vectorstore")
        self.folderPath_l2_faiss = os.path.join(self.folderPath_l2_main, "faiss_index")

        # ----- folders -----
        self.folderPath_all = os.path.join(self.folderPath_l2_main, "all")
        self.folderPath_result = os.path.join(self.folderPath_l2_main, "results")
        os.makedirs(self.folderPath_result, exist_ok=True)
        self.folderPath_eval = os.path.join(self.folderPath_l2_main, "evaluations")
        os.makedirs(self.folderPath_eval, exist_ok=True)

        # ----- final results -----
        self.filePath_causes = os.path.join(self.folderPath_result, "causes.txt")
        self.filePath_academic_writing = os.path.join(self.folderPath_result, "academic_writing.txt")

        # ---- DRAG ----
        self.filePath_result_drag = os.path.join(self.folderPath_result, f"result_drag_{embeddingType}_{topic}.json")
        self.filePath_result_iter_drag = os.path.join(self.folderPath_result, f"result_iter_drag_{embeddingType}_{topic}.json")

        self.filePath_eval_drag = os.path.join(self.folderPath_eval, f"eval_drag_{embeddingType}_{topic}.json")
        self.filePath_eval_iter_drag = os.path.join(self.folderPath_eval, f"eval_iter_drag_{embeddingType}_{topic}.json")


PATHS = FilePathL2("_", "_")


def load_retrieval_config():
    retriever_config = {}
    with open(PATHS.filePath_method_config, "r") as f:
        retriever_config = yaml.load(f, Loader=yaml.FullLoader)
    return retriever_config


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
    topic: str, rag_config, results: List[QueryResult], performance: Dict[str, Any],
    results_key: str, performance_key: str
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
