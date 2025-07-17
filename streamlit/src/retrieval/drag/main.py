import os
import sys
from tqdm import tqdm
import json
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from retrieval.drag.utils.files_io import (
    FilePathL2,
    load_retrieval_config,
    prepare_results_payload,
    save_payload_to_json
)
from retrieval.vectorstore_setup import (
    encode_pdf_to_vectorstore,
    get_vectorstore
)
from retrieval.drag.utils.data_types import (
    QueryResult,
    RAGExample,
    RAGConfig
)
from retrieval.drag.rag import DRAG, IterDRAG


TOPIC, MODEL_TYPE, CHUNKING_TYPE, EMBEDDING_TYPE, K, M, N = None, None, None, None, 50, 8, 5
# K: Number of documents to retrieve (k in paper)
# M: Number of demonstration examples (m in paper)
# N: Number of iterations (n in paper)
if len(sys.argv) > 7:
    TOPIC = sys.argv[1]
    MODEL_TYPE = sys.argv[2]
    CHUNKING_TYPE = int(sys.argv[3])
    EMBEDDING_TYPE = sys.argv[4]
    K = int(sys.argv[5])
    M = int(sys.argv[6])
    N = int(sys.argv[7])
else:
    raise ValueError("Please provide a topic, a model type, and a chunking type.")

PATHS = FilePathL2(TOPIC, EMBEDDING_TYPE)


def load_knowledge_documents(path_vectorstore, vectorstore, chunk_settings) -> List[Document]:
    """Load L2 knowledge documents"""
    files = os.listdir(PATHS.folderPath_all)
    for file in tqdm(files, desc="Loading documents"):
        path_file = os.path.join(PATHS.folderPath_all, file)
        vectorstore = encode_pdf_to_vectorstore(path_file, path_vectorstore, vectorstore, chunk_settings)


def load_examples(dataset_name: str) -> List[RAGExample]:
    """Load demonstration examples"""
    dataset = []
    with open(f'/home/nlplab/atwolin/thesis/data/L2-knowledge/{dataset_name}.json', 'r') as f:
        dataset = json.load(f)
    examples = []
    for item in dataset:
        example = RAGExample(
            documents=[],  # Will be filled during processing
            query=item['question'],
            answer=item['answer']
        )
        examples.append(example)
    return examples  # Limit for example


def main():
    # Load program information
    print("Initializing program settings")

    retrieve_method_doc = load_retrieval_config()

    model_settings = retrieve_method_doc["model"][MODEL_TYPE]['settings']
    chunk_settings = retrieve_method_doc["chunking"][str(CHUNKING_TYPE)]
    embedding_settings = retrieve_method_doc["embedding_model"][str(EMBEDDING_TYPE)]
    chunk_settings["length_function"] = len
    query = retrieve_method_doc["query_retrieval"]["rewrite"][TOPIC]["fourOneNano"]
    print(f"Topic: {TOPIC}\nModel Type: {MODEL_TYPE}\nQuery: {query}\nModel Settings: {model_settings}")

    # Initialize models and stores
    print(f"{'=' * 50}\nInitializing models and stores")
    embedding_model = OpenAIEmbeddings(**embedding_settings)

    path_vectorstore = os.path.join(PATHS.folderPath_l2_faiss, f"knowledge_{TOPIC}_{CHUNKING_TYPE}")
    document_store = get_vectorstore(
        f"knowledge_{TOPIC}_{CHUNKING_TYPE}_{EMBEDDING_TYPE}", path_vectorstore
    )

    # Load documents and index them
    print("Loading and indexing documents...")
    load_knowledge_documents(path_vectorstore, document_store, chunk_settings)
    num_documents_in_vectorsore = len(document_store.index_to_docstore_id)
    print(f"Number of documents in document_store: {num_documents_in_vectorsore}")

    # Load configuration
    config = RAGConfig(
        num_documents=K,    # k in paper
        num_shots=M,        # m in paper
        max_iterations=N,   # n in paper
        max_doc_length=CHUNKING_TYPE,
        max_context_length=1_000_000,
        model_settings=model_settings
    )

    # Initialize DRAG and IterDRAG
    drag_system = DRAG(document_store, embedding_model, config)
    iter_drag_system = IterDRAG(document_store, embedding_model, config)

    datasets = [f"qa_examples_{TOPIC}"]
    all_results = {}

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")
        examples = load_examples(dataset_name)

        # Process queries with both DRAG and IterDRAG
        drag_results = []
        iter_drag_results = []

        # Test with different context lengths
        context_lengths = [1000000]

        for length in context_lengths:
            print(f"\nTesting with context length: {length}")
            config.max_context_length = length

            # DRAG processing
            print("Running DRAG...")
            result = drag_system.process_query(query, examples)
            if result.effective_context_length <= length:
                drag_results.append(result)

            # IterDRAG processing
            print("Running IterDRAG...")
            result = iter_drag_system.process_query(query, examples)
            if result.effective_context_length <= length:
                iter_drag_results.append(result)

        # Store results for this dataset
        all_results[dataset_name] = {
            'drag': drag_results,
            'iter_drag': iter_drag_results
        }

    # Analyze and print results
    for dataset_name, results in all_results.items():
        print(f"\n{'*' * 50}\nResults for {dataset_name}:")

        # Calculate metrics for DRAG
        print("\nDRAG Performance:")
        drag_performance = calculate_performance_metrics(results['drag'])
        print_metrics(drag_performance)

        # Calculate metrics for IterDRAG
        print("\nIterDRAG Performance:")
        iter_drag_performance = calculate_performance_metrics(results['iter_drag'])
        print_metrics(iter_drag_performance)
    print(f"{'*' * 50}\n")

    # Store wrapped JSON results
    drag_payload = prepare_results_payload(
        topic=TOPIC,
        rag_config=config,
        results=drag_results,
        performance=drag_performance,
        results_key="drag_results",
        performance_key="drag_performance"
    )
    save_payload_to_json(drag_payload, PATHS.filePath_result_drag)
    # save_payload_to_json(drag_payload, f"./process/retrieval/output/result_{EMBEDDING_TYPE}_drag.json")

    iter_payload = prepare_results_payload(
        topic=TOPIC,
        rag_config=config,
        results=iter_drag_results,
        performance=iter_drag_performance,
        results_key="iter_drag_results",
        performance_key="iter_drag_performance"
    )
    save_payload_to_json(iter_payload, PATHS.filePath_result_iter_drag)
    # save_payload_to_json(drag_payload, f"./process/retrieval/output/result_{EMBEDDING_TYPE}_iter_drag.json")


def calculate_performance_metrics(results: List[QueryResult]) -> Dict:
    """Calculate performance metrics for a list of results"""
    performance = {
        'avg_context_length': sum(r.effective_context_length for r in results) / len(results),
        'avg_confidence': sum(r.confidence for r in results) / len(results),
        'num_results': len(results)
    }

    # Calculate additional metrics for IterDRAG results
    if any(hasattr(r, 'sub_queries') for r in results):
        avg_iterations = sum(len(r.sub_queries) for r in results) / len(results)
        performance['avg_iterations'] = avg_iterations

    return performance


def print_metrics(metrics: Dict):
    """Print metrics in a formatted way"""
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.2f}")


if __name__ == "__main__":
    main()
