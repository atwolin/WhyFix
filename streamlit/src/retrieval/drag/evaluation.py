import os
import json
from tqdm import tqdm

from retrieval.drag.utils.files_io import (
    FilePathL2,
    prepare_results_payload,
    save_payload_to_json
)
from retrieval.drag.utils.data_types import QueryResult
from retrieval.drag.utils.metrics import RAGMetrics


# Paths to results and ground‐truths
SM_SMALL_CAUSES = FilePathL2("causes", "small")
SM_SMALL_ACADEMIC_WRITING = FilePathL2("academic_writing", "small")
SM_LARGE_CAUSES = FilePathL2("causes", "large")
SM_LARGE_ACADEMIC_WRITING = FilePathL2("academic_writing", "large")

GROUND_TRUTHS = {
    "causes": os.path.join(SM_SMALL_CAUSES.folderPath_main, "ground_truth_causes.txt"),
    "academic_writing": os.path.join(SM_SMALL_ACADEMIC_WRITING.folderPath_main, "ground_truth_academic_writing.txt")
}

RESULT_FILES_SMALL = {
    "drag_causes": SM_SMALL_CAUSES.filePath_eval_drag,
    "drag_academic_writing": SM_SMALL_ACADEMIC_WRITING.filePath_eval_drag,
    "iter_drag_causes": SM_SMALL_CAUSES.filePath_eval_iter_drag,
    "iter_drag_academic_writing": SM_SMALL_ACADEMIC_WRITING.filePath_eval_iter_drag
}
RESULT_FILES_LARGE = {
    "drag_causes": SM_LARGE_CAUSES.filePath_eval_drag,
    "drag_academic_writing": SM_LARGE_ACADEMIC_WRITING.filePath_eval_drag,
    "iter_drag_causes": SM_LARGE_CAUSES.filePath_eval_iter_drag,
    "iter_drag_academic_writing": SM_LARGE_ACADEMIC_WRITING.filePath_eval_iter_drag
}


def load_ground_truth(topic: str) -> str:
    path = GROUND_TRUTHS.get(topic)
    with open(path, "r") as f:
        return f.read().strip()


def evaluate(input_file, experiment: str, topic: str):
    metrics = RAGMetrics()
    gt = load_ground_truth(topic)
    data = []

    with open(input_file, "r") as f:
        data = json.load(f)  # list of experiment dicts
    # with open(experiment, "r") as f:
    #     data = json.load(f)

    for payload in tqdm(data, desc=f"Evaluating {experiment} on topic {topic}"):
        for exp_name in payload.keys():
            experiment_result = payload[exp_name]

            result_key = "drag_results" if "drag_results" in experiment_result.keys() else "iter_drag_results"
            performance_key = "drag_performance" if "drag_performance" in experiment_result.keys() else "iter_drag_performance"

            query_result_flat = experiment_result[result_key][0]
            query_result = QueryResult(
                query=query_result_flat["query"],
                documents=[],
                answer=query_result_flat["answer"],
                confidence=float(query_result_flat["confidence"]),
                sub_queries=query_result_flat.get("sub_queries", []),
                intermediate_answers=query_result_flat.get("intermediate_answers", []),
                effective_context_length=int(query_result_flat.get("effective_context_length", 0))
            )
            metrics.update(query_result_flat['answer'], gt, query_result)

            experiment_result[performance_key].update(metrics.get_metrics())

            updated_payload = prepare_results_payload(
                topic=topic,
                rag_config=experiment_result.get("rag_config", {}),
                results=[query_result],
                performance=experiment_result[performance_key],
                results_key=result_key,
                performance_key=performance_key
            )

            save_payload_to_json(updated_payload, input_file)
            # save_payload_to_json(updated_payload, "./process/retrieval/output/eval_test.json")

    return metrics


def main():
    topics = ["causes", "academic_writing"]
    # evaluate DRAG
    print("=== DRAG Evaluation ===")
    for topic in topics:
        experiment_name = "drag_" + topic
        evaluate(RESULT_FILES_SMALL[experiment_name], experiment_name, topic)
        evaluate(RESULT_FILES_LARGE[experiment_name], experiment_name, topic)

    # evaluate IterDRAG
    print("=== IterDRAG Evaluation ===")
    for topic in topics:
        experiment_name = "iter_drag_" + topic
        evaluate(RESULT_FILES_SMALL[experiment_name], experiment_name, topic)
        evaluate(RESULT_FILES_LARGE[experiment_name], experiment_name, topic)

    print("=== Evaluation Completed ===")


if __name__ == "__main__":
    main()
