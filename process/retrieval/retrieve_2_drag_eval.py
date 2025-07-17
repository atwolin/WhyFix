import json
import os
from tqdm import tqdm
from ..utils.metrics import RAGMetrics
from ..utils.data_types import (
    QueryResult,
    prepare_results_payload,
    save_payload_to_json
)
from ..utils.helper_functions import SupportMaterial

# Paths to results and ground‐truths
SM_CUASES = SupportMaterial("causes", "large")
SM_ACADEMIC_WRITING = SupportMaterial("academic_writing", "large")

GROUND_TRUTHS = {
    "causes": SM_CUASES.filePath_l2_ground_truth_causes,
    "academic_writing": SM_ACADEMIC_WRITING.filePath_l2_ground_truth_academic_writing
}
RESULT_FILES = {
    "drag_causes": SM_CUASES.filePath_result_drag,
    "drag_academic_writing": SM_ACADEMIC_WRITING.filePath_result_drag,
    "iter_drag_causes": SM_CUASES.filePath_result_iter_drag,
    "iter_drag_academic_writing": SM_ACADEMIC_WRITING.filePath_result_iter_drag
}
EVAL_FILES = {
    "drag_causes": SM_CUASES.filePath_eval_drag,
    "drag_academic_writing": SM_ACADEMIC_WRITING.filePath_eval_drag,
    "iter_drag_causes": SM_CUASES.filePath_eval_iter_drag,
    "iter_drag_academic_writing": SM_ACADEMIC_WRITING.filePath_eval_iter_drag
}


def load_ground_truth(topic: str) -> str:
    path = GROUND_TRUTHS.get(topic)
    with open(path, "r") as f:
        return f.read().strip()


def evaluate(experiment: str, topic: str):
    metrics = RAGMetrics()
    gt = load_ground_truth(topic)
    data = []

    with open(RESULT_FILES[experiment], "r") as f:
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

            save_payload_to_json(updated_payload, EVAL_FILES[experiment])
            # save_payload_to_json(updated_payload, "./process/retrieval/output/eval_test.json")

    return metrics


def main():
    topics = ["causes", "academic_writing"]

    # m0 = evaluate("/home/nlplab/atwolin/thesis/experiment/drag_result_causes.json", topics[0])
    # m0.print_metrics()
    # return

    # evaluate DRAG
    print("=== DRAG Evaluation ===")
    for topic in topics:
        drag_metrics = evaluate("drag_" + topic, topic)
        # drag_metrics.print_metrics()

    # evaluate IterDRAG
    print("=== IterDRAG Evaluation ===")
    for topic in topics:
        iter_drag_metrics = evaluate("iter_drag_" + topic, topic)
        # iter_drag_metrics.print_metrics()

    print("=== Evaluation Completed ===")


if __name__ == "__main__":
    main()
