import os
import json
import pandas as pd

from retrieval.drag.utils.files_io import (
    FilePathL2,
)


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


def transform_eval_data(data, rag_type):
    """
    Transforms the evaluation data into a DataFrame suitable for analysis.
    Supports two types of performance keys:
     1) perf["metric"]["mean"]
     2) perf["avg_metric"]
    """
    records = []
    for payload in data:
        for exp_name, exp in payload.items():
            results = exp.get("drag_results") or exp.get("iter_drag_results", [])
            records.append({
                "rag_type":        rag_type,
                "topic":           exp.get("topic_name", exp_name.split("_")[0]),
                "rag_topic":       ' '.join(exp.get("topic_name", exp_name.split("_")[0]).split('_')).capitalize() + " - " + rag_type,
                "chunk_size":      exp["rag_config"]["max_doc_length"],
                "num_documents":   exp["rag_config"]["num_documents"],
                "num_shots":       exp["rag_config"]["num_shots"],
                "max_iterations":  exp["rag_config"]["max_iterations"],
                "answer":          results[0]["answer"],
            })

    return pd.DataFrame(records)


def load_df(df, eval_files):
    for experiment_name in eval_files.keys():
        data = []
        rag_type = "IterDRAG" if "iter" in experiment_name else "DRAG"
        with open(eval_files[experiment_name], "r") as f:
            data = json.load(f)
        df = pd.concat([df, transform_eval_data(data, rag_type)], ignore_index=True)
    return df


def main():
    df = pd.DataFrame()
    df_small = load_df(df, RESULT_FILES_SMALL)
    df_large = load_df(df, RESULT_FILES_LARGE)

    # Small
    df_small_answer_causes = df_small[
        (df_small['rag_type'] == 'IterDRAG') &
        (df_small['topic'] == 'causes') &
        (df_small['chunk_size'] == 1000) &
        (df_small['num_shots'] == 8) &
        (df_small['max_iterations'] == 5) &
        (df_small['num_documents'] == 700)
    ]
    answer_small_causes = df_small_answer_causes['answer'].values[0]

    df_small_answer_academic_writing = df_small[
        (df_small['rag_type'] == 'DRAG') &
        (df_small['topic'] == 'academic_writing') &
        (df_small['chunk_size'] == 500) &
        (df_small['num_shots'] == 8) &
        (df_small['max_iterations'] == 1) &
        (df_small['num_documents'] == 50)
    ]
    answer_small_academic_writing = df_small_answer_academic_writing['answer'].values[0]

    with open("./process/retrieval/output/answers_small.json", "w") as f:
        json.dump({
            "causes": answer_small_causes,
            "academic_writing": answer_small_academic_writing
        }, f, indent=4)

    with open("./process/retrieval/output/answers.txt", "w") as f:
        f.write(f"Causes Answer:\n{answer_small_causes}\n\n")
        f.write(f"Academic Writing Answer:\n{answer_small_academic_writing}\n")

    # Large
    df_large_answer_causes = df_large[
        (df_large['rag_type'] == 'IterDRAG') &
        (df_large['topic'] == 'causes') &
        (df_large['chunk_size'] == 1000) &
        (df_large['num_shots'] == 8) &
        (df_large['max_iterations'] == 5) &
        (df_large['num_documents'] == 700)
    ]
    answer_large_causes = df_large_answer_causes['answer'].values[0]

    df_large_answer_academic_writing = df_large[
        (df_large['rag_type'] == 'DRAG') &
        (df_large['topic'] == 'academic_writing') &
        (df_large['chunk_size'] == 500) &
        (df_large['num_shots'] == 8) &
        (df_large['max_iterations'] == 1) &
        (df_large['num_documents'] == 50)
    ]
    answer_large_academic_writing = df_large_answer_academic_writing['answer'].values[0]
    with open("./process/retrieval/output/answers_large.json", "w") as f:
        json.dump({
            "causes": answer_large_causes,
            "academic_writing": answer_large_academic_writing
        }, f, indent=4)
    with open("./process/retrieval/output/answers.txt", "a") as f:
        f.write(f"\n\nCauses Answer (Large):\n{answer_large_causes}\n\n")
        f.write(f"Academic Writing Answer (Large):\n{answer_large_academic_writing}\n")


if __name__ == "__main__":
    main()
