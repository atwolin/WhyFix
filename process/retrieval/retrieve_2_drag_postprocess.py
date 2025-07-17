import os
import sys
import json
import pandas as pd

from ..utils.helper_functions import SupportMaterial

# Paths to results and ground‐truths
SM_SMALL_CAUSES = SupportMaterial("causes", "small")
SM_SMALL_ACADEMIC_WRITING = SupportMaterial("academic_writing", "small")
SM_LARGE_CAUSES = SupportMaterial("causes", "large")
SM_LARGE_ACADEMIC_WRITING = SupportMaterial("academic_writing", "large")

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
                "rag_topic":       ' '.join(exp.get("topic_name", exp_name.split("_")[0]).split('_')).capitalize() + " - " + \
                                   rag_type,
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

    # Small, causes selected
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

    with open("./process/retrieval/output/answers_small.txt", "w") as f:
        f.write(f"Causes Answer:\n{answer_small_causes}\n\n")
        f.write(f"Academic Writing Answer:\n{answer_small_academic_writing}\n")

    df_small_answer_academic_writing_large = df_small[
        (df_small['rag_type'] == 'IterDRAG') &
        (df_small['topic'] == 'academic_writing') &
        (df_small['chunk_size'] == 1000) &
        (df_small['num_shots'] == 8) &
        (df_small['max_iterations'] == 5) &
        (df_small['num_documents'] == 700)
    ]
    with open("./process/retrieval/output/answers_small_extreme_academic_writing.txt", "w") as f:
        f.write(f"Academic Writing Answer (Large):\n{df_small_answer_academic_writing_large['answer'].values[0]}\n")

    # Large, academic_writing selected
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
        (df_large['num_shots'] == 0) &
        (df_large['max_iterations'] == 1) &
        (df_large['num_documents'] == 50)
    ]
    answer_large_academic_writing = df_large_answer_academic_writing['answer'].values[0]
    with open("./process/retrieval/output/answers_large.json", "w") as f:
        json.dump({
            "causes": answer_large_causes,
            "academic_writing": answer_large_academic_writing
        }, f, indent=4)
    with open("./process/retrieval/output/answers_large.txt", "w") as f:
        f.write(f"\n\nCauses Answer (Large):\n{answer_large_causes}\n\n")
        f.write(f"Academic Writing Answer (Large):\n{answer_large_academic_writing}\n")

    df_large_answer_academic_writing_large = df_large[
        (df_large['rag_type'] == 'IterDRAG') &
        (df_large['topic'] == 'academic_writing') &
        (df_large['chunk_size'] == 1000) &
        (df_large['num_shots'] == 8) &
        (df_large['max_iterations'] == 5) &
        (df_large['num_documents'] == 700)
    ]
    with open("./process/retrieval/output/answers_large_extreme_academic_writing.txt", "w") as f:
        f.write(f"Academic Writing Answer (Large):\n{df_large_answer_academic_writing_large['answer'].values[0]}\n")

    print("Answers saved to output files.")


if __name__ == "__main__":
    main()
