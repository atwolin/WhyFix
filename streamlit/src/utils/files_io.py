import os
import yaml


class FilePath():
    def __init__(self):
        # /home/ikm-admin/Work/atwolin/thesis/thesis-system/src
        # /home/nlplab/atwolin/thesis/thesis-LinguaSynth-Explainer/src
        self.folderPath_thesis = os.path.abspath(__file__).rsplit('/', 2)[0]
        # Configuration file
        self.filePath_llm_method_config = os.path.join(self.folderPath_thesis, "utils/llm_config.yaml")
        self.filePath_retrieval_method_config = os.path.join(self.folderPath_thesis, "utils/retrieval_config.yaml")

        # CLC FEC dataset
        self.folderPath_fce = os.path.join(self.folderPath_thesis, "data/fce")
        # self.folderPath_fce_raw_data = os.path.join(self.folderPath_thesis, "data/fce-released-dataset/dataset")
        self.filePath_fce_essays = os.path.join(self.folderPath_fce, "fce_essays.csv")
        self.filePath_fce_sentences = os.path.join(self.folderPath_fce, "fce_sentences.csv")
        self.filePath_fce_sentences_withIndex = os.path.join(self.folderPath_fce, "fce_sentences_with_index.csv")
        # --- information files ---
        self.filePath_fce_dictInfo = os.path.join(self.folderPath_fce, "fce_sentences_dictionary.csv")
        # self.filePath_fce_allInfo = os.path.join(self.folderPath_fce, "fce_sentences_information.csv")  # dictionary + L2 knowledge
        # --- sample files ---
        self.filePath_fce_sample = os.path.join(self.folderPath_fce, "fce_sample.csv")
        self.filePath_fce_sample_withCollocation = os.path.join(self.folderPath_fce, "fce_sample_with_collocation.json")

        # Longman Dictionary of Common Errors
        self.folderPath_longman = os.path.join(self.folderPath_thesis, "data/longman")
        self.filePath_longman_errant_m2 = os.path.join(self.folderPath_longman, "m2_with_problem_word_copy.txt")
        self.filePath_longman_sentences = os.path.join(self.folderPath_longman, "longman_sentences.csv")
        self.filePath_longman_sentences_withIndex = os.path.join(self.folderPath_longman, "longman_sentences_with_index.csv")
        # --- information files ---
        self.filePath_longman_dictInfo = os.path.join(self.folderPath_longman, "longman_sentences_dictionary.csv")
        # self.filePath_longman_allInfo = os.path.join(self.folderPath_longman, "longman_sentences_information.csv")
        # --- sample files ---
        self.filePath_longman_sample = os.path.join(self.folderPath_longman, "longman_sample.csv")
        self.filePath_longman_sample_withCollocation = os.path.join(self.folderPath_longman, "longman_sample_with_collocation.json")

        # Runtime data
        self.folderPath_runtime = os.path.join(self.folderPath_thesis, "data/runtime")
        os.makedirs(self.folderPath_runtime, exist_ok=True)
        self.filePath_runtime_m2 = os.path.join(self.folderPath_runtime, "m2_with_problem_word.txt")
        self.filePath_runtime_sentences = os.path.join(self.folderPath_runtime, "runtime_sentences.csv")
        self.filePath_runtime_sentences_withIndex = os.path.join(self.folderPath_runtime, "runtime_sentences_with_index.csv")
        # --- information files ---
        self.filePath_runtime_dictInfo = os.path.join(self.folderPath_runtime, "runtime_sentences_dictionary.csv")
        self.filePath_runtime_withCollocation = os.path.join(self.folderPath_runtime, "runtime_sentences_with_collocation.json")
        self.filePath_runtime_allInfo = os.path.join(self.folderPath_runtime, "runtime_response.json")

        # Dictionary
        self.folderPath_dictionary = os.path.join(self.folderPath_thesis, "data/dictionary")
        self.filePath_dictionary_expample_sentences = os.path.join(self.folderPath_dictionary, "combined.v2.txt")
        self.filePath_dictionary_cambridge = os.path.join(self.folderPath_dictionary, "cambridge_parse.words.v2.csv")

        # L2 knowledge
        self.folderPath_l2_main = os.path.join(self.folderPath_thesis, "data/l2-knowledge")
        self.folderPath_result = os.path.join(self.folderPath_l2_main, "results")
        self.filePath_causes = os.path.join(self.folderPath_result, "causes_txt")
        self.filePath_academic_writing = os.path.join(self.folderPath_result, "academic_writing.txt")

        # Collocations
        self.folderPath_collocation_main = os.path.join(self.folderPath_thesis, "data/collocations")

        self.folderPath_collocation_faiss = os.path.join(self.folderPath_collocation_main, "faiss_index")

        self.filePath_collocation_txt_raw = os.path.join(self.folderPath_collocation_main, "collocations_0518.txt")
        # self.filePath_collocation_csv_raw = os.path.join(self.folderPath_collocation_main, "data_0516.csv")
        # self.filePath_collocation_csv_collection = os.path.join(self.folderPath_collocation_main, "collocation_collection.csv")
        self.filePath_collocation_full_csv = os.path.join(self.folderPath_collocation_main, "collocation_full.csv")
        self.filePath_collocation_full_txt = os.path.join(self.folderPath_collocation_main, "collocation_full.txt")
        self.filePath_collocation_simplified_csv = os.path.join(self.folderPath_collocation_main, "collocation_simplified.csv")
        self.filePath_collocation_simplified_txt = os.path.join(self.folderPath_collocation_main, "collocation_simplified.txt")

        #  Unit test for prompts
        # self.folderPath_longman_performanceTest = os.path.join(self.folderPath_longman, "performance-test")

        # Batch API
        self.folderPath_batch = os.path.join(self.folderPath_thesis, "data/batch")
        self.folderPath_batchFile = os.path.join(self.folderPath_batch, "batchFile")
        self.folderPath_batchResult = os.path.join(self.folderPath_batch, "batchResult")
        os.makedirs(self.folderPath_batch, exist_ok=True)
        os.makedirs(self.folderPath_batchFile, exist_ok=True)
        os.makedirs(self.folderPath_batchResult, exist_ok=True)

        # Structured data
        self.folderPath_structured_data = os.path.join(self.folderPath_thesis, "data/results/structured_data")


PATHS = FilePath()


# def load_experiment():
def load_method_config(config_type: str):
    methods = {}
    if config_type == "llm":
        with open(PATHS.filePath_llm_method_config, "r") as f:
            methods = yaml.load(f, Loader=yaml.FullLoader)
    elif config_type == "retrieval":
        with open(PATHS.filePath_retrieval_method_config, "r") as f:
            methods = yaml.load(f, Loader=yaml.FullLoader)
    return methods


def to_serializable(obj):
    if hasattr(obj, "__dict__"):
        # Recursively convert __dict__ values
        return {k: to_serializable(v) for k, v in obj.__dict__.items()}
    elif hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif hasattr(obj, "dict"):  # fallback for Pydantic v1
        return obj.dict()
    elif isinstance(obj, list):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj
