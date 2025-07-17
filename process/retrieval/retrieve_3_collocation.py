import os
import sys
import regex as re
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
import spacy

# from langchain_core.documents import Document
# from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import (
    # BM25Retriever,
    ElasticSearchBM25Retriever,
)
from langchain.retrievers.ensemble import EnsembleRetriever
# from langchain_elasticsearch import (
#     DenseVectorStrategy,
# )
# from langchain_elasticsearch.vectorstores import ElasticsearchStore
from elasticsearch import Elasticsearch
from pydantic import Field, BaseModel
from ..utils.helper_functions import (
    SupportMaterial,
    load_yaml,
    get_vectorstore,
)
from model.llm_setup import (
    FilePath,
    # lemmatize_word_list
)

RETRIEVAL_CHAIN, RELEVANCE_CHAIN = None, None
MODEL_TYPE = None
EMBEDDING_TYPE = None
IS_FUSION = True
TOP_K = 5
# DATA = []
if len(sys.argv) > 4:
    MODEL_TYPE = sys.argv[1]
    IS_FUSION = sys.argv[2].lower() == 'true'
    LONGMAN_SAMPLE_TYPE = sys.argv[3]
    EMBEDDING_TYPE = sys.argv[4]
else:
    raise ValueError("Please provide a model type as an argument.")
print(f"Using model type: {MODEL_TYPE}, Fusion: {IS_FUSION}, Longman sample type: {LONGMAN_SAMPLE_TYPE}, Embedding type: {EMBEDDING_TYPE}")


load_dotenv(override=True)

SM = SupportMaterial('_', '_')
PATHS = FilePath(EMBEDDING_TYPE)
RETRIEVAL_DOC = load_yaml()
NLP = spacy.load("en_core_web_sm")


# def load_data():
#     filepath = SM.filePath_collocation_txt_raw
#     with open(filepath, 'r', encoding='utf-8') as file:
#         for line in file:
#             if line.strip():
#                 DATA.append(line.strip())


def create_or_get_retrivers(embedding_settings, top_k: int = 5):
    # Vector-based retrieval setup
    if embedding_settings == 'small':
        vectorstore_name = "simplified_collocation_collection"
    else:
        vectorstore_name = f"simplified_collocation_collection_{embedding_settings}"
    vectorstore_path = os.path.join(SM.folderPath_collocation_faiss, vectorstore_name)
    collocation_store = get_vectorstore(vectorstore_name, vectorstore_path, embeddingSettings=RETRIEVAL_DOC['embedding_model'][embedding_settings])
    faiss_retriever = collocation_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

    print(f"Number of documents in collocation vector store: {len(collocation_store.index_to_docstore_id)}")

    # Keyword-based retrieval setup
    elasticsearch_url = os.getenv("ES_URL")
    index_name = os.getenv("ES_INDEX_NAME", "collocation_collection")
    client_elastic = Elasticsearch(
        elasticsearch_url,
        api_key=os.getenv("ES_API_KEY"),
    )
    bm25_retriever = ElasticSearchBM25Retriever(
        client=client_elastic,
        index_name=index_name,
    )
    return faiss_retriever, bm25_retriever


# def create_exact_match_retriever(sentences):
#     words = re.findall(r'\[-(.*?)-\]', sentences)
#     lemmatized_words = lemmatize_word_list(words)
#     exact_matches = []
#     for line in DATA:
#         entry_word = line.split(',')[0].strip()
#         if entry_word in lemmatized_words:
#             exact_matches.append(line.strip())
#     print(f"Found {len(exact_matches)} exact matches for words: {lemmatized_words}")
#     return exact_matches


def get_full_collocation_info(simplified_collocations):
    """
    Matches simplified collocation information with full collocation data.

    Args:
        simplified_collocations (list): A list of strings, where each string
                                        contains the first four fields of a collocation
                                        (e.g., "write,VERB,write activity,mention activity").
    Returns:
        list: A list of strings, where each string is the full collocation
              information if a match is found based on the first four fields.
    """
    collocation_full_info = []
    full_data = []
    with open(SM.filePath_collocation_full_txt, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                full_data.append(line.strip())
    # Iterate through the simplified collocations and match with full data
    for simplified_line in simplified_collocations:
        for full_line in full_data:
            # Extract the first four fields from the full line
            full_parts = full_line.split('|')[:4]
            full_prefix = '|'.join(full_parts)
            if simplified_line == full_prefix:
                collocation_full_info.append(full_line.strip())

    return collocation_full_info


def process_other_category(row):
    error_component = row['error_component']

    # Extract changed component from 'component_change_freq_details'
    # e.g., (big-great) 131 -> great
    change_match = re.search(r'\((.*?)-(.*?)\)', row['component_change_freq_details'])
    changed_component = change_match.group(2) if change_match else None

    # Extract category and pivot word from 'collocation_pivot_and_category_details'
    # e.g., pleasure < Happiness and Sadness -> pivot: pleasure, category: Happiness and Sadness
    pivot_category_match = re.search(r'(.*?) < (.*)', row['collocation_pivot_and_category_details'])
    pivot_word = pivot_category_match.group(1).strip() if pivot_category_match else None
    category = pivot_category_match.group(2).strip() if pivot_category_match else None

    component_change_category = f"{error_component} -> {changed_component} + {category}"
    example_collocation = f"{error_component} -> {changed_component} + {pivot_word}"

    return pd.Series([component_change_category, example_collocation])


def get_other_category_collocation(collocations, pivot_word_lemma, pivot_word_pos):
    df_full_collocation = pd.read_csv(SM.filePath_collocation_full_csv)

    # If collocations is empty or not structured as expected
    if not collocations:
        df_selected_collocation = pd.DataFrame(columns=['error_component_total_freq_in_pivot_category_details'])
    else:
        df_selected_collocation = pd.DataFrame([collocation.split('|') for collocation in collocations], columns=df_full_collocation.columns)

    df_other_category_collocation = df_full_collocation[df_full_collocation.apply(
        lambda row:
            row['error_component'] == pivot_word_lemma and
            row['error_component_pos'] == pivot_word_pos and
            row['error_component_total_freq_in_pivot_category_details'] not in df_selected_collocation['error_component_total_freq_in_pivot_category_details'].values,
        axis=1
    )].drop_duplicates(subset=['error_component_total_freq_in_pivot_category_details'])

    scores = []
    for item in df_other_category_collocation['error_component_total_freq_in_pivot_category_details']:
        value = item.rsplit(')', 1)[-1].strip()
        category = item.rsplit(')', 1)[0].lstrip('(').strip()
        if "misc-ap" in category:
            value = '0'
        if value:
            scores.append(int(value))
        else:
            scores.append(0)

    if not df_other_category_collocation.empty:
        df_other_category_collocation['error_component_total_freq_in_pivot_category_score'] = scores
        df_other_category_collocation = df_other_category_collocation.sort_values(
            by='error_component_total_freq_in_pivot_category_score',
            ascending=False
        ).head(3)
    else:
        # Ensure 'error_component_total_freq_in_pivot_category_score' column exists if df is empty but scores were meant to be added
        # However, if df_other_category_collocation is empty, scores list will be empty, so this is fine.
        # If it needs to have this column even when empty for schema consistency:
        df_other_category_collocation['error_component_total_freq_in_pivot_category_score'] = pd.Series(dtype='int')

    if df_other_category_collocation.empty:
        df_format = pd.DataFrame(columns=['component_change_category', 'example_collocation'])
    else:
        df_format = df_other_category_collocation.apply(process_other_category, axis=1)
        df_format.columns = ['component_change_category', 'example_collocation']

    return df_other_category_collocation, df_format


def process_target_words_lemma_pos(formattd_sentences, learner_sentences):
    # 1-1. Get original error strings from formatted_sentences
    original_error_strings = re.findall(r'\[-(.*?)-\]', formattd_sentences)

    # 1-2. Process learner_sentences with spaCy to get tokens with lemma and POS
    learner_sentence_doc = NLP(learner_sentences)
    learner_tokens = list(learner_sentence_doc)  # For easier indexed access

    # 1-3. For each original_error_string, find its lemma and POS from learner_sentence_doc.
    # This simplified matching assumes errors are single tokens and appear in order.
    # More robust matching might be needed for complex cases (e.g., phrases, tokenization differences).
    target_infos = []
    current_learner_idx = 0
    for err_texts in original_error_strings:
        for err_text in err_texts.split():
            err_text = re.sub(r'[^\w\s]', '', err_text)
            found_in_learner = False
            # Try to find err_text as a single token (case-insensitive) starting from current_learner_idx
            for i in range(current_learner_idx, len(learner_tokens)):
                if learner_tokens[i].text.lower() == err_text.lower():
                    target_infos.append({
                        'text': learner_tokens[i].text,  # Store original text from learner sentence
                        'lemma': learner_tokens[i].lemma_,
                        'pos': learner_tokens[i].pos_
                    })
                    current_learner_idx = i + 1  # Move search start for the next error string
                    found_in_learner = True
                    break
            if not found_in_learner:
                # Fallback if direct match fails
                # Use external lemmatizer for lemma, and mark POS as UNKNOWN.
                # This assumes `lemmatize_word_list` can handle `err_text`.
                lemma_fallback = err_text  # Default lemma to err_text itself
            #     try:
            #         lemmatized_result = lemmatize_word_list([err_text])
            #         if lemmatized_result:
            #             lemma_fallback = lemmatized_result[0]
            #     except Exception:
            #         pass # Keep err_text as lemma if lemmatization fails

                target_infos.append({'text': err_text, 'lemma': lemma_fallback, 'pos': 'UNKNOWN'})
                print(f"Warning: Could not directly match '{err_text}' in learner sentence. Using lemma '{lemma_fallback}' and UNKNOWN POS.")
    # print(f"Processed {len(target_infos)} target words with lemma and POS information.")
    return target_infos


def get_ensemble_retriever(fusion: bool, embedding_settings: str, top_k: int = 5):
    faiss_retriever, bm25_retriever = create_or_get_retrivers(embedding_settings)
    # Vector-based retrieval
    if not fusion:
        return faiss_retriever

    # Keyword-based retrieval
    # bm25_retriever.k = top_k

    # Ensemble retriever combining both methods
    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    return ensemble_retriever


def create_response_chain(model_settings):
    class RetrievalResponse(BaseModel):
        response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")
    retrieval_prompt = PromptTemplate(
        input_variables=["sentences"],
        template=(
            "You will be given a sentence '{sentences}' with a suggested revision shown as [-original-]{{+revised+}}.\n"
            "Your task is to first evaluate whether the [-original-] and {{+revised+}} words has a strong collocational "
            "relationship with the surrounding words in the sentence.\n"
            "Based on this evaluation, output only 'Yes' (if a relevant collocation exists) or 'No' (if it does not).\n"
            "Example:\n"
            "There is a [-big-]{{+strong+}} possibility that they may not come.\n"
            "Output: Yes\n"
        )
    )

    class RelevanceResponse(BaseModel):
        response: str = Field(
            ...,
            title="Determines if collocations are related",
            description="Output only 'Relevant' or 'Irrelevant'."
        )
    relevance_prompt = PromptTemplate(
        input_variables=["sentences", "collocations"],
        template=(
            "Given the revised sentence '{sentences}', where edits are shown in bracketed notation—"
            "original text in [- –] and revised text in {{+ +}} "
            "determine if the following retrieving related collocation is relevant to the original text in [- –].\n"
            "Output only 'Relevant' or 'Irrelevant'."
            "Example:\n"
            "There is a [-big-]{{+strong+}} possibility that they may not come.\n"
            "related_collocations:\n"
            "big ||| ADJ ||| big success ||| great success ||| 3 ||| (big-great) 131 ||| success < Doing Things ||| (big-great Doing Things acc) 5 ||| (big-great Doing Things uniq) 3 ||| (big-x) 500 ||| (big-x-Doing Things) 6"
            "\nOutput: Relevant\n"
            "Now, please evaluate the following collocations:\n"
            "<related_collocations>\n{collocations}\n</related_collocations>"
        )
    )

    llm = ChatOpenAI(**model_settings)

    # Create LLMChains for each step
    global RETRIEVAL_CHAIN, RELEVANCE_CHAIN
    RETRIEVAL_CHAIN = retrieval_prompt | llm.with_structured_output(RetrievalResponse)
    RELEVANCE_CHAIN = relevance_prompt | llm.with_structured_output(RelevanceResponse)


def self_rag(retriever, formattd_sentences: str, learner_sentences: str):
    """
    Process a sentence to determine if retrieval is necessary and retrieve relevant collocations.
    """
    # print(f"\n{'=' * 50}\nProcessing query: {sentences}")

    # Step 1: Determine if retrieval is necessary
    # print(f"{'-' * 50}\nStep 1: Determine if retrieval is necessary")
    input_data = {"sentences": formattd_sentences}
    retrieval_decision = RETRIEVAL_CHAIN.invoke(input_data).response.strip().lower()
    # print(f"\nRetrieval decision: {retrieval_decision}")

    if retrieval_decision == "yes":
        # Step 2: Retrieve relevant documents
        # print(f"{'-' * 50}\nStep 2: Retrieve relevant documents")
        docs = retriever.invoke(formattd_sentences)
        retrieved_simplified_collocations = [doc.page_content for doc in docs]
        # Exact match retrieval
        # exact_matches = create_exact_match_retriever(sentences)
        # if exact_matches:
        #     collocations.extend(exact_matches)
        full_retrieved_collocations = list(set(get_full_collocation_info(retrieved_simplified_collocations)))
        print(f"Retrieved {len(full_retrieved_collocations)} documents.")
        # return collocations

        # Step 3: Evaluate relevance of retrieved documents
        # print(f"{'-' * 50}\nStep 3: Evaluate relevance of retrieved documents")
        relevant_contexts = []
        for i, context_item in tqdm(enumerate(full_retrieved_collocations), desc="Evaluating relevance", colour="green"):  # Iterate over each retrieved doc
            # Pass one context_item at a time for relevance evaluation
            input_data_relevance = {"sentences": formattd_sentences, "collocations": context_item}
            relevance = RELEVANCE_CHAIN.invoke(input_data_relevance).response.strip().lower()
            # print(f"Document {i+1} relevance: {relevance}")
            if relevance == "relevant":
                relevant_contexts.append(context_item)

        target_infos = process_target_words_lemma_pos(formattd_sentences, learner_sentences)
        df_other_categories_list = []
        df_other_categories_formatted_list = []

        for info in target_infos:
            pivot_lemma = info['lemma']
            pivot_pos = info['pos']

            if pivot_pos == 'UNKNOWN':
                print(f"Skipping get_other_category_collocation for original text '{info['text']}' (lemma '{pivot_lemma}') due to UNKNOWN POS.")
                df_other_categories_list.append(pd.DataFrame())  # Append empty DataFrame
                df_other_categories_formatted_list.append(pd.DataFrame())  # Append empty DataFrame
                continue

            # Call get_other_category_collocation with lemma and POS
            df_single_other_category, df_single_other_category_formatted = get_other_category_collocation(
                relevant_contexts,
                pivot_lemma,
                pivot_pos
            )
            df_other_categories_list.append(df_single_other_category)
            df_other_categories_formatted_list.append(df_single_other_category_formatted)

        # Concatenate results
        df_other_categories = pd.concat(df_other_categories_list, ignore_index=True) if df_other_categories_list else pd.DataFrame()
        df_other_categories_formatted = pd.concat(df_other_categories_formatted_list, ignore_index=True) if df_other_categories_formatted_list else pd.DataFrame()

        print(f"Number of relevant contexts: {len(relevant_contexts)}")

        return relevant_contexts, df_other_categories, df_other_categories_formatted


def process_one_df(df, name, retriever, output_path):
    # Start process self-rag for FCE and Longman samples
    tqdm.pandas(desc=f"Processing {name}")

    # self_rag returns a tuple: (relevant_contexts, df_other_categories, df_other_categories_formatted)
    # or None if retrieval is not deemed necessary or an issue occurs.
    results_from_apply = df.progress_apply(lambda row: self_rag(retriever, row['formatted_sentence'], row['learner_sentence']), axis=1)

    # Initialize lists for new columns
    collocations_str_list = []
    other_categories_json_list = []
    other_categories_formatted_json_list = []

    for result_tuple in results_from_apply:
        if result_tuple is not None:
            relevant_contexts, df_other_categories, df_other_categories_formatted = result_tuple

            # Process relevant_contexts (list of strings) into a single string
            collocations_str = "\n".join(map(str, relevant_contexts)) if relevant_contexts else ""
            collocations_str_list.append(collocations_str)

            # Convert df_other_categories to JSON string
            # An empty DataFrame will result in "[]"
            other_categories_json_list.append(
                df_other_categories.to_json(orient='records', force_ascii=False)
                if isinstance(df_other_categories, pd.DataFrame)
                else "[]"  # Default to empty JSON array string if not a DataFrame
            )

            # Convert df_other_categories_formatted to JSON string
            other_categories_formatted_json_list.append(
                df_other_categories_formatted.to_json(orient='records', force_ascii=False)
                if isinstance(df_other_categories_formatted, pd.DataFrame)
                else "[]"  # Default to empty JSON array string if not a DataFrame
            )
        else:
            # Handle cases where self_rag returned None
            collocations_str_list.append("")  # Default for no collocations
            other_categories_json_list.append("[]")  # Default for empty DataFrame (JSON array string)
            other_categories_formatted_json_list.append("[]")  # Default for empty DataFrame

    # Combine results with original dataframes
    df_with_results = df.copy()
    df_with_results["collocations"] = collocations_str_list
    df_with_results["other_categories_json"] = other_categories_json_list
    df_with_results["other_categories_formatted_json"] = other_categories_formatted_json_list

    # Save results to text file
    df_with_results.to_json(output_path, index=False, orient='records', force_ascii=False)


def main():
    model_settings = RETRIEVAL_DOC["model"][MODEL_TYPE]['settings']
    # load_data()
    create_response_chain(model_settings)
    retriever = get_ensemble_retriever(IS_FUSION, EMBEDDING_TYPE, top_k=TOP_K)

    # ---- Test cases ----
    # # sentences = "Her car was involved in a [-big-]{+serious+} accident."
    # # sentences = "To [-accomplish-]{+achieve+} world unity, we need peace."
    # # df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo)
    # # sentences = "It started pouring with rain and we all got {+soaked+}[-completely wet-]."
    # # learner_sentences = "It started pouring with rain and we all got completely wet."
    # # df_example3 = df_longman.loc[1731].copy()
    # # results, df_other_categories, df_other_categories_formatted = self_rag(retriever, sentences, learner_sentences)
    # # with open("./process/retrieval/output/test_retrieval_v3.txt", "w", encoding="utf-8") as f:
    # #     for result in results:
    # #         f.write(f"{result}\n")
    # # df_other_categories.to_json("./process/retrieval/output/test_df_soak.json", orient='records', lines=True)
    # # df_other_categories_formatted.to_json("./process/retrieval/output/test_df_soak_formatted.json", orient='records', lines=True)
    # # df_example3['collocations'] = "\n".join(map(str, results))
    # # df_example3['other_categories_json'] = df_other_categories
    # # df_example3['other_categories_formatted_json'] = df_other_categories_formatted
    # # df_example3 = df_example3.to_frame().T  # Convert Series to DataFrame
    # # df_example3.to_json("./process/retrieval/output/test_retrieval_soak.json", orient='records')

    # Completely wet
    # df_test = pd.read_csv(PATHS.filePath_test_dictInfo)
    # process_one_df(df_test, "Test", retriever, PATHS.filePath_test_withCollocation)
    # print("Test case processed successfully.")

    # Test two sample from full datasets
    # df_fce = pd.read_csv(PATHS.filePath_fce_dictInfo_filtered)
    # df_fce = df_fce.loc[0:1].copy()  # Apply slicing for testing
    # process_one_df(df_fce, "FCE", retriever, PATHS.filePath_fce_withCollocation)
    # df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo_filtered)
    # df_longman = df_longman.loc[0:1].copy()  # Apply slicing for testing
    # process_one_df(df_longman, "Longman", retriever, PATHS.filePath_longman_withCollocation)
    # return

    # ---- Process FCE and Longman samples ----
    # df_fce = pd.read_csv(PATHS.filePath_fce_sample_filtered)
    # process_one_df(df_fce, "FCE", retriever, PATHS.filePath_fce_sample_withCollocation)
    # if LONGMAN_SAMPLE_TYPE == 'R':
    #     df_longman = pd.read_csv(PATHS.filePath_longman_sample_one_replace)
    #     process_one_df(df_longman, "Longman", retriever, PATHS.filePath_longman_sample_one_replace_withCollocation)
    # else:
    #     df_longman = pd.read_csv(PATHS.filePath_longman_sample_filtered)
    #     process_one_df(df_longman, "Longman", retriever, PATHS.filePath_longman_sample_withCollocation)

    # ---- Process full FCE and Longman datasets ----
    df_fce = pd.read_csv(PATHS.filePath_fce_dictInfo_filtered)
    process_one_df(df_fce, "FCE", retriever, PATHS.filePath_fce_withCollocation)
    df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo_filtered)
    process_one_df(df_longman, "Longman", retriever, PATHS.filePath_longman_withCollocation)

    print("Processing completed.")


if __name__ == "__main__":
    main()
