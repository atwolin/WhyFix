# src/app/sentence_entry_page.py
import streamlit as st
import sys
import os
import pandas as pd
import errant
from sentence_transformers.cross_encoder import CrossEncoder

# --- Path Setup ---
APP_DIR_MV = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR_MV should be /home/ikm-admin/Work/atwolin/thesis/thesis-system/src
SRC_DIR_MV = os.path.dirname(APP_DIR_MV)
if SRC_DIR_MV not in sys.path:
    sys.path.insert(0, SRC_DIR_MV)  # Insert at the beginning for priority

# --- Project Imports ---
from preprocess.format_to_bracket import (
    format_runtime_to_bracket,
)
from app.utils.ui_utils import (
    process_formatted_sentence,
    formatted_to_original_edited,
)
from retrieval.dictionary_info import (
    concat_cambridge_data_runtime,
)
from preprocess.preprocess_setup import (
    NLP as spacy_nlp_for_dict,
)
from retrieval.collocation_info import (
    RETRIEVAL_DOC as COLLOCATION_CONFIG,
    find_collocations_runtime,
)
from model.get_response import (
    chat_baseline,
    chat_dictionary,
    chat_l2_knowledge_resource, # This function doesn't return explanations in the same format
    chat_l2_knowledge,
    chat_collocations,
    chat_mix,
    ExplanationSchema,      # Assuming this is the general schema for non-collocation specific
    GeneralExplanationSchema,
    CollocationSchema,
    GeneralCollocationSchema
)
from utils.files_io import (
    FilePath,
)


PATHS = FilePath()


# --- Data Loading Functions ---
@st.cache_resource
def get_errant_annotator():
    """Loads and caches the ERRANT annotator."""
    with st.spinner("載入 ERRANT 語言模型中..."):
        annotator = errant.load('en')
    # st.success("ERRANT annotator 載入成功!") # 成功後訊息可省略或簡化
    return annotator


@st.cache_resource
def get_cross_encoder_model_cached():
    with st.spinner("載入 CrossEncoder 語言模型中..."):
        cross_encoder_model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    # CrossEncoder is loaded globally in dictionary_info.py.
    # To cache it here, we'd ideally have dictionary_info.py provide a function to get it.
    # For now, we rely on its global loading but acknowledge caching it directly here would be better.
    # Or, ensure dictionary_info.MODEL is used directly if it's already loaded.
    # This is a placeholder for proper caching of the CrossEncoder if it's not handled well by its own loading.
    # If dictionary_info.MODEL is already the loaded model, we can just use it.
    # from sentence_transformers.cross_encoder import CrossEncoder
    # return CrossEncoder("cross-encoder/stsb-distilroberta-base", device="cuda:0")
    return cross_encoder_model  # Assuming dictionary_info.MODEL is the loaded instance


@st.cache_resource
def get_spacy_nlp_for_dict_cached():
    # Similar to CrossEncoder, NLP is loaded globally in preprocess.preprocess_setup
    # and imported into dictionary_info.
    return spacy_nlp_for_dict  # Assuming dictionary_info.NLP is the loaded instance


def _initialize_session_state():
    """Initializes all relevant session state variables."""
    if 'is_formatted_input_mode' not in st.session_state:
        st.session_state.is_formatted_input_mode = False  # Default: Original and Edited Sentences mode
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'retrieved_collocations_str' not in st.session_state:  # For storing raw collocation string
        st.session_state.retrieved_collocations_str = None
    if 'display_html1' not in st.session_state:
        st.session_state.display_html1 = ""
    if 'display_html2' not in st.session_state:
        st.session_state.display_html2 = ""
    if 'submitted_once' not in st.session_state:
        st.session_state.submitted_once = False

    # Default original and edited sentences
    DEFAULT_LEARNER_SENTENCE = "It started pouring with rain and we all got completely wet."
    DEFAULT_EDITOR_SENTENCE = "It started pouring with rain and we all got soaked."
    if "learner_sentence_input" not in st.session_state:
        st.session_state.learner_sentence_input = DEFAULT_LEARNER_SENTENCE
    if "editor_sentence_input" not in st.session_state:
        st.session_state.editor_sentence_input = DEFAULT_EDITOR_SENTENCE

    # DEFAULT_FORMATTED_SENTENCE = "It was a breakfast like any other until the dishes started to [- rattle. -] {+ shake. +}"
    DEFAULT_FORMATTED_SENTENCE = "It started pouring with rain and we all got {+soaked+}[-completely wet-]."
    if "formatted_sentence_input" not in st.session_state:
        st.session_state.formatted_sentence_input = DEFAULT_FORMATTED_SENTENCE
    elif st.session_state.formatted_sentence_input is None:
        st.session_state.formatted_sentence_input = DEFAULT_FORMATTED_SENTENCE

    # For LLM Explanation
    if 'llm_for_general' not in st.session_state:
        st.session_state.llm_for_general = False  # Default to lexical error explanation
    if 'llm_selected_function_name' not in st.session_state:
        st.session_state.llm_selected_function_name = "chat_baseline"  # Default function
    if 'llm_with_collocation_examples' not in st.session_state:
        st.session_state.llm_with_collocation_examples = True  # Default for relevant functions
    if 'llm_response' not in st.session_state:
        st.session_state.llm_response = None
    if 'llm_processing_error' not in st.session_state:
        st.session_state.llm_processing_error = None
    if 'generate_explanation_clicked' not in st.session_state:
        st.session_state.generate_explanation_clicked = False


def _load_resources():
    """Loads models and annotator, showing status."""
    with st.status("Initializing resources...", expanded=False) as status_init:
        annotator = get_errant_annotator()
        get_cross_encoder_model_cached()
        get_spacy_nlp_for_dict_cached()
        if annotator:
            status_init.update(label="Resources initialized successfully!", state="complete")
        else:
            status_init.update(label="ERRANT annotator failed to load.", state="error")
    return annotator


# --- Data Fetching/Processing Functions ---
def process_align_lines(learner_sentence, editor_sentence, annotator):
    """
    Generates HTML for visual comparison.
    Core of stage 1: (learner, editor) -> formatted_sentence -> (html1, html2)
    """
    if not learner_sentence or not editor_sentence:
        return None, "<i>Please enter both the original and edited sentences.</i>", ""

    # format_runtime_to_bracket returns a DataFrame. We'll work with its first row as a dict.
    processed_data_df = format_runtime_to_bracket(learner_sentence, editor_sentence, annotator=annotator)
    if processed_data_df.empty:
        return None, "<i>Could not process sentences for ERRANT alignment.</i>", ""

    processed_data_dict = processed_data_df.to_dict(orient="records")[0]

    if processed_data_dict is None or not isinstance(processed_data_dict, dict) or not processed_data_dict.get("formatted_sentence"):
        # Return the raw dict/None if it's not structured as expected for error inspection
        return processed_data_dict, "<i>Could not generate difference markup.</i>", ""

    line1_html, line2_html = process_formatted_sentence(processed_data_dict["formatted_sentence"])
    return processed_data_dict, line1_html, line2_html


def retrieve_linguistic_data(processed_data_dict, collocation_model_type: str, collocation_is_fusion: bool):
    # ---- Dictionary Information Part ----
    learner_info_json, editor_info_json = None, None
    with st.spinner("Retrieving dictionary information..."):
        if (
            processed_data_dict is not None
            and isinstance(processed_data_dict, dict)
            and all(k in processed_data_dict for k in ["learner_sentence", "learner_word", "editor_sentence", "editor_word"])
            and processed_data_dict.get("learner_word") is not None
            and processed_data_dict.get("editor_word") is not None
        ):
            _ = get_cross_encoder_model_cached()
            _ = get_spacy_nlp_for_dict_cached()

            learner_info, editor_info = concat_cambridge_data_runtime()
            learner_info_json = learner_info
            editor_info_json = editor_info
            # col1, col2 = st.columns(2)
            # with col1:
            #     st.markdown(f"**Learner's Word ({processed_data_dict.get('learner_word', 'N/A')})**")
            #     st.json(learner_info, expanded=True)
            # with col2:
            #     st.markdown(f"**Editor's Word ({processed_data_dict.get('editor_word', 'N/A')})**")
            #     st.json(editor_info, expanded=True)
        elif isinstance(processed_data_dict, dict) and "formatted_sentence" in processed_data_dict:
            st.info("Dictionary analysis requires original and edited sentences to extract specific words. This step will be skipped for formatted sentence input.")
        else:
            st.warning("Missing data required for dictionary analysis (e.g., learner/editor words not extracted).")

    # ---- Collocation Information Part ----
    df_collocations_to_display = None
    no_collocations_message = None
    with st.spinner("Retrieving collocation information..."):
        formatted_sentence_for_colloc = processed_data_dict.get("formatted_sentence") if isinstance(processed_data_dict, dict) else None

        if (
            formatted_sentence_for_colloc is None or
            (isinstance(formatted_sentence_for_colloc, str) and formatted_sentence_for_colloc.strip() == "") or
            (isinstance(formatted_sentence_for_colloc, (pd.Series, pd.DataFrame)) and formatted_sentence_for_colloc.empty)
        ):
            st.warning("Missing formatted sentence for collocation analysis.")
            return

        data_with_collocations = find_collocations_runtime(model_type=collocation_model_type, is_fusion=collocation_is_fusion)
        if data_with_collocations is not None and data_with_collocations.get("collocations"):
            collocations_str = data_with_collocations["collocations"]
            if isinstance(collocations_str, str) and collocations_str.strip():
                collocations = [c.split('|') for c in collocations_str.split('\n') if c.strip()]
                if not collocations:
                    no_collocations_message = "No collocations found related to this sentence edit."
                    return
                # Ensure all inner lists have the same length as headers, pad if necessary
                num_cols = 11
                info_headers = "error_component|error_component_pos|error_collocation|correct_collocation|collocation_correction_freq|component_change_freq_details|collocation_pivot_and_category_details|component_change_pivot_category_accum_freq_details|component_change_pivot_category_uniq_freq_details|error_component_total_freq_details|error_component_total_freq_in_pivot_category_details".split('|')

                # Pad rows with fewer than num_cols elements
                padded_collocations = []
                for row in collocations:
                    padded_row = row + [None] * (num_cols - len(row)) if len(row) < num_cols else row[:num_cols]
                    padded_collocations.append(padded_row)

                if padded_collocations:
                    df_collocations = pd.DataFrame(padded_collocations, columns=info_headers)
                    df_collocations_to_display = df_collocations
                    # st.dataframe(df_collocations, use_container_width=True)
                else:
                    no_collocations_message = "No valid collocation data to display."
                    # st.info(no_collocations_message)
            else:
                no_collocations_message = "No collocations found related to this sentence edit."
                # st.info(no_collocations_message)
        else:
            no_collocations_message = "No collocations found related to this sentence edit."
            # st.info(no_collocations_message)

    st.session_state.processed_data = pd.DataFrame([processed_data_dict])  # Store processed data in session state

    return {
        "learner_dict_info": learner_info_json,  # 實際應為獲取到的數據
        "editor_dict_info": editor_info_json,    # 實際應為獲取到的數據
        "collocation_df": df_collocations_to_display,
        "no_collocations_message": no_collocations_message,
        "learner_word": processed_data_dict.get('learner_word', 'N/A'),  # 傳遞詞彙以供顯示
        "editor_word": processed_data_dict.get('editor_word', 'N/A')     # 傳遞詞彙以供顯示
    }


def _handle_form_submission(annotator):
    """Processes sentences on button click and updates session state for results."""
    st.session_state.submitted_once = True
    st.session_state.llm_response = None  # Reset previous LLM response
    st.session_state.llm_processing_error = None  # Reset previous LLM error
    st.session_state.generate_explanation_clicked = False  # Reset this flag

    # Use sentences from session state, which are kept up-to-date
    learner_s = st.session_state.learner_sentence_input
    editor_s = st.session_state.editor_sentence_input
    formatted_s = st.session_state.formatted_sentence_input

    data_dict_result = None
    html1_result = ""
    html2_result = ""

    valid_input = False
    if st.session_state.is_formatted_input_mode:
        if formatted_s and formatted_s.strip():
            # Learner/Editor sentences are already derived when formatted_sentence_input changes
            # Pass these derived sentences for alignment.
            data_dict_result, html1_result, html2_result = process_align_lines(
                learner_s, editor_s, annotator
            )
            valid_input = True
        else:
            html1_result = "<i>Please enter the formatted sentence.</i>"
    else:  # Original and Edited mode
        if learner_s and learner_s.strip() and editor_s and editor_s.strip():
            data_dict_result, html1_result, html2_result = process_align_lines(
                learner_s, editor_s, annotator
            )
            valid_input = True
        elif not (learner_s and learner_s.strip()):
            html1_result = "<i>Please enter the original sentence.</i>"
        elif not (editor_s and editor_s.strip()):
            html1_result = "<i>Please enter the edited sentence.</i>"
        else:  # Should not happen if previous two are exhaustive for empty
            html1_result = "<i>Please enter both the original and edited sentences.</i>"

    st.session_state.processed_data = data_dict_result if valid_input and isinstance(data_dict_result, dict) else None
    st.session_state.display_html1 = html1_result
    st.session_state.display_html2 = html2_result if valid_input and isinstance(data_dict_result, dict) else ""

    # Retrieve linguistic data after processing sentences, this will also store collocation string
    processed_data = st.session_state.processed_data
    has_processed_data = (
        processed_data is not None and
        (
            (isinstance(processed_data, dict) and bool(processed_data)) or
            (isinstance(processed_data, pd.DataFrame) and not processed_data.empty)
        )
    )
    if has_processed_data:  # Only show if sentences have been processed
        linguistic_results = retrieve_linguistic_data(
            st.session_state.processed_data,
            collocation_model_type=st.session_state.collo_model_type_select,
            collocation_is_fusion=st.session_state.collo_fusion_select
        )
        st.session_state.learner_dict_info_display = linguistic_results.get("learner_dict_info")
        st.session_state.editor_dict_info_display = linguistic_results.get("editor_dict_info")
        st.session_state.colloc_df_display = linguistic_results.get("collocation_df")
        st.session_state.no_collocations_message_display = linguistic_results.get("no_collocations_message")
        st.session_state.display_learner_word = linguistic_results.get("learner_word")
        st.session_state.display_editor_word = linguistic_results.get("editor_word")
    else:
        st.session_state.learner_dict_info_display = None
        st.session_state.editor_dict_info_display = None
        st.session_state.colloc_df_display = None
        st.session_state.no_collocations_message_display = "Missing data for linguistic analysis."
        st.session_state.display_learner_word = "N/A"
        st.session_state.display_editor_word = "N/A"


def _display_processed_linguistic_info():
    """Displays dictionary and collocation information from session state."""
    if st.session_state.submitted_once:  # 或者檢查特定 session state 鍵是否存在
        # ---- Dictionary Information Part ----
        with st.expander("📚 Dictionary Information", expanded=False):
            if st.session_state.get("learner_dict_info_display") is not None or st.session_state.get("editor_dict_info_display") is not None:
                # Simulating retrieval for display, actual data comes from session state
                # In a real scenario, concat_cambridge_data_runtime would be called during _handle_form_submission
                # and its results (learner_info, editor_info) stored in session_state.
                # Here, we'd use those stored values.
                # For demonstration, let's assume they are stored:
                learner_info_to_display = st.session_state.get("learner_dict_info_display")
                editor_info_to_display = st.session_state.get("editor_dict_info_display")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Learner's Word ({st.session_state.get('display_learner_word', 'N/A')})**")
                    if learner_info_to_display:
                        st.json(learner_info_to_display, expanded=True)
                    else:
                        st.info("No dictionary information for learner's word.")
                with col2:
                    st.markdown(f"**Editor's Word ({st.session_state.get('display_editor_word', 'N/A')})**")
                    if editor_info_to_display:
                        st.json(editor_info_to_display, expanded=True)
                    else:
                        st.info("No dictionary information for editor's word.")
            elif "formatted_sentence_input" in st.session_state and st.session_state.is_formatted_input_mode: # Check if input was from formatted mode
                # This condition might need adjustment based on how `processed_data` is structured for formatted input
                is_dict_processed = isinstance(st.session_state.get("processed_data"), pd.DataFrame) and not st.session_state.processed_data.empty and \
                                    ("learner_word" not in st.session_state.processed_data.columns or \
                                     st.session_state.processed_data["learner_word"].iloc[0] is None)

                if is_dict_processed and (st.session_state.processed_data["learner_word"].iloc[0] is None and st.session_state.processed_data["editor_word"].iloc[0] is None) :
                    st.info("Dictionary analysis requires original and edited sentences to extract specific words. This step was skipped for formatted sentence input without word edits or direct word input.")
                else: # Fallback if specific words were somehow extracted or if it's not purely formatted input.
                    st.warning("Missing data required for dictionary analysis.")

            else:
                st.warning("Dictionary data not available or not processed.")


        # ---- Collocation Information Part ----
        with st.expander("🔗 Collocation Information", expanded=False):
            if st.session_state.get("colloc_df_display") is not None and not st.session_state.colloc_df_display.empty:
                st.dataframe(st.session_state.colloc_df_display, use_container_width=True)
            elif st.session_state.get("no_collocations_message_display"):
                st.info(st.session_state.no_collocations_message_display)
            else:
                st.warning("Collocation data not available or not processed.")


def _handle_generate_explanation():
    """Handles the LLM explanation generation."""
    st.session_state.generate_explanation_clicked = True
    st.session_state.llm_response = None
    st.session_state.llm_processing_error = None

    processed_data = st.session_state.processed_data
    has_processed_data = (
        processed_data is not None and
        (
            (isinstance(processed_data, dict) and bool(processed_data)) or
            (isinstance(processed_data, pd.DataFrame) and not processed_data.empty)
        )
    )
    if not has_processed_data:  # Only show if sentences have been processed
        st.warning("Please process sentences first to get data for explanation.")
        st.session_state.generate_explanation_clicked = False # 重置以免顯示空結果
        return

    # Prepare DataFrame for chat functions
    # The chat functions expect a DataFrame. We create one from processed_data.
    # We also add the retrieved collocations string if available and relevant.
    # current_data = st.session_state.processed_data.copy() # Make a copy
    # if st.session_state.retrieved_collocations_str:
    #     current_data['collocations'] = st.session_state.retrieved_collocations_str # Key expected by some formatters
    current_data_path = PATHS.filePath_runtime_withCollocation  # Use the runtime file path directly
    file_path_runtime = PATHS.filePath_runtime_allInfo

    # df_for_llm = st.session_state.processed_data
    df_for_llm = pd.read_json(current_data_path) if os.path.exists(current_data_path) else df_for_llm

    print(current_data_path)

    # Map selected function name to actual function
    chat_function_map = {
        "chat_baseline": chat_baseline,
        "chat_dictionary": chat_dictionary,
        "chat_l2_knowledge": chat_l2_knowledge,
        "chat_collocations": chat_collocations,
        "chat_mix": chat_mix,
    }
    selected_function = chat_function_map.get(st.session_state.llm_selected_function_name)

    if not selected_function:
        st.session_state.llm_processing_error = "Invalid LLM function selected."
        return

    with st.spinner(f"Generating explanation using {st.session_state.llm_selected_function_name}..."):
        for_general_val = st.session_state.llm_for_general

        # Default model_type and role, can be made configurable later
        model_type_llm = "gpt-4.1-nano"  # Or from st.selectbox if you add it
        role_llm = "linguist"       # Or from st.text_input if you add it

        df_for_llm = chat_l2_knowledge_resource(
            df_for_llm,
            file_path=file_path_runtime,
            model_type=model_type_llm,
            role=role_llm
        )
        # st.session_state.processed_data

        print("Processed data for LLM:", st.session_state.processed_data)

        explanation_result = None
        if selected_function in [chat_collocations, chat_mix]:
            with_examples_val = st.session_state.llm_with_collocation_examples
            explanation_result = selected_function(
                df_for_llm,
                file_path=file_path_runtime,
                for_general=for_general_val,
                with_collocaiton_examples=with_examples_val,  # Note: 'collocaiton' typo in original get_response.py
                model_type=model_type_llm,
                role=role_llm
            )
        elif selected_function == chat_l2_knowledge:  # Does not have for_general
            explanation_result = selected_function(
                df_for_llm,
                file_path=file_path_runtime,
                model_type=model_type_llm,
                role=role_llm
            )
        else:  # For chat_baseline, chat_dictionary
            explanation_result = selected_function(
                df_for_llm,
                file_path=file_path_runtime,
                for_general=for_general_val,
                model_type=model_type_llm,
                role=role_llm
            )
        st.session_state.llm_response = explanation_result
        # The functions in get_response.py save the df to file_path_runtime,
        # and return the parsed Pydantic model.
        # We can inspect df_for_llm if needed, e.g., df_for_llm.to_dict()

    # except Exception as e:
    #     st.session_state.llm_processing_error = f"Error during explanation generation: {str(e)}"
    #     st.error(st.session_state.llm_processing_error)


def _display_llm_explanation_results():
    """Displays the LLM generated explanation."""
    if st.session_state.generate_explanation_clicked:
        st.subheader("💬 LLM Generated Explanation")
        if st.session_state.llm_processing_error:
            st.error(st.session_state.llm_processing_error)
            return

        response_data = st.session_state.llm_response
        if response_data is None:
            st.info("No explanation generated or an issue occurred.")
            return

        # The response_data is a Pydantic model (e.g., ExplanationSchema, CollocationSchema)
        # We convert it to a dictionary to access its fields easily.
        payload = {}
        if hasattr(response_data, 'model_dump'):
            payload = response_data.model_dump()
        elif isinstance(response_data, dict):  # Should ideally be a Pydantic model
            payload = response_data
        else:
            st.warning(f"Unexpected response type: {type(response_data)}")
            st.json(response_data)  # Display raw if unknown
            return

        if not payload:
            st.info("Explanation payload is empty.")
            return

        # Mimicking the structure from your display_results function
        if "explanation_en" in payload:
            with st.expander("Explanation (English)", expanded=True):
                st.write(payload["explanation_en"])
                if "example_en" in payload and payload["example_en"] and isinstance(payload["example_en"], list):
                    st.markdown("**Examples:**")
                    for ex_en in payload["example_en"]:
                        st.markdown(f"- {ex_en}")

        if "explanation_zh_tw" in payload:  # For GeneralExplanationSchema
            with st.expander("解釋 (繁體中文)", expanded=True):
                st.write(payload["explanation_zh_tw"])
                if "example_zh_tw" in payload and payload["example_zh_tw"] and isinstance(payload["example_zh_tw"], list):
                    st.markdown("**範例:**")
                    for ex_zh in payload["example_zh_tw"]:
                        st.markdown(f"- {ex_zh}")

        # For CollocationSchema / GeneralCollocationSchema
        if "corresponding_collocation_en" in payload:
            with st.expander("Corresponding Collocation (English)", expanded=True):
                if isinstance(payload["corresponding_collocation_en"], list) and payload["corresponding_collocation_en"]:
                    for item in payload["corresponding_collocation_en"]:
                        st.markdown(str(item))  # Items could be strings or other structures
                elif isinstance(payload["corresponding_collocation_en"], str):  # If it's a single string
                    st.markdown(payload["corresponding_collocation_en"])

                if "corresponding_collocation_examples_en" in payload and payload["corresponding_collocation_examples_en"] and isinstance(payload["corresponding_collocation_examples_en"], list):
                    st.markdown("**Examples:**")
                    for ex_en in payload["corresponding_collocation_examples_en"]:
                        st.markdown(f"- {ex_en}")

        if "other_collocations_en" in payload:
            with st.expander("Other Collocations (English)", expanded=True):
                if isinstance(payload["other_collocations_en"], list) and payload["other_collocations_en"]:
                    st.markdown("**Collocations:**")
                    for item in payload["other_collocations_en"]:
                        st.markdown(str(item))
                elif isinstance(payload["other_collocations_en"], str):
                    st.markdown(payload["other_collocations_en"])

                if "other_collocations_examples_en" in payload and payload["other_collocations_examples_en"] and isinstance(payload["other_collocations_examples_en"], list):
                    st.markdown("**Examples:**")
                    for ex_en in payload["other_collocations_examples_en"]:
                        st.markdown(f"- {ex_en}")

        # Fallback for any other fields in the payload
        st.markdown("---")
        with st.expander("Raw LLM Response Data", expanded=False):
            st.json(payload)


def _display_results_and_feedback(annotator):  # annotator might not be needed here unless for re-processing
    """Displays processing results, linguistic data, or relevant feedback messages."""
    # This function will now primarily display the ERRANT alignment and linguistic info.
    # LLM results are handled by _display_llm_explanation_results
    if st.session_state.submitted_once:
        # Linguistic data (Dictionary, Collocations table) is displayed within retrieve_linguistic_data
        # when _handle_form_submission calls it. So, no need to call it again here explicitly for display.
        # We just need to display the alignment HTML.

        # If submitted but no processed data (implying an error before linguistic data retrieval)
        # and no error message in display_html1, it implies an issue or cleared state.
        processed_data = st.session_state.processed_data
        is_empty = (
            processed_data is None or
            (isinstance(processed_data, pd.DataFrame) and processed_data.empty) or
            (isinstance(processed_data, dict) and not processed_data)
        )
        if is_empty and not ("<i>" in st.session_state.display_html1):
            st.info("No valid data to process or display results for. Process sentences first.")

        st.header("📊 Results")
        if st.session_state.display_html1:
            if "<i>" not in st.session_state.display_html1:  # Success
                st.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap; text-align: center;">{st.session_state.display_html1}</div>',
                    unsafe_allow_html=True
                )
                if st.session_state.display_html2:
                    st.markdown(
                        f'<div style="font-family: monospace; white-space: pre-wrap; text-align: center;">{st.session_state.display_html2}</div>',
                        unsafe_allow_html=True
                    )
            else:  # Error message from submission handling
                st.markdown(st.session_state.display_html1, unsafe_allow_html=True)

    else:  # Not submitted_once: Show initial guidance or warnings for empty fields
        if st.session_state.is_formatted_input_mode:
            if not st.session_state.formatted_sentence_input.strip():
                st.info("Please enter the formatted sentence and click 'Process Sentences'.")
        else:  # Original and Edited mode
            if not st.session_state.learner_sentence_input.strip() and not st.session_state.editor_sentence_input.strip():
                st.info("Please enter both original and edited sentences, then click 'Process Sentences'.")
            elif not st.session_state.learner_sentence_input.strip():
                st.info("Please enter the original sentence.")
            elif not st.session_state.editor_sentence_input.strip():
                st.info("Please enter the edited sentence.")


# --- UI Presentation Functions ---
def _render_input_widgets():
    """Renders all input widgets and handles immediate logic related to them."""
    st.toggle(
        label="Enable Formatted Sentence Input Mode",
        key="is_formatted_input_mode",
        help="If enabled, you can paste sentences with edit markers directly (e.g., He [-go-]{+went+} to school.).\nIf disabled, please enter the original and edited sentences separately."
    )

    col_input, col_settings = st.columns([2, 1])

    with col_input:
        st.header("Enter Sentences")
        if not st.session_state.is_formatted_input_mode:
            st.text_area(
                label="Original Sentence:",
                placeholder="E.g., He go to school yesterday.",
                height=100,
                key="learner_sentence_input"
            )
            st.text_area(
                label="Edited Sentence:",
                placeholder="E.g., He went to school yesterday.",
                height=100,
                key="editor_sentence_input"
            )
        else:  # Formatted Sentence mode
            st.text_area(
                label="Formatted Sentence:",
                placeholder="E.g., He [-go-]{+went+} to school yesterday.",
                height=100,
                key="formatted_sentence_input"
            )
            # Update derived learner/editor sentences in session state immediately
            if st.session_state.formatted_sentence_input:
                derived_l, derived_e = formatted_to_original_edited(
                    st.session_state.formatted_sentence_input
                )
                st.session_state.learner_sentence_input = derived_l
                st.session_state.editor_sentence_input = derived_e
            else:
                st.session_state.learner_sentence_input = ""
                st.session_state.editor_sentence_input = ""

    with col_settings:
        st.markdown("**Collocation Analysis Settings**")
        collo_model_keys = list(COLLOCATION_CONFIG["model"].keys())
        default_collo_model_index = 0
        if "gpt" in collo_model_keys:
            default_collo_model_index = collo_model_keys.index("gpt")

        st.selectbox(
            "Collocation Analysis LLM Type:",
            options=collo_model_keys,
            index=default_collo_model_index,
            key="collo_model_type_select"
        )
        st.checkbox(
            "Enable Hybrid Retrieval (Fusion)?",
            value=True,
            key="collo_fusion_select"
        )

        st.markdown("---")
        st.markdown("**LLM Explanation Generation Settings**")
        # st.toggle(
        #     label="Generic Explanation",
        #     key="llm_for_general",
        #     value=st.session_state.llm_for_general, # Use value from session state
        #     help="If True, requests a general explanation of comparing two sentences. If False, requests an explanation specific to the learner's error."
        # )

        # Define available chat functions for the user
        available_chat_functions = {
            "Baseline Explanation": "chat_baseline",
            "Dictionary-Augmented Explanation": "chat_dictionary",
            "(Not for General use) L2 Knowledge-Augmented Explanation": "chat_l2_knowledge",
            "Collocation-Augmented Explanation": "chat_collocations",
            "Mixed RAG Explanation": "chat_mix",
        }
        st.selectbox(
            "Choose Explanation Type:",
            options=list(available_chat_functions.keys()),
            # Format function to get the key for session state from the display name
            format_func=lambda x: x,
            key="llm_selected_function_display_name", # Temp key to store display name
            help="Select the type of explanation to generate."
        )
        # Update the actual function name in session state based on display name
        st.session_state.llm_selected_function_name = available_chat_functions[st.session_state.llm_selected_function_display_name]

        # Conditional checkbox for collocation examples
        if st.session_state.llm_selected_function_name in ["chat_collocations", "chat_mix"]:
            st.checkbox(
                "Include Collocation Examples in LLM Explanation?",
                key="llm_with_collocation_examples",
                value=st.session_state.llm_with_collocation_examples, # Use value from session state
                help="Applies if 'Collocation-Augmented' or 'Mixed RAG' explanation is selected."
            )
        # You could add more settings here, e.g., for model_type, role for the LLM explanation part


def sentence_input_page_content():
    """
    Renders a Streamlit page for users to input learner and editor sentences.
    Orchestrates UI rendering, state initialization, and processing logic.
    """
    st.title("📝 WhyFix: Explain Your Sentence Edits")
    # st.write("Enter the original sentences and the edited version below.")

    # Initialize session state variables if they don't exist
    _initialize_session_state()
    annotator = _load_resources()

    if not annotator:  # Check moved after status for clarity
        st.error("ERRANT annotator failed to load. Page functionality may be limited.")
        return

    # Input Fields
    _render_input_widgets()

    # Submission Button
    if st.button("Process Sentences", key="process_sentences_button"):
        _handle_form_submission(annotator)

    # 無論哪個按鈕被按下，只要提交過一次，就嘗試顯示結果
    if st.session_state.get("submitted_once", False):
        _display_processed_linguistic_info()      # 新增：顯示字典和詞語搭配資訊
        _display_results_and_feedback(annotator)  # Displaying Inputs (for confirmation)


    # Separator and button for LLM Explanation
    processed_data = st.session_state.processed_data
    has_processed_data = (
        processed_data is not None and
        (
            (isinstance(processed_data, dict) and bool(processed_data)) or
            (isinstance(processed_data, pd.DataFrame) and not processed_data.empty)
        )
    )
    if has_processed_data:  # Only show if sentences have been processed
        st.markdown("---")
        if st.button("Generate LLM Explanation", key="generate_explanation_button"):
            _handle_generate_explanation()  # This now handles the LLM call

    # Display LLM Explanation Results
    # This will display the results if 'generate_explanation_clicked' is true
    _display_llm_explanation_results()


# --- How to run this new page (Example) ---
# To test this page independently, you can add this at the end of your file:
#
if __name__ == "__main__":
    # To run the original page:
    if SRC_DIR_MV not in sys.path:  # Use SRC_DIR_MV from the top
        sys.path.insert(0, SRC_DIR_MV)
    sentence_input_page_content()
else:
    # To run the new sentence input page:
    sentence_input_page_content()

# Make sure to comment out one or the other when testing.
# For a multi-page app, you'd typically use a more structured approach,
# for example, using a dictionary of pages and a sidebar selection.
