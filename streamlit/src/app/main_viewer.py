import streamlit as st
import os
import sys
import json
import pandas as pd


# --- Path Setup ---
APP_DIR_MV = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR_MV should be /home/ikm-admin/Work/atwolin/thesis/thesis-system/src
SRC_DIR_MV = os.path.dirname(APP_DIR_MV)
if SRC_DIR_MV not in sys.path:
    sys.path.insert(0, SRC_DIR_MV)  # Insert at the beginning for priority

# --- Project Imports ---
from app.utils.ui_utils import (
    process_formatted_sentence
)


# --- Data Loading Functions ---
@st.cache_data
def load_data_from_string(json_string):
    """Loads data from a string where each line is a JSON object."""
    data = []
    if not json_string:
        st.error("Error: Input JSON string is empty.")
        return data
    for i, line in enumerate(json_string.strip().split('\n')):
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON on line {i+1}: {e} in line: {line[:100]}...")  # Show part of the line
                continue
    return data


@st.cache_data
def load_jsonl_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            json_string = f.read()
        if not json_string.strip():
            st.warning(f"警告：檔案 {os.path.basename(file_path)} 為空。")
            return data
        for i, line in enumerate(json_string.strip().split('\n')):
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    st.error(f"檔案 {os.path.basename(file_path)} 第 {i+1} 行 JSON 解碼錯誤: {e} (內容: {line[:100]}...)")
                    continue
    except FileNotFoundError:
        st.error(f"錯誤：找不到資料檔案 {file_path}")
        return None
    except Exception as e:
        st.error(f"讀取檔案 {file_path} 時發生錯誤: {e}")
        return None
    return data


# --- Data Fetching/Processing Functions ---
# def get_learner_sentences_and_indices(sentence_data):
#     """Extracts learner sentences and their original indices."""
#     return [(s.get("learner_sentence"), s.get("index")) for s in sentence_data if s.get("learner_sentence") and s.get("index") is not None]


def get_formatted_sentences_and_indices(sentence_data):
    """Extracts learner sentences and their original indices."""
    indices = []
    if sentence_data:
        # 嘗試不同的 index 欄位名稱
        for s in sentence_data:
            formatted_sentence = s.get("formatted_sentence")
            if formatted_sentence:
                # 嘗試找到 index 欄位 (可能是 'index' 或 'data_index')
                index_value = s.get("index")
                if index_value is None:
                    index_value = s.get("data_index")

                if index_value is not None:
                    indices.append((formatted_sentence, index_value))
                else:
                    # 如果沒有 index 欄位，使用資料的順序作為 index
                    indices.append((formatted_sentence, len(indices)))

    if not indices and sentence_data:
        # 檢查資料結構並提供更詳細的錯誤訊息
        sample_keys = list(sentence_data[0].keys()) if sentence_data else []
        st.warning(f"在提供的句子資料中，未找到帶有 'index' 或 'data_index' 的 'formatted_sentence'。資料欄位: {sample_keys}")
    return indices


def get_unique_values_for_filters(explanation_data, fields):
    """Gets unique values for specified filter fields from explanation data."""
    unique_values = {}
    if not explanation_data:
        for field in fields: unique_values[field] = []
        return unique_values
    for field in fields:
        field_values = [item.get(field) for item in explanation_data if item.get(field) is not None]
        unique_values[field] = sorted(list(set(field_values)))
    return unique_values


def find_sentence_index(selected_sentence_text, sentences_with_indices):
    """Finds the data_index for the selected learner_sentence."""
    for sentence, index in sentences_with_indices:
        if sentence == selected_sentence_text:
            return index
    return None


def filter_explanations(explanation_data, target_data_index, target_dataset, selected_filters):
    """Filters explanations based on the selected sentence's data_index and other filter criteria."""
    if target_data_index is None or not explanation_data:
        return []
    filtered = []

    for exp in explanation_data:
        # 支援 data_index 和 index 兩種欄位名稱
        exp_index = exp.get("data_index")
        if exp_index is None:
            exp_index = exp.get("index")

        if exp_index != target_data_index:
            continue

        # 支援 dataset 和 data_source 兩種欄位名稱
        exp_dataset = exp.get("dataset")
        if exp_dataset is None:
            exp_dataset = exp.get("data_source")

        if exp_dataset not in target_dataset:
            continue

        match = True
        for field, selected_values in selected_filters.items():
            if selected_values:
                if exp.get(field) not in selected_values:
                    match = False
                    break
        if match:
            filtered.append(exp)
    return filtered


# --- UI Presentation Functions ---
def display_selection_interface(sentences_with_indices, unique_filter_options):
    """Displays the selection widgets for sentence and filters."""
    st.header("🔍 Select Data to Display")
    col1, col2 = st.columns(2)

    selected_sentence_text = None
    selected_filters = {}

    with col1:
        st.subheader("Choose a Sentence")
        sentence_options = [s[0] for s in sentences_with_indices if s[0] is not None]
        if sentence_options:
            selected_sentence_text = st.selectbox(
                "Select a sentence:",
                options=sentence_options,
                index=0,
                placeholder="Choose an option"
            )
        else:
            st.warning("No sentences available for selection.")

    with col2:
        st.subheader("Filter Explanations")
        filter_fields = ["method", "input_type", "output_length"]
        for field in filter_fields:
            options = unique_filter_options.get(field, [])
            # If options are available, default to the first one
            if options:
                default_value = [options[0]] if options else []  # 'default' value is the first option
                selected_filters[field] = st.multiselect(
                    f"Select {field.replace('_', ' ').title()}(s):",
                    options=options,
                    default=default_value  # 加上 default 參數
                )
            else:
                st.caption(f"No options available for {field.replace('_', ' ').title()}.")
                selected_filters[field] = []  # Ensure key exists

    return selected_sentence_text, selected_filters


def print_retrieval_data_for_item(item_data):
    """Prints data for DictionaryInfo, L2KnowledgeInfo, and CollocationInfo from an item."""
    if not item_data:
        st.warning("No item data provided to print.")
        return

    st.subheader("Retrieved Data Details:")

    # DictionaryInfo
    # st.markdown("---")
    st.markdown("#### Dictionary Info")
    dict_info_keys_learner = [
        "word_learner", "lemmaWord_learner", "pos_learner", "level_learner",
        "definition_learner", "examples_learner", "in_akl_learner"
    ]
    dict_info_data_learner = []
    for key in dict_info_keys_learner:
        value = item_data.get(key, "Not available")
        if isinstance(value, list):
            value_str = "; ".join(map(str, value)) if value else "Not available"
        elif isinstance(value, bool):
            value_str = str(value)
        else:
            value_str = value if value is not None else "Not available"
        dict_info_data_learner.append({"Attribute": key, "Value": value_str})

    st.dataframe(dict_info_data_learner, use_container_width=True)

    st.markdown("#### Dictionary Info (Editor)")
    dict_info_keys_editor = [
        "word_editor", "lemmaWord_editor", "pos_editor", "level_editor",
        "definition_editor", "examples_editor", "in_akl_editor"
    ]
    dict_info_data_editor = []
    for key in dict_info_keys_editor:
        value = item_data.get(key, "Not available")
        if isinstance(value, list):
            value_str = "; ".join(map(str, value)) if value else "Not available"
        elif isinstance(value, bool):
            value_str = str(value)
        else:
            value_str = value if value is not None else "Not available"
        dict_info_data_editor.append({"Attribute": key, "Value": value_str})

    st.dataframe(dict_info_data_editor, use_container_width=True)


    # L2KnowledgeInfo
    st.markdown("---")
    st.markdown("#### L2 Knowledge Info")
    l2_info_keys = [
        "causes_fifty", "academic_writing_fifty",
        "causes_eighty", "academic_writing_eighty"
    ]
    # for key in l2_info_keys:
    #     if key in item_data:
    #         st.text(f"{key}:")
    #         if isinstance(item_data[key], list):
    #             for i, val_item in enumerate(item_data[key]):
    #                 st.text(f"  - {val_item}")
    #         else:
    #             st.text(f"  {item_data[key]}")
    #     else:
    #         st.text(f"{key}: Not available")
    l2_info_data = []
    for key in l2_info_keys:
        value = item_data.get(key)
        if value is not None:
            # If the value is a list, st.dataframe handles it well by showing list items.
            # If it's a single string (though not expected by schema), it will also be shown.
            l2_info_data.append({"Category": key, "Details": value})
        else:
            l2_info_data.append({"Category": key, "Details": "Not available"})

    if l2_info_data:
        # Transpose for better readability if desired, or display as is.
        # For now, displaying as is, where each key is a row.
        st.dataframe(l2_info_data, use_container_width=True)
        # To make it more readable if lists are long, you might consider custom rendering
        # or further processing of the 'Details' column if st.dataframe's default list display isn't ideal.
        # For example, joining list items into a multi-line string:
        # for item_row in l2_info_data:
        #     if isinstance(item_row["Details"], list):
        #         item_row["Details"] = "\n".join([f"- {d}" for d in item_row["Details"]])
        # st.dataframe(l2_info_data, use_container_width=True)
    else:
        st.text("No L2 Knowledge Info available.")

    # CollocationInfo
    st.markdown("---")
    st.markdown("#### Collocation Info")
    collo_info_keys = [
        "collocations", "other_categories_json", "other_categories_formatted_json"
    ]
    collo_field_names = [
        "error_component", "error_component_pos", "error_collocation", "correct_collocation",
        "collocation_correction_freq",
        "component_change_freq_details", "collocation_pivot_and_category_details",
        "component_change_pivot_category_accum_freq_details",
        "component_change_pivot_category_uniq_freq_details",
        "error_component_total_freq_details",
        "error_component_total_freq_in_pivot_category_details",
    ]
    collo_example_field_names = [
        "component_change_category", "example_collocation"
    ]
    # for key in collo_info_keys:
    #     if key in item_data:
    #         st.text(f"{key}: {item_data[key]}")
    #     else:
    #         st.text(f"{key}: Not available")
    # 1. collocations row
    collocations_rows = [entry.split('|') for entry in item_data['collocations'].split('\n') if entry.strip()]
    df_collocations = pd.DataFrame(collocations_rows, columns=collo_field_names)

    # 2. other_categories_json row
    other_categories_json = json.loads(item_data['other_categories_json'])
    df_other_categories = pd.DataFrame(other_categories_json, columns=collo_field_names)

    # 3. example row
    other_categories_formatted_json = json.loads(item_data['other_categories_formatted_json'])
    df_other_collocation_example = pd.DataFrame(other_categories_formatted_json, columns=collo_example_field_names)

    if not df_collocations.empty:
        st.dataframe(df_collocations, use_container_width=True)
    if not df_other_categories.empty:
        st.dataframe(df_other_categories, use_container_width=True)
    if not df_other_collocation_example.empty:
        st.dataframe(df_other_collocation_example, use_container_width=True)
    if df_collocations.empty and df_other_categories.empty and df_other_collocation_example.empty:
        st.text("No Collocation Info available.")

    st.markdown("---")


def display_results(explanations, all_sentence_data):
    """Displays the filtered explanations with their content_payload and attributes."""
    st.header("📊 Results")
    if not explanations:
        st.info("No explanations match the current selection.")
        return

    # --- Display retrieval data ---
    index_of_first_explanation = explanations[0].get("data_index")
    sentence_info = next(
        (s for s in all_sentence_data if s.get("index") == index_of_first_explanation), None
    )
    if sentence_info:
        with st.expander("View Retrieval Data", expanded=False):
            print_retrieval_data_for_item(sentence_info)
    else:
        st.warning("Retrieval data could not be loaded for this explanation set.")

    for i, exp_item in enumerate(explanations):
        with st.container(border=True):
            col_header, col_badges_container = st.columns([3, 2], gap="medium")

            with col_header:
                st.subheader(f"Explanation Set {i+1}")

            with col_badges_container:
                badge_fields = ["dataset", "input_type", "method", "output_length"]
                icons = {"dataset": "📚", "input_type": "⌨️", "method": "🔬", "output_length": "📏"}
                badge_html_parts = []
                for field_name in badge_fields:
                    val = exp_item.get(field_name)
                    if val:
                        badge_html_parts.append(f"<span title='{field_name}' style='display:inline-block; background-color:#e0e0e0; color:#333; padding:2px 8px; margin:2px; border-radius:10px; font-size:0.85em;'>{icons.get(field_name, 'ℹ️')} {val}</span>")
                if badge_html_parts:
                    st.markdown(" ".join(badge_html_parts), unsafe_allow_html=True)

            # --- Display formatted_sentence ---
            # st.markdown("---")  # 視覺分隔線
            # current_data_index = exp_item.get("data_index")
            # sentence_info = next(
            #     (s for s in all_sentence_data if s.get("index") == current_data_index), None
            # )
            if not sentence_info:
                st.warning("Formatted sentence not available for this item.")
                st.markdown("---")  # Separator for the next explanation item
                continue

            formatted_sentence_original = sentence_info.get("formatted_sentence")
            if formatted_sentence_original:
                line1_html, line2_html = process_formatted_sentence(formatted_sentence_original)

                # st.markdown("##### Sentence Comparison:")  # 小標題
                # 應用 monospace 字體並確保空白被正確處理
                # 使用 <div> 並添加 white-space: pre-wrap; 樣式
                st.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap; text-align: center;">{line1_html}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div style="font-family: monospace; white-space: pre-wrap; text-align: center;">{line2_html}</div>',
                    unsafe_allow_html=True
                )
                # st.markdown(
                #     f'<div style="font-family: Consolas, \'Courier New\', monospace; white-space: pre-wrap; text-align: left; padding: 5px; border: 1px solid #e0e0e0; border-radius: 4px; background-color: #f9f9f9;">{line1_html}<br>{line2_html}</div>',
                #     unsafe_allow_html=True
                # )
            else:
                st.warning("Formatted sentence not available for this item.")

            # --- Display content_payload ---
            payload = exp_item.get("content_payload", {})
            if payload:
                st.markdown("---")  # Visual separator
                if "explanation_en" in payload:
                    with st.expander("Explanation", expanded=True):
                        st.write(payload["explanation_en"])
                        if "example_en" in payload and payload["example_en"]:
                            st.markdown("**Examples:**")
                            for ex_en in payload["example_en"]:
                                st.markdown(f"- {ex_en}")

                if "explanation_zh_tw" in payload:
                    with st.expander("解釋", expanded=True):
                        st.write(payload["explanation_zh_tw"])
                        if "example_zh_tw" in payload and payload["example_zh_tw"]:
                            st.markdown("**範例:**")
                            for ex_zh in payload["example_zh_tw"]:
                                st.markdown(f"- {ex_zh}")

                if "corresponding_collocation_en" in payload:
                    with st.expander("Corresponding Collocation", expanded=True):
                        if len(payload["corresponding_collocation_en"]):
                            print(f"@corresponding_collocation_en: {payload['corresponding_collocation_en']}")
                            for collocation_category in payload["corresponding_collocation_en"]:
                                st.markdown(collocation_category)
                            if "corresponding_collocation_examples_en" in payload and payload["corresponding_collocation_examples_en"]:
                                print(f"@corresponding_collocation_examples_en: {payload['corresponding_collocation_examples_en']}")
                                st.markdown("**Examples:**")
                                for ex_en in payload["corresponding_collocation_examples_en"]:
                                    st.markdown(f"- {ex_en}")

                if "other_collocations_en" in payload:
                    with st.expander("Other Collocations", expanded=True):
                        if len(payload["other_collocations_en"]):
                            print(f"@other_collocations_en: {payload['other_collocations_en']}")
                            st.markdown("**Collocations:**")
                            # Display each collocation category
                            for collocation_category in payload["other_collocations_en"]:
                                st.markdown(collocation_category)
                            if "other_collocations_examples_en" in payload and payload["other_collocations_examples_en"]:
                                print(f"@other_collocations_examples_en: {payload['other_collocations_examples_en']}")
                                st.markdown("**Examples:**")
                                for ex_en in payload["other_collocations_examples_en"]:
                                    st.markdown(f"- {ex_en}")

            else:
                st.warning("No content payload available for this entry.")


        st.markdown("---")


# --- Main Application ---
def main_viewer_page_content():
    st.title("📄 WhyFix: Explanation Viewer")

    # Provided data strings
    sentence_info_str = ""
    explanation_info_str = ""

    # thesis-system/src/data/batch/df_examples_updated_0524-1106.jsonl
    # thesis-system/src/data/batch/structured_api_data_0524-1106.jsonl

    # src/data/batch/structured_api_data_fce_t_eighty_0622-1458-fix.jsonl

    with open("../data/batch/df_examples_updated_0622-1458.jsonl", 'r', encoding='utf-8') as f:
        sentence_info_str = f.read()
    # with open("../data/batch/structured_api_data_0622-1458.jsonl", 'r', encoding='utf-8') as f:
        # explanation_info_str = f.read()
    structured_data_folder = "../data/batch"
    files = os.listdir(structured_data_folder)
    for file_name in files:
        if file_name.startswith("structured_api_data_") and file_name.endswith("fix.jsonl"):
            with open(os.path.join(structured_data_folder, file_name), 'r', encoding='utf-8') as f:
                explanation_info_str = f.read()
            # break

    # Load data
    sentence_data_full = load_data_from_string(sentence_info_str)
    explanation_data = load_data_from_string(explanation_info_str)

    if sentence_data_full is None or explanation_data is None:
        # Errors handled within load_jsonl_data, main function can stop if critical data is missing.
        st.error("Failed to load data. Please check the input format.")
        return
    if not sentence_data_full:
        st.error("Failed to load sentence data. Please check the input format.")
        return
    if not explanation_data:
        st.error("Failed to load explanation data. Please check the input format.")
        return

    # Select data source
    data_source_display_map = {
        "FCE dataset": "fce-dataset",
        "Longman Dictionary of Common Errors": "longman-dictionary-of-common-errors"
    }
    # data_source_to_explanation_filter_map = {
    #     "fce-dataset": "fce",
    #     "longman-dictionary-of-common-errors": "longman"
    # }
    displayed_data_source_options = list(data_source_display_map.keys())
    if not displayed_data_source_options:
        st.error("設定錯誤：沒有定義可用的資料來源選項。")
        return

    selected_data_source_display_name = st.selectbox(
        "Select Data Source:",
        options=displayed_data_source_options, index=0, key="data_source_selector"
    )
    selected_data_source_internal = data_source_display_map.get(selected_data_source_display_name)

    active_sentence_data = []
    if selected_data_source_internal and sentence_data_full:
        active_sentence_data = [s for s in sentence_data_full if s.get("data_source") == selected_data_source_internal]
        if not active_sentence_data:
            st.warning(f"在 '{selected_data_source_display_name}' 資料來源下未找到任何句子資料。")
    elif not sentence_data_full:
        st.warning("句子資料未載入或為空。")

    # sentences_with_indices = get_formatted_sentences_and_indices(active_sentence_data)

    # Prepare data for UI
    # sentences_with_indices = get_learner_sentences_and_indices(sentence_data)
    sentences_with_indices = get_formatted_sentences_and_indices(active_sentence_data)
    unique_filter_options = get_unique_values_for_filters(
        explanation_data,
        ["method", "input_type", "output_length"]
    )
    # If sentences_with_indices is empty, then selectbox will fail
    if not sentences_with_indices and active_sentence_data:  # sentence_data might exist but no valid items extracted
        st.error("Could not find any valid sentences with indices.")
        return  # Stop if no sentences can be selected

    # Display selection interface
    selected_sentence_text, selected_filters = display_selection_interface(
        sentences_with_indices,
        unique_filter_options
    )

    # Process and display results
    if selected_sentence_text:
        target_data_index = find_sentence_index(selected_sentence_text, sentences_with_indices)
        if target_data_index is not None:
            filtered_explanations = filter_explanations(explanation_data, target_data_index, selected_data_source_internal, selected_filters)
            display_results(filtered_explanations, sentence_data_full)
        else:
            st.error("Could not find the selected sentence's index. Data might be inconsistent.")
    else:
        st.info("Please select a sentence to see explanations.")


if __name__ == "__main__":
    # APP_DIR_MV = os.path.dirname(os.path.abspath(__file__))
    # SRC_DIR_MV = os.path.join(APP_DIR_MV, '..')
    # if SRC_DIR_MV not in sys.path:
    #     sys.path.append(SRC_DIR_MV)
    if SRC_DIR_MV not in sys.path:  # Use SRC_DIR_MV from the top
        sys.path.insert(0, SRC_DIR_MV)
    main_viewer_page_content()

else:
    main_viewer_page_content()
