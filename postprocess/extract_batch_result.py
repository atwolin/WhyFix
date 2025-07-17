import os
import sys
import pandas as pd
import json

from model.llm_setup import (
    load_experiment,
    FilePath,
)

from model.methods_combine_step3 import (
    update_df_with_knownledge
)
from postprocess.evaluation_data_extract import (
    convert_data_to_source_blocks,
)

# Define file paths and parameters
LONGMAN_SAMPLE_TYPE = None
EMBEDDING_SIZE = None
ROLE = 'linguist'
DATE = ""
if len(sys.argv) > 3:
    LONGMAN_SAMPLE_TYPE = sys.argv[1]
    EMBEDDING_SIZE = sys.argv[2]
    DATE = sys.argv[3]
else:
    raise ValueError("Please provide the date for batch results as a command line argument.")

PATHS = FilePath(EMBEDDING_SIZE)

# Load experiment configuration
experimentDoc = load_experiment()


# 定義 custom_id 各部分對應的欄位名稱
custom_id_field_names = [
    "dataset", "sample_size", "data_index", "method",
    "model_type", "model_temperature", "input_type",
    "model_role_setting", "output_length"
]

# These lookups are derived from experimentDoc
METHODS = experimentDoc.get('methods', {})
MODELS = experimentDoc.get('model_names', {})
MODEL_TEMPERATURES = experimentDoc.get('temp', {})
INPUT_TYPES = experimentDoc.get('input_sent', {})
ROLE_SETTING = experimentDoc.get('role_names', {})
OUTPUT_LENGTHS = experimentDoc.get('output_len', {})


def extract_one_batch_jsonl(folder_batch_result):
    """
    Parses a JSONL batch result file.

    Args:
        folder_batch_result (str): The name of the folder (relative to PATHS.folderPath_batchResult)
                                   containing the 'batchResult.jsonl' file.

    Returns:
        list: A list of dictionaries, where each dictionary contains the parsed data
              from one line of the JSONL file.
    """
    # 按行分割輸入的文字 (JSONL 格式)
    path_batch_result = os.path.join(PATHS.folderPath_batchResult, folder_batch_result, "batchResult.jsonl")
    structured_data = []  # 儲存所有結構化結果的列表

    print(f"INFO: Attempting to read batch results from: {path_batch_result}")
    if not os.path.exists(path_batch_result):
        print(f"ERROR: File not found: {path_batch_result}")
        return structured_data  # Return empty list if file doesn't exist

    with open(path_batch_result, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            # 跳過可能的空行
            if not line.strip():
                continue

            structured_entry = {}  # 為每一行初始化一個新的 structured_entry
            custom_id = "Unknown"  # 初始化 custom_id 以便在錯誤日誌中使用

            try:
                # 解析每一行的 JSON 字串
                data = json.loads(line)
                custom_id = data.get("custom_id")
                structured_entry["original_custom_id"] = custom_id

                # 1. 解析 custom_id
                if custom_id:
                    custom_id_parts = custom_id.split('_')
                    if len(custom_id_parts) == len(custom_id_field_names):
                        for i, field_name in enumerate(custom_id_field_names):
                            part_value = custom_id_parts[i]
                            if field_name == "dataset":
                                if part_value == "lg":
                                    structured_entry[field_name] = "longman"
                                elif part_value == "fce":
                                    structured_entry[field_name] = "fce"
                                else:
                                    structured_entry[field_name] = part_value
                            elif field_name == "sample_size":
                                if part_value.startswith("sp"):
                                    structured_entry[field_name] = part_value.replace("sp", "30_samples")
                                elif part_value.startswith("fu"):
                                    structured_entry[field_name] = part_value.replace("fu", "full")
                                else:
                                    structured_entry[field_name] = part_value
                            elif field_name == "data_index":  # data_index 嘗試轉為整數
                                try:
                                    structured_entry[field_name] = int(part_value)
                                except ValueError:
                                    structured_entry[field_name] = part_value  # 若轉換失敗則保留字串
                                    print(f"INFO: custom_id '{custom_id}' (line {line_number}) - data_index '{part_value}' could not be converted to int, kept as string.")
                            elif field_name == "method":
                                structured_entry[field_name] = METHODS.get(part_value, part_value)  # Use .get for safety
                            elif field_name == "model_type":
                                structured_entry[field_name] = MODELS.get(part_value, part_value)
                            elif field_name == "model_temperature":
                                structured_entry[field_name] = MODEL_TEMPERATURES.get(part_value, part_value)
                            elif field_name == "input_type":
                                structured_entry[field_name] = INPUT_TYPES.get(part_value, part_value)
                            elif field_name == "model_role_setting":
                                structured_entry[field_name] = ROLE_SETTING.get(part_value, part_value)
                            elif field_name == "output_length":
                                structured_entry[field_name] = OUTPUT_LENGTHS.get(part_value, part_value)
                            else:
                                structured_entry[field_name] = part_value
                    else:
                        msg = f"custom_id '{custom_id}' (line {line_number}) parts count ({len(custom_id_parts)}) does not match expected ({len(custom_id_field_names)})."
                        print(f"WARNING: {msg}")
                        structured_entry["custom_id_parsing_error"] = msg
                        structured_entry["raw_custom_id_parts"] = custom_id_parts  # Store raw parts for debugging
                        # Fill with None or "Error" for expected fields if parsing fails partially
                        for fn in custom_id_field_names:
                            if fn not in structured_entry:
                                structured_entry[fn] = "ErrorInParsingCustomId"
                else:
                    structured_entry["custom_id_parsing_error"] = "custom_id field is missing"
                    print(f"WARNING: custom_id missing in line {line_number}: '{line[:100]}...'")
                    # Fill with None or "Error" for expected fields if custom_id is missing
                    for fn in custom_id_field_names:
                        structured_entry[fn] = "MissingCustomId"

                # 2. 提取 response 中的 text 內容
                text_content_payload = None  # 初始化
                response_data = data.get("response")

                if response_data and response_data.get("status_code") == 200:
                    body = response_data.get("body")
                    if body and body.get("output") and isinstance(body.get("output"), list) and len(body.get("output")) > 0:
                        first_output = body.get("output")[0]
                        if first_output.get("content") and isinstance(first_output.get("content"), list) and len(first_output.get("content")) > 0:
                            first_content = first_output.get("content")[0]
                            if first_content.get("type") == "output_text" and "text" in first_content:
                                text_json_string = first_content.get("text")
                                if isinstance(text_json_string, str):
                                    try:
                                        text_content_payload = json.loads(text_json_string)
                                    except json.JSONDecodeError as e_inner:

                                        msg = f"無法解析 custom_id '{custom_id}' 中的 text 欄位JSON：{e_inner} (line {line_number}): {e_inner}"
                                        print(f"警告：{msg}")
                                        text_content_payload = {"error": msg, "original_text": text_json_string}
                                else:
                                    msg = f"custom_id '{custom_id}' 中的 text 欄位不是有效的字串 (實際類型: {type(text_json_string)})。"
                                    print(f"警告：{msg}")
                                    text_content_payload = {"error": msg}
                            else:
                                text_content_payload = {"error": "在第一個 content 項目中找不到 text 欄位或類型不為 output_text"}
                        else:
                            text_content_payload = {"error": "第一個 output 項目的 content 列表為空或無效"}
                    else:
                        text_content_payload = {"error": "response body 的 output 列表為空或無效"}
                elif response_data:  # status_code 不是 200 或其他 response 錯誤
                    error_info = {"error": f"Response 狀態非 200 或資料缺失 (status: {response_data.get('status_code')})"}
                    if response_data.get("error"):  # OpenAI Batch API 可能直接在 response 下提供 error
                        error_info["api_error_details"] = response_data.get("error")
                    text_content_payload = error_info
                else:  # response 欄位本身缺失
                    text_content_payload = {"error": "Response 欄位缺失"}

                structured_entry["content_payload"] = text_content_payload
                structured_data.append(structured_entry)

            except json.JSONDecodeError as e_outer:
                print(f"嚴重警告：無法解析整行JSON：{e_outer} - 行內容：{line[:100]}...")
                structured_data.append({
                    "original_custom_id": "ErrorInOuterJSONParsing",
                    "line_content_snippet": line[:200],
                    "parsing_error": str(e_outer),
                    "content_payload": {"error": "無法解析最外層的JSON行"}
                })
            except Exception as e_general:  # 捕獲其他未預期的錯誤
                print(f"嚴重警告：處理 custom_id '{custom_id}' (或未知 custom_id，若在提取前出錯) 時發生未預期錯誤：{e_general} - 行: {line[:100]}")
                structured_data.append({
                    "original_custom_id": custom_id if custom_id != "Unknown" else "ErrorBeforeCustomIdExtraction",
                    "line_content_snippet": line[:200],
                    "unexpected_error": str(e_general),
                    "content_payload": {"error": "處理過程中發生未預期錯誤"}
                })
    return structured_data


def main():
    """
    Main function to orchestrate the parsing of batch results,
    structuring the data, saving it, and then updating sample
    DataFrames with this new knowledge.
    """
    print(f"INFO: Starting processing for date: {DATE}")
    all_structured_data = []

    # Convert the list of dictionaries to a Pandas DataFrame
    df_fce = pd.read_csv(PATHS.filePath_fce_dictInfo_filtered)
    df_longman = pd.read_csv(PATHS.filePath_longman_dictInfo_filtered)
    # df_fce = df_fce.set_index('index')
    # df_longman = df_longman.set_index('index')

    datasets = {"lg_fu": 'df_longman', "fce_fu": 'df_fce'}
    sentenceTypes = ["t", "tf"]
    outputLenTypes = ["fifty", "eighty"]
    for datasetType in datasets.keys():
        for sentenceType in sentenceTypes:
            if datasetType.startswith("l") and sentenceType == "tf":
                continue
            for outputLength in outputLenTypes:
                print(f"Processing dataset: {datasetType}, sentenceType: {sentenceType}, outputLength: {outputLength}")
                all_structured_data = []
                first_three_folder = f"first_three_methods_{datasetType}_{sentenceType}_{outputLength}-{DATE}"
                all_structured_data += extract_one_batch_jsonl(first_three_folder)

                part_2_folder = f"part2_{datasetType}_{sentenceType}_{outputLength}-{DATE}"
                all_structured_data += extract_one_batch_jsonl(part_2_folder)

                print(f"INFO: Total records extracted from batch results: {len(all_structured_data)}")

                if not all_structured_data:
                    print("WARNING: No data extracted from batch result files. Exiting.")
                    return  # Exit if no data to process

                df_structured_data = pd.DataFrame(all_structured_data)

                # 為每筆資料尋找對應的 formatted_sentence
                for idx, row in df_structured_data.iterrows():
                    dataset = row['dataset']
                    data_index = row['data_index']

                    if dataset == 'fce':
                        matching_rows = df_fce[df_fce['index'].astype(int) == data_index]
                        if not matching_rows.empty:
                            df_structured_data.at[idx, 'formatted_sentence'] = (
                                matching_rows['formatted_sentence'].iloc[0] + " + " + matching_rows['following_sentence'].iloc[0]
                                if isinstance(matching_rows['following_sentence'].iloc[0], str)
                                else matching_rows['formatted_sentence'].iloc[0]
                            )
                    else:  # dataset == 'longman'
                        matching_rows = df_longman[df_longman['index'].astype(int) == data_index]
                        if not matching_rows.empty:
                            df_structured_data.at[idx, 'formatted_sentence'] = matching_rows['formatted_sentence'].iloc[0]

                # 將結果儲存到新的 JSON 檔案
                try:
                    output_filename = os.path.join(PATHS.folderPath_structured_data, f"structured_api_data_{dataset}_{sentenceType}_{outputLength}_{DATE}.jsonl")
                    df_structured_data.to_json(output_filename, orient='records', lines=True, force_ascii=False)
                    print(f"\n結構化結果已成功儲存到 {output_filename}\n\n")
                except IOError as e:
                    print(f"\nERROR: Could not write to file. Error: {e}")

    # # Store results from different methods to one json file
    # first_three_folder = f"first_three_methods-{DATE}"
    # part_2_folder = f"part2-{DATE}"

    # first_three_structured_data = extract_one_batch_jsonl(first_three_folder)
    # part_2_structured_data = extract_one_batch_jsonl(part_2_folder)
    # all_structured_data = first_three_structured_data + part_2_structured_data
    # print(f"INFO: Total records extracted from batch results: {len(all_structured_data)}")

    # if not all_structured_data:
    #     print("WARNING: No data extracted from batch result files. Exiting.")
    #     return  # Exit if no data to process

    # df_structured_data = pd.DataFrame(all_structured_data)

    # # 為每筆資料尋找對應的 formatted_sentence
    # for idx, row in df_structured_data.iterrows():
    #     dataset = row['dataset']
    #     data_index = row['data_index']
    #     if dataset == 'fce':
    #         matching_rows = df_fce[df_fce['index'].astype(int) == data_index]
    #         if not matching_rows.empty:
    #             df_structured_data.at[idx, 'formatted_sentence'] = (
    #                 matching_rows['formatted_sentence'].iloc[0] + " + " + matching_rows['following_sentence'].iloc[0]
    #                 if isinstance(matching_rows['following_sentence'].iloc[0], str)
    #                 else matching_rows['formatted_sentence'].iloc[0]
    #             )
    #     else:  # dataset == 'longman'
    #         matching_rows = df_longman[df_longman['index'].astype(int) == data_index]
    #         if not matching_rows.empty:
    #             df_structured_data.at[idx, 'formatted_sentence'] = matching_rows['formatted_sentence'].iloc[0]

    # # 將結果儲存到新的 JSON 檔案
    # try:
    #     output_filename = os.path.join(PATHS.folderPath_structured_data, f"structured_api_data_{DATE}.jsonl")
    #     df_structured_data.to_json(output_filename, orient='records', lines=True, force_ascii=False)
    #     print(f"\n結構化結果已成功儲存到 {output_filename}")
    # except IOError as e:
    #     print(f"\nERROR: Could not write to file. Error: {e}")

    # source_data_blocks = convert_data_to_source_blocks()
    # print(f"INFO: Extracted {len(source_data_blocks)} data blocks for evaluation.")
    # # Optionally, save the data to a Python file
    # with open(f'/home/nlplab/atwolin/thesis/data/results/source_data_blocks_{DATE}.py', 'w') as f:
    #     f.write("source_data_blocks = ")
    #     f.write(repr(source_data_blocks))

    # with open(f'/home/nlplab/atwolin/thesis/data/results/source_data_blocks_{DATE}.json', 'w') as f:
    #     json.dump(source_data_blocks, f, indent=4, ensure_ascii=False)

    # return

    # --- Extract this new jsonl file to format for evaluation ---
    # evaluation_data_blocks = create_source_data_blocks(output_filename)
    # print(f"INFO: Extracted {len(evaluation_data_blocks)} data blocks for evaluation.")
    # with open(os.path.join(PATHS.folderPath_structured_data, f"source_data_blocks_{DATE}.json"), "w", encoding="utf-8") as f:
    #     json.dump(evaluation_data_blocks, f, indent=4, ensure_ascii=False)
    # print(f"INFO: Evaluation data blocks saved to source_data_blocks_{DATE}.py")

    # --- Load DataFrames and update them ---
    print("\nINFO: Loading data...")

    # ---- Test ----
    # df_longman_sample = pd.read_json(
    #     "/home/nlplab/atwolin/thesis/data/fce-released-dataset/data/fce_sample_with_collocation.json"
    # ).loc[0:1]
    # df_longman_updated = update_df_with_knownledge(df_longman_sample, date_batch_result=DATE)
    # output_filename_updated = os.path.join(PATHS.folderPath_structured_data, "df_examples_updated.json")
    # df_longman_updated.to_json(output_filename_updated, orient='records', lines=True, force_ascii=False)

    # Complete wet
    # df_test = pd.read_json(PATHS.filePath_test_withCollocation)
    # df_test = df_test.set_index('index')
    # # Update DataFrames with knowledge from batch results
    # df_test_updated = pd.DataFrame()  # Initialize as empty
    # if not df_test.empty:
    #     print("\nINFO: Updating test sample data...")
    #     df_test_updated = update_df_with_knownledge(df_test, "fifty", date_batch_result=DATE)
    #     df_test_updated = update_df_with_knownledge(df_test_updated, "eighty", date_batch_result=DATE)
    #     print(f"INFO: test data update complete. Resulting test DataFrame has {len(df_test_updated)} records.")
    # else:
    #     print("INFO: Skipping test data update as test sample data was not loaded.")
    # print(f"INFO: Preparing to concatenate {len(df_test_updated)} updated DataFrames.")
    # if len(df_test_updated) > 0:
    #     output_filename_updated = os.path.join(PATHS.folderPath_structured_data, f"df_examples_updated_{DATE}.jsonl")  # Changed to .jsonl
    #     try:
    #         df_test_updated.to_json(output_filename_updated, orient='records', lines=True, force_ascii=False)
    #         print(f"\nINFO: 更新後的 DataFrame 已成功儲存到 {output_filename_updated}")
    #     except IOError as e:
    #         print(f"\nERROR: Could not write updated DataFrame to file {output_filename_updated}. Error: {e}")
    #     except Exception as e:  # Catch any other exceptions during save
    #         print(f"\nERROR: An unexpected error occurred while saving updated DataFrame: {e}")
    # else:
    #     print("INFO: No updated data to save (test may not have been processed or were empty).")
    # return

    # Test two sample from full datasets
    # df_fce = pd.read_json(PATHS.filePath_fce_withCollocation)
    # df_fce = df_fce.loc[0:1].copy()  # Apply slicing
    # df_longman = pd.read_json(PATHS.filePath_longman_withCollocation)
    # df_longman = df_longman.loc[0:1].copy()  # Apply slicing
    # df_fce = df_fce.set_index('index')
    # df_longman = df_longman.set_index('index')

    # # Update DataFrames with knowledge from batch results
    # df_fce_updated = pd.DataFrame()  # Initialize as empty
    # if not df_fce.empty:
    #     print("\nINFO: Updating FCE full data...")
    #     df_fce_updated = update_df_with_knownledge(df_fce, "fifty", date_batch_result=DATE)
    #     df_fce_updated = update_df_with_knownledge(df_fce_updated, "eighty", date_batch_result=DATE)
    #     print(f"INFO: FCE data update complete. Resulting FCE DataFrame has {len(df_fce_updated)} records.")
    # else:
    #     print("INFO: Skipping FCE data update as FCE full data was not loaded.")

    # df_longman_updated = pd.DataFrame()  # Initialize as empty
    # if not df_longman.empty:
    #     print("\nINFO: Updating Longman full data...")
    #     df_longman_updated = update_df_with_knownledge(df_longman, "fifty", date_batch_result=DATE)
    #     df_longman_updated = update_df_with_knownledge(df_longman_updated, "eighty", date_batch_result=DATE)
    #     # print(f"INFO: {df_longman_updated.columns}")
    #     print(f"INFO: Longman data update complete. Resulting Longman DataFrame has {len(df_longman_updated)} records.")
    # else:
    #     print("INFO: Skipping Longman data update as Longman full data was not loaded.")
    # print(f"INFO: FCE updated records: {len(df_fce_updated)}, Longman updated records: {len(df_longman_updated)}")

    # ---- Main ----
    # df_fce_sample = pd.read_json(PATHS.filePath_fce_sample_withCollocation)
    # df_fce_sample = df_fce_sample.set_index('index')
    # if LONGMAN_SAMPLE_TYPE == 'R':
    #     df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_one_replace_withCollocation)
    # else:
    #     df_longman_sample = pd.read_json(PATHS.filePath_longman_sample_withCollocation)
    # df_longman_sample = df_longman_sample.set_index('index')

    # # Update DataFrames with knowledge from batch results
    # df_fce_updated = pd.DataFrame()  # Initialize as empty
    # if not df_fce_sample.empty:
    #     print("\nINFO: Updating FCE sample data...")
    #     df_fce_updated = update_df_with_knownledge(df_fce_sample, "fifty", date_batch_result=DATE)
    #     df_fce_updated = update_df_with_knownledge(df_fce_updated, "eighty", date_batch_result=DATE)
    #     print(f"INFO: FCE data update complete. Resulting FCE DataFrame has {len(df_fce_updated)} records.")
    # else:
    #     print("INFO: Skipping FCE data update as FCE sample data was not loaded.")

    # df_longman_updated = pd.DataFrame()  # Initialize as empty
    # if not df_longman_sample.empty:
    #     print("\nINFO: Updating Longman sample data...")
    #     df_longman_updated = update_df_with_knownledge(df_longman_sample, "fifty", date_batch_result=DATE)
    #     df_longman_updated = update_df_with_knownledge(df_longman_updated, "eighty", date_batch_result=DATE)
    #     # print(f"INFO: {df_longman_updated.columns}")
    #     print(f"INFO: Longman data update complete. Resulting Longman DataFrame has {len(df_longman_updated)} records.")
    # else:
    #     print("INFO: Skipping Longman data update as Longman sample data was not loaded.")

    # ---- Full ----
    df_fce = pd.read_json(PATHS.filePath_fce_withCollocation)
    df_longman = pd.read_json(PATHS.filePath_longman_withCollocation)
    df_fce = df_fce.set_index('index')
    df_longman = df_longman.set_index('index')

    # Update DataFrames with knowledge from batch results
    df_fce_updated = pd.DataFrame()  # Initialize as empty
    if not df_fce.empty:
        datasets = {"lg_fu": 'df_longman', "fce_fu": 'df_fce'}
        sentenceTypes = ["t", "tf"]
        outputLenTypes = ["fifty", "eighty"]
        for dataset in datasets.keys():
            # df_examples = datasets[dataset]
            for sentenceType in sentenceTypes:
                if dataset.startswith("l") and sentenceType == "tf":
                    continue
                for outputLength in outputLenTypes:
                    print(f"Processing dataset: {dataset}, sentenceType: {sentenceType}, outputLength: {outputLength}")
                    df_fce_updated = update_df_with_knownledge(df_fce, dataset, sentenceType, outputLength, date_batch_result=DATE)

        # # print("\nINFO: Updating FCE full data...")
        # df_fce_updated = update_df_with_knownledge(df_fce, "", "", "fifty", date_batch_result=DATE)
        # df_fce_updated = update_df_with_knownledge(df_fce_updated, "", "", "eighty", date_batch_result=DATE)
        print(f"INFO: FCE data update complete. Resulting FCE DataFrame has {len(df_fce_updated)} records.")
    else:
        print("INFO: Skipping FCE data update as FCE full data was not loaded.")

    df_longman_updated = pd.DataFrame()  # Initialize as empty
    if not df_longman.empty:
        datasets = {"lg_fu": 'df_longman', "fce_fu": 'df_fce'}
        sentenceTypes = ["t", "tf"]
        outputLenTypes = ["fifty", "eighty"]
        for dataset in datasets.keys():
            # df_examples = datasets[dataset]
            for sentenceType in sentenceTypes:
                if dataset.startswith("l") and sentenceType == "tf":
                    continue
                for outputLength in outputLenTypes:
                    print(f"Processing dataset: {dataset}, sentenceType: {sentenceType}, outputLength: {outputLength}")
                    df_longman_updated = update_df_with_knownledge(df_longman, dataset, sentenceType, outputLength, date_batch_result=DATE)

        # print("\nINFO: Updating Longman full data...")
        # df_longman_updated = update_df_with_knownledge(df_longman, "", "", "fifty", date_batch_result=DATE)
        # df_longman_updated = update_df_with_knownledge(df_longman_updated, "", "", "eighty", date_batch_result=DATE)
        # print(f"INFO: {df_longman_updated.columns}")
        print(f"INFO: Longman data update complete. Resulting Longman DataFrame has {len(df_longman_updated)} records.")
    else:
        print("INFO: Skipping Longman data update as Longman full data was not loaded.")
    print(f"INFO: FCE updated records: {len(df_fce_updated)}, Longman updated records: {len(df_longman_updated)}")

    # Concatenate updated DataFrames
    # Only concatenate if both DataFrames are not empty, or handle cases where one might be.
    updated_dfs_to_concat = []
    if not df_fce_updated.empty:
        updated_dfs_to_concat.append(df_fce_updated)
    if not df_longman_updated.empty:
        # df_longman_ori = pd.read_csv(PATHS.filePath_longman_original, encoding='utf-8')
        # # Combine the explanation columns - handling NaN values properly
        # df_longman_updated['longman_ori'] = df_longman_ori['longman_explantion'].fillna('').astype(str)
        # Add the "see note" explanation if it exists (with a separator if both exist)
        # mask = df_longman_ori['longman_see_note_explantion'].notna() & (df_longman_ori['longman_see_note_explantion'] != '')
        # df_longman_updated.loc[mask, 'longman_ori'] += '\n\n' + df_longman_ori.loc[mask, 'longman_see_note_explantion']
        # Clean up any rows that might have just spaces or newlines
        # df_longman_updated['longman_ori'] = df_longman_updated['longman_ori'].str.strip()

        updated_dfs_to_concat.append(df_longman_updated)
    print(f"INFO: Preparing to concatenate {len(updated_dfs_to_concat)} updated DataFrames.")

    if updated_dfs_to_concat:
        df_updated = pd.concat(updated_dfs_to_concat)
        print(f"\nINFO: Combined updated data. Total records: {len(df_updated)}")

        # 修改輸出格式以匹配 0524-1106 版本
        # 1. 添加 index 欄位（如果 data_index 存在的話）
        if 'data_index' in df_updated.columns:
            df_updated['index'] = df_updated['data_index']

        # 2. 保留所有重要的分析欄位
        # 不移除任何欄位，保持完整的數據結構

        # 3. 重新命名欄位以匹配舊格式
        # 檢查是否有 'dataset' 欄位需要重新命名為 'data_source'
        if 'dataset' in df_updated.columns and 'data_source' not in df_updated.columns:
            df_updated = df_updated.rename(columns={'dataset': 'data_source'})

        # 4. 確保輸出欄位順序與 0524-1106 版本一致
        desired_columns = [
            'index', 'learner_word', 'editor_word', 'learner_sentence', 'editor_sentence',
            'formatted_sentence', 'sentence_in_dataset', 'preceding_sentence', 'following_sentence',
            'text_in_dataset', 'data_source', 'note', 'word_learner', 'lemmaWord_learner',
            'pos_learner', 'level_learner', 'definition_learner', 'examples_learner', 'in_akl_learner',
            'word_editor', 'lemmaWord_editor', 'pos_editor', 'level_editor', 'definition_editor',
            'examples_editor', 'in_akl_editor', 'error_type', 'collocations',
            'other_categories_json', 'other_categories_formatted_json',
            'causes_fce_fu_tf_eighty', 'academic_writing_fce_fu_tf_eighty',
            'causes', 'academic_writing'
        ]

        # 只保留存在的欄位
        available_columns = [col for col in desired_columns if col in df_updated.columns]
        df_updated = df_updated[available_columns]

        output_filename_updated = os.path.join(PATHS.folderPath_structured_data, f"df_examples_updated_{DATE}.jsonl")  # Changed to .jsonl
        try:
            df_updated.to_json(output_filename_updated, orient='records', lines=True, force_ascii=False)
            print(f"\nINFO: 更新後的 DataFrame 已成功儲存到 {output_filename_updated}")
            print("INFO: 輸出格式已調整為與 0524-1106 版本相容的格式")
        except IOError as e:
            print(f"\nERROR: Could not write updated DataFrame to file {output_filename_updated}. Error: {e}")
        except Exception as e:  # Catch any other exceptions during save
            print(f"\nERROR: An unexpected error occurred while saving updated DataFrame: {e}")
    else:
        print("INFO: No updated data to save (FCE and/or Longman samples may not have been processed or were empty).")


if __name__ == "__main__":
    main()
