import os
import json
from pandas import pd

from utils.files_io import (
    FilePath,
    load_method_config,
)


CUSTOM_ID_FIELD_NAMES = [
    "dataset", "sample_size", "data_index", "method",
    "model_type", "model_temperature", "input_type",
    "model_role_setting", "output_length"
]
PATHS = FilePath()
experimentDoc = load_method_config("llm")
METHODS = experimentDoc['methods']
MODELS = experimentDoc['model_type']['gpt']
MODEL_TEMPERATURES = experimentDoc['temp']
INPUT_TYPES = experimentDoc['input_sent']
OUTPUT_LENGTHS = experimentDoc['output_len']

DATE = ""
if len(os.sys.argv) > 1:
    DATE = os.sys.argv[1]
else:
    raise ValueError("Please provide the date for batch results as a command line argument.")


def add_batch_results_to_df(df, folder_batch_result, column_name):
    """
    customID: f"{datasetType}_{idx}_L2_fourOneNano_zero_{sentenceType}_linguist_{outputLength}"
    idx: df.index
    """
    # 1. 讀取 batchResult.jsonl
    # path_batch_result = PATHS.folderPath_batchResult + folder_batch_result + "batchResult.json"
    path_batch_result = os.path.join(PATHS.folderPath_batchResult, folder_batch_result, "batchResult.jsonl")
    results = {}
    with open(path_batch_result, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):  # 跳過空行或註解
                continue
            obj = json.loads(line)
            custom_id = obj['custom_id']
            # 2. 解析 index
            # custom_id 格式: {datasetType}_{idx}_L2_fourOneNano_zero_{sentenceType}_linguist_{outputLength}
            idx = int(custom_id.split('_')[2])
            # 3. 取得你要的內容（假設是 matched）
            matched = []
            try:
                matched = json.loads(obj['response']['body']['output'][0]['content'][0]['text'])['matched']
            except Exception:
                pass
            results[idx] = matched

    # 4. 寫入 DataFrame
    df[column_name] = df.index.map(lambda idx: results.get(idx, []))
    return df


def update_df_with_knownledge(df, date_batch_result):
    df_updated = df.copy()
    df_updated = add_batch_results_to_df(
        df_updated,
        f"ragL2_causes-{date_batch_result}",
        'causes'
    )
    df_updated = add_batch_results_to_df(
        df_updated,
        f"ragL2_academic_writing-{date_batch_result}",
        'academic_writing'
    )
    return df_updated


def extract_one_batch_jsonl(folder_batch_result):
    # 按行分割輸入的文字 (JSONL 格式)
    path_batch_result = os.path.join(PATHS.folderPath_batchResult, folder_batch_result, "batchResult.jsonl")
    structured_data = []  # 儲存所有結構化結果的列表
    with open(path_batch_result, 'r') as f:
        for line in f:
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
                    if len(custom_id_parts) == len(CUSTOM_ID_FIELD_NAMES):
                        for i, field_name in enumerate(CUSTOM_ID_FIELD_NAMES):
                            if field_name == "dataset":
                                if custom_id_parts[i] == "lg":
                                    structured_entry[field_name] = "longman"
                                elif custom_id_parts[i] == "fce":
                                    structured_entry[field_name] = "fce"
                            elif field_name == "sample_size":
                                if custom_id_parts[i].startswith("sp"):
                                    structured_entry[field_name] = custom_id_parts[i].replace("sp", "small_sample_")
                                else:
                                    structured_entry[field_name] = custom_id_parts[i]
                            elif field_name == "data_index":  # data_index 嘗試轉為整數
                                try:
                                    structured_entry[field_name] = int(custom_id_parts[i])
                                except ValueError:
                                    structured_entry[field_name] = custom_id_parts[i]  # 若轉換失敗則保留字串
                                    print(f"資訊：custom_id '{custom_id}' 中的 data_index '{custom_id_parts[i]}' 無法轉換為整數，將保留為字串。")
                            elif field_name == "method":
                                structured_entry[field_name] = METHODS[custom_id_parts[i]]
                            elif field_name == "model_type":
                                structured_entry[field_name] = MODELS[custom_id_parts[i]]
                            elif field_name == "model_temperature":
                                structured_entry[field_name] = MODEL_TEMPERATURES[custom_id_parts[i]]
                            elif field_name == "input_type":
                                structured_entry[field_name] = INPUT_TYPES[custom_id_parts[i]]
                            elif field_name == "output_length":
                                structured_entry[field_name] = OUTPUT_LENGTHS[custom_id_parts[i]]
                            else:
                                structured_entry[field_name] = custom_id_parts[i]
                    else:
                        msg = f"custom_id '{custom_id}' 的部分數量 ({len(custom_id_parts)}) 與預期 ({len(CUSTOM_ID_FIELD_NAMES)}) 不符。"
                        print(f"警告：{msg}")
                        structured_entry["custom_id_parsing_error"] = msg
                        structured_entry["raw_custom_id_parts"] = custom_id_parts
                else:
                    structured_entry["custom_id_parsing_error"] = "custom_id 欄位缺失"
                    print(f"警告：行 '{line[:100]}...' 中缺少 custom_id。")

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
                                        msg = f"無法解析 custom_id '{custom_id}' 中的 text 欄位JSON：{e_inner}"
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


def update_df_with_knownledge_fce_and_longman():
    # Store results from different methods to one json file
    first_three_folder = f"first_three_methods-{DATE}"
    ragL2_explanation_folder = f"ragL2_explanation-{DATE}"

    first_three_structured_data = extract_one_batch_jsonl(first_three_folder)
    ragL2_explanation_structured_data = extract_one_batch_jsonl(ragL2_explanation_folder)
    all_structured_data = first_three_structured_data + ragL2_explanation_structured_data
    # 將最終結果轉換為格式化的 JSON 字串 (用於預覽或調試)
    output_json_string_preview = json.dumps(all_structured_data, indent=2, ensure_ascii=False)
    df_structured_data = pd.DataFrame(all_structured_data)
    print("\n--- 預覽結果 ---")
    print(output_json_string_preview)
    # 將結果儲存到新的 JSON 檔案
    output_filename = os.path.join(PATHS.folderPath_structured_data, "structured_api_data.json")
    output_filename_app = os.path.join(PATHS.folderPath_app, "structured_api_data_app.json")
    df_structured_data.to_json(output_filename, orient='records', lines=True, force_ascii=False)
    df_structured_data.to_json(output_filename_app, orient='records', lines=True, force_ascii=False)
    print(f"\n結構化結果已成功儲存到 {output_filename}")

    # ---- test ----
    # df_longman_sample = pd.read_json(
    #     "/home/nlplab/atwolin/thesis/data/fce-released-dataset/data/fce_sample_with_collocation.json"
    # ).loc[0:1]
    # df_longman_updated = update_df_with_knownledge(df_longman_sample, date_batch_result=DATE)
    # output_filename_updated = os.path.join(PATHS.folderPath_structured_data, "df_examples_updated.json")
    # df_longman_updated.to_json(output_filename_updated, orient='records', lines=True, force_ascii=False)
    # print(f"\n更新後的 DataFrame 已成功儲存到 {output_filename_updated}")
    # return

    # ---- experiment ----
    df_fce_sample = pd.read_csv(PATHS.filePath_fce_sample_withCollocation)
    df_fce_sample.index = df_fce_sample['index']
    df_longman_sample = pd.read_csv(PATHS.filePath_longman_sample_withCollocation)
    df_longman_sample.index = df_longman_sample['index']

    df_fce_updated = update_df_with_knownledge(df_fce_sample, date_batch_result=DATE)
    df_longman_updated = update_df_with_knownledge(df_longman_sample, date_batch_result=DATE)

    # Store updated DataFrames to JSON files
    df_updated = pd.concat([df_fce_updated, df_longman_updated], ignore_index=True)
    output_filename_updated = os.path.join(PATHS.folderPath_structured_data, "df_examples_updated.json")
    df_updated.to_json(output_filename_updated, orient='records', lines=True, force_ascii=False)
    print(f"\n更新後的 DataFrame 已成功儲存到 {output_filename_updated}")


def update_df():
    pass


if __name__ == "__main__":
    update_df_with_knownledge_fce_and_longman()
