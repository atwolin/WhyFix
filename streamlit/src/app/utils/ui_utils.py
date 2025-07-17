# src/app/ui_utils.py
import re

# 輔助函數：產生帶樣式的 HTML span
# 這個函式原先在 main_viewer.py 中，被 process_formatted_sentence 使用
def _create_html_span(text: str, span_type: str) -> str:
    """
    根據類型產生帶樣式的 HTML span。
    span_type: 'learner_delete', 'editor_insert',
               'learner_replace_from', 'editor_replace_to',
               'placeholder_general' (用於在對應行列創建空白佔位)
    """
    styles = {
        'learner_delete': 'color: #D32F2F; background-color: #FFEBEE;',
        'editor_insert': 'color: #388E3C; background-color: #E8F5E9;',
        'learner_replace_from': 'color: #D32F2F; background-color: #FFEBEE;',
        'editor_replace_to': 'color: #388E3C; background-color: #E8F5E9;',
        'placeholder_general': ('background-color: transparent; '
                                'color: transparent;')
    }
    current_style = styles.get(span_type, '')
    # 基礎樣式確保佔位符與實際內容有相似的佈局行為
    base_span_style = 'padding: 2px 3px; border-radius: 3px; display: inline-block; white-space: pre-wrap;'

    if span_type == 'placeholder_general' and not text.strip():
        if not text:
            pass

    return (f'<span style="{current_style} {base_span_style}">'
            f'{text}</span>')


# 這個函式原先在 main_viewer.py 中，現在移至此處
def process_formatted_sentence(sentence_str: str):
    """
    處理包含特殊標記的句子字串，將其轉換為兩行 HTML 字串以進行並排比較顯示。
    特殊標記格式:
    - [-delete-]
    - {+insert+}
    - [-replace_from-]{+replace_to+}
    """
    if not sentence_str:
        return "<i>無法處理空句子。</i>", ""

    # 正則表達式，用於匹配三種編輯類型
    diff_pattern = re.compile(
        r"\[-(?P<l_repl>.*?)-\]\{\+(?P<e_repl>.*?)\+\}|"  # 替換
        r"\[-(?P<l_del>.*?)-\]|"                         # 刪除
        r"\{\+(?P<e_ins>.*?)\+\}"                        # 插入
    )

    if not diff_pattern.search(sentence_str):
        # 如果沒有任何標記，則兩行都顯示原始句子 (或可選擇一行顯示原始，一行空白)
        return sentence_str, sentence_str

    learner_line_parts = []  # 學習者句子 (第一行) 的 HTML 組件
    editor_line_parts = []   # 編輯者句子 (第二行) 的 HTML 組件
    current_pos = 0          # 目前在原始句子字串中處理到的位置

    for match in diff_pattern.finditer(sentence_str):
        # 1. 添加匹配之間的普通文本
        learner_line_parts.append(sentence_str[current_pos:match.start()])
        editor_line_parts.append(sentence_str[current_pos:match.start()])

        current_pos = match.end()  # 更新處理位置

        # 從匹配中獲取不同組的文本
        l_repl_text = match.group("l_repl")  # 學習者替換掉的文本
        e_repl_text = match.group("e_repl")  # 編輯者替換上的文本
        l_del_text = match.group("l_del")    # 學習者刪除的文本
        e_ins_text = match.group("e_ins")    # 編輯者插入的文本

        if l_repl_text is not None and e_repl_text is not None:
            # 情況 1: 標準替換 [-L-]{+E+}
            l_padded = l_repl_text
            e_padded = e_repl_text
            # 為了對齊，對較短的文本進行填充 (使用 non-breaking space)
            # 注意：這裡的填充邏輯可能需要根據實際視覺效果調整
            # 簡單的 ljust 可能不適用於 HTML，因為空格的渲染方式不同。
            # _create_html_span 內部處理 display: inline-block，長度差異主要體現在內容。
            # 若要嚴格對齊字符，可能需要更複雜的 HTML/CSS 或 JS。
            # 目前的 _create_html_span 樣式已包含 padding，有助於視覺分隔。
            # 長度不一時，顏色塊的長度會不同。
            if len(e_repl_text) > len(l_repl_text):
                l_padded = l_repl_text.ljust(len(e_repl_text))
            elif len(l_repl_text) > len(e_repl_text):
                e_padded = e_repl_text.ljust(len(l_repl_text))

            learner_line_parts.append(_create_html_span(l_padded, 'learner_replace_from'))
            editor_line_parts.append(_create_html_span(e_padded, 'editor_replace_to'))

        elif l_del_text is not None:
            # 情況 2: 單純刪除 [-L-]
            learner_line_parts.append(_create_html_span(l_del_text, 'learner_delete'))
            # 在編輯者句子對應位置插入相同視覺長度的空白佔位符
            num_chars = len(l_del_text)
            if num_chars > 0:
                placeholder_text = "&nbsp;" * num_chars # 使用 non-breaking spaces
                editor_line_parts.append(_create_html_span(placeholder_text, 'placeholder_general'))

        elif e_ins_text is not None:
            # 情況 3: 單純插入 {+E+}
            editor_line_parts.append('&nbsp;')
            editor_line_parts.append(_create_html_span(e_ins_text, 'editor_insert'))
            # 在學習者句子對應位置插入相同視覺長度的空白佔位符
            num_chars = len(e_ins_text)
            if num_chars > 0:
                placeholder_text = "&nbsp;" * (num_chars + 1) # 使用 non-breaking spaces
                learner_line_parts.append(_create_html_span(placeholder_text, 'placeholder_general'))

    # 2. 添加最後一個匹配之後的剩餘普通文本
    learner_line_parts.append(sentence_str[current_pos:])
    editor_line_parts.append(sentence_str[current_pos:])

    return "".join(learner_line_parts), "".join(editor_line_parts)


def formatted_to_original_edited(formatted_sentence: str) -> tuple[str, str]:
    """
    Converts a formatted sentence string (with [-delete-], {+insert+},
    [-replace_from-]{+replace_to+} markup) back into an original (learner)
    sentence and an edited (editor) sentence.

    Args:
        formatted_sentence: The input string with diff markup.

    Returns:
        A tuple containing two strings: (learner_sentence, editor_sentence).
    """
    if not formatted_sentence:
        return "", ""

    # Regular expression to find all three types of markup
    # It's the same pattern used in process_formatted_sentence
    diff_pattern = re.compile(
        r"\[-(?P<l_repl>.*?)-\]\{\+(?P<e_repl>.*?)\+\}|"  # Replacement: [-learner-]{+editor+}
        r"\[-(?P<l_del>.*?)-\]|"                         # Deletion: [-learner_delete-]
        r"\{\+(?P<e_ins>.*?)\+\}"                        # Insertion: {+editor_insert+}
    )

    learner_parts = []
    editor_parts = []
    current_pos = 0

    for match in diff_pattern.finditer(formatted_sentence):
        # Append the text segment before the current match (common to both)
        plain_text_before_match = formatted_sentence[current_pos:match.start()]
        learner_parts.append(plain_text_before_match)
        editor_parts.append(plain_text_before_match)

        # Update current position to the end of the match
        current_pos = match.end()

        # Extract matched groups
        l_repl_text = match.group("l_repl")
        e_repl_text = match.group("e_repl")
        l_del_text = match.group("l_del")
        e_ins_text = match.group("e_ins")

        if l_repl_text is not None and e_repl_text is not None:
            # Replacement: [-learner text-]{+editor text+}
            learner_parts.append(l_repl_text)
            editor_parts.append(e_repl_text)
        elif l_del_text is not None:
            # Deletion: [-learner text-]
            # Learner sentence includes the deleted text
            # Editor sentence does not include this part
            learner_parts.append(l_del_text)
        elif e_ins_text is not None:
            # Insertion: {+editor text+}
            # Learner sentence does not include this part
            # Editor sentence includes the inserted text
            editor_parts.append(e_ins_text)

    # Append any remaining text after the last match (common to both)
    remaining_text = formatted_sentence[current_pos:]
    learner_parts.append(remaining_text)
    editor_parts.append(remaining_text)

    learner_sentence = "".join(learner_parts)
    editor_sentence = "".join(editor_parts)

    return learner_sentence, editor_sentence
