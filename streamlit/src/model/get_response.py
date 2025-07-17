# model/get_response.py
from utils.files_io import (
    load_method_config,
)
from model.model_setup import (
    format_prompt_baseline,
    format_prompt_ragDictionary,
    format_prompt_ragL2_causes,
    format_prompt_ragL2_academicWriting,
    format_prompt_ragL2_explanation,
    format_prompt_ragCollocation,
    format_prompt_ragMix,
    chat_response,
    KnowledgeMatchSchema,
    ExplanationSchema,
    CollocationSchema,
    GeneralExplanationSchema,
    GeneralCollocationSchema,
)


MODEL_DOC = load_method_config("llm")


def chat_baseline(df, file_path, for_general, model_type="gpt-4.1-nano", role="linguist"):
    response_baseline = None
    if not for_general:
        prompt_baseline = MODEL_DOC['prompt_v2']['baseline']
        prompts_baseline = format_prompt_baseline(df.loc[0], prompt_baseline, role=role)
        response_baseline = chat_response(
            prompts_baseline, run=True, modelType=model_type, formatSchema=ExplanationSchema
        )
    else:
        prompt_baseline = MODEL_DOC['prompt_general']['baseline']
        prompts_baseline = format_prompt_baseline(df.loc[0], prompt_baseline, role=role)
        response_baseline = chat_response(
            prompts_baseline, run=True, modelType=model_type, formatSchema=GeneralExplanationSchema
        )
    df['baseline_explanation'] = response_baseline.output_parsed.model_dump()
    print("Baseline explanation completed.")
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    return response_baseline.output_parsed


def chat_dictionary(df, file_path, for_general, model_type="gpt-4.1-nano", role="linguist"):
    response_ragDictionary = None
    if not for_general:
        prompt_ragDictionary = MODEL_DOC['prompt_v2']['ragDictionary']
        prompt_ragDictionary = format_prompt_ragDictionary(df.loc[0], prompt_ragDictionary, role=role)
        response_ragDictionary = chat_response(
            prompt_ragDictionary, run=True, modelType=model_type, formatSchema=ExplanationSchema
        )
    else:
        prompt_ragDictionary = MODEL_DOC['prompt_general']['ragDictionary']
        prompt_ragDictionary = format_prompt_ragDictionary(df.loc[0], prompt_ragDictionary, role=role)
        response_ragDictionary = chat_response(
            prompt_ragDictionary, run=True, modelType=model_type, formatSchema=GeneralExplanationSchema
        )
    df['dictionary_explanation'] = response_ragDictionary.output_parsed.model_dump()
    print("Dictionary explanation completed.")
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    return response_ragDictionary.output_parsed


def chat_l2_knowledge_resource(df, file_path, model_type="gpt-4.1-nano", role="linguist"):
    prompt_ragL2_causes = MODEL_DOC['prompt']['causes']
    prompt_ragL2_academicWriting = MODEL_DOC['prompt']['academic_writing']

    prompt_ragL2_causes = format_prompt_ragL2_causes(df.loc[0], prompt_ragL2_causes, role=role)
    prompt_ragL2_academicWriting = format_prompt_ragL2_academicWriting(df.loc[0], prompt_ragL2_academicWriting, role=role)

    response_ragL2_causes = chat_response(
        prompt_ragL2_causes, run=True, modelType=model_type, formatSchema=KnowledgeMatchSchema
    )
    response_ragL2_academicWriting = chat_response(
        prompt_ragL2_academicWriting, run=True, modelType=model_type, formatSchema=KnowledgeMatchSchema
    )
    content_ragL2_causes = response_ragL2_causes.output_parsed
    content_ragL2_academicWriting = response_ragL2_academicWriting.output_parsed
    response_ragL2_causes = '\n'.join(content_ragL2_causes.model_dump()['matched_items'])
    response_ragL2_academicWriting = '\n'.join(content_ragL2_academicWriting.model_dump()['matched_items'])
    df['causes'] = response_ragL2_causes
    df['academic_writing'] = response_ragL2_academicWriting
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    print("L2 knowledge resource completed.")

    return df


def chat_l2_knowledge(df, file_path, model_type="gpt-4.1-nano", role="linguist"):
    prompt_ragL2_explanation = MODEL_DOC['prompt_v2']['ragL2']
    prompt_ragL2_explanation = format_prompt_ragL2_explanation(
        df.loc[0], prompt_ragL2_explanation, role=role
    )
    response_ragL2_explanation = chat_response(
        prompt_ragL2_explanation, run=True, modelType=model_type, formatSchema=ExplanationSchema
    )

    df['L2_explanation'] = response_ragL2_explanation.output_parsed.model_dump()
    print("Explanation with L2 knowledge completed.")
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    return response_ragL2_explanation.output_parsed


def chat_collocations(df, file_path, for_general, with_collocaiton_examples, model_type="gpt-4.1-nano", role="linguist"):
    response_ragCollocation = None
    if not with_collocaiton_examples and not for_general:
        prompt_ragCollocation = MODEL_DOC['prompt_v2']['ragCollocation']
        prompt_ragCollocation = format_prompt_ragCollocation(df.loc[0], prompt_ragCollocation, role=role)
        response_ragCollocation = chat_response(
            prompt_ragCollocation, run=True, modelType=model_type, formatSchema=ExplanationSchema
        )
    elif not with_collocaiton_examples and for_general:
        prompt_ragCollocation = MODEL_DOC['prompt_general']['ragCollocation']
        prompt_ragCollocation = format_prompt_ragCollocation(df.loc[0], prompt_ragCollocation, role=role)
        response_ragCollocation = chat_response(
            prompt_ragCollocation, run=True, modelType=model_type, formatSchema=GeneralExplanationSchema
        )
    elif with_collocaiton_examples and not for_general:
        prompt_ragCollocation = MODEL_DOC['prompt_v2']['ragCollocation_v2']
        prompt_ragCollocation = format_prompt_ragCollocation(df.loc[0], prompt_ragCollocation, role=role)
        response_ragCollocation = chat_response(
            prompt_ragCollocation, run=True, modelType=model_type, formatSchema=CollocationSchema
        )
    elif with_collocaiton_examples and for_general:
        prompt_ragCollocation = MODEL_DOC['prompt_general']['ragCollocation_v2']
        prompt_ragCollocation = format_prompt_ragCollocation(df.loc[0], prompt_ragCollocation, role=role)
        response_ragCollocation = chat_response(
            prompt_ragCollocation, run=True, modelType=model_type, formatSchema=GeneralCollocationSchema
        )

    df['collocation_explanation'] = response_ragCollocation.output_parsed.model_dump()
    print("Collocation explanation completed.")
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    return response_ragCollocation.output_parsed


def chat_mix(df, file_path, for_general, with_collocaiton_examples, model_type="gpt-4.1-nano", role="linguist"):
    response_ragMix = None
    if not with_collocaiton_examples and not for_general:
        prompt_ragMix = MODEL_DOC['prompt_v2']['ragMix']
        prompt_ragMix = format_prompt_ragMix(df.loc[0], prompt_ragMix, role=role)
        response_ragMix = chat_response(
            prompt_ragMix, run=True, modelType=model_type, formatSchema=ExplanationSchema
        )
    elif not with_collocaiton_examples and for_general:
        prompt_ragMix = MODEL_DOC['prompt_general']['ragMix']
        prompt_ragMix = format_prompt_ragMix(df.loc[0], prompt_ragMix, role=role)
        response_ragMix = chat_response(
            prompt_ragMix, run=True, modelType=model_type, formatSchema=GeneralExplanationSchema
        )
    elif with_collocaiton_examples and not for_general:
        prompt_ragMix = MODEL_DOC['prompt_v2']['ragMix_v2']
        prompt_ragMix = format_prompt_ragMix(df.loc[0], prompt_ragMix, role=role)
        response_ragMix = chat_response(
            prompt_ragMix, run=True, modelType=model_type, formatSchema=CollocationSchema
        )
    elif with_collocaiton_examples and for_general:
        prompt_ragMix = MODEL_DOC['prompt_general']['ragMix_v2']
        prompt_ragMix = format_prompt_ragMix(df.loc[0], prompt_ragMix, role=role)
        response_ragMix = chat_response(
            prompt_ragMix, run=True, modelType=model_type, formatSchema=GeneralCollocationSchema
        )

    df['mix_explanation'] = response_ragMix.output_parsed.model_dump()
    print("Mixed explanation completed.")
    df.to_json(file_path, orient='records', lines=True, force_ascii=False)
    return response_ragMix.output_parsed
