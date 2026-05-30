# Requires Bedrock Guardrail created in AWS console
# Store guardrail ID in config.py as BEDROCK_GUARDRAIL_ID
# AWS console: Amazon Bedrock → Guardrails → Create guardrail

import json
import time
import boto3
import mlflow
from datetime import datetime

from config import (
    BEDROCK_REGION,
    BEDROCK_MODEL_ID,
    BEDROCK_GUARDRAIL_ID,
    BEDROCK_GUARDRAIL_VERSION,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    ACTIVE_PROMPT_VERSION,
)
from prompt_manager import load_prompt, format_prompt


class BedrockCallError(Exception):
    pass


def summarize_document(
    document_text: str,
    prompt_version: str = None,
    apply_guardrails: bool = True
) -> dict:
    """
    Summarize a document using Amazon Bedrock with optional guardrails.

    Steps:
    1. Load prompt template via prompt_manager.load_prompt(prompt_version)
    2. Format prompt with document_text
    3. Call Bedrock with or without guardrails based on apply_guardrails
    4. Strip markdown fences from response before JSON parsing
    5. Log to MLflow: prompt_version, latency_ms, guardrails_fired,
       input_length, output_length
    6. Return dict with summary fields plus metadata

    If guardrails fire (BLOCKED or ANONYMIZED):
    - Set guardrails_fired = True
    - Set guardrails_action to the action taken
    - If BLOCKED: return a safe error response instead of the raw output
    - If ANONYMIZED: return the anonymized output with a notice

    Args:
        document_text: extracted text from PDF
        prompt_version: which prompt version to use (default: ACTIVE_PROMPT_VERSION)
        apply_guardrails: whether to apply Bedrock Guardrails (default: True)

    Returns:
        Dict with summary fields and metadata
    """
    if prompt_version is None:
        prompt_version = ACTIVE_PROMPT_VERSION

    prompt_meta = load_prompt(prompt_version)
    prompt = format_prompt(prompt_meta['template'], document_text)

    bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    })

    guardrails_fired = False
    guardrails_action = None
    start_time = time.time()

    try:
        if apply_guardrails:
            # invoke_model supports guardrails via parameters; action returned in response headers
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=body,
                guardrailIdentifier=BEDROCK_GUARDRAIL_ID,
                guardrailVersion=BEDROCK_GUARDRAIL_VERSION,
                trace='ENABLED'
            )
            result = json.loads(response['body'].read())
            raw_text = result['content'][0]['text']

            # Guardrail action is returned in the HTTP response headers
            guardrail_action_header = response.get('ResponseMetadata', {}) \
                .get('HTTPHeaders', {}) \
                .get('x-amzn-bedrock-guardrail-action', '').upper()

            if guardrail_action_header == 'BLOCKED':
                guardrails_fired = True
                guardrails_action = 'BLOCKED'
                latency_ms = int((time.time() - start_time) * 1000)
                _log_to_mlflow(prompt_version, latency_ms, True, len(document_text), 0)
                return {
                    "summary": "Content blocked by safety guardrails.",
                    "key_points": [],
                    "main_argument": "Content blocked.",
                    "sentiment": "neutral",
                    "recommended_action": "Review the document for policy-violating content.",
                    "prompt_version": prompt_version,
                    "latency_ms": latency_ms,
                    "guardrails_fired": True,
                    "guardrails_action": "BLOCKED",
                    "model_id": BEDROCK_MODEL_ID,
                }
            elif guardrail_action_header == 'ANONYMIZED':
                guardrails_fired = True
                guardrails_action = 'ANONYMIZED'

        else:
            response = bedrock.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                body=body
            )
            result = json.loads(response['body'].read())
            raw_text = result['content'][0]['text']

    except Exception as e:
        raise BedrockCallError(f'Bedrock call failed: {str(e)}')

    latency_ms = int((time.time() - start_time) * 1000)

    # Strip markdown fences before JSON parsing
    clean_text = raw_text.strip()
    if clean_text.startswith('```'):
        clean_text = clean_text.split('```')[1]
        if clean_text.startswith('json'):
            clean_text = clean_text[4:]

    try:
        parsed = json.loads(clean_text.strip())
    except json.JSONDecodeError as e:
        raise BedrockCallError(f'Failed to parse Bedrock JSON response: {str(e)}\nRaw: {clean_text[:200]}')

    _log_to_mlflow(prompt_version, latency_ms, guardrails_fired, len(document_text), len(raw_text))

    result_dict = {**parsed}
    result_dict.update({
        "prompt_version": prompt_version,
        "latency_ms": latency_ms,
        "guardrails_fired": guardrails_fired,
        "guardrails_action": guardrails_action,
        "model_id": BEDROCK_MODEL_ID,
    })

    if guardrails_fired and guardrails_action == 'ANONYMIZED':
        result_dict['_guardrails_notice'] = 'PII detected and anonymised in this summary.'

    return result_dict


def _log_to_mlflow(prompt_version: str, latency_ms: int, guardrails_fired: bool,
                   input_length: int, output_length: int):
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        with mlflow.start_run(run_name=f'summarize_{prompt_version}_{timestamp}'):
            mlflow.log_param('prompt_version', prompt_version)
            mlflow.log_param('model_id', BEDROCK_MODEL_ID)
            mlflow.log_metric('latency_ms', latency_ms)
            mlflow.log_metric('input_length', input_length)
            mlflow.log_metric('output_length', output_length)
            mlflow.log_metric('guardrails_fired', int(guardrails_fired))
    except Exception:
        # MLflow logging is non-critical — don't fail the request
        pass
