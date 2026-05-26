# evaluate.py
# Runs all prompt versions against the evaluation set and logs results to MLflow.
# This is the core LLMOps evaluation pattern:
# 1. Before deploying a new prompt version, run this script.
# 2. Check the MLflow UI (localhost:5000) to compare versions.
# 3. Only update ACTIVE_PROMPT_VERSION if the new version beats the current one.
# Usage: python evaluate.py
# Requires: AWS credentials configured, MLflow server running (docker-compose up)

import json
import time
from datetime import datetime
from pathlib import Path

import mlflow
from rouge_score import rouge_scorer

from config import (
    EVAL_DIR,
    EVAL_RESULTS_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    ROUGE_THRESHOLD,
)
from bedrock_client import summarize_document
from prompt_manager import list_prompt_versions


def run_evaluation():
    docs_dir = EVAL_DIR / 'documents'
    refs_dir = EVAL_DIR / 'references'

    doc_files = sorted(docs_dir.glob('doc*.txt'))
    if not doc_files:
        print('No evaluation documents found in eval/documents/')
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    prompt_versions = list_prompt_versions()
    results_table = []

    for prompt_meta in prompt_versions:
        version = prompt_meta['version']
        print(f'\nEvaluating prompt {version}...')

        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        latencies = []
        guardrails_count = 0

        for doc_file in doc_files:
            doc_num = doc_file.stem.replace('doc', '')
            ref_file = refs_dir / f'ref{doc_num}.txt'

            if not ref_file.exists():
                print(f'  Warning: no reference found for {doc_file.name}, skipping')
                continue

            document_text = doc_file.read_text()
            reference_summary = ref_file.read_text().strip()

            try:
                result = summarize_document(
                    document_text=document_text,
                    prompt_version=version,
                    apply_guardrails=True
                )
            except Exception as e:
                print(f'  Error on {doc_file.name}: {e}')
                continue

            generated_summary = result.get('summary', '')
            latencies.append(result.get('latency_ms', 0))
            if result.get('guardrails_fired'):
                guardrails_count += 1

            scores = scorer.score(reference_summary, generated_summary)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

            print(f'  {doc_file.name}: ROUGE-L={scores["rougeL"].fmeasure:.3f}, latency={result.get("latency_ms")}ms')

        if not rouge1_scores:
            print(f'  No results for {version}, skipping')
            continue

        rouge1_avg = sum(rouge1_scores) / len(rouge1_scores)
        rouge2_avg = sum(rouge2_scores) / len(rouge2_scores)
        rougeL_avg = sum(rougeL_scores) / len(rougeL_scores)
        latency_avg = sum(latencies) / len(latencies)

        with mlflow.start_run(run_name=f'eval_{version}'):
            mlflow.log_param('prompt_version', version)
            mlflow.log_param('prompt_description', prompt_meta['description'])
            mlflow.log_metric('rouge1_avg', rouge1_avg)
            mlflow.log_metric('rouge2_avg', rouge2_avg)
            mlflow.log_metric('rougeL_avg', rougeL_avg)
            mlflow.log_metric('latency_ms_avg', latency_avg)
            mlflow.log_metric('guardrails_fired_count', guardrails_count)

        results_table.append({
            'version': version,
            'rouge1': rouge1_avg,
            'rouge2': rouge2_avg,
            'rougeL': rougeL_avg,
            'latency_ms': latency_avg,
            'guardrails_fired': guardrails_count,
        })

    if not results_table:
        print('No results to display.')
        return

    # Print comparison table
    print('\n' + '=' * 70)
    print(f'{"Prompt":<8} {"ROUGE-1":<10} {"ROUGE-2":<10} {"ROUGE-L":<10} {"Latency":<12} {"Guardrails"}')
    print('-' * 70)
    for row in results_table:
        print(f'{row["version"]:<8} {row["rouge1"]:<10.3f} {row["rouge2"]:<10.3f} '
              f'{row["rougeL"]:<10.3f} {row["latency_ms"]:<12.0f} {row["guardrails_fired"]}')
    print('=' * 70)

    best = max(results_table, key=lambda x: x['rougeL'])
    print(f'\nBest prompt by ROUGE-L: {best["version"]} ({best["rougeL"]:.3f})')
    if best['rougeL'] >= ROUGE_THRESHOLD:
        print(f'Recommendation: set ACTIVE_PROMPT_VERSION = \'{best["version"]}\' in config.py')
    else:
        print(f'Warning: best ROUGE-L ({best["rougeL"]:.3f}) is below threshold ({ROUGE_THRESHOLD}).')
        print('None of the prompt versions meet the deployment threshold.')

    # Save results to eval_results/
    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    output_path = EVAL_RESULTS_DIR / f'eval_{timestamp}.json'
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'results': results_table,
            'best_version': best['version'],
            'best_rougeL': best['rougeL'],
            'rouge_threshold': ROUGE_THRESHOLD,
        }, f, indent=2)
    print(f'\nResults saved to {output_path}')


if __name__ == '__main__':
    run_evaluation()
