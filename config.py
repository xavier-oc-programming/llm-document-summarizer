from pathlib import Path

# AWS
BEDROCK_REGION = 'us-east-1'
BEDROCK_MODEL_ID = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
BEDROCK_GUARDRAIL_ID = 'o2eo4z9blnfp'  # Set after creating in AWS console
BEDROCK_GUARDRAIL_VERSION = '1'

# Prompt versioning
PROMPTS_DIR = Path('prompts')
ACTIVE_PROMPT_VERSION = 'v3'  # Change to switch prompt versions
# Changing ACTIVE_PROMPT_VERSION is the only code change needed
# to switch prompt versions. All other files read from this constant.

# MLflow — local file backend by default (no server needed)
# To use the Docker Compose MLflow server instead, set:
#   MLFLOW_TRACKING_URI = 'http://localhost:5000'
MLFLOW_TRACKING_URI = 'mlruns'
MLFLOW_EXPERIMENT_NAME = 'llm-document-summarizer'

# Evaluation
EVAL_DIR = Path('eval')
EVAL_RESULTS_DIR = Path('eval_results')
ROUGE_THRESHOLD = 0.3  # Minimum acceptable ROUGE-L score
# If a prompt version scores below this threshold on the
# evaluation set, it should not be deployed.

# Paths
UPLOAD_DIR = Path('uploads')
MAX_PDF_SIZE_MB = 10
MAX_TEXT_LENGTH = 8000  # chars — truncate before sending to Bedrock
# Bedrock has context window limits. Truncating at 8000 chars
# (~2000 tokens) leaves room for the prompt template and response.

# Azure
AZURE_APP_NAME = 'llm-document-summarizer'
AZURE_RESOURCE_GROUP = 'llm-summarizer-rg'
AZURE_LOCATION = 'westeurope'
