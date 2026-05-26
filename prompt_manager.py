import yaml
from pathlib import Path
from config import PROMPTS_DIR, ACTIVE_PROMPT_VERSION, MAX_TEXT_LENGTH


def load_prompt(version: str = None) -> dict:
    """
    Load a prompt template from the prompts/ directory.

    Args:
        version: prompt version string e.g. 'v1', 'v2', 'v3'.
                 If None, loads ACTIVE_PROMPT_VERSION from config.

    Returns:
        Dict with keys: version, created, description, template

    Raises:
        FileNotFoundError if the prompt file does not exist.
    """
    if version is None:
        version = ACTIVE_PROMPT_VERSION

    prompt_file = PROMPTS_DIR / f'{version}_summarize.yaml'
    if not prompt_file.exists():
        raise FileNotFoundError(f'Prompt file not found: {prompt_file}')

    with open(prompt_file, 'r') as f:
        data = yaml.safe_load(f)

    return data


def list_prompt_versions() -> list[dict]:
    """
    List all available prompt versions in prompts/ directory.
    Returns list of dicts with version, created, description for each.
    Sorted by version ascending.
    """
    versions = []
    for prompt_file in sorted(PROMPTS_DIR.glob('*_summarize.yaml')):
        with open(prompt_file, 'r') as f:
            data = yaml.safe_load(f)
        versions.append({
            'version': data.get('version'),
            'created': data.get('created'),
            'description': data.get('description'),
        })
    return sorted(versions, key=lambda x: x['version'])


def format_prompt(template: str, document_text: str) -> str:
    """
    Fill in the {document_text} placeholder in a prompt template.
    Truncates document_text to MAX_TEXT_LENGTH before inserting.
    Returns the complete prompt string ready to send to Bedrock.
    """
    truncated = document_text[:MAX_TEXT_LENGTH]
    return template.format(document_text=truncated)
