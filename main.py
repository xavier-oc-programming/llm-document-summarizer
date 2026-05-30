import json
from pathlib import Path

import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from config import (
    ACTIVE_PROMPT_VERSION,
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    UPLOAD_DIR,
    MAX_PDF_SIZE_MB,
    EVAL_RESULTS_DIR,
)
from pdf_extractor import extract_text_from_bytes
from bedrock_client import summarize_document
from prompt_manager import list_prompt_versions, load_prompt

app = FastAPI(
    title="LLM Document Summarizer",
    description="Summarizes PDF documents using Amazon Bedrock (Claude Haiku) with versioned prompts, MLflow tracking, and Bedrock Guardrails. LLMOps patterns: prompt versioning, automated evaluation, PII redaction.",
    version="1.0.0"
)

API_READY = False


async def _startup():
    global API_READY
    try:
        boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    except Exception:
        pass
    UPLOAD_DIR.mkdir(exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(exist_ok=True)
    try:
        load_prompt(ACTIVE_PROMPT_VERSION)
        API_READY = True
    except Exception:
        API_READY = False

app.add_event_handler("startup", _startup)


class SummarizeResponse(BaseModel):
    summary: str
    key_points: list[str]
    main_argument: str
    sentiment: str
    recommended_action: str
    document_type: str | None = None
    confidence: str | None = None
    risk_flags: list[str] | None = None
    prompt_version: str
    latency_ms: int
    guardrails_fired: bool
    guardrails_action: str | None = None
    page_count: int
    word_count: int
    truncated: bool
    model_id: str


class HealthResponse(BaseModel):
    status: str
    api_ready: bool
    active_prompt_version: str
    model_id: str


class TextSummarizeRequest(BaseModel):
    text: str
    prompt_version: str | None = None


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "LLM Document Summarizer API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        api_ready=API_READY,
        active_prompt_version=ACTIVE_PROMPT_VERSION,
        model_id=BEDROCK_MODEL_ID,
    )


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF and receive a structured summary.

    Accepts: multipart/form-data with a 'file' field containing a PDF.
    Max size: 10MB.

    Returns a structured summary including key points, main argument,
    sentiment, and recommended action. All outputs pass through Bedrock
    Guardrails before returning — PII is anonymised, harmful content blocked.
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF.")

    if not API_READY:
        raise HTTPException(status_code=503, detail="API not ready — check AWS credentials and prompt configuration.")

    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File size {size_mb:.1f}MB exceeds maximum {MAX_PDF_SIZE_MB}MB.")

    try:
        extraction = extract_text_from_bytes(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = summarize_document(document_text=extraction['text'])

    return SummarizeResponse(
        summary=result.get('summary', ''),
        key_points=result.get('key_points', []),
        main_argument=result.get('main_argument', ''),
        sentiment=result.get('sentiment', 'neutral'),
        recommended_action=result.get('recommended_action', ''),
        document_type=result.get('document_type'),
        confidence=result.get('confidence'),
        risk_flags=result.get('risk_flags'),
        prompt_version=result['prompt_version'],
        latency_ms=result['latency_ms'],
        guardrails_fired=result['guardrails_fired'],
        guardrails_action=result.get('guardrails_action'),
        page_count=extraction['page_count'],
        word_count=extraction['word_count'],
        truncated=extraction['truncated'],
        model_id=result['model_id'],
    )


@app.post("/summarize/text", response_model=SummarizeResponse)
async def summarize_text(request: TextSummarizeRequest):
    """
    Summarize raw text directly without PDF extraction.
    Useful for testing without uploading files.
    """
    if not API_READY:
        raise HTTPException(status_code=503, detail="API not ready — check AWS credentials and prompt configuration.")

    result = summarize_document(
        document_text=request.text,
        prompt_version=request.prompt_version,
    )

    word_count = len(request.text.split())

    return SummarizeResponse(
        summary=result.get('summary', ''),
        key_points=result.get('key_points', []),
        main_argument=result.get('main_argument', ''),
        sentiment=result.get('sentiment', 'neutral'),
        recommended_action=result.get('recommended_action', ''),
        document_type=result.get('document_type'),
        confidence=result.get('confidence'),
        risk_flags=result.get('risk_flags'),
        prompt_version=result['prompt_version'],
        latency_ms=result['latency_ms'],
        guardrails_fired=result['guardrails_fired'],
        guardrails_action=result.get('guardrails_action'),
        page_count=1,
        word_count=word_count,
        truncated=len(request.text) > 8000,
        model_id=result['model_id'],
    )


@app.get("/api/prompt-versions")
async def get_prompt_versions():
    return list_prompt_versions()


@app.get("/api/active-prompt")
async def get_active_prompt():
    prompt = load_prompt(ACTIVE_PROMPT_VERSION)
    return {
        "version": prompt['version'],
        "description": prompt['description'],
        "template": prompt['template'],
    }


@app.get("/api/eval-results")
async def get_eval_results():
    result_files = sorted(EVAL_RESULTS_DIR.glob('eval_*.json'), reverse=True)
    if not result_files:
        return {"message": "No evaluation results yet. Run python evaluate.py to generate."}

    with open(result_files[0], 'r') as f:
        return json.load(f)


@app.get("/demo", response_class=HTMLResponse, include_in_schema=False)
async def demo():
    html_path = Path('templates/index.html')
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Demo page not found</h1>", status_code=404)
