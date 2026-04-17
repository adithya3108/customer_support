from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("POLICY_DATA_PATH", BASE_DIR / "data" / "policies.json"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "").strip()
SARVAM_MODEL = os.getenv("SARVAM_MODEL", "sarvam-m").strip()
SARVAM_API_URL = os.getenv("SARVAM_API_URL", "https://api.sarvam.ai/v1/chat/completions").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").strip().lower()
LOW_SCORE_THRESHOLD = float(os.getenv("BM25_LOW_SCORE_THRESHOLD", "0.8"))


class PolicyDoc(BaseModel):
    title: str
    content: str


class SupportRequest(BaseModel):
    complaint: str = Field(..., min_length=3)
    mode: Literal["strict", "friendly"] = "strict"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class RetrievedDocument(BaseModel):
    title: str
    content: str
    score: float


class SupportResponse(BaseModel):
    response: str
    documents: List[RetrievedDocument]
    provider: str
    mode: str
    temperature: float
    max_tokens: int
    prompt: str
    fallback_used: bool


app = FastAPI(title="AI-Assisted Customer Support Response Generator")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_policy_docs() -> List[PolicyDoc]:
    if DATA_PATH.exists():
        payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        return [PolicyDoc(**item) for item in payload]

    return [
        PolicyDoc(
            title="Refund Policy",
            content="Customers can request a refund within 7 days of delivery if the item is unused and returned with original packaging.",
        ),
        PolicyDoc(
            title="Return Policy",
            content="Returns are accepted within 10 days for damaged, defective, or incorrect items after customer support approval.",
        ),
        PolicyDoc(
            title="Delivery Delay",
            content="If delivery is delayed by more than 5 business days beyond the promised date, the customer may request status review or escalation.",
        ),
    ]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


POLICY_DOCS = load_policy_docs()
TOKENIZED_CORPUS = [tokenize(f"{doc.title} {doc.content}") for doc in POLICY_DOCS]
BM25 = BM25Okapi(TOKENIZED_CORPUS)


def choose_provider() -> str:
    if LLM_PROVIDER in {"sarvam", "openai"}:
        return LLM_PROVIDER
    if OPENAI_API_KEY:
        return "openai"
    if SARVAM_API_KEY:
        return "sarvam"
    return "openai"


def retrieve_docs(query: str, top_k: int = 3) -> List[RetrievedDocument]:
    query_tokens = tokenize(query)
    scores = BM25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
    results: List[RetrievedDocument] = []

    for index, score in ranked:
        doc = POLICY_DOCS[index]
        results.append(
            RetrievedDocument(
                title=doc.title,
                content=doc.content,
                score=float(score),
            )
        )

    return results


def build_prompt(query: str, docs: List[RetrievedDocument], mode: str) -> str:
    docs_text = "\n\n".join(
        f"Title: {doc.title}\nPolicy: {doc.content}" for doc in docs
    )

    if mode == "strict":
        return (
            "You are a professional customer support assistant.\n"
            "Use ONLY the provided policy context.\n"
            "Do not add extra assumptions.\n\n"
            f"Context:\n{docs_text}\n\n"
            f"Customer Issue:\n{query}\n\n"
            "Give a clear and concise response."
        )

    return (
        "You are a polite and empathetic support agent.\n"
        "Use the policy context but respond in a friendly tone.\n\n"
        f"Context:\n{docs_text}\n\n"
        f"Customer Issue:\n{query}\n\n"
        "Write a warm but policy-based response."
    )


def call_openai(prompt: str, temperature: float, max_tokens: int) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


def call_sarvam(prompt: str, temperature: float, max_tokens: int) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set.")

    payload = {
        "model": SARVAM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": 1,
        "max_tokens": max_tokens,
    }
    request_obj = urllib.request.Request(
        SARVAM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "api-subscription-key": SARVAM_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request_obj, timeout=90) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise RuntimeError(f"Sarvam request failed: {exc.code} {error_body}") from exc

    try:
        return response_data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Sarvam response did not contain an assistant message.") from exc


def run_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    provider = choose_provider()
    if provider == "sarvam":
        return call_sarvam(prompt, temperature, max_tokens)
    return call_openai(prompt, temperature, max_tokens)


def log_request(complaint: str, mode: str, temperature: float, max_tokens: int, docs: List[RetrievedDocument], prompt: str) -> None:
    print(f"Complaint: {complaint}", flush=True)
    print(f"Mode: {mode} temperature={temperature} max_tokens={max_tokens}", flush=True)
    print("Retrieved docs:", flush=True)
    for doc in docs:
        print(f"  - {doc.title} score={doc.score:.4f}", flush=True)
    print("Prompt used:", flush=True)
    print(prompt, flush=True)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "source_dataset": str(DATA_PATH),
        "documents": len(POLICY_DOCS),
        "provider": choose_provider(),
    }


@app.post("/api/generate", response_model=SupportResponse)
def generate_response(payload: SupportRequest) -> SupportResponse:
    complaint = payload.complaint.strip()
    mode = payload.mode
    temperature = payload.temperature if payload.temperature is not None else (0.2 if mode == "strict" else 0.7)
    max_tokens = payload.max_tokens if payload.max_tokens is not None else (150 if mode == "strict" else 200)

    print("Incoming support request.", flush=True)
    print(f"Query: {complaint}", flush=True)

    docs = retrieve_docs(complaint, top_k=3)
    best_score = docs[0].score if docs else 0.0
    if not docs or best_score < LOW_SCORE_THRESHOLD:
        fallback_response = "Please escalate this issue to a human support agent."
        print(f"Fallback triggered. best_score={best_score:.4f}", flush=True)
        fallback_prompt = "No relevant policy found. Respond with: Please escalate this issue to a human support agent."
        log_request(complaint, mode, temperature, max_tokens, docs, fallback_prompt)
        return SupportResponse(
            response=fallback_response,
            documents=docs,
            provider=choose_provider(),
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=fallback_prompt,
            fallback_used=True,
        )

    prompt = build_prompt(complaint, docs, mode)
    log_request(complaint, mode, temperature, max_tokens, docs, prompt)

    try:
        response_text = run_llm(prompt, temperature, max_tokens)
        print(f"LLM response preview: {response_text[:400]}", flush=True)
    except Exception as exc:
        print(f"LLM call failed: {exc}", flush=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return SupportResponse(
        response=response_text,
        documents=docs,
        provider=choose_provider(),
        mode=mode,
        temperature=temperature,
        max_tokens=max_tokens,
        prompt=prompt,
        fallback_used=False,
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting customer support backend on http://127.0.0.1:8000", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8002, reload=False)
