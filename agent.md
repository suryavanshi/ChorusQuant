# agent.md — GPT‑5‑Codex Implementation Guide

## Mission
Build a **multi‑agent financial analysis framework** that achieves **very high‑accuracy PDF/Excel/scan extraction** and verified summaries, **without** using external agent frameworks (no LangChain/LangGraph/ReAct wrappers). Each “agent” is a plain Python module with a clear interface.

---

## Repo Structure (monorepo)
```
financial-agents/
├─ agents/
│  ├─ orchestrator/                 # Task routing, DAG orchestration, parallelism
│  │  ├─ orchestrator.py
│  │  └─ planner.py                 # optional planning utilities
│  ├─ ingest/
│  │  ├─ sniff_mime.py              # file type, triage
│  │  ├─ pdf_text.py                # PyMuPDF text; page blocks; figure detection
│  │  ├─ ocr.py                     # Tesseract/OCR API + preproc (OpenCV)
│  │  ├─ tables.py                  # Camelot/pdfplumber table extraction
│  │  ├─ sheets.py                  # CSV/XLSX parsing (pandas/openpyxl)
│  │  └─ vision_charts.py           # chart/graph caption + value extraction
│  ├─ normalize/
│  │  ├─ schema.py                  # Pydantic models: DocumentData, Table, Field
│  │  └─ map_to_schema.py           # heuristics + small LLM calls to map
│  ├─ analysis/
│  │  ├─ portfolio.py               # allocation, concentration, fees
│  │  ├─ sec_reports.py             # KPIs from 10-K/10-Q sections
│  │  └─ narrative.py               # summaries, Q&A over chunks
│  ├─ validate/
│  │  ├─ numeric_crosscheck.py      # regex+fuzzy checks against source text
│  │  ├─ table_consistency.py       # totals tie‑out, footnote unit/scale tests
│  │  └─ evidence_picker.py         # citation spans for each asserted fact
│  ├─ synthesize/
│  │  ├─ report_md.py               # Markdown/JSON outputs + citations
│  │  └─ exports.py                 # CSV/Excel exports of structured outputs
│  └─ llm/
│     ├─ client.py                  # call_llm(), call_vision() w/ base_url+model
│     └─ prompts/                   # jinja templates per task
├─ pipelines/
│  ├─ batch_driver.py               # multi-file jobs, retries, checkpoints
│  └─ api_server.py                 # optional REST for jobs
├─ tests/
│  ├─ unit/                         # pure function tests
│  ├─ golden/                       # ground truth JSON for PDFs (field/table level)
│  └─ eval_harness/                 # metrics: field-F1, cell-F1, numeric error
├─ assets/                          # sample docs (redacted)
├─ scripts/                         # CLIs: ingest, eval, demo
├─ configs/
│  ├─ models.yaml                   # switch models via env / config
│  ├─ prompts.yaml                  # task prompts & output schemas
│  └─ routing.yaml                  # rules: when to use OCR/vision/tables
├─ SECURITY.md
├─ LICENSE
└─ README.md
```

---

## Design Principles
1. **Deterministic orchestration.** Clear Python control flow for routing; avoid hidden “agent magic.”
2. **Layout‑aware extraction first; LLM second.** Prefer PyMuPDF/pdfplumber/Camelot and OCR to get *ground truth* text and tables; use LLMs for interpretation/normalization.
3. **Verification‑before‑synthesis.** Every numeric claim and key phrase must point to evidence spans. Add tie‑outs and unit/scale normalization.
4. **Schema‑first outputs.** Map all extractions to typed Pydantic models; produce both JSON (machines) and Markdown (humans).
5. **Batch‑native.** Process multiple docs concurrently with per‑file checkpoints; idempotent re‑runs.
6. **Swap‑friendly models.** `LLM_BASE_URL` + `LLM_MODEL` envs; retry/backoff + streaming support.

---

## Core Interfaces (sketch)

```python
# agents/normalize/schema.py
from pydantic import BaseModel
from typing import List, Optional

class Table(BaseModel):
    title: Optional[str]
    columns: List[str]
    rows: List[List[str]]
    page_span: List[int]

class Evidence(BaseModel):
    text: str
    page: int
    bbox: Optional[tuple]  # (x0, y0, x1, y1)

class PortfolioHolding(BaseModel):
    name: str
    ticker: Optional[str]
    value: float
    currency: str
    weight_pct: Optional[float]
    evidence: List[Evidence] = []

class DocumentData(BaseModel):
    text_pages: List[str]
    tables: List[Table]
    images: List[str]  # paths
    meta: dict
```

```python
# agents/llm/client.py
def call_llm(messages: list[dict], *, model=None, base_url=None, **kwargs) -> str: ...
def call_vision(image_paths: list[str], prompt: str, *, model=None, base_url=None, **kw) -> str: ...
```

---

## Accuracy Guardrails
- **Dual‑path extraction:** text path (PyMuPDF) + OCR path (Tesseract) with **auto‑diff** reconciliation.
- **Table QA:** sum checks, column type checks, header detection recovery, thousands/decimal separator normalization.
- **Number verifier:** regex+fuzzy finder returns *exact source spans*; mismatch ⇒ block synthesis & raise retry.
- **Chart agent:** detect axis labels/units; recover series points when present; fallback to captioned trends.
- **Scale/units:** normalize (K/M/B, thousands vs exact) with footnote detection; propagate to all metrics.
- **No‑hallucination policy:** synthesis must include a per‑assertion evidence handle.

---

## Evaluation (goldens + metrics)
- **Field‑level:** precision/recall/F1 for required fields (Total Assets, Account #, etc.).
- **Table cell‑wise:** micro‑F1 (exact match), tolerance rules for numerics (abs/rel error thresholds).
- **Document coverage:** % pages touched; % assertions with evidence.
- **Regression CI:** fail build if δF1 > threshold on golden set.

---

## Config & Secrets
- `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`, `VISION_MODEL` via env.
- `OCR_ENGINE` (“tesseract” | “textract” | “gcv”), `OCR_LANGS`.
- Routing thresholds in `configs/routing.yaml` (min text density, image area %, etc.).

---

## Coding Standards
- Python 3.11, Ruff + Black, mypy on Pydantic models.
- Unit tests first for pure extraction utilities.
- Deterministic seeds; no global state in agents.
- Thin prompts with strict JSON schemas (jsonschema validation).

---

## Runbook (common CLIs)
```
python scripts/ingest.py --in docs/ --out runs/ingest.jsonl
python scripts/eval.py --pred runs/ingest.jsonl --gold tests/golden/*.json
python scripts/demo.py --in docs/sample/ --report out/report.md --json out/struct.json
python pipelines/batch_driver.py --job jobs/job.yaml
```

---

## Definition of Done (phase 1)
- ≥99.0% field‑F1 on brokerage statement core fields (n=200 docs, mixed layouts).
- ≥98.0% table cell‑F1 on holdings tables (tolerance: ±0.5% or ±$1.00).
- 100% numeric assertions have evidence spans (page + excerpt).
- End‑to‑end batch job completes with retries and resumability.
