from __future__ import annotations

import json
import os
import pickle
import re
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# Paths (robust to your repo layout)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these files exist: {[str(p) for p in paths]}")


def _maybe_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# Your repo seems to sometimes keep these in root, sometimes in data/
PROCESSED_CORPUS_PATH = _first_existing(
    PROJECT_ROOT / "data" / "processed_corpus.pkl",
    PROJECT_ROOT / "processed_corpus.pkl",
)

ICD_NAME_MAP_PATH = _maybe_existing(
    PROJECT_ROOT / "data" / "icd_name_map.json",
    PROJECT_ROOT / "icd_name_map.json",
)

DEFAULT_MODEL_PATH = (
    (PROJECT_ROOT / "data" / "model.pkl") if (PROJECT_ROOT / "data").exists() else (PROJECT_ROOT / "model.pkl")
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()

# ============================================================
# Config (keep your familiar params)
# ============================================================
def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v is None else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v is None else float(v)


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


TOP_CODES = _env_int("TOP_CODES", 15)
K_NEIGHBORS = _env_int("K_NEIGHBORS", 40)
ALPHA = _env_float("ALPHA", 0.65)
BETA = _env_float("BETA", 0.15)

USE_FOCUSED_INDEX = _env_bool("USE_FOCUSED_INDEX", True)

# ============================================================
# Qazcode LLM (optional) — you will set these env vars
# ============================================================
# Enable these if you want:
LLM_QUERY_EXPAND = _env_bool("LLM_QUERY_EXPAND", False)  # best "LLM helps similarity" option
LLM_RERANK = _env_bool("LLM_RERANK", False)              # safe "choose top3 from candidates"
LLM_CANDIDATES = _env_int("LLM_CANDIDATES", 30)
LLM_TIMEOUT_S = _env_int("LLM_TIMEOUT_S", 20)
LLM_CACHE = _env_bool("LLM_CACHE", True)


QAZCODE_BASE_URL = os.getenv("QAZCODE_BASE_URL", "https://hub.qazcode.ai").rstrip("/")  # e.g. https://hub.qazcode.ai/v1  (or without /v1)
QAZCODE_API_KEY = os.getenv("QAZCODE_API_KEY", "sk-BDVloWBwHCr5oltlXwyhtA")
QAZCODE_MODEL = os.getenv("QAZCODE_MODEL", "oss-120b")

# ============================================================
# API schemas (no explanation, as you asked)
# ============================================================
class DiagnoseRequest(BaseModel):
    symptoms: Optional[str] = ""


class Diagnosis(BaseModel):
    rank: int
    diagnosis: str
    icd10_code: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ============================================================
# Text preprocessing (strong)
# ============================================================
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_WS = re.compile(r"\s+")
KEEP_SHORT = {"не", "no"}  # keep negation

SYMPTOM_KEYS = [
    "жалоб", "симптом", "клиническ", "анамнез", "объектив", "осмотр",
    "complaint", "symptom", "clinical", "history", "anamnes",
]

BOILER_PATTERNS = [
    r"утвержден[ао]?\s+приказ",
    r"приказ\s*№",
    r"министерств[ао]",
    r"приложени[ея]\s+\d+",
    r"клиническ(ий|ие)\s+протокол",
    r"протокол\s+диагностик",
    r"официальн(ое|ая)\s+издани",
]


def normalize_text(text: Any) -> str:
    """Tokenize to a compact bag-of-words string (keeps digits + negations)."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    toks: list[str] = []
    for t in TOKEN_RE.findall(text.lower()):
        if t.isdigit():
            toks.append(t)
            continue
        if t in KEEP_SHORT:
            toks.append(t)
            continue
        if len(t) >= 3:
            toks.append(t)

    return " ".join(toks)


def strip_boilerplate(text: str) -> str:
    if not text:
        return ""
    t = text.replace("\u00a0", " ")
    t = _WS.sub(" ", t).strip()
    low = t.lower()
    for pat in BOILER_PATTERNS:
        low = re.sub(pat, " ", low)
    return _WS.sub(" ", low).strip()


def _window(text: str, center: int, left: int = 500, right: int = 900) -> str:
    a = max(0, center - left)
    b = min(len(text), center + right)
    return text[a:b]


def make_protocol_index_text(raw_text: str, icd_codes: list[str]) -> str:
    """
    Focused indexing:
      - snippets around ICD mentions (+ base code)
      - snippets around symptom-ish sections
      - fallback: first chunk
    """
    t = raw_text or ""
    if not t.strip():
        return ""

    t = t.replace("\u00a0", " ")
    t = _WS.sub(" ", t)
    t_clean = strip_boilerplate(t) or t
    tl = t_clean.lower()

    chunks: list[str] = []

    # A) ICD windows
    for code in icd_codes or []:
        c = str(code).strip()
        if not c:
            continue
        base = c.split(".")[0]
        for needle in {c, base}:
            if not needle:
                continue
            j = tl.find(needle.lower())
            if j != -1:
                chunks.append(_window(t_clean, j))

    # B) symptom windows
    for key in SYMPTOM_KEYS:
        j = tl.find(key)
        if j != -1:
            chunks.append(_window(t_clean, j, left=400, right=1400))

    # de-dup lightly
    uniq: list[str] = []
    seen = set()
    for ch in chunks:
        k = ch[:250]
        if k not in seen:
            uniq.append(ch)
            seen.add(k)

    if not uniq:
        uniq = [t_clean[:2000]]

    return normalize_text(" ".join(uniq))


# ============================================================
# Model state loaded from model.pkl
# ============================================================
_word_vec: TfidfVectorizer | None = None
_char_vec: TfidfVectorizer | None = None
_word_X = None  # sparse
_char_X = None  # sparse
_proto_codes: list[list[str]] = []
_code_freq: Counter[str] = Counter()

ICD_NAME: dict[str, str] = {}


@dataclass
class LLMStatus:
    enabled: bool
    reason: str = ""


def llm_status() -> LLMStatus:
    if not (LLM_QUERY_EXPAND or LLM_RERANK):
        return LLMStatus(False, "LLM disabled by config")
    if not QAZCODE_BASE_URL:
        return LLMStatus(False, "QAZCODE_BASE_URL missing")
    if not QAZCODE_API_KEY:
        return LLMStatus(False, "QAZCODE_API_KEY missing")
    if not QAZCODE_MODEL:
        return LLMStatus(False, "QAZCODE_MODEL missing")
    return LLMStatus(True, "")


# ============================================================
# Build + save model.pkl
# ============================================================
def build_and_save_model(model_path: Path) -> None:
    """
    Builds TF-IDF over protocol texts and saves everything needed for retrieval:
      - vectorizers
      - sparse matrices
      - proto_codes
      - code_freq
    """
    protocols = pickle.load(open(PROCESSED_CORPUS_PATH, "rb"))

    proto_codes = [list(p.get("icd_codes", [])) for p in protocols]
    raw_texts = [p.get("text", "") or "" for p in protocols]

    if USE_FOCUSED_INDEX:
        texts = [make_protocol_index_text(raw_texts[i], proto_codes[i]) for i in range(len(protocols))]
    else:
        texts = [normalize_text(strip_boilerplate(t)) for t in raw_texts]

    code_freq = Counter(str(c).strip() for codes in proto_codes for c in codes if str(c).strip())

    # TF-IDF (fit happens here)
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=120_000,
        sublinear_tf=True,
        lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        max_features=180_000,
        sublinear_tf=True,
        lowercase=True,
    )

    word_X = word_vec.fit_transform(texts)
    char_X = char_vec.fit_transform(texts)

    payload = {
        "word_vec": word_vec,
        "char_vec": char_vec,
        "word_X": word_X,
        "char_X": char_X,
        "proto_codes": proto_codes,
        "code_freq": code_freq,
        "meta": {
            "n_protocols": len(protocols),
            "top_codes": TOP_CODES,
            "k_neighbors": K_NEIGHBORS,
            "alpha": ALPHA,
            "beta": BETA,
            "focused_index": USE_FOCUSED_INDEX,
            "corpus_path": str(PROCESSED_CORPUS_PATH),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model_path: Path) -> None:
    global _word_vec, _char_vec, _word_X, _char_X, _proto_codes, _code_freq
    with open(model_path, "rb") as f:
        m = pickle.load(f)
    _word_vec = m["word_vec"]
    _char_vec = m["char_vec"]
    _word_X = m["word_X"]
    _char_X = m["char_X"]
    _proto_codes = m["proto_codes"]
    _code_freq = m["code_freq"]


# ============================================================
# Retrieval (kNN + aggregation with alpha/beta like your version)
# ============================================================
def retrieve_candidates(symptoms: str, top_codes: int) -> list[str]:
    text = normalize_text(symptoms)
    if not text.strip():
        return [c for c, _ in _code_freq.most_common(top_codes)]

    assert _word_vec is not None and _char_vec is not None and _word_X is not None and _char_X is not None

    qw = _word_vec.transform([text])
    qc = _char_vec.transform([text])

    sw = (_word_X @ qw.T).toarray().ravel()
    sc = (_char_X @ qc.T).toarray().ravel()
    s = ALPHA * sw + (1.0 - ALPHA) * sc

    k = min(K_NEIGHBORS, len(s))
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx])]

    code_score = defaultdict(float)
    for rank, i in enumerate(idx, start=1):
        sim = float(s[i])
        if sim <= 0:
            continue
        w = sim / rank
        for code in _proto_codes[i]:
            code = str(code).strip()
            if code:
                code_score[code] += BETA * w

    # small stability prior
    for code, f in _code_freq.items():
        if code in code_score:
            code_score[code] += 0.0005 * np.log1p(f)

    best = sorted(code_score.items(), key=lambda x: x[1], reverse=True)
    return [c for c, _ in best[:top_codes]]


# ============================================================
# Qazcode LLM helpers (OpenAI-compatible chat/completions)
# ============================================================
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_expand_cache: dict[str, str] = {}


def _chat_url(base: str) -> str:
    # Accept base like:
    #   https://hub.qazcode.ai/v1
    #   https://hub.qazcode.ai
    # Ensure we call .../chat/completions
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


async def _qazcode_chat(client: httpx.AsyncClient, messages: list[dict[str, str]], max_tokens: int) -> str:
    payload = {
        "model": QAZCODE_MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {QAZCODE_API_KEY}",
        "Content-Type": "application/json",
    }
    r = await client.post(_chat_url(QAZCODE_BASE_URL), headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


async def llm_expand_query(client: httpx.AsyncClient, symptoms: str) -> str:
    s = (symptoms or "").strip()
    if not s:
        return ""
    if LLM_CACHE and s in _expand_cache:
        return _expand_cache[s]

    expanded = ""
    try:
        content = await _qazcode_chat(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You improve lexical retrieval. Convert the symptoms into a compact list of clinical keywords "
                        "and synonyms (prefer Russian medical terms when possible). "
                        "No ICD codes. No prose. Return ONLY JSON: {\"expanded\":\"...\"}."
                    ),
                },
                {"role": "user", "content": f"Symptoms: {s}"},
            ],
            max_tokens=160,
        )
        try:
            obj = json.loads(content)
        except Exception:
            m = _JSON_OBJ_RE.search(content)
            obj = json.loads(m.group(0)) if m else {}
        expanded = str(obj.get("expanded", "")).strip()
    except Exception:
        expanded = ""

    if LLM_CACHE:
        _expand_cache[s] = expanded
    return expanded


async def llm_pick_top3(client: httpx.AsyncClient, symptoms: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return []

    cand_lines = [f"- {c}: {ICD_NAME.get(c, '')}" for c in candidates]
    cand_text = "\n".join(cand_lines)

    picks: list[str] = []
    try:
        content = await _qazcode_chat(
            client,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Pick EXACTLY 3 ICD-10 codes ONLY from the candidate list. "
                        "Do NOT invent codes. Output JSON only: {\"codes\":[\"A\",\"B\",\"C\"]}."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Symptoms:\n{symptoms}\n\nCandidate ICD-10 codes:\n{cand_text}\n",
                },
            ],
            max_tokens=140,
        )
        try:
            obj = json.loads(content)
        except Exception:
            m = _JSON_OBJ_RE.search(content)
            obj = json.loads(m.group(0)) if m else {}
        raw = obj.get("codes", [])
        if isinstance(raw, list):
            picks = [str(x).strip() for x in raw if str(x).strip()]
    except Exception:
        picks = []

    allowed = set(candidates)
    picks = [c for c in picks if c in allowed]

    # fill deterministically to 3
    for c in candidates:
        if len(picks) >= 3:
            break
        if c not in picks:
            picks.append(c)

    return picks[:3]


# ============================================================
# FastAPI app
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ICD_NAME

    print("\n🏥 Mock Diagnostic Server (model.pkl + kNN TF-IDF + optional Qazcode LLM)")
    print("=" * 72)
    print("Endpoint: /diagnose (POST)  Body: {\"symptoms\": \"...\"}   Docs: /docs")
    print("=" * 72)

    # Load ICD names
    if ICD_NAME_MAP_PATH:
        ICD_NAME = json.loads(ICD_NAME_MAP_PATH.read_text(encoding="utf-8"))
        print(f"Loaded ICD name map: {ICD_NAME_MAP_PATH}")
    else:
        ICD_NAME = {}
        print("ICD name map not found; will use fallback names.")

    # Load or build model.pkl
    if MODEL_PATH.exists():
        load_model(MODEL_PATH)
        print(f"Loaded model: {MODEL_PATH}")
    else:
        print(f"Model not found: {MODEL_PATH}")
        print("Building TF-IDF index and saving model.pkl...")
        build_and_save_model(MODEL_PATH)
        load_model(MODEL_PATH)
        print(f"Built+saved model: {MODEL_PATH}")

    print(f"Params: TOP_CODES={TOP_CODES}, K_NEIGHBORS={K_NEIGHBORS}, ALPHA={ALPHA}, BETA={BETA}, focused={USE_FOCUSED_INDEX}")

    st = llm_status()
    print(f"LLM query expand: {'ON' if LLM_QUERY_EXPAND else 'OFF'}")
    print(f"LLM rerank:       {'ON' if LLM_RERANK else 'OFF'}")
    if (LLM_QUERY_EXPAND or LLM_RERANK) and not st.enabled:
        print(f"⚠️  LLM not usable: {st.reason}")

    app.state.llm_client = httpx.AsyncClient(timeout=float(LLM_TIMEOUT_S))
    try:
        yield
    finally:
        await app.state.llm_client.aclose()


app = FastAPI(title="Mock Diagnostic Server", lifespan=lifespan)


@app.post("/diagnose", response_model=DiagnoseResponse)
async def handle_diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    symptoms = request.symptoms or ""

    # LLM (optional)
    st = llm_status()
    retrieval_input = symptoms

    if LLM_QUERY_EXPAND and st.enabled:
        extra = await llm_expand_query(app.state.llm_client, symptoms)
        if extra:
            retrieval_input = f"{symptoms} {extra}"

    # retrieve candidates (same kNN+alpha+beta logic)
    want = max(TOP_CODES, LLM_CANDIDATES) if (LLM_RERANK and st.enabled) else TOP_CODES
    candidates = retrieve_candidates(retrieval_input, top_codes=want)

    # choose top3
    if LLM_RERANK and st.enabled and candidates:
        top3 = await llm_pick_top3(app.state.llm_client, symptoms, candidates[:LLM_CANDIDATES])
    else:
        top3 = candidates[:3]

    diagnoses: list[Diagnosis] = []
    for i, code in enumerate(top3):
        name = ICD_NAME.get(code, f"МКБ-10 {code}")
        diagnoses.append(Diagnosis(rank=i + 1, icd10_code=code, diagnosis=name))

    return DiagnoseResponse(diagnoses=diagnoses)