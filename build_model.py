import pickle
from pathlib import Path
from collections import Counter
import json
import re
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

PKL_PATH = (DATA_DIR / "processed_corpus.pkl") if (DATA_DIR / "processed_corpus.pkl").exists() else (PROJECT_ROOT / "processed_corpus.pkl")
OUT_PATH = DATA_DIR / "model.pkl"

# --- same preprocessing helpers (copy from your server) ---
TOKEN_RE = re.compile(r"[a-zа-я0-9]+", re.IGNORECASE)
_WS = re.compile(r"\s+")
KEEP_SHORT = {"не", "no"}

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
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    toks = []
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
    t = raw_text or ""
    if not t.strip():
        return ""
    t = t.replace("\u00a0", " ")
    t = _WS.sub(" ", t)
    t_clean = strip_boilerplate(t) or t
    tl = t_clean.lower()

    chunks = []
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

    for key in SYMPTOM_KEYS:
        j = tl.find(key)
        if j != -1:
            chunks.append(_window(t_clean, j, left=400, right=1400))

    uniq, seen = [], set()
    for ch in chunks:
        k = ch[:250]
        if k not in seen:
            uniq.append(ch)
            seen.add(k)

    if not uniq:
        uniq = [t_clean[:2000]]

    return normalize_text(" ".join(uniq))

def main():
    protocols = pickle.load(open(PKL_PATH, "rb"))
    proto_codes = [list(p.get("icd_codes", [])) for p in protocols]
    raw_texts = [p.get("text", "") or "" for p in protocols]

    texts = [make_protocol_index_text(raw_texts[i], proto_codes[i]) for i in range(len(protocols))]
    code_freq = Counter(str(c).strip() for codes in proto_codes for c in codes if str(c).strip())

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

    model = {
        "word_vec": word_vec,
        "char_vec": char_vec,
        "word_X": word_X,
        "char_X": char_X,
        "proto_codes": proto_codes,
        "code_freq": code_freq,
        "n_protocols": len(protocols),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ Saved model to: {OUT_PATH}")
    print(f"Protocols: {len(protocols)} | Nonempty codes: {sum(1 for x in proto_codes if x)}")

if __name__ == "__main__":
    main()