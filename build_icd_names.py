import re
import json
import pickle
from collections import Counter
from pathlib import Path

CORPUS_PKL = Path("data/processed_corpus.pkl")
OUT_JSON = Path("data/icd_name_map.json")

# Some PDFs use a non-standard dot: '․' (U+2024) or comma.
DOTS = r"\.|,|\u2024"

# ICD like A84, G91.0, D19․1 (we will normalize to D19.1)
CODE_RE = rf"[A-Z]\d{{2}}(?:[{DOTS}]\d)?"

PAT = re.compile(
    rf"(?<![A-Z0-9])({CODE_RE})\s*[:\-–—]?\s*([А-Яа-яA-Za-z][^;:\n\r]{{3,140}})",
    flags=re.IGNORECASE,
)

BAD_PREFIX = re.compile(r"^(код|коды|мкб|мкб-10|icd|протокол|раздел)\b", re.I)

def norm_code(code: str) -> str:
    code = code.upper().strip()
    code = code.replace("\u2024", ".").replace(",", ".")
    return code

def clean_name(s: str) -> str:
    s = " ".join(s.split())
    # stop if another ICD code starts inside the tail
    s = re.split(rf"(?<![A-Z0-9]){CODE_RE}", s)[0].strip()
    s = s.strip(" .,-–—:;()[]«»\"'")
    if BAD_PREFIX.search(s):
        return ""
    if not (3 <= len(s) <= 90):
        return ""
    return s

def main():
    if not CORPUS_PKL.exists():
        raise FileNotFoundError(f"Missing {CORPUS_PKL}. Build processed_corpus.pkl first.")

    corpus = pickle.load(open(CORPUS_PKL, "rb"))

    votes: dict[str, Counter] = {}

    for doc in corpus:
        text = doc.get("text", "") or ""
        for code_raw, raw_name in PAT.findall(text):
            code = norm_code(code_raw)
            name = clean_name(raw_name)
            if not name:
                continue
            votes.setdefault(code, Counter())[name] += 1

    icd2name = {}
    for code, c in votes.items():
        name, _ = c.most_common(1)[0]
        icd2name[code] = name

    OUT_JSON.write_text(json.dumps(icd2name, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved", OUT_JSON, "entries:", len(icd2name))

if __name__ == "__main__":
    main()
