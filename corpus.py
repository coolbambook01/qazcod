import json
import pickle
from pathlib import Path

INPUT_PATH = Path("data/protocols_corpus.jsonl")
OUTPUT_PATH = Path("data/processed_corpus.pkl")

def load_corpus_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def clean_text(text: str) -> str:
    text = (text or "").replace("\n", " ").replace("\t", " ")
    return " ".join(text.split())

def main():
    print("Loading corpus...")
    protocols = load_corpus_jsonl(INPUT_PATH)
    print("Total protocols loaded:", len(protocols))

    processed = []
    for p in protocols:
        processed.append({
            "protocol_id": p.get("protocol_id"),
            "source_file": p.get("source_file", ""),
            "title": p.get("title", ""),
            "icd_codes": p.get("icd_codes", []),
            "text": clean_text(p.get("text", "")),
        })

    print("Saving processed corpus...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        pickle.dump(processed, f)

    print("Done:", OUTPUT_PATH)

if __name__ == "__main__":
    main()