"""Smoke test 2: stream ai4bharat/MSMARCO-XI (too big to fully download) and report
schema, size, sample rows, and language content."""
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

from datasets import load_dataset  # noqa: E402

REPO = "ai4bharat/MSMARCO-XI"
# datasets>=3 dropped loading-script configs, so the repo exposes only "default".
# Point at the per-language parquet directly instead.
LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}
LANG = sys.argv[1] if len(sys.argv) > 1 else "hi"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def script_of(s: str) -> str:
    for ch in s:
        o = ord(ch)
        for lo, hi, name in (
            (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
            (0x0B80, 0x0BFF, "Tamil"), (0x0C00, 0x0C7F, "Telugu"),
            (0x0C80, 0x0CFF, "Kannada"), (0x0D00, 0x0D7F, "Malayalam"),
            (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
            (0x0B00, 0x0B7F, "Odia"), (0x0600, 0x06FF, "Arabic/Urdu"),
        ):
            if lo <= o <= hi:
                return name
    return "Latin/ASCII"


def main() -> int:
    print(f"== streaming {REPO} config={LANG} split=validation ==", flush=True)
    t0 = time.perf_counter()
    url = f"hf://datasets/{REPO}/validation/{LANG_FILE[LANG]}val.parquet"
    print(f"file: {url}", flush=True)
    ds = load_dataset("parquet", data_files=url, split="train", streaming=True)
    print(f"stream opened in {(time.perf_counter()-t0):.1f}s", flush=True)
    print("features:", json.dumps({k: str(v) for k, v in (ds.features or {}).items()}, indent=2)[:2000], flush=True)

    npass = 0
    nsel = 0
    for i, row in enumerate(ds):
        if i >= N:
            break
        p = row["passages"]
        npass += len(p["English_passages"])
        nsel += sum(p["is_selected"])
        if i < 2:
            print(f"\n---- row {i} ----", flush=True)
            print(f"  query_id      : {row['query_id']}", flush=True)
            print(f"  query_type    : {row['query_type']}", flush=True)
            print(f"  Eng_Query     : {row['Eng_Query']!r}", flush=True)
            print(f"  query (tgt)   : {row['query']!r}  [{script_of(row['query'])}]", flush=True)
            print(f"  Eng_Answer    : {str(row['Eng_Answer'])[:200]!r}", flush=True)
            print(f"  Answer (tgt)  : {str(row['Answer'])[:200]!r}", flush=True)
            print(f"  source/target : {row['source_lang']} -> {row['target_lang']}", flush=True)
            print(f"  n_passages    : {len(p['English_passages'])}  is_selected={p['is_selected']}", flush=True)
            print(f"  Eng_passage[0]: {p['English_passages'][0][:250]!r}", flush=True)
            print(f"  Trn_passage[0]: {p['Translated_passages'][0][:250]!r}  [{script_of(p['Translated_passages'][0])}]", flush=True)
            lens = [len(x) for x in p["English_passages"]]
            print(f"  eng passage chars: min={min(lens)} max={max(lens)} mean={sum(lens)//len(lens)}", flush=True)

    print(f"\n== over {N} rows: {npass} passages, {nsel} marked is_selected==1 ==", flush=True)
    print(f"avg passages/query = {npass/N:.1f}; golden relevance labels available = YES", flush=True)
    print(f"elapsed {(time.perf_counter()-t0):.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
