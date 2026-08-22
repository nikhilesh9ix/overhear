"""Download one per-language MSMARCO-XI parquet locally (streaming reads are too slow)."""
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "hf_cache"))

from huggingface_hub import hf_hub_download  # noqa: E402

LANG_FILE = {
    "as": "asm", "bn": "ben", "gu": "guj", "hi": "hin", "kn": "kan", "ml": "mal",
    "mr": "mar", "ne": "nep", "or": "ori", "pa": "pan", "sa": "san", "ta": "tam",
    "te": "tel", "ur": "urd",
}

lang = sys.argv[1] if len(sys.argv) > 1 else "hi"
split = sys.argv[2] if len(sys.argv) > 2 else "validation"
fn = f"{split}/{LANG_FILE[lang]}{'val' if split == 'validation' else 'train'}.parquet"

t0 = time.perf_counter()
print(f"downloading {fn} ...", flush=True)
p = hf_hub_download("ai4bharat/MSMARCO-XI", fn, repo_type="dataset",
                    token=os.environ.get("HF_TOKEN") or None)
sz = pathlib.Path(p).stat().st_size / 1e6
dt = time.perf_counter() - t0
print(f"done: {p}\n{sz:.1f} MB in {dt:.1f}s ({sz/max(dt,0.001):.1f} MB/s)", flush=True)
