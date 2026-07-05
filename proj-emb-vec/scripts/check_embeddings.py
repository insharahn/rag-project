import numpy as np, json
from pathlib import Path
from loader import load_corpus

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "embeddings"

corpus = load_corpus()
expected_ids = [c["chunk_id"] for c in corpus]
n = len(expected_ids)
print(f"corpus: {n} chunks\n")

models = ["bge-large", "e5-large", "multilingual-e5-large", "bge-m3", "instructor-xl"]

print(f"{'model':<24}{'shape':<14}{'dim':<6}{'norm':<8}{'NaN?':<6}{'aligned?'}")
print("-" * 70)
for m in models:
    npy = EMB / f"{m}.npy"
    meta = EMB / f"{m}.meta.json"
    if not npy.exists():
        print(f"{m:<24}MISSING .npy")
        continue
    vecs = np.load(npy)
    md = json.loads(meta.read_text())

    norm = float(np.linalg.norm(vecs[0]))          # should be ~1.0
    has_nan = bool(np.isnan(vecs).any())
    aligned = (md["chunk_ids"] == expected_ids)     # exact ID order match
    count_ok = vecs.shape[0] == n

    flag = "OK" if (aligned and count_ok and not has_nan) else "*** CHECK ***"
    print(f"{m:<24}{str(vecs.shape):<14}{vecs.shape[1]:<6}"
          f"{norm:<8.3f}{str(has_nan):<6}{aligned} {flag}")