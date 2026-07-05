# scripts/embed_queries.py
import json, time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
queries = json.loads((ROOT/"eval"/"eval_queries.json").read_text(encoding="utf-8"))["queries"]
texts = [q["query"] for q in queries]

OUT = ROOT/"embeddings"/"queries"; OUT.mkdir(parents=True, exist_ok=True)

# (name, hf_id, query_prefix)   — NOTE: query prefix, not passage
STD = [
    ("bge-large",             "BAAI/bge-large-en-v1.5",         ""),           # bge query instruction is optional; omit for symmetry
    ("bge-m3",                "BAAI/bge-m3",                    ""),
    ("e5-large",              "intfloat/e5-large-v2",           "query: "),    # NOT passage:
    ("multilingual-e5-large", "intfloat/multilingual-e5-large", "query: "),
]

for name, hf_id, prefix in STD:
    print(f"=== {name} (prefix={prefix!r}) ===")
    m = SentenceTransformer(hf_id, device="cpu")
    t = time.time()
    v = m.encode([prefix+x for x in texts], normalize_embeddings=True,
                 convert_to_numpy=True).astype(np.float32)
    np.save(OUT/f"{name}.npy", v)
    print(f"  {v.shape} in {time.time()-t:.0f}s")

# order MUST match eval_queries.json order — it's the index into `queries`
(OUT/"query_order.json").write_text(
    json.dumps([q["query"] for q in queries], ensure_ascii=False),
    encoding="utf-8")
print("saved query vectors + order")