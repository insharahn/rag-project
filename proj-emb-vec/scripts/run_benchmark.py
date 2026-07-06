"""
scripts/run_benchmark.py

Runs the full grid: every embedding model x every vector DB.
For each (model, db): build the index, run all 45 eval queries, capture the
top-K retrieved chunk_ids + per-query search latency + build time + memory.

Writes one raw file: results/benchmark_raw.json
Everything (model-quality metrics, DB latency/recall) derives from it later.

Docker must be up for Milvus:
    docker compose up -d
    python scripts/run_benchmark.py
"""
import json, time
from pathlib import Path
import numpy as np
import gc

try:
    import psutil
    def rss_mb(): return psutil.Process().memory_info().rss / 1e6
except ImportError:
    print("(psutil not installed — memory will be null. `pip install psutil` to enable)")
    def rss_mb(): return None
    
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# --- adapter imports: adjust class names here if yours differ ---
from src.db.faiss_db import FaissDB
from src.db.chroma_db import ChromaDB
from src.db.milvus_db import MilvusDB

ROOT = Path(__file__).resolve().parent.parent
EMB  = ROOT / "embeddings"
QEMB = EMB / "queries"
RES  = ROOT / "results"; RES.mkdir(exist_ok=True)

MODELS = ["bge-large", "bge-m3", "e5-large", "multilingual-e5-large", "instructor-xl"]
DBS    = {"faiss": FaissDB, "chroma": ChromaDB, "milvus": MilvusDB}

def warmup_milvus():
    """Cold-start Milvus once so the first real run isn't hit by etcd timeouts / lazy init."""
    try:
        import numpy as np
        db = MilvusDB()
        dummy = np.random.rand(10, 8).astype(np.float32)
        db.build(dummy, [f"w{i}" for i in range(10)])
        db.search(dummy[0], 3)
        db.teardown()
        print("[warmup] milvus ready")
    except Exception as e:
        print(f"[warmup] milvus warmup skipped: {e}")

K          = 10   # retrieve top-10 -> lets us compute recall@1/3/5/10 + MRR
WARMUP     = 5    # warmup searches discarded before timing
LAT_REPS   = 20   # repeat each query's search N times for a real latency distribution
                  # (45 queries x 20 = 900 samples -> meaningful p95/p99)

queries = json.loads((ROOT / "eval" / "eval_queries.json").read_text(encoding="utf-8"))["queries"]


def load_corpus(model):
    vecs = np.load(EMB / f"{model}.npy")
    ids  = json.loads((EMB / f"{model}.meta.json").read_text(encoding="utf-8"))["chunk_ids"]
    assert len(vecs) == len(ids), f"{model}: vec/id length mismatch"
    return vecs, ids


def load_query_vecs(model):
    qv = np.load(QEMB / f"{model}.npy")
    assert len(qv) == len(queries), (
        f"{model}: {len(qv)} query vecs but {len(queries)} eval queries — "
        f"re-run embed_queries.py, order must match eval_queries.json")
    return qv


def rank_of(answer_id, retrieved_ids):
    """1-based rank of the true answer in the retrieved list, or None if absent."""
    try:
        return retrieved_ids.index(answer_id) + 1
    except ValueError:
        return None


def run():
    warmup_milvus()
    runs = []
    for model in MODELS:
        print(f"\n########## {model} ##########")
        corpus_vecs, corpus_ids = load_corpus(model)
        qvecs = load_query_vecs(model)
        dim = int(corpus_vecs.shape[1])

        for db_name, DBClass in DBS.items():
            print(f"  -- {db_name} --", flush=True)
            db = DBClass()

            # ---- build (median of N, de-noises Docker/OS) ----
            BUILD_REPS = 2 if db_name == "milvus" else 5
            build_times = []
            mem_mb = None
            for rep in range(BUILD_REPS):
                import gc; gc.collect()
                m0 = rss_mb()
                t0 = time.perf_counter()
                db.build(corpus_vecs, corpus_ids)
                build_times.append(time.perf_counter() - t0)
                m1 = rss_mb()
                if rep == 0:  # measure memory on first build only (later builds may hit warm caches)
                    mem_mb = (m1 - m0) if (m0 is not None and db_name != "milvus") else None
                if rep < BUILD_REPS - 1:
                    db.teardown()          # tear down between rebuilds
                    db = DBClass()         # fresh instance for next build
            build_s = float(np.median(build_times))

            # ---- warmup ----
            for i in range(min(WARMUP, len(qvecs))):
                db.search(qvecs[i], K)

            # ---- correctness pass (once; search is deterministic) ----
            per_query = []
            for qi, q in enumerate(queries):
                results = db.search(qvecs[qi], K)          # [(chunk_id, score), ...] higher=better
                retrieved = [cid for cid, _ in results]
                per_query.append({
                    "query_idx": qi,
                    "language": q["language"],
                    "answer_chunk_id": q["answer_chunk_id"],
                    "retrieved": retrieved,
                    "rank": rank_of(q["answer_chunk_id"], retrieved),
                })

            # ---- latency pass (repeated -> distribution) ----
            lat_ms = []
            for _ in range(LAT_REPS):
                for qi in range(len(qvecs)):
                    t = time.perf_counter()
                    db.search(qvecs[qi], K)
                    lat_ms.append((time.perf_counter() - t) * 1000.0)

            db.teardown()

            hits1 = sum(1 for r in per_query if r["rank"] == 1)
            print(f"     recall@1={hits1}/{len(queries)}  "
                  f"build={build_s:.2f}s  p50={np.percentile(lat_ms,50):.2f}ms"
                  + (f"  mem={mem_mb:.0f}MB" if mem_mb else ""))

            runs.append({
                "model": model, "db": db_name, "dim": dim,
                "build_seconds": round(build_s, 4),
                "index_memory_mb": round(mem_mb, 1) if mem_mb else None,
                "search_latencies_ms": [round(x, 4) for x in lat_ms],
                "per_query": per_query,
            })

    out = {
        "config": {
            "k": K, "n_queries": len(queries), "latency_reps": LAT_REPS,
            "models": MODELS, "dbs": list(DBS),
            "note_milvus": "Milvus search latency includes a localhost TCP round-trip "
                           "that in-process FAISS/Chroma do not incur; memory is out-of-process.",
        },
        "runs": runs,
    }
    (RES / "benchmark_raw.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"\n[done] {len(runs)} runs -> results/benchmark_raw.json")


if __name__ == "__main__":
    run()
