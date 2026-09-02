"""
scripts/run_benchmark.py

Docker must be up for Milvus:
    docker compose up -d
    python scripts/run_benchmark.py
"""
import json, time
from pathlib import Path
import numpy as np

try:
    import psutil
    def rss_mb(): return psutil.Process().memory_info().rss / 1e6
except ImportError:
    print("(psutil not installed — memory will be null)")
    def rss_mb(): return None

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.db.faiss_db import FaissDB
from src.db.chroma_db import ChromaDB
from src.db.milvus_db import MilvusDB

ROOT = Path(__file__).resolve().parent.parent
DBS  = {"faiss": FaissDB, "chroma": ChromaDB, "milvus": MilvusDB}


MODELS = ["bge-large", "bge-m3", "e5-large", "multilingual-e5-large", "instructor-xl"] 
K        = 10
WARMUP   = 5
LAT_REPS = 20

# ---  strategy -> path map ---
STRATEGY_PATHS = {
    "semantic":  {
        "emb":     ROOT / "embeddings",
        "eval":    ROOT / "eval" / "eval_queries.json",
        "results": ROOT / "results",
    },
    "fixed": {
        "emb":     ROOT / "embeddings_fixed",
        "eval":    ROOT / "eval_fixed" / "eval_queries.json",
        "results": ROOT / "results_fixed",
    },
    "recursive": {
        "emb":     ROOT / "embeddings_recursive",
        "eval":    ROOT / "eval_recursive" / "eval_queries.json",
        "results": ROOT / "results_recursive",
    },
}

STRATEGIES_TO_RUN = ["recursive"]  # remove any, add any


def warmup_milvus():
    try:
        db = MilvusDB()
        dummy = np.random.rand(10, 8).astype(np.float32)
        db.build(dummy, [f"w{i}" for i in range(10)])
        db.search(dummy[0], 3)
        db.teardown()
        print("[warmup] milvus ready")
    except Exception as e:
        print(f"[warmup] milvus warmup skipped: {e}")


def load_corpus_vecs(model, emb_dir):
    vecs = np.load(emb_dir / f"{model}.npy")
    ids  = json.loads((emb_dir / f"{model}.meta.json").read_text(encoding="utf-8"))["chunk_ids"]
    assert len(vecs) == len(ids), f"{model}: vec/id length mismatch"
    return vecs, ids


def load_query_vecs(model, emb_dir, n_queries):
    qv = np.load(emb_dir / "queries" / f"{model}.npy")
    assert len(qv) == n_queries, (
        f"{model}: {len(qv)} query vecs but {n_queries} eval queries")
    return qv


def rank_of(answer_id, retrieved_ids):
    try:
        return retrieved_ids.index(answer_id) + 1
    except ValueError:
        return None


def run_strategy(strategy: str, paths: dict):
    emb_dir  = paths["emb"]
    eval_path = paths["eval"]
    res_dir  = paths["results"]
    res_dir.mkdir(exist_ok=True)

    queries = json.loads(eval_path.read_text(encoding="utf-8"))["queries"]
    # filter to only ok-remapped queries for fixed/recursive
    # (semantic has no remap_status field so the filter is a no-op there)
    queries = [q for q in queries if q.get("remap_status", "ok") == "ok"]

    print(f"\n{'='*60}")
    print(f"STRATEGY: {strategy}  ({len(queries)} queries, emb={emb_dir.name})")
    print(f"{'='*60}")

    runs = []
    # resume: load any existing checkpoint for this strategy
    out_path = res_dir / "benchmark_raw.json"
    completed_keys = set()
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            runs = existing.get("runs", [])
            completed_keys = {(r["model"], r["db"]) for r in runs}
            print(f"  [resume] loaded {len(runs)} existing runs from checkpoint")
        except Exception:
            pass

    for model in MODELS:
        print(f"\n########## {model} ##########")
        corpus_vecs, corpus_ids = load_corpus_vecs(model, emb_dir)
        qvecs = load_query_vecs(model, emb_dir, len(queries))
        dim = int(corpus_vecs.shape[1])

        for db_name, DBClass in DBS.items():
            if (model, db_name) in completed_keys:
                print(f"  -- {db_name} -- [skipped, already in checkpoint]")
                continue
            
            print(f"  -- {db_name} --", flush=True)
            db = DBClass()

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
                if rep == 0:
                    mem_mb = (m1 - m0) if (m0 is not None and db_name != "milvus") else None
                if rep < BUILD_REPS - 1:
                    db.teardown()
                    db = DBClass()
            build_s = float(np.median(build_times))

            for i in range(min(WARMUP, len(qvecs))):
                db.search(qvecs[i], K)

            per_query = []
            for qi, q in enumerate(queries):
                results  = db.search(qvecs[qi], K)
                retrieved = [cid for cid, _ in results]
                per_query.append({
                    "query_idx":      qi,
                    "language":       q["language"],
                    "answer_chunk_id": q["answer_chunk_id"],
                    "retrieved":      retrieved,
                    "rank":           rank_of(q["answer_chunk_id"], retrieved),
                })

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
            
        # checkpoint after each model (all 3 DBs done) so a Milvus crash
        # doesn't lose completed models
        out = {
            "config": {
                "strategy": strategy,
                "k": K, "n_queries": len(queries), "latency_reps": LAT_REPS,
                "models": MODELS, "dbs": list(DBS),
                "note_milvus": "Milvus latency includes localhost TCP round-trip.",
            },
            "runs": runs,
        }
        out_path = res_dir / "benchmark_raw.json"
        out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [checkpoint] {model} saved ({len(runs)} runs so far)")

    out = {
        "config": {
            "strategy": strategy,
            "k": K, "n_queries": len(queries), "latency_reps": LAT_REPS,
            "models": MODELS, "dbs": list(DBS),
            "note_milvus": "Milvus latency includes localhost TCP round-trip.",
        },
        "runs": runs,
    }
    out_path = res_dir / "benchmark_raw.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[{strategy}] {len(runs)} runs -> {out_path}")


def run():
    warmup_milvus()
    for strategy in STRATEGIES_TO_RUN:
        run_strategy(strategy, STRATEGY_PATHS[strategy])


if __name__ == "__main__":
    run()