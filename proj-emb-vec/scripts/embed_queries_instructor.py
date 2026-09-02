# scripts/embed_queries_instructor.py  — pinned-stack env (sentence-transformers==2.2.2 + InstructorEmbedding)
import json, numpy as np
from pathlib import Path
from InstructorEmbedding import INSTRUCTOR

ROOT = Path(__file__).resolve().parent.parent
queries = json.loads((ROOT/"eval"/"eval_queries.json").read_text(encoding="utf-8"))["queries"]

Q_INSTRUCTION = "Represent the question for retrieving supporting documents:"  # query-side, differs from the doc instruction
m = INSTRUCTOR("hkunlp/instructor-xl")
pairs = [[Q_INSTRUCTION, q["query"]] for q in queries]
v = np.asarray(m.encode(pairs, normalize_embeddings=True), dtype=np.float32)
np.save(ROOT/"embeddings"/"queries"/"instructor-xl.npy", v)
print(v.shape)