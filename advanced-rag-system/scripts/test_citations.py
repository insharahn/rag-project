import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.pipeline import retrieve
from generation.citation_generator import generate_answer

queries = [
    "the outsider guy's job",              # expected: low confidence, hedge
    "Albert Camus death philosophy",        # expected: high confidence, cited answer
]

for q in queries:
    print(f"QUERY: {q}")
    chunks = retrieve(q, top_k=5)
    result = generate_answer(q, chunks)
    print(f"CONFIDENCE: {result['confidence']} (top_score={result['top_score']:.4f})")
    print(f"ANSWER:\n{result['answer']}\n")
    if result["confidence"] == "high":
        print("SOURCES:")
        for i, src in result["sources"].items():
            print(f"  [{i}] {src['source_doc']} (rerank={src['rerank_score']:.4f})")
    print("-" * 60)