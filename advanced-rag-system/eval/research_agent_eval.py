# eval/research_agent_eval.py
"""
Tests whether the research agent's expand/don't-expand decision matches
expectation on a small, deliberately chosen set of cases — not the full
45-query set, to conserve LLM budget (each case costs 2-4 LLM calls).
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node
from agents.research_agent import research_node

TEST_CASES = [
    {"query": "What attributes does the Car class define in Python?", "expected_expand": False},
    {"query": "Who designed the C programming language and when?", "expected_expand": False},
    {"query": "What four basic functions does Matplotlib use for plotting?", "expected_expand": False},
    {"query": "How does the StringTokenizer work in the Java code example?", "expected_expand": False},
    {"query": "What is the definition of a sentence according to the grammar text?", "expected_expand": False},
    {"query": "What was Meursault's job and did his employer offer him a new position?", "expected_expand": True},
]

results = []
for case in TEST_CASES:
    state = {"query": case["query"], "top_k": 5, "history": []}
    state = retrieval_node(state)
    top_score = state["top_score"]
    state = research_node(state)
    actual_expand = state["research_expanded"]
    correct = actual_expand == case["expected_expand"]

    results.append({
        "query": case["query"],
        "expected_expand": case["expected_expand"],
        "actual_expand": actual_expand,
        "top_score": round(top_score, 3),
        "correct": correct,
    })

    status = "✓" if correct else "✗"
    print(f"{status} [{top_score:.3f}] expand={actual_expand} (expected {case['expected_expand']}) — {case['query'][:60]}")

accuracy = sum(r["correct"] for r in results) / len(results)
print(f"\nAccuracy: {sum(r['correct'] for r in results)}/{len(results)} = {accuracy:.2%}")

RESULTS_PATH = Path(__file__).resolve().parent / "agent_results" / "research_agent_eval.json"
RESULTS_PATH.parent.mkdir(exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Saved to {RESULTS_PATH}")