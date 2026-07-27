# eval/validation_agent_eval.py
"""
Tests whether the validation agent actually catches bad drafts — not
just whether it passes good ones (already demonstrated). Uses real
retrieved chunks with hand-corrupted draft answers, so we know the
correct verdict in advance.
"""
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.retrieval_agent import retrieval_node
from agents.validation_agent import validation_node

TEST_CASES = [
    {
        "query": "What attributes does the Car class define in Python?",
        "corruption_type": "invented_fact",
        # deliberately claims an attribute NOT in the real chunk
        "corrupted_draft": "The Car class defines attributes for make, model, year, and top_speed_mph, which is used to calculate acceleration curves [1].",
        "should_pass": False,
    },
    {
        "query": "Who designed the C programming language and when?",
        "corruption_type": "wrong_citation",
        # correct-sounding fact, but cites a chunk number that (after retrieval)
        # will likely not correspond to where this specific fact appears
        "corrupted_draft": "C was designed by Dennis Ritchie in 1972 at Bell Labs [3].",
        "should_pass": False,  # only valid if [3] isn't actually the supporting chunk — verify against real retrieval below
    },
    {
        "query": "What did Segur buy with money from Tallard?",
        "corruption_type": "dodges_question",
        # doesn't actually answer what was asked, talks around it instead
        "corrupted_draft": "Tallard was a general involved in the war in Italy that year [1]. Financial arrangements between generals were common during this period.",
        "should_pass": False,
    },
]

results = []
for case in TEST_CASES:
    state = {"query": case["query"], "top_k": 5, "history": []}
    state = retrieval_node(state)

    print(f"\nQUERY: {case['query']}")
    print(f"Retrieved top chunk: {state['retrieved_chunks'][0][0]} (score {state['top_score']:.3f})")
    print("Retrieved sources (for citation-number sanity check):")
    for i, (cid, chunk, score) in enumerate(state["retrieved_chunks"], 1):
        print(f"  [{i}] {cid}")

    # inject the hand-written corrupted draft directly, bypassing summarization
    state["draft_answer"] = case["corrupted_draft"]
    state["draft_sources"] = {
        str(i): {"chunk_id": cid, "source_doc": chunk.get("source_doc", cid), "rerank_score": score}
        for i, (cid, chunk, score) in enumerate(state["retrieved_chunks"], 1)
    }

    state = validation_node(state)

    passed = state["validation_passed"]
    correct = passed == case["should_pass"]

    results.append({
        "query": case["query"],
        "corruption_type": case["corruption_type"],
        "corrupted_draft": case["corrupted_draft"],
        "should_pass": case["should_pass"],
        "actual_passed": passed,
        "validation_issues": state.get("validation_issues"),
        "correct": correct,
    })

    status = "✓" if correct else "✗"
    print(f"{status} validation_passed={passed} (expected {case['should_pass']})")
    print(f"  issues: {state.get('validation_issues')}")

accuracy = sum(r["correct"] for r in results) / len(results)
print(f"\n{'='*60}\nAccuracy: {sum(r['correct'] for r in results)}/{len(results)} = {accuracy:.2%}")

RESULTS_PATH = Path(__file__).resolve().parent / "agent_results" / "validation_agent_eval.json"
RESULTS_PATH.parent.mkdir(exist_ok=True)
RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")