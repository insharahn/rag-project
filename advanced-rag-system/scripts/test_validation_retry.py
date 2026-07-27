# scripts/test_validation_retry.py
"""
Tests the validation retry loop: run several queries through the full
workflow, and print out whether validation passed, whether a retry fired,
and what the final answer looks like — so we can see the loop mechanics
regardless of which specific query happens to trigger a first-pass failure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.workflow import workflow

test_cases = [
   # {"query": "What is a class in OOP?", "language": "en", "top_k": 5, "history": []},
    {"query": "What was Meursault's job and did his employer offer him a new position?", "language": "en", "top_k": 5, "history": []},
]

for case in test_cases:
    print(f"\n{'='*70}")
    print(f"QUERY: {case['query']}")
    print('='*70)

    result = workflow.invoke(case)

    print(f"\nresearch_expanded:     {result.get('research_expanded')}")
    print(f"validation_passed:     {result.get('validation_passed')}")
    print(f"_retry_pass fired:     {result.get('_retry_pass', False)}")
    print(f"validation_issues:     {result.get('validation_issues')}")
    print(f"\nFinal answer:\n{result.get('final_answer', result.get('draft_answer'))}")