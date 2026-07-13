# scripts/test_double_normalize.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graphs.build_graph import normalize_entity

raw = "남녀"
once = normalize_entity(raw)
twice = normalize_entity(once)

print(f"raw:   {raw!r}")
print(f"once:  {once!r}")
print(f"twice: {twice!r}")
print(f"once == twice: {once == twice}")
print(f"once bytes:  {once.encode('utf-8')}")
print(f"twice bytes: {twice.encode('utf-8')}")