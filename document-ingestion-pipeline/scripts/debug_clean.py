"""One-off debug: check whether a cleaned document's text still contains
an expected phrase. Useful for confirming that something that looks
'missing' in a terminal-pasted preview is actually present in the real
string — terminal copy/paste can silently trim whitespace at line-wrap
points, which looks identical to a real bug but isn't one.
"""

import sys

from pipeline.ingest import ingest_document
from pipeline.clean import clean_text


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python -m scripts.debug_clean "<path>" "<phrase to check for>"')
        sys.exit(1)

    path, phrase = sys.argv[1], sys.argv[2]
    cleaned = clean_text(ingest_document(path)["full_text"])

    print(f"Phrase {phrase!r} found: {phrase in cleaned}")