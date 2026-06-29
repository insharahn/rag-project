"""Extract main text content from HTML documents.

Unlike a raw HTML-to-text dump, trafilatura strips navigation, ads, and
other boilerplate to keep just the article-like main content — similar
to what a browser's "reader mode" does.
"""

import sys

import trafilatura


def extract_html_text(path: str) -> dict:
    """Extract main text content from a local HTML file.

    Returns a dict with:
      - full_text: extracted main content (None if extraction failed)
      - word_count: number of words in the extracted text
    """
    with open(path, "rb") as f:
        html_content = f.read()

    full_text = trafilatura.extract(html_content)
    word_count = len(full_text.split()) if full_text else 0

    return {
        "full_text": full_text,
        "word_count": word_count,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pipeline/extract_html.py <path-to-html>")
        sys.exit(1)

    result = extract_html_text(sys.argv[1])

    if result["full_text"] is None:
        print("Extraction failed — trafilatura found no main content.")
        print("This can happen on very short pages, or pages with an unusual layout.")
        sys.exit(1)

    print(f"Word count: {result['word_count']}")
    print("\n--- First 500 characters of extracted text ---\n")
    print(result["full_text"][:500])