from collections import Counter, defaultdict
from loader import load_corpus

def alpha_ratio(text):
    ns = [c for c in text if not c.isspace()]
    if not ns:
        return 0.0
    return sum(c.isalpha() for c in ns) / len(ns)

def main():
    corpus = load_corpus()

    docs = defaultdict(list)
    for c in corpus:
        docs[c["source_doc"]].append(c)

    lang_docs   = Counter(cs[0]["language"] for cs in docs.values())
    lang_chunks = Counter(c["language"] for c in corpus)

    print(f"\n=== CORPUS ===")
    print(f"docs: {len(docs)}   chunks: {len(corpus)}   "
          f"avg chunks/doc: {len(corpus)/len(docs):.1f}")

    print(f"\n=== BY LANGUAGE (docs / chunks) ===")
    for lang in lang_docs:
        print(f"  {lang or '??':<4} {lang_docs[lang]:>3} docs   {lang_chunks[lang]:>4} chunks")

    print(f"\n=== 10 LOWEST alpha-ratio docs (likely garbled) ===")
    ranked = sorted(docs.items(),
                    key=lambda kv: alpha_ratio(" ".join(c["text"] for c in kv[1])))
    for name, cs in ranked[:10]:
        r = alpha_ratio(" ".join(c["text"] for c in cs))
        flag = "  <-- CHECK" if r < 0.55 else ""
        print(f"  {r:.2f}  {name}{flag}")

if __name__ == "__main__":
    main()