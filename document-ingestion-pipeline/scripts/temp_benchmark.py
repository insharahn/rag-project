from pipeline.ingest import ingest_document
from pipeline.clean import clean_text
from pipeline.chunk_semantic import chunk_semantic
from benchmark.run_benchmark import semantic_coherence  # your fixed version

for path in ["pdfs/1.-Of-Mice-Men-Full-Text.pdf", "htmls/Travel literature - Wikipedia.html", "pdfs\CAMUS, Albert - The Stranger.pdf"]:
    cleaned = clean_text(ingest_document(path)["full_text"]) or ""
    chunks = chunk_semantic(cleaned)
    print(path, semantic_coherence(chunks))