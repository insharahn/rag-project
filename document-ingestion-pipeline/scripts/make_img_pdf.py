"""Dev utility: turn a normal text PDF into an image-only PDF
(no text layer).
Used for testing OCR.
"""

import sys
import fitz  # PyMuPDF


def make_fake_scan(src_path: str, dst_path: str, dpi: int = 200):
    src = fitz.open(src_path)
    out = fitz.open()

    for page in src:
        pix = page.get_pixmap(dpi=dpi)
        new_page = out.new_page(width=pix.width, height=pix.height)
        new_page.insert_image(new_page.rect, pixmap=pix)

    out.save(dst_path)
    out.close()
    src.close()
    print(f"Wrote image-only PDF to {dst_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/make_fake_scan.py <source.pdf> <output.pdf>")
        sys.exit(1)
    make_fake_scan(sys.argv[1], sys.argv[2])