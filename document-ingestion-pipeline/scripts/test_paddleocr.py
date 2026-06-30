"""Standalone sanity check for PaddleOCR's Urdu model -- lightweight
CRNN-based engine (not the heavier PaddleOCR-VL), officially supports
'ur' directly. Tests on a couple of pages before deciding whether to
integrate for real.
"""

import sys
import time
import os

import fitz
import cv2
import numpy as np
from paddleocr import PaddleOCR


def safe_render_page(page, dpi=150, max_side=2500):
    """
    Render PDF page safely with size limits.
    
    Args:
        page: fitz page object
        dpi: resolution (lower = smaller image)
        max_side: maximum dimension in pixels
    
    Returns:
        tuple: (image_path, original_dims, resized_dims)
    """
    # Render at specified DPI
    pix = page.get_pixmap(dpi=dpi)
    img_path = "paddleocr_test_page.png"
    pix.save(img_path)
    
    # Read and check dimensions
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    print(f"Original image size: {w}x{h}")
    
    # Resize if too large
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, img)
        print(f"Resized to: {new_w}x{new_h}")
        return img_path, (w, h), (new_w, new_h)
    
    return img_path, (w, h), (w, h)


def process_pdf_page_safely(pdf_path, page_number=0, dpi=120):
    """
    Process a PDF page with safe rendering and optional tiling.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_number]
    
    # Get page dimensions for tiling
    rect = page.rect
    w, h = rect.width, rect.height
    
    print(f"Page dimensions: {w}x{h} points")
    print(f"Rendering at {dpi} DPI...")
    
    # For large pages, consider tiling
    # Using points dimension - if page is taller than 1000pt at 120dpi, tile it
    if h > 1000 and w > 700:
        print("Large page detected - using tiling approach...")
        return process_with_tiling(page, doc, dpi, tiles=2)
    else:
        # Single page processing
        img_path, original_dims, resized_dims = safe_render_page(page, dpi)
        doc.close()
        return img_path, [img_path]  # Return single image in list


def process_with_tiling(page, doc, dpi=120, tiles=2):
    """
    Split large page into vertical tiles for stable processing.
    """
    rect = page.rect
    w, h = rect.width, rect.height
    
    # Calculate tile dimensions
    tile_height = h / tiles
    tile_paths = []
    
    for i in range(tiles):
        # Create clip rectangle for this tile
        y0 = i * tile_height
        y1 = (i + 1) * tile_height
        
        # Add slight overlap for text that might be cut
        overlap = 20  # points
        if i > 0:
            y0 -= overlap
        if i < tiles - 1:
            y1 += overlap
            
        clip_rect = fitz.Rect(0, max(0, y0), w, min(h, y1))
        
        # Render just this tile
        pix = page.get_pixmap(dpi=dpi, clip=clip_rect)
        tile_path = f"paddleocr_tile_{i}.png"
        pix.save(tile_path)
        tile_paths.append(tile_path)
        
        # Resize tile if needed
        img = cv2.imread(tile_path)
        h_tile, w_tile = img.shape[:2]
        if max(h_tile, w_tile) > 2500:
            scale = 2500 / max(h_tile, w_tile)
            new_w = int(w_tile * scale)
            new_h = int(h_tile * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(tile_path, img)
            print(f"Tile {i} resized to: {new_w}x{new_h}")
    
    doc.close()
    return None, tile_paths


def main(pdf_path: str, page_number: int = 0):
    """
    Main OCR function with safe configuration.
    
    Args:
        pdf_path: Path to PDF file
        page_number: Page index (0-based)
    """
    # Enable debugging for Paddle crashes
    os.environ["PYTHONFAULTHANDLER"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    print("Loading PaddleOCR Urdu model...")
    print("(First use downloads weights -- much smaller than EasyOCR's)")
    
    # Initialize with ONLY valid parameters for current PaddleOCR version
    # Using simpler initialization to avoid deprecated/unknown params
    try:
        # Try with minimal parameters first
        ocr = PaddleOCR(
            lang="ur",
            use_angle_cls=False,
            enable_mkldnn=False,
        )
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        print("Trying with even simpler initialization...")
        ocr = PaddleOCR(
            lang="ur",
            enable_mkldnn=False,
        )
    
    # Process with safe rendering
    print(f"\nProcessing PDF: {pdf_path}")
    print(f"Page {page_number + 1}...")
    
    try:
        _, image_paths = process_pdf_page_safely(pdf_path, page_number, dpi=120)
        
        if not image_paths:
            print("No images to process")
            return
        
        print(f"\nRunning OCR on {len(image_paths)} tile(s)...")
        start = time.time()
        
        results = []
        for idx, img_path in enumerate(image_paths):
            print(f"\nProcessing tile {idx + 1}/{len(image_paths)}...")
            try:
                # For newer PaddleOCR versions, result is a list of dictionaries
                result = ocr.predict(img_path)
                results.append(result)
                print(f"✓ Tile {idx + 1} processed")
            except Exception as e:
                print(f"✗ Error on tile {idx + 1}: {e}")
                continue
        
        elapsed = time.time() - start
        print(f"\n{'='*60}")
        print(f"Total time for {len(image_paths)} tile(s): {elapsed:.1f}s")
        if len(image_paths) > 0:
            print(f"Average per tile: {elapsed/len(image_paths):.1f}s")
        print(f"{'='*60}")
        
        # Parse and display results
        print("\n--- OCR Results ---")
        for tile_idx, result in enumerate(results):
            print(f"\nTile {tile_idx + 1} results:")
            for res in result:
                res.print()
                res.save_to_json(save_path="paddleocr_test_output")
        print("\nAlso saved to paddleocr_test_output/")
        
        # Summary
        if total_text:
            print(f"\n{'='*60}")
            print(f"Total text blocks detected: {len(total_text)}")
            print("\nFirst few detections:")
            for i, text in enumerate(total_text[:5]):
                print(f"  {i+1}. {text}")
        
        # Clean up tile files
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except:
                pass
                
    except Exception as e:
        print(f"\n✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Force UTF-8 output
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    if len(sys.argv) < 2:
        print("Usage: python script.py <pdf_path> [page_number]")
        print("Example: python script.py urdu_novel.pdf 0")
        print("  (page_number is 0-based, defaults to 0)")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    print(f"Processing: {pdf_path}")
    print(f"Page: {page_number + 1}")
    print("-" * 60)
    
    main(pdf_path, page_number)