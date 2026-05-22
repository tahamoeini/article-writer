import os
import fitz
import json
import requests
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "corpus" / "pdfs"
OUTPUT_DIR = BASE_DIR / "corpus" / "processed"
GROBID_URL = os.environ.get("GROBID_URL", "http://localhost:8070") + "/api/processHeaderDocument"

def extract_basic_metadata(pdf_path: Path):
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata
        first_page_text = doc[0].get_text("text")[:1000]
    year_match = re.search(r'\b(19|20)\d{2}\b', first_page_text)
    year = year_match.group(0) if year_match else (meta.get("creationDate", "")[2:6] or "Unknown")
    return {
        "title": meta.get("title") or pdf_path.stem,
        "author": meta.get("author") or "Unknown",
        "year": year,
        "filename": pdf_path.name
    }

def process_corpus():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting academic document preprocessing pipeline...")
    
    for pdf_path in PDF_DIR.glob("*.pdf"):
        print(f"Extracting structural text: {pdf_path.name}")
        base_meta = extract_basic_metadata(pdf_path)
        structured_sections = []
        
        # Try to parse via GROBID
        grobid_success = False
        try:
            with open(pdf_path, 'rb') as f:
                response = requests.post(GROBID_URL, files={'input': f}, timeout=15)
                if response.status_code == 200:
                    grobid_success = True
                    # If GROBID successfully generates XML, parse sections here
                    # For this production fallback structure, we combine layout markers:
        except Exception:
            print("  ⚠️ GROBID offline or timed out. Falling back to structured PyMuPDF extraction.")
            
        # Extract via structural layout blocks
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text_blocks = page.get_text("blocks")
                for block_idx, b in enumerate(text_blocks):
                    block_text = b[4].strip()
                    if len(block_text) > 40:
                        structured_sections.append({
                            "text": block_text,
                            "page": page_num + 1,
                            "paragraph_index": block_idx,
                            "metadata": base_meta
                        })
                        
        out_json_path = OUTPUT_DIR / f"{pdf_path.stem}.json"
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(structured_sections, f, indent=2, ensure_ascii=False)
            
    print("Pre-processing successfully completed.")

if __name__ == "__main__":
    process_corpus()