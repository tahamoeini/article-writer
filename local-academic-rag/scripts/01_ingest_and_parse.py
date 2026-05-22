import os
import fitz  # PyMuPDF
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_DIR = BASE_DIR / "corpus" / "pdfs"
OUTPUT_DIR = BASE_DIR / "corpus" / "processed"

def resolve_output_dir() -> Path:
    preferred_dir = OUTPUT_DIR
    probe_path = preferred_dir / ".write_test"
    try:
        preferred_dir.mkdir(parents=True, exist_ok=True)
        with open(probe_path, "w", encoding="utf-8") as probe_file:
            probe_file.write("")
        probe_path.unlink(missing_ok=True)
        return preferred_dir
    except OSError:
        fallback_root = Path(os.environ.get("TEMP", str(Path.home() / "AppData" / "Local" / "Temp")))
        fallback_dir = fallback_root / "local-academic-rag" / "corpus" / "processed"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        return fallback_dir

def extract_basic_metadata(pdf_path: Path):
    """Fallback parser to grab author/year if GROBID metadata is missing."""
    with fitz.open(pdf_path) as doc:
        meta = doc.metadata
        first_page_text = doc[0].get_text("text")[:1000]
    
    # Try to extract a plausible 4-digit year from the front text
    import re
    year_match = re.search(r'\b(19|20)\d{2}\b', first_page_text)
    year = year_match.group(0) if year_match else (meta.get("creationDate", "")[2:6] or "Unknown")
    
    return {
        "title": meta.get("title") or pdf_path.stem,
        "author": meta.get("author") or "Unknown",
        "year": year,
        "filename": pdf_path.name
    }

def process_corpus():
    print("Starting academic document preprocessing pipeline...")
    output_dir = resolve_output_dir()
    if output_dir != OUTPUT_DIR:
        print(f"Using fallback output directory: {output_dir}")
    
    for pdf_path in PDF_DIR.glob("*.pdf"):
        print(f"Extracting structural text: {pdf_path.name}")
        try:
            base_meta = extract_basic_metadata(pdf_path)

            # Open and map page boundaries to match chunks back to hard page numbers
            structured_sections = []

            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text("blocks") # Keeps structural paragraphs together

                    for block_idx, b in enumerate(text):
                        block_text = b[4].strip()
                        if len(block_text) > 40: # Skip noise, headers, footers
                            structured_sections.append({
                                "text": block_text,
                                "page": page_num + 1,
                                "paragraph_index": block_idx,
                                "metadata": base_meta
                            })

            # Write clean json structure out for our indexer
            output_stem = pdf_path.stem.lstrip("-") or "document"
            out_json_path = output_dir / f"{output_stem}.json"
            out_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(structured_sections, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Skipping unreadable file: {pdf_path.name} ({exc})")
            continue
            
    print("Pre-processing successfully completed.")

if __name__ == "__main__":
    process_corpus()