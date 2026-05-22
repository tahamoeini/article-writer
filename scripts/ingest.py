import argparse
import concurrent.futures
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

try:
    from scripts.config import RuntimeSettings
except ModuleNotFoundError:
    from config import RuntimeSettings

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None


MIN_BLOCK_LENGTH = 40
TEI_NAMESPACE = {"tei": "http://www.tei-c.org/ns/1.0"}
MAX_WORKERS = 32


def empty_ingest_summary() -> dict[str, object]:
    return {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "failures": [],
    }


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_year(first_page_text: str, creation_date: str) -> str:
    year_match = re.search(r"\b(19|20)\d{2}\b", first_page_text)
    if year_match:
        return year_match.group(0)

    creation_year_match = re.search(r"(19|20)\d{2}", creation_date or "")
    return creation_year_match.group(0) if creation_year_match else "Unknown"


def extract_basic_metadata(pdf_path: Path) -> dict[str, str | int]:
    if pymupdf is None:
        raise RuntimeError(
            "A PDF parser is required. Install PyMuPDF (preferred) or fitz package before running ingestion."
        )

    with pymupdf.open(pdf_path) as document:
        metadata = document.metadata or {}
        first_page_text = document[0].get_text("text")[:2000] if len(document) else ""
        year = extract_year(first_page_text, metadata.get("creationDate", ""))
        return {
            "title": metadata.get("title") or pdf_path.stem,
            "author": metadata.get("author") or "Unknown",
            "year": year,
            "filename": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": len(document),
        }


def extract_grobid_metadata(pdf_path: Path, settings: RuntimeSettings) -> dict[str, str] | None:
    try:
        import requests
    except ModuleNotFoundError:
        # Continue without GROBID enrichment when requests is unavailable.
        return None

    try:
        with pdf_path.open("rb") as handle:
            response = requests.post(
                settings.grobid_process_url,
                files={"input": handle},
                timeout=20,
            )
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return None

    title = root.findtext(".//tei:titleStmt/tei:title", default="", namespaces=TEI_NAMESPACE).strip()
    author_nodes = root.findall(".//tei:sourceDesc//tei:author", TEI_NAMESPACE)
    authors = []
    for node in author_nodes:
        name_parts = [part.text.strip() for part in node.findall(".//tei:persName/*", TEI_NAMESPACE) if part.text]
        if name_parts:
            authors.append(" ".join(name_parts))

    year_candidates = [
        root.findtext(".//tei:publicationStmt/tei:date", default="", namespaces=TEI_NAMESPACE),
        root.findtext(".//tei:imprint/tei:date", default="", namespaces=TEI_NAMESPACE),
    ]
    year = ""
    for candidate in year_candidates:
        match = re.search(r"(19|20)\d{2}", candidate or "")
        if match:
            year = match.group(0)
            break

    parsed = {}
    if title:
        parsed["title"] = title
    if authors:
        parsed["author"] = "; ".join(authors)
    if year:
        parsed["year"] = year
    return parsed or None


def extract_structured_sections(pdf_path: Path, metadata: dict[str, str | int]) -> list[dict[str, object]]:
    if pymupdf is None:
        raise RuntimeError(
            "A PDF parser is required. Install PyMuPDF (preferred) or fitz package before running ingestion."
        )

    structured_sections: list[dict[str, object]] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            text_blocks = page.get_text("blocks")

            for block_index, block in enumerate(text_blocks):
                block_text = normalize_text(block[4])
                if len(block_text) < MIN_BLOCK_LENGTH:
                    continue

                structured_sections.append(
                    {
                        "id": f"{pdf_path.stem}:{page_index + 1}:{block_index}",
                        "text": block_text,
                        "page": page_index + 1,
                        "paragraph_index": block_index,
                        "block_type": "paragraph",
                        "metadata": metadata,
                    }
                )

    return structured_sections


def process_pdf(pdf_path: Path, settings: RuntimeSettings) -> Path:
    base_metadata = extract_basic_metadata(pdf_path)
    grobid_metadata = extract_grobid_metadata(pdf_path, settings)
    if grobid_metadata:
        base_metadata.update({key: value for key, value in grobid_metadata.items() if value})
        base_metadata["extraction_method"] = "grobid+layout"
    else:
        base_metadata["extraction_method"] = "layout"

    structured_sections = extract_structured_sections(pdf_path, base_metadata)
    output_path = settings.processed_dir / f"{pdf_path.stem}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(structured_sections, handle, indent=2, ensure_ascii=False)

    return output_path


def resolve_pdf_files(settings: RuntimeSettings, selected_files: list[str] | None = None) -> list[Path]:
    if not selected_files:
        return sorted(settings.pdf_dir.glob("*.pdf"))

    pdf_files = []
    for filename in selected_files:
        pdf_path = Path(filename)
        if pdf_path.name != filename or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Invalid PDF selection: {filename}")

        resolved_path = (settings.pdf_dir / filename).resolve()
        try:
            resolved_path.relative_to(settings.pdf_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"Invalid PDF selection: {filename}") from exc

        if not resolved_path.exists() or not resolved_path.is_file():
            raise ValueError(f"Selected PDF was not found: {filename}")

        pdf_files.append(resolved_path)

    return sorted(dict.fromkeys(pdf_files))


def process_pdf_safely(pdf_path: Path, settings: RuntimeSettings) -> dict[str, str]:
    process_pdf(pdf_path, settings)
    return {"filename": pdf_path.name, "status": "processed"}


def process_corpus(
    force: bool = False,
    settings_overrides: Mapping[str, str] | None = None,
    selected_files: list[str] | None = None,
    max_workers: int = 1,
) -> dict[str, object]:
    settings = RuntimeSettings.from_env(settings_overrides)
    settings.ensure_runtime_dirs()
    max_workers = max(1, min(max_workers, MAX_WORKERS))

    pdf_files = resolve_pdf_files(settings, selected_files)
    if not pdf_files:
        print(f"No PDF files found in {settings.pdf_dir}.")
        return empty_ingest_summary()

    summary = empty_ingest_summary()
    scheduled_files = []
    print(
        "Starting academic document preprocessing for "
        f"{len(pdf_files)} PDF files with {max_workers} worker(s)..."
    )

    for pdf_path in pdf_files:
        output_path = settings.processed_dir / f"{pdf_path.stem}.json"
        if output_path.exists() and not force:
            summary["skipped"] += 1
            print(f"Skipping existing file: {pdf_path.name}")
            continue

        scheduled_files.append(pdf_path)

    if max_workers == 1:
        for pdf_path in scheduled_files:
            try:
                print(f"Extracting structural text: {pdf_path.name}")
                process_pdf_safely(pdf_path, settings)
                summary["processed"] += 1
                print(f"Completed extraction: {pdf_path.name}")
            except Exception as exc:
                summary["failed"] += 1
                summary["failures"].append({"filename": pdf_path.name, "error": str(exc)})
                print(f"Failed extraction: {pdf_path.name}: {exc}")
    else:
        for pdf_path in scheduled_files:
            print(f"Queueing extraction: {pdf_path.name}")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_pdf_safely, pdf_path, settings): pdf_path for pdf_path in scheduled_files
            }
            for future in concurrent.futures.as_completed(futures):
                pdf_path = futures[future]
                try:
                    future.result()
                    summary["processed"] += 1
                    print(f"Completed extraction: {pdf_path.name}")
                except Exception as exc:
                    summary["failed"] += 1
                    summary["failures"].append({"filename": pdf_path.name, "error": str(exc)})
                    print(f"Failed extraction: {pdf_path.name}: {exc}")

    print(
        "Pre-processing completed. "
        f"processed={summary['processed']}, skipped={summary['skipped']}, failed={summary['failed']}, "
        f"output_dir={settings.processed_dir}"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured text and metadata from PDFs.")
    parser.add_argument("--force", action="store_true", help="Rebuild JSON output for files that already exist.")
    parser.add_argument("--workers", type=int, default=1, help=f"Number of PDF files to process in parallel, up to {MAX_WORKERS}.")
    parser.add_argument("--file", action="append", dest="selected_files", help="PDF filename from corpus/pdfs to ingest. Repeat for multiple files.")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    process_corpus(
        force=arguments.force,
        selected_files=arguments.selected_files,
        max_workers=arguments.workers,
    )
