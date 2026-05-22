import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from scripts.config import RuntimeSettings

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf


MIN_BLOCK_LENGTH = 40
TEI_NAMESPACE = {"tei": "http://www.tei-c.org/ns/1.0"}


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def extract_year(first_page_text: str, creation_date: str) -> str:
    year_match = re.search(r"\b(19|20)\d{2}\b", first_page_text)
    if year_match:
        return year_match.group(0)

    creation_year_match = re.search(r"(19|20)\d{2}", creation_date or "")
    return creation_year_match.group(0) if creation_year_match else "Unknown"


def extract_basic_metadata(pdf_path: Path) -> dict[str, str | int]:
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


def process_corpus(force: bool = False) -> tuple[int, int]:
    settings = RuntimeSettings.from_env()
    settings.ensure_runtime_dirs()

    pdf_files = sorted(settings.pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {settings.pdf_dir}.")
        return 0, 0

    processed_count = 0
    skipped_count = 0
    print(f"Starting academic document preprocessing for {len(pdf_files)} PDF files...")

    for pdf_path in pdf_files:
        output_path = settings.processed_dir / f"{pdf_path.stem}.json"
        if output_path.exists() and not force:
            skipped_count += 1
            print(f"Skipping existing file: {pdf_path.name}")
            continue

        print(f"Extracting structural text: {pdf_path.name}")
        process_pdf(pdf_path, settings)
        processed_count += 1

    print(
        "Pre-processing completed. "
        f"processed={processed_count}, skipped={skipped_count}, output_dir={settings.processed_dir}"
    )
    return processed_count, skipped_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured text and metadata from PDFs.")
    parser.add_argument("--force", action="store_true", help="Rebuild JSON output for files that already exist.")
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    process_corpus(force=arguments.force)