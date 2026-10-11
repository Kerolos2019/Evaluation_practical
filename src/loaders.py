"""Load raw text out of the files under datasets/ so they can be chunked."""


import csv

from dataclasses import dataclass

from pathlib import Path


from pypdf import PdfReader



@dataclass
class LoadedDocument:
    text: str
    source: str



def _load_csv(path: Path) -> str:
    rows_text = []

    with path.open("r", encoding="utf-8", newline="") as f:
        
        reader = csv.DictReader(f) # {"name": "ACME Corp", "address": "123 Main St", "city": "Anytown", "state": "CA", "zip": "12345"}
        for row in reader:

            row_text = ", ".join(f"{key}: {value}" for key, value in row.items()) # o/p = "name: ACME Corp, address: 123 Main St, city: Anytown, state: CA, zip: 12345"
            rows_text.append(row_text)

    return "\n".join(rows_text)



def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))

    pages_text = [page.extract_text() or "" for page in reader.pages]

    return "\n".join(pages_text)



def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


_LOADERS = {
    ".csv": _load_csv,
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
}



def load_documents(datasets_dir: Path) -> list[LoadedDocument]:
    """Walk datasets_dir recursively and load every supported file into text."""
    documents = []

    for path in sorted(datasets_dir.rglob("*")):

        if not path.is_file():
            continue

        loader = _LOADERS.get(path.suffix.lower())
        if loader is None:
            continue

        text = loader(path)

        if text.strip():

            documents.append(LoadedDocument(text=text, source=str(path.relative_to(datasets_dir))))

    return documents
