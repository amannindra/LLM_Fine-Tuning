"""Merge the PDFs directly in a folder in natural filename order."""

import argparse
from pathlib import Path
import re

from pypdf import PdfReader, PdfWriter

def merge_pdfs(folder: Path) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    output = folder / "merged_output.pdf"
    files = sorted(
        (path for path in folder.iterdir()
         if path.is_file() and path.suffix.lower() == ".pdf"
         and path.resolve() != output.resolve()),
        key=lambda path: [int(part) if part.isdigit() else part
                          for part in re.split(r"(\d+)", path.name.casefold())],
    )
    if not files:
        raise ValueError(f"No input PDFs found in {folder}")

    expected_pages = 0
    with PdfWriter() as merger:
        for index, pdf in enumerate(files, start=1):
            reader = PdfReader(pdf)
            count = len(reader.pages)
            print(f"{index}. {pdf.name} ({count} pages)")
            merger.append(reader)
            expected_pages += count
        merger.write(output)

    actual_pages = len(PdfReader(output).pages)
    if actual_pages != expected_pages:
        raise RuntimeError(f"Expected {expected_pages} pages, got {actual_pages}")
    print(f"Saved {actual_pages} pages to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        default=Path("/Users/amannindra/Resume/UC Merced/CSE031 Lectures"))
    args = parser.parse_args()
    merge_pdfs(args.folder)
