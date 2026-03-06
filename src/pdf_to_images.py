from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError

from .dataset_io import ensure_dir, stable_page_id


@dataclasses.dataclass(frozen=True)
class PdfPageImage:
    pdf_path: Path
    page_number: int  # 1-indexed
    page_id: str
    image_path: Path


def get_pdf_page_count(pdf_path: Path, poppler_path: Optional[str] = None) -> int:
    try:
        info = pdfinfo_from_path(str(pdf_path), poppler_path=poppler_path)
    except PDFInfoNotInstalledError as e:
        raise RuntimeError(
            "Poppler is required for PDF processing. On Windows, install Poppler and either add it to PATH "
            "or set pdf_to_images.poppler_path in your config to the Poppler 'bin' directory."
        ) from e
    except Exception as e:  # pdf2image can raise platform-specific errors
        raise RuntimeError(f"Failed to read PDF info for: {pdf_path}") from e

    try:
        return int(info["Pages"])
    except Exception as e:
        raise RuntimeError(f"Could not determine page count for: {pdf_path}") from e


def convert_pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    *,
    dpi: int = 300,
    image_format: str = "png",
    poppler_path: Optional[str] = None,
    selected_pages: Optional[Set[int]] = None,
) -> List[PdfPageImage]:
    """
    Convert selected pages of a PDF to images.

    Notes:
    - Page numbering is 1-indexed.
    - For discontiguous selections we convert page-by-page for correctness and clarity.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    ensure_dir(out_dir)

    total_pages = get_pdf_page_count(pdf_path, poppler_path=poppler_path)
    if selected_pages is None:
        pages = list(range(1, total_pages + 1))
    else:
        pages = sorted(p for p in selected_pages if 1 <= p <= total_pages)

    results: List[PdfPageImage] = []
    for page_number in pages:
        page_id = stable_page_id(pdf_path, page_number)
        filename = f"{pdf_path.stem}_p{page_number:04d}.{image_format}"
        image_path = out_dir / filename

        try:
            images = convert_from_path(
                str(pdf_path),
                dpi=dpi,
                fmt=image_format,
                first_page=page_number,
                last_page=page_number,
                poppler_path=poppler_path,
            )
        except PDFInfoNotInstalledError as e:
            raise RuntimeError(
                "Poppler is required for PDF processing. On Windows, install Poppler and either add it to PATH "
                "or set pdf_to_images.poppler_path in your config to the Poppler 'bin' directory."
            ) from e
        except PDFPageCountError as e:
            raise RuntimeError(f"Could not read page count for PDF: {pdf_path}") from e
        except Exception as e:
            raise RuntimeError(f"Failed converting {pdf_path} page {page_number}") from e

        if len(images) != 1:
            raise RuntimeError(f"Expected 1 image for {pdf_path} page {page_number}, got {len(images)}")

        # pdf2image returns PIL Images
        images[0].save(image_path)

        results.append(
            PdfPageImage(
                pdf_path=pdf_path,
                page_number=page_number,
                page_id=page_id,
                image_path=image_path,
            )
        )

    return results

