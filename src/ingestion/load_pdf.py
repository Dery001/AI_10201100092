# ============================================================
# load_pdf.py – Load and extract raw text from PDF using PyMuPDF
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

from typing import List, Dict, Optional
import fitz  # PyMuPDF


def load_pdf_pages(path: str) -> Optional[List[Dict]]:
    """
    Open a PDF and return a list of page dicts:
        { "page_number": int, "text": str }
    Returns None if the file cannot be opened.
    """
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"[load_pdf] Cannot open PDF: {e}")
        return None

    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")  # plain text extraction
        pages.append({"page_number": i + 1, "text": text})

    doc.close()
    return pages
