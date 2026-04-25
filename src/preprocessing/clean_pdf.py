# ============================================================
# clean_pdf.py – Clean raw PDF page text
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import re
from typing import List, Dict
from src.utils.helpers import normalize_whitespace


# Patterns that indicate header/footer boilerplate
BOILERPLATE_PATTERNS = [
    r"^\s*\d+\s*$",               # lone page numbers
    r"^.*confidential.*$",
    r"^.*all rights reserved.*$",
    r"^.*ministry of finance.*$",  # repeated headers
]


def is_boilerplate(line: str) -> bool:
    line_lower = line.strip().lower()
    for pat in BOILERPLATE_PATTERNS:
        if re.match(pat, line_lower, re.IGNORECASE):
            return True
    return False


def clean_page_text(text: str) -> str:
    """
    Remove boilerplate lines and normalise whitespace from a single PDF page.
    """
    lines = text.split("\n")
    cleaned = [line for line in lines if not is_boilerplate(line)]
    joined = "\n".join(cleaned)
    return normalize_whitespace(joined)


def clean_pdf_pages(pages: List[Dict]) -> List[Dict]:
    """
    Apply clean_page_text to every page dict and return the cleaned list.
    """
    cleaned = []
    for page in pages:
        raw = page.get("text", "")
        cleaned_text = clean_page_text(raw)
        if cleaned_text:  # skip completely empty pages
            cleaned.append({
                "page_number": page["page_number"],
                "text": cleaned_text,
            })
    return cleaned
