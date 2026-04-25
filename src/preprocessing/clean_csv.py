# ============================================================
# clean_csv.py – Clean election CSV and convert rows to text
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import pandas as pd
from typing import List, Dict
from src.utils.helpers import normalize_whitespace, extract_keywords


def clean_election_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise column names, drop fully empty rows,
    and fill NaNs with sensible defaults.
    """
    # Lowercase and strip column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Drop rows where every cell is empty
    df = df.dropna(how="all").reset_index(drop=True)

    # Fill remaining NaN values
    df = df.fillna("unknown")

    # Strip string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    return df


def row_to_text(row: pd.Series, columns: List[str]) -> str:
    """Convert a DataFrame row into a readable natural-language sentence."""
    parts = []
    for col in columns:
        val = row.get(col, "unknown")
        label = col.replace("_", " ").title()
        parts.append(f"{label}: {val}")
    return ". ".join(parts) + "."


def csv_to_chunks(df: pd.DataFrame, source: str = "election_csv") -> List[Dict]:
    """
    Convert each row of the cleaned election DataFrame
    into a chunk dict with metadata.
    """
    chunks = []
    columns = list(df.columns)

    for idx, row in df.iterrows():
        text = row_to_text(row, columns)
        text = normalize_whitespace(text)
        keywords = extract_keywords(text, top_n=8)

        # Try to extract year from common column names
        year = "unknown"
        for col in ["year", "election_year", "date"]:
            if col in row and str(row[col]).isdigit():
                year = str(row[col])
                break

        chunk = {
            "chunk_id": f"csv_{idx}",
            "source": source,
            "chunk_type": "election_row",
            "text": text,
            "section_title": None,
            "year": year,
            "keywords": keywords,
            "row_index": int(idx),
        }
        chunks.append(chunk)

    return chunks
