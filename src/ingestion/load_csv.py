# ============================================================
# load_csv.py – Load Ghana Election Results CSV
# Student: [YOUR NAME] | Index: [YOUR INDEX NUMBER]
# ============================================================

import pandas as pd
from typing import Optional


def load_election_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Load the Ghana election results CSV into a DataFrame.
    Returns None if the file is not found.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
        return df
    except FileNotFoundError:
        print(f"[load_csv] File not found: {path}")
        return None
    except Exception as e:
        print(f"[load_csv] Error loading CSV: {e}")
        return None
