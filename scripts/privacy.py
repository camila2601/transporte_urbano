"""Helpers for basic anonymization and privacy-preserving transformations.

This module provides small, auditable utilities to:
- hash identifiers (non-reversible string hashing with optional salt),
- round coordinates (reduce precision),
- apply optional Laplace noise (very small) to numeric fields,
- anonymize a pandas DataFrame in-place or returning a copy.

Note: These are basic tools for demonstration. For production-grade privacy (differential privacy guarantees) consult specialized libraries and experts.
"""
from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def hash_id(id_value: str, salt: str = "") -> str:
    """Return a SHA256 hex digest for id_value with an optional salt.

    Use this to replace raw identifiers before sharing data.
    """
    txt = f"{salt}{id_value}".encode("utf-8")
    return hashlib.sha256(txt).hexdigest()


def round_coords(lat: float, lon: float, decimals: int = 3) -> Tuple[float, float]:
    """Round latitude and longitude to `decimals` decimal places.

    Reducing decimal places reduces spatial precision (approx ~111km/degree latitude).
    """
    return round(float(lat), decimals), round(float(lon), decimals)


def add_laplace_noise(value: float, scale: float = 1e-5) -> float:
    """Add Laplace noise centered at 0 with given scale (small by default).

    Note: Use carefully; this is a toy helper — not a DP library.
    """
    return float(value + np.random.laplace(loc=0.0, scale=scale))


def anonymize_dataframe(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    lat_col: str = "pickup_latitude",
    lon_col: str = "pickup_longitude",
    salt: str = "",
    round_decimals: int = 3,
    add_noise: bool = False,
    noise_scale: float = 1e-5,
    in_place: bool = False,
) -> pd.DataFrame:
    """Return an anonymized copy (or modify in place) of the dataframe.

    Steps performed:
    - If id_col provided, replace with SHA256 hash (with salt).
    - Round coordinates to `round_decimals`.
    - Optionally add Laplace noise to coordinates.
    """
    if not in_place:
        df = df.copy()

    if id_col is not None and id_col in df.columns:
        df[id_col] = df[id_col].astype(str).apply(lambda x: hash_id(x, salt=salt))

    if lat_col in df.columns and lon_col in df.columns:
        lats = df[lat_col].astype(float)
        lons = df[lon_col].astype(float)

        if add_noise:
            lats = lats.apply(lambda v: add_laplace_noise(v, scale=noise_scale))
            lons = lons.apply(lambda v: add_laplace_noise(v, scale=noise_scale))

        df[lat_col] = lats.round(round_decimals)
        df[lon_col] = lons.round(round_decimals)

    return df


if __name__ == "__main__":
    # tiny smoke test when run directly
    sample = pd.DataFrame(
        {
            "id": ["a1", "b2"],
            "pickup_latitude": [40.712776, 40.7812],
            "pickup_longitude": [-74.005974, -73.9665],
        }
    )
    print("Original:\n", sample)
    anon = anonymize_dataframe(sample, id_col="id", salt="somesalt", round_decimals=3)
    print("Anonymized:\n", anon)
