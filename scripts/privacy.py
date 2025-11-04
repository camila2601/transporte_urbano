from __future__ import annotations

import hashlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def hash_id(id_value: str, salt: str = "") -> str:

    txt = f"{salt}{id_value}".encode("utf-8")
    return hashlib.sha256(txt).hexdigest()


def round_coords(lat: float, lon: float, decimals: int = 3) -> Tuple[float, float]:

    return round(float(lat), decimals), round(float(lon), decimals)


def add_laplace_noise(value: float, scale: float = 1e-5) -> float:
   
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
