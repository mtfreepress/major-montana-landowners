"""Shared paths and I/O helpers for the landowner analysis scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import pyogrio


PROJECT_DIR = Path(__file__).resolve().parent
RAW_PARCELS = PROJECT_DIR / "raw" / "Montana_Cadastral" / "OWNERPARCEL.shp"
OWNER_GROUPS_PATH = PROJECT_DIR / "owner-name-grouping.json"
PUBLIC_LANDOWNERS_PATH = PROJECT_DIR / "known-public-landowners.json"
OUTPUT_DIR = PROJECT_DIR / "outputs"
GEODATA_OUTPUT_DIR = PROJECT_DIR / "geodata-outputs"
CLEANED_DIR = PROJECT_DIR / "cleaned"


def require_file(path: Path) -> Path:
    """Return *path* or raise a useful error if it does not exist."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def ensure_output_dirs(*directories: Path) -> None:
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with require_file(path).open(encoding="utf-8") as file:
        return json.load(file)


def read_parcel_attributes(columns: Iterable[str]) -> pd.DataFrame:
    """Read selected DBF attributes without loading expensive parcel geometry."""
    return gpd.read_file(
        require_file(RAW_PARCELS),
        columns=list(columns),
        ignore_geometry=True,
        engine="pyogrio",
    )


def iter_parcel_attributes(
    columns: Iterable[str], chunk_size: int = 100_000
):
    """Yield selected DBF attributes in bounded-memory chunks."""
    path = require_file(RAW_PARCELS)
    feature_count = pyogrio.read_info(path)["features"]
    selected_columns = list(columns)
    for offset in range(0, feature_count, chunk_size):
        yield pyogrio.read_dataframe(
            path,
            columns=selected_columns,
            read_geometry=False,
            skip_features=offset,
            max_features=chunk_size,
        )


def owner_name_cleaner() -> dict[str, str]:
    cleaner: dict[str, str] = {}
    for group in load_json(OWNER_GROUPS_PATH):
        canonical_name = group["owner"]
        for alternate_name in group["alternatesOwnershipNames"]:
            existing = cleaner.get(alternate_name)
            if existing is not None and existing != canonical_name:
                raise ValueError(
                    f"Owner name {alternate_name!r} belongs to both "
                    f"{existing!r} and {canonical_name!r}"
                )
            cleaner[alternate_name] = canonical_name
    return cleaner


def public_landowners() -> set[str]:
    return set(load_json(PUBLIC_LANDOWNERS_PATH))


def sql_string(value: str) -> str:
    """Quote a string literal for an OGR attribute filter."""
    return "'" + value.replace("'", "''") + "'"
