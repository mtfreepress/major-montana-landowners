#!/usr/bin/env python3
"""Summarize or map parcels for one grouped owner without loading the whole state."""

from __future__ import annotations

import argparse
import re

import geopandas as gpd

from analysis_common import (
    PROJECT_DIR,
    RAW_PARCELS,
    ensure_output_dirs,
    owner_name_cleaner,
    require_file,
    sql_string,
)


MAP_COLUMNS = [
    "CountyName",
    "OwnerGroup",
    "OwnerName",
    "OwnerAddre",
    "OwnerAdd_1",
    "OwnerAdd_2",
    "OwnerCity",
    "OwnerState",
    "PropType",
    "TotalAcres",
    "geometry",
]
SOURCE_COLUMNS = [column for column in MAP_COLUMNS if column != "OwnerGroup"]


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")


def read_owner_parcels(owner_name: str) -> gpd.GeoDataFrame:
    cleaner = owner_name_cleaner()
    raw_names = sorted(
        {name for name, canonical in cleaner.items() if canonical == owner_name}
        | {owner_name}
    )
    where = '"OwnerName" IN (' + ", ".join(map(sql_string, raw_names)) + ")"
    parcels = gpd.read_file(
        require_file(RAW_PARCELS),
        columns=SOURCE_COLUMNS,
        where=where,
        engine="pyogrio",
    ).to_crs(epsg=4326)
    parcels["OwnerGroup"] = parcels["OwnerName"].replace(cleaner)
    return parcels[MAP_COLUMNS]


def print_summary(owner_name: str, parcels: gpd.GeoDataFrame) -> None:
    print(f"## {owner_name}")
    print(f"{parcels['TotalAcres'].sum():,.0f} acres / {len(parcels)} parcels")
    print("Owned as", parcels["OwnerName"].unique())
    print("Property types", parcels["PropType"].unique())
    print("Counties", parcels["CountyName"].unique())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "owner", help='canonical or raw owner name, for example "TED TURNER"'
    )
    parser.add_argument(
        "--geojson", action="store_true", help="write the selected parcels as GeoJSON"
    )
    parser.add_argument(
        "--map-html", action="store_true", help="write an interactive HTML map"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parcels = read_owner_parcels(args.owner)
    if parcels.empty:
        raise SystemExit(f"No parcels found for owner: {args.owner}")

    print_summary(args.owner, parcels)
    output_dir = PROJECT_DIR / "major-owners"
    filename = safe_filename(args.owner)

    if args.geojson:
        ensure_output_dirs(output_dir)
        path = output_dir / f"{filename}.geojson"
        parcels.to_file(path, driver="GeoJSON", engine="pyogrio")
        print(f"Wrote {path}")

    if args.map_html:
        ensure_output_dirs(output_dir)
        path = output_dir / f"{filename}.html"
        parcels.explore(tiles="OpenStreetMap").save(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
