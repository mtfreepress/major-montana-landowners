#!/usr/bin/env python3
"""Identify the largest private Montana landowners and export their parcels."""

from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely import STRtree, union_all

from analysis_common import (
    CLEANED_DIR,
    GEODATA_OUTPUT_DIR,
    OUTPUT_DIR,
    RAW_PARCELS,
    ensure_output_dirs,
    iter_parcel_attributes,
    owner_name_cleaner,
    public_landowners,
    require_file,
    sql_string,
)


TOP_COUNT = 20
BLOCK_ANALYSIS_CRS = "EPSG:5070"
METERS_PER_MILE = 1609.344
MAP_COLUMNS = [
    "CountyName",
    "OwnerName_Grouped",
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
SOURCE_MAP_COLUMNS = [column for column in MAP_COLUMNS if column != "OwnerName_Grouped"]
BLOCK_COLUMNS = ["BlockID", "BlockAcres", "BlockParcelCount"]
PARCEL_OUTPUT_COLUMNS = MAP_COLUMNS + BLOCK_COLUMNS


def create_owner_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE owners (
            grouped_name TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            parcel_count INTEGER NOT NULL,
            total_acres REAL NOT NULL,
            first_seen INTEGER NOT NULL,
            is_public INTEGER NOT NULL,
            PRIMARY KEY (grouped_name, owner_name)
        ) WITHOUT ROWID
        """
    )
    return connection


def store_owner_chunk(
    connection: sqlite3.Connection,
    parcels: pd.DataFrame,
    cleaner: dict[str, str],
    public_owners: set[str],
    offset: int,
) -> None:
    parcels = parcels.loc[parcels["OwnerName"].notna()].copy()
    parcels["OwnerName_Grouped"] = parcels["OwnerName"].replace(cleaner)
    parcels["first_seen"] = range(offset, offset + len(parcels))
    grouped = (
        parcels.groupby(["OwnerName_Grouped", "OwnerName"], sort=False)
        .agg(
            parcel_count=("PARCELID", "count"),
            total_acres=("TotalAcres", "sum"),
            first_seen=("first_seen", "min"),
        )
        .reset_index()
    )
    connection.executemany(
        """
        INSERT INTO owners (
            grouped_name, owner_name, parcel_count, total_acres,
            first_seen, is_public
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT (grouped_name, owner_name) DO UPDATE SET
            parcel_count = parcel_count + excluded.parcel_count,
            total_acres = total_acres + excluded.total_acres,
            first_seen = MIN(first_seen, excluded.first_seen)
        """,
        (
            (
                group_name,
                owner_name,
                int(count),
                float(acres),
                int(first_seen),
                int(group_name in public_owners),
            )
            for group_name, owner_name, count, acres, first_seen
            in grouped.itertuples(index=False)
        ),
    )
    connection.commit()


def aggregate_ranking(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, dict[str, dict[str, tuple[float, int]]]]:
    rows = connection.execute(
        """
        SELECT grouped_name, SUM(parcel_count), SUM(total_acres)
        FROM owners
        WHERE is_public = 0
        GROUP BY grouped_name
        ORDER BY SUM(total_acres) DESC
        LIMIT ?
        """,
        (TOP_COUNT,),
    ).fetchall()

    records = []
    owner_stats: dict[str, dict[str, tuple[float, int]]] = {}
    for rank, (group_name, count, acres) in enumerate(rows, start=1):
        subowners = connection.execute(
            """
            SELECT owner_name, total_acres, parcel_count
            FROM owners
            WHERE grouped_name = ?
            ORDER BY first_seen
            """,
            (group_name,),
        ).fetchall()
        included_names = [row[0] for row in subowners]
        owner_stats[group_name] = {
            name: (owner_acres, owner_count)
            for name, owner_acres, owner_count in subowners
        }
        records.append(
            {
                "Rank": rank,
                "Owner": group_name,
                "NumParcels": count,
                "TotalAcres": acres,
                "IncludedOwnerNames": included_names,
            }
        )
    return pd.DataFrame.from_records(records), owner_stats


def format_ranking(
    ranking: pd.DataFrame,
    owner_stats: dict[str, dict[str, tuple[float, int]]],
) -> str:
    blocks: list[str] = []

    for row in ranking.itertuples(index=False):
        lines = [
            f"#{row.Rank} / {row.Owner}",
            f"{row.TotalAcres:,.0f} acres - {row.NumParcels} parcels",
            "  Includes:",
        ]
        for owner_name in row.IncludedOwnerNames:
            acres, count = owner_stats[row.Owner][owner_name]
            lines.append(
                f"    {owner_name} --> {acres:,.0f} acres/ {count} parcels"
            )
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def read_selected_geometry(
    ranking: pd.DataFrame, cleaner: dict[str, str]
) -> gpd.GeoDataFrame:
    selected_names = sorted(
        {
            name
            for names in ranking["IncludedOwnerNames"]
            for name in names
        }
    )
    if not selected_names:
        raise RuntimeError("No parcel owner names matched the final ranking")

    where = '"OwnerName" IN (' + ", ".join(map(sql_string, selected_names)) + ")"
    selected = gpd.read_file(
        require_file(RAW_PARCELS),
        columns=SOURCE_MAP_COLUMNS,
        where=where,
        engine="pyogrio",
    ).to_crs(epsg=4326)
    selected["OwnerName_Grouped"] = selected["OwnerName"].replace(cleaner)
    return selected[MAP_COLUMNS]


def safe_filename(owner_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", owner_name).strip("-")


class DisjointSet:
    """Union-find structure used to build transitive parcel components."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def assign_landholding_blocks(
    ranking: pd.DataFrame,
    parcels: gpd.GeoDataFrame,
    gap_miles: float,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Annotate parcels with transitive proximity components and dissolve them."""
    if parcels.crs is None:
        raise ValueError("Selected parcel geometry has no CRS")

    # Normalize the retained/exported geometry up front. Only the temporary
    # ``projected`` copy is used for proximity analysis.
    annotated = parcels.to_crs(epsg=4326).reset_index(drop=True)
    annotated["BlockID"] = pd.Series(pd.NA, index=annotated.index, dtype="string")
    annotated["BlockAcres"] = float("nan")
    annotated["BlockParcelCount"] = 0
    projected = annotated.to_crs(BLOCK_ANALYSIS_CRS)
    rank_by_owner = dict(zip(ranking["Owner"], ranking["Rank"], strict=True))
    gap_meters = gap_miles * METERS_PER_MILE

    for owner_name, owner_rows in annotated.groupby(
        "OwnerName_Grouped", sort=False
    ):
        if owner_name not in rank_by_owner:
            raise ValueError(f"Selected parcel owner is not ranked: {owner_name}")

        positions = owner_rows.index.to_list()
        owner_geometries = projected.loc[positions, "geometry"].to_numpy()
        components = DisjointSet(len(positions))
        tree = STRtree(owner_geometries)
        pair_indexes = tree.query(
            owner_geometries,
            predicate="dwithin",
            distance=gap_meters,
        )
        for left, right in zip(pair_indexes[0], pair_indexes[1], strict=True):
            # The bulk query includes self-pairs and usually both pair directions.
            if left < right:
                components.union(int(left), int(right))

        members_by_root: dict[int, list[int]] = {}
        for local_position, source_position in enumerate(positions):
            root = components.find(local_position)
            members_by_root.setdefault(root, []).append(source_position)

        component_stats = []
        for members in members_by_root.values():
            acres = float(annotated.loc[members, "TotalAcres"].sum())
            component_stats.append((members, acres))
        # Source position breaks equal-acre ties deterministically.
        component_stats.sort(key=lambda item: (-item[1], min(item[0])))

        owner_rank = int(rank_by_owner[owner_name])
        for block_number, (members, acres) in enumerate(
            component_stats, start=1
        ):
            block_id = f"{owner_rank}-{block_number:03d}"
            annotated.loc[members, "BlockID"] = block_id
            annotated.loc[members, "BlockAcres"] = acres
            annotated.loc[members, "BlockParcelCount"] = len(members)

    block_records = []
    for block_id, block_parcels in annotated.groupby("BlockID", sort=False):
        owners = block_parcels["OwnerName_Grouped"].unique()
        if len(owners) != 1:
            raise ValueError(f"Block {block_id} contains multiple owners")
        block_records.append(
            {
                "OwnerName_Grouped": owners[0],
                "Rank": int(rank_by_owner[owners[0]]),
                "BlockID": block_id,
                "BlockAcres": float(block_parcels["TotalAcres"].sum()),
                "BlockParcelCount": len(block_parcels),
                # Union the original WGS84 geometries, never buffered geometries.
                "geometry": union_all(block_parcels.geometry.to_numpy()),
            }
        )
    blocks = gpd.GeoDataFrame(block_records, geometry="geometry", crs=annotated.crs)
    blocks = blocks.sort_values(["Rank", "BlockID"]).reset_index(drop=True)
    return annotated, blocks


def validate_landholding_blocks(
    parcels: gpd.GeoDataFrame, blocks: gpd.GeoDataFrame
) -> None:
    """Validate block membership, acreage, parcel counts, ownership, and CRS."""
    if parcels["BlockID"].isna().any() or (parcels["BlockID"] == "").any():
        raise ValueError("Every selected parcel must have exactly one BlockID")
    if parcels.crs is None or parcels.crs.to_epsg() != 4326:
        raise ValueError("Parcel output geometry must be in EPSG:4326")
    if blocks.crs is None or blocks.crs.to_epsg() != 4326:
        raise ValueError("Block output geometry must be in EPSG:4326")
    if not blocks["BlockID"].is_unique:
        raise ValueError("Block output must contain one feature per BlockID")
    if set(parcels["BlockID"]) != set(blocks["BlockID"]):
        raise ValueError("Parcel and block outputs contain different BlockIDs")
    if (
        parcels.groupby("BlockID")["OwnerName_Grouped"].nunique() > 1
    ).any():
        raise ValueError("A block cannot contain parcels from multiple owners")

    for owner_name, owner_parcels in parcels.groupby("OwnerName_Grouped"):
        owner_blocks = blocks.loc[blocks["OwnerName_Grouped"] == owner_name]
        parcel_acres = float(owner_parcels["TotalAcres"].sum())
        block_acres = float(owner_blocks["BlockAcres"].sum())
        if not math.isclose(parcel_acres, block_acres, rel_tol=1e-12, abs_tol=1e-6):
            raise ValueError(f"Block acreage does not reconcile for {owner_name}")
        if int(owner_blocks["BlockParcelCount"].sum()) != len(owner_parcels):
            raise ValueError(f"Block parcel counts do not reconcile for {owner_name}")


def print_block_summary(blocks: gpd.GeoDataFrame) -> None:
    for owner_name, owner_blocks in blocks.groupby(
        "OwnerName_Grouped", sort=False
    ):
        print(owner_name)
        for block in owner_blocks.sort_values("BlockID").itertuples(index=False):
            block_number = str(block.BlockID).rsplit("-", 1)[1]
            print(
                f"{block_number}: {block.BlockAcres:,.0f} acres / "
                f"{block.BlockParcelCount:,} parcels"
            )
        print()


def write_owner_geojson(
    rank: int,
    owner_name: str,
    owner_parcels: gpd.GeoDataFrame,
) -> Path:
    path = GEODATA_OUTPUT_DIR / f"{rank}-{safe_filename(owner_name)}.geojson"
    owner_parcels.to_file(path, driver="GeoJSON", engine="pyogrio")
    return path


def export_geodata(
    ranking: pd.DataFrame,
    parcels: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    workers: int,
) -> None:
    jobs = []
    for row in ranking.itertuples(index=False):
        owner_parcels = parcels.loc[
            parcels["OwnerName_Grouped"] == row.Owner, PARCEL_OUTPUT_COLUMNS
        ].copy()
        jobs.append((row.Rank, row.Owner, owner_parcels))

    # Every worker receives a distinct GeoDataFrame and writes a distinct file.
    # This avoids sharing a GDAL dataset between threads.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda args: write_owner_geojson(*args), jobs))

    top_ten = ranking.head(10)[["Owner", "Rank"]]
    combined = parcels.loc[
        parcels["OwnerName_Grouped"].isin(top_ten["Owner"])
    ].merge(
        top_ten,
        left_on="OwnerName_Grouped",
        right_on="Owner",
        how="left",
        validate="many_to_one",
    )
    combined = combined.drop(columns="Owner")
    # Keep the detailed combined GeoJSON parcel-level. The merged block export
    # below is the lightweight overview layer for initial map loading.
    combined[MAP_COLUMNS + ["Rank"]].to_file(
        GEODATA_OUTPUT_DIR / "top-10-combined.geojson",
        driver="GeoJSON",
        engine="pyogrio",
    )
    # Preserve the existing itemized CSV schema; block annotations are part of
    # the per-owner parcel GeoJSON outputs only.
    combined.drop(columns=["geometry", *BLOCK_COLUMNS]).to_csv(
        OUTPUT_DIR / "top-10-itemized.csv", index=False
    )

    # A block is one feature even when its dissolved geometry is multipart
    # (for example, parcels linked across one missing section). Interior holes
    # are retained. Sort by total-holdings rank and then block size/order.
    top_ten_blocks = (
        blocks.loc[blocks["Rank"].isin(top_ten["Rank"])]
        .sort_values(["Rank", "BlockID"], kind="stable")
        .reset_index(drop=True)
    )
    top_ten_blocks.to_file(
        GEODATA_OUTPUT_DIR / "merged-top-10-block.geojson",
        driver="GeoJSON",
        engine="pyogrio",
    )


def write_cleaned_parcels(cleaner: dict[str, str]) -> None:
    """Write the legacy large all-parcel shapefile with grouped owner names."""
    print("Loading all parcel geometry for optional cleaned shapefile...")
    parcels = gpd.read_file(require_file(RAW_PARCELS), engine="pyogrio").to_crs(
        epsg=4326
    )
    parcels["OwnerName_Grouped"] = parcels["OwnerName"].replace(cleaner)
    ensure_output_dirs(CLEANED_DIR)
    parcels.to_file(
        CLEANED_DIR / "parcels-with-mtfp-groupings.shp", engine="pyogrio"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel GeoJSON writers (default: up to 4; use 1 to disable)",
    )
    parser.add_argument(
        "--block-gap-miles",
        type=float,
        default=1.0,
        help="maximum gap between parcels in one block (default: 1 mile)",
    )
    parser.add_argument(
        "--write-cleaned",
        action="store_true",
        help="also create the legacy, very large all-parcel grouped shapefile",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if not math.isfinite(args.block_gap_miles) or args.block_gap_miles < 0:
        parser.error("--block-gap-miles must be a finite, non-negative number")
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs(OUTPUT_DIR, GEODATA_OUTPUT_DIR)
    cleaner = owner_name_cleaner()
    public_owners = public_landowners()

    with tempfile.TemporaryDirectory(prefix="landowner-ranking-") as temp_dir:
        connection = create_owner_database(Path(temp_dir) / "owners.sqlite3")
        processed_count = 0
        try:
            print("Reading parcel attributes in 25,000-row chunks...")
            for parcels in iter_parcel_attributes(
                ["PARCELID", "OwnerName", "TotalAcres"], 25_000
            ):
                store_owner_chunk(
                    connection,
                    parcels,
                    cleaner,
                    public_owners,
                    processed_count,
                )
                processed_count += len(parcels)
                print(f"Processed {processed_count:,} records", end="\r")
            print()
            ranking, owner_stats = aggregate_ranking(connection)
        finally:
            connection.close()

    ranking.to_json(
        OUTPUT_DIR / "final-top-20-list.json",
        orient="records",
        indent=4,
        index=False,
    )
    report = format_ranking(ranking, owner_stats)
    print(report, end="")
    (OUTPUT_DIR / "final-top-20-list.txt").write_text(report, encoding="utf-8")

    print("Reading geometry for the top landowners...")
    selected_parcels = read_selected_geometry(ranking, cleaner)
    print(
        "Grouping parcels into landholding blocks with a "
        f"{args.block_gap_miles:g}-mile gap..."
    )
    selected_parcels, blocks = assign_landholding_blocks(
        ranking, selected_parcels, args.block_gap_miles
    )
    validate_landholding_blocks(selected_parcels, blocks)
    print_block_summary(blocks)
    print(f"Writing geodata with {args.workers} worker(s)...")
    export_geodata(ranking, selected_parcels, blocks, args.workers)

    if args.write_cleaned:
        write_cleaned_parcels(cleaner)


if __name__ == "__main__":
    main()
