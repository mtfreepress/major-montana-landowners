#!/usr/bin/env python3
"""Rank private Montana landowners without grouping owner-name variants."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from analysis_common import (
    OUTPUT_DIR,
    ensure_output_dirs,
    iter_parcel_attributes,
    public_landowners,
)


def create_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute(
        """
        CREATE TABLE owners (
            owner_name TEXT PRIMARY KEY,
            parcel_count INTEGER NOT NULL,
            total_acres REAL NOT NULL,
            is_public INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    return connection


def store_chunk(
    connection: sqlite3.Connection,
    parcels: pd.DataFrame,
    public_owners: set[str],
) -> None:
    grouped = (
        parcels.loc[parcels["OwnerName"].notna()]
        .groupby("OwnerName", sort=False)
        .agg(parcel_count=("PARCELID", "count"), total_acres=("TotalAcres", "sum"))
        .reset_index()
    )
    connection.executemany(
        """
        INSERT INTO owners (owner_name, parcel_count, total_acres, is_public)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (owner_name) DO UPDATE SET
            parcel_count = parcel_count + excluded.parcel_count,
            total_acres = total_acres + excluded.total_acres
        """,
        (
            (owner, int(count), float(acres), int(owner in public_owners))
            for owner, count, acres in grouped.itertuples(index=False)
        ),
    )
    connection.commit()


def main() -> None:
    ensure_output_dirs(OUTPUT_DIR)
    public_owners = public_landowners()

    with tempfile.TemporaryDirectory(prefix="naive-landowner-ranking-") as temp_dir:
        connection = create_database(Path(temp_dir) / "owners.sqlite3")
        processed_count = 0
        try:
            print("Reading parcel attributes in 25,000-row chunks...")
            for parcels in iter_parcel_attributes(
                ["PARCELID", "OwnerName", "TotalAcres"], 25_000
            ):
                store_chunk(connection, parcels, public_owners)
                processed_count += len(parcels)
                print(f"Processed {processed_count:,} records", end="\r")
            print()
            rows = connection.execute(
                """
                SELECT owner_name, parcel_count, total_acres
                FROM owners
                WHERE is_public = 0
                ORDER BY total_acres DESC
                LIMIT 10
                """
            ).fetchall()
        finally:
            connection.close()

    ranking = pd.DataFrame(rows, columns=["OwnerName", "NumParcels", "TotalAcres"])
    ranking.insert(0, "Rank", range(1, len(ranking) + 1))

    print(ranking.to_string(index=False))
    ranking.to_json(
        OUTPUT_DIR / "naive-top-10-list.json",
        orient="records",
        indent=4,
        index=False,
    )


if __name__ == "__main__":
    main()
