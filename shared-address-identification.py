#!/usr/bin/env python3
"""Find large owner-name clusters that share a tax mailing address."""

from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis_common import iter_parcel_attributes, public_landowners


ADDRESS_COLUMNS = [
    "OwnerAddre",
    "OwnerAdd_1",
    "OwnerAdd_2",
    "OwnerCity",
    "OwnerState",
]
INPUT_COLUMNS = ["PARCELID", "OwnerName", *ADDRESS_COLUMNS, "TotalAcres"]
DEFAULT_SEARCHES = [
    "N5B CAPITAL",
    "LUCERNE CO",
    "CISCO TX",
    "PO BOX 11350",
    "200 S 23RD AVE STE D9",
    "32 S EWING",
]


def join_addresses(parcels: pd.DataFrame) -> pd.Series:
    # str.cat is vectorized; DataFrame.apply/agg across millions of rows is not.
    joined = parcels[ADDRESS_COLUMNS[0]].astype("string").fillna("")
    for column in ADDRESS_COLUMNS[1:]:
        joined = joined.str.cat(
            parcels[column].astype("string").fillna(""), sep=" "
        )
    return joined.str.replace(r"\s+", " ", regex=True).str.strip()


@dataclass
class SearchResult:
    parcel_count: int = 0
    acreage: float = 0.0
    addresses: set[str] = field(default_factory=set)
    owner_names: set[str] = field(default_factory=set)


def update_searches(
    private_parcels: pd.DataFrame,
    searches: dict[str, SearchResult],
) -> None:
    for text, result in searches.items():
        matches = private_parcels.loc[
            private_parcels["joined_address"].str.contains(
                text, regex=False, na=False
            )
        ]
        result.parcel_count += len(matches)
        result.acreage += matches["TotalAcres"].sum()
        result.addresses.update(matches["joined_address"].dropna().unique())
        result.owner_names.update(matches["OwnerName"].dropna().unique())


def create_group_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(
        """
        CREATE TABLE owner_groups (
            joined_address TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            parcel_count INTEGER NOT NULL,
            total_acres REAL NOT NULL,
            PRIMARY KEY (joined_address, owner_name)
        ) WITHOUT ROWID
        """
    )
    return connection


def store_grouped_chunk(
    connection: sqlite3.Connection, private_parcels: pd.DataFrame
) -> None:
    grouped = (
        private_parcels.groupby(["joined_address", "OwnerName"], sort=False)
        .agg(parcel_count=("PARCELID", "count"), total_acres=("TotalAcres", "sum"))
        .reset_index()
    )
    connection.executemany(
        """
        INSERT INTO owner_groups (
            joined_address, owner_name, parcel_count, total_acres
        ) VALUES (?, ?, ?, ?)
        ON CONFLICT (joined_address, owner_name) DO UPDATE SET
            parcel_count = parcel_count + excluded.parcel_count,
            total_acres = total_acres + excluded.total_acres
        """,
        (
            (address, owner, int(count), float(acres))
            for address, owner, count, acres in grouped.itertuples(index=False)
        ),
    )
    connection.commit()


def top_addresses(
    connection: sqlite3.Connection, limit: int
) -> list[dict[str, object]]:
    rows = connection.execute(
        """
        SELECT joined_address, SUM(parcel_count), SUM(total_acres)
        FROM owner_groups
        GROUP BY joined_address
        ORDER BY SUM(total_acres) DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    results: list[dict[str, object]] = []
    for address, parcel_count, total_acres in rows:
        owner_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT owner_name
                FROM owner_groups
                WHERE joined_address = ?
                ORDER BY owner_name
                """,
                (address,),
            )
        ]
        results.append(
            {
                "joined_address": address,
                "OwnerName": owner_names,
                "PARCELID": parcel_count,
                "TotalAcres": total_acres,
            }
        )
    return results


def print_search(text: str, result: SearchResult) -> None:
    print(f'### SEARCH FOR: "{text}"')
    print("Parcel Count", result.parcel_count)
    print("Acreage", result.acreage)
    print("Addresses", sorted(result.addresses))
    print("OwnerNames", sorted(result.owner_names))
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top", type=int, default=30, help="number of shared addresses to print"
    )
    parser.add_argument(
        "--search",
        action="append",
        dest="searches",
        help="literal address fragment to search for; may be repeated",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25_000,
        help="parcel records held in memory at once (default: 25000)",
    )
    args = parser.parse_args()
    if args.top < 0:
        parser.error("--top cannot be negative")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    search_texts = args.searches if args.searches is not None else DEFAULT_SEARCHES
    search_results = {text: SearchResult() for text in search_texts}
    public_owners = public_landowners()
    processed_count = 0

    with tempfile.TemporaryDirectory(prefix="landowner-addresses-") as temp_dir:
        connection = create_group_database(Path(temp_dir) / "groups.sqlite3")
        try:
            print(f"Reading parcel attributes in {args.chunk_size:,}-row chunks...")
            for parcels in iter_parcel_attributes(INPUT_COLUMNS, args.chunk_size):
                processed_count += len(parcels)
                parcels["joined_address"] = join_addresses(parcels)
                private_parcels = parcels.loc[
                    ~parcels["OwnerName"].isin(public_owners)
                ]
                update_searches(private_parcels, search_results)
                store_grouped_chunk(connection, private_parcels)
                print(f"Processed {processed_count:,} records", end="\r")
            print()
            print(json.dumps(top_addresses(connection, args.top), indent=4))
        finally:
            connection.close()

    for text, result in search_results.items():
        print_search(text, result)


if __name__ == "__main__":
    main()
