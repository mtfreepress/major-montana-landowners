#!/usr/bin/env python3
"""Write a reporting-friendly summary of the final top-20 landowners."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis_common import OUTPUT_DIR, iter_parcel_attributes, require_file


TOP_COUNT = 20
DEFAULT_RANKING = OUTPUT_DIR / "final-top-20-list.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "final-top-20-summary.txt"
UNKNOWN_COUNTY = "Unknown county"


@dataclass
class CountyStats:
    acres: float = 0.0
    parcels: int = 0


@dataclass
class OwnerStats:
    acres: float = 0.0
    parcels: int = 0
    counties: dict[str, CountyStats] = field(
        default_factory=lambda: defaultdict(CountyStats)
    )
    largest_parcel_acres: float = -math.inf
    largest_parcel_county: str = UNKNOWN_COUNTY
    largest_parcel_id: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking",
        type=Path,
        default=DEFAULT_RANKING,
        help=f"final-analysis JSON input (default: {DEFAULT_RANKING})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"text summary output (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="parcel records read at a time (default: 100,000)",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    return args


def load_ranking(path: Path) -> list[dict]:
    with require_file(path).open(encoding="utf-8") as file:
        ranking = json.load(file)

    if not isinstance(ranking, list) or not ranking:
        raise ValueError(f"Ranking must be a non-empty JSON list: {path}")
    ranking = ranking[:TOP_COUNT]
    required = {
        "Rank",
        "Owner",
        "NumParcels",
        "TotalAcres",
        "IncludedOwnerNames",
    }
    for row in ranking:
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"Ranking entry {row!r} is missing: {', '.join(sorted(missing))}"
            )
    return ranking


def owner_name_lookup(ranking: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in ranking:
        for recorded_name in row["IncludedOwnerNames"]:
            existing = lookup.get(recorded_name)
            if existing is not None and existing != row["Owner"]:
                raise ValueError(
                    f"Recorded owner name {recorded_name!r} is assigned to both "
                    f"{existing!r} and {row['Owner']!r}"
                )
            lookup[recorded_name] = row["Owner"]
    return lookup


def normalize_county(value) -> str:
    if pd.isna(value):
        return UNKNOWN_COUNTY
    county = str(value).strip()
    return county or UNKNOWN_COUNTY


def collect_stats(
    ranking: list[dict], chunk_size: int
) -> dict[str, OwnerStats]:
    name_to_owner = owner_name_lookup(ranking)
    stats = {row["Owner"]: OwnerStats() for row in ranking}

    for parcels in iter_parcel_attributes(
        ["PARCELID", "OwnerName", "CountyName", "TotalAcres"], chunk_size
    ):
        selected = parcels.loc[parcels["OwnerName"].isin(name_to_owner)]
        for parcel in selected.itertuples(index=False):
            owner = name_to_owner[parcel.OwnerName]
            owner_stats = stats[owner]
            county = normalize_county(parcel.CountyName)
            acres = 0.0 if pd.isna(parcel.TotalAcres) else float(parcel.TotalAcres)
            has_parcel_id = not pd.isna(parcel.PARCELID)

            owner_stats.acres += acres
            owner_stats.counties[county].acres += acres
            if has_parcel_id:
                owner_stats.parcels += 1
                owner_stats.counties[county].parcels += 1
            if acres > owner_stats.largest_parcel_acres:
                owner_stats.largest_parcel_acres = acres
                owner_stats.largest_parcel_county = county
                owner_stats.largest_parcel_id = (
                    str(parcel.PARCELID) if has_parcel_id else "unknown"
                )

    return stats


def validate_stats(ranking: list[dict], stats: dict[str, OwnerStats]) -> None:
    for row in ranking:
        owner_stats = stats[row["Owner"]]
        if owner_stats.parcels != int(row["NumParcels"]):
            raise ValueError(
                f"Parcel count for {row['Owner']} does not match the ranking "
                f"({owner_stats.parcels:,} vs. {int(row['NumParcels']):,}). "
                "Rerun major-landowners-final-analysis.py."
            )
        if not math.isclose(
            owner_stats.acres,
            float(row["TotalAcres"]),
            rel_tol=1e-9,
            abs_tol=0.01,
        ):
            raise ValueError(
                f"Acreage for {row['Owner']} does not match the ranking "
                f"({owner_stats.acres:,.3f} vs. {float(row['TotalAcres']):,.3f}). "
                "Rerun major-landowners-final-analysis.py."
            )


def parcel_label(count: int) -> str:
    return "parcel" if count == 1 else "parcels"


def format_summary(ranking: list[dict], stats: dict[str, OwnerStats]) -> str:
    lines = [
        "MONTANA'S TOP 20 PRIVATE LANDOWNERS",
        "====================================",
        "",
        "A reporting summary based on outputs/final-top-20-list.json and the",
        "Montana Cadastral parcel attributes. Acreage is the source TotalAcres",
        "field; parcels are cadastral records, not necessarily separate ranches",
        "or contiguous holdings. Owner-name variants are manually consolidated.",
        "",
    ]

    for row in ranking:
        owner = row["Owner"]
        owner_stats = stats[owner]
        counties = sorted(
            owner_stats.counties.items(),
            key=lambda item: (-item[1].acres, item[0]),
        )
        county_names = sorted(owner_stats.counties)
        largest_county, largest_county_stats = counties[0]
        largest_share = (
            largest_county_stats.acres / owner_stats.acres * 100
            if owner_stats.acres
            else 0.0
        )
        average_parcel = (
            owner_stats.acres / owner_stats.parcels if owner_stats.parcels else 0.0
        )

        lines.extend(
            [
                f"#{int(row['Rank'])} {owner}",
                f"Total holdings: {owner_stats.acres:,.0f} acres across "
                f"{owner_stats.parcels:,} {parcel_label(owner_stats.parcels)}",
                f"Counties ({len(county_names)}): {', '.join(county_names)}",
                "County breakdown:",
            ]
        )
        for county, county_stats in counties:
            share = (
                county_stats.acres / owner_stats.acres * 100
                if owner_stats.acres
                else 0.0
            )
            lines.append(
                f"  - {county}: {county_stats.acres:,.0f} acres "
                f"({share:.1f}%); {county_stats.parcels:,} "
                f"{parcel_label(county_stats.parcels)}"
            )

        variant_count = len(row["IncludedOwnerNames"])
        variant_label = "name" if variant_count == 1 else "names"
        lines.extend(
            [
                "Other potentially useful numbers:",
                f"  - Largest county concentration: {largest_county}, "
                f"{largest_county_stats.acres:,.0f} acres ({largest_share:.1f}%)",
                f"  - Average acreage per parcel: {average_parcel:,.1f}",
                f"  - Largest recorded parcel: "
                f"{owner_stats.largest_parcel_acres:,.0f} acres in "
                f"{owner_stats.largest_parcel_county} "
                f"(parcel ID {owner_stats.largest_parcel_id})",
                f"  - Consolidated from {variant_count:,} recorded owner {variant_label}",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    ranking = load_ranking(args.ranking)
    print(f"Reading parcel attributes for {len(ranking)} ranked owners...")
    stats = collect_stats(ranking, args.chunk_size)
    validate_stats(ranking, stats)
    summary = format_summary(ranking, stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
