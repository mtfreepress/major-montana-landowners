#!/usr/bin/env python3
"""Rank Montana public landholders and summarize land by jurisdiction."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from analysis_common import (
    OUTPUT_DIR,
    ensure_output_dirs,
    iter_parcel_attributes,
    public_landowners,
)


DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "major-public-landowners.json"
DEFAULT_TEXT_OUTPUT = OUTPUT_DIR / "major-public-landowners.txt"

FEDERAL = "Federal"
STATE = "State of Montana"
TRIBAL_TRUST = "Tribal/individual trust"
LOCAL = "Local government"
JURISDICTION_ORDER = (FEDERAL, STATE, TRIBAL_TRUST, LOCAL)


@dataclass(frozen=True)
class Classification:
    jurisdiction: str
    landholder: str


@dataclass
class AliasStats:
    acres: float = 0.0
    parcels: int = 0


@dataclass
class LandholderStats:
    acres: float = 0.0
    parcels: int = 0
    aliases: dict[str, AliasStats] = field(
        default_factory=lambda: defaultdict(AliasStats)
    )


def normalize_owner_name(value: str) -> str:
    """Normalize punctuation and spacing without changing ownership meaning."""
    normalized = re.sub(r"[^A-Z0-9]+", " ", value.upper())
    return " ".join(normalized.split())


def has_any(value: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in value for phrase in phrases)


def tribal_landholder(normalized: str) -> str | None:
    """Return a tribe name, or a generic trust label, for tribal/trust records."""
    is_trust = bool(
        re.search(
            r"\b(?:USA|U S A|UNITED STATES(?: OF AMERICA)?) IN TRUST\b",
            normalized,
        )
        or "USA IN TRUST" in normalized
    )
    tribal_markers = (
        "TRIBE",
        "TRIBAL",
        "INDIAN RESERVATION",
        "FLATHEAD RESERVATION",
        "FORT BELKNAP",
        "FORT PECK INDIAN",
        "BUREAU OF INDIAN AFFAIRS",
        "SALISH KOOTENAI",
        "SALISH AND KOOTENAI",
        "SALISH KOOTENAI HOUSING AUTHORITY",
    )
    has_tribal_marker = has_any(normalized, tribal_markers) or bool(
        re.search(r"\b(?:CSKT|CKST)\b", normalized)
    )
    if not is_trust and not has_tribal_marker:
        return None

    if "CROW" in normalized:
        return "Crow Tribe / trust land"
    if has_any(normalized, ("SALISH", "KOOTENAI")) or re.search(
        r"\b(?:CSKT|CKST)\b", normalized
    ):
        return "Confederated Salish and Kootenai Tribes / trust land"
    if "NORTHERN CHEYENNE" in normalized or "CHEYENNES" in normalized:
        return "Northern Cheyenne Tribe / trust land"
    if "BLACKFEET" in normalized:
        return "Blackfeet Nation / trust land"
    if "FORT BELKNAP" in normalized:
        return "Fort Belknap Indian Community / trust land"
    if "FORT PECK" in normalized or has_any(
        normalized, ("ASSINIBOINE", "SIOUX TRIBES")
    ):
        return "Fort Peck Assiniboine and Sioux Tribes / trust land"
    if has_any(normalized, ("CHIPPEWA CREE", "ROCKY BOYS")):
        return "Chippewa Cree Tribe / trust land"
    if "BUREAU OF INDIAN AFFAIRS" in normalized:
        return "Bureau of Indian Affairs / trust land"
    return "Tribal or individual trust land (unspecified)"


def federal_landholder(normalized: str) -> str | None:
    """Consolidate federal agency aliases found in cadastral owner names."""
    if (
        "NATIONAL FOREST" in normalized
        or re.search(r"\bFOREST SER(?:VICE|IVCE|VCE|VIDE|IECE|RICE)?\b", normalized)
        or "USDA FORERST SERVICE" in normalized
        or "USDA FORST SERVICE" in normalized
        or "ESDA FOREST SERVICE" in normalized
        or "USDS FOREST SERVICE" in normalized
    ):
        return "U.S. Forest Service"

    if (
        re.search(r"\bBLM\b", normalized)
        or re.search(
            r"\bBUREAU (?:OF )?LAND (?:MANAGEMENT|MANGEMENT|MGMT|MGNT|MGT)\b",
            normalized,
        )
        or normalized == "MONTANA STATE OFFICE"
    ):
        return "U.S. Bureau of Land Management"

    if re.search(r"\bFISH (?:AND )?WILDLIFE SERVICES?\b", normalized) or (
        re.match(r"^(?:USDI|USA|U S A|US|U S|UNITED STATES)\b", normalized)
        and re.search(r"\bFISH (?:AND )?WILDLIFE\b", normalized)
    ):
        return "U.S. Fish and Wildlife Service"

    if (
        "NATIONAL PARK SERVICE" in normalized
        or "NAT L PARK SERVICE" in normalized
        or "BIGHORN CANYON NRA" in normalized
    ):
        return "U.S. National Park Service"

    if "BUREAU OF RECLAMATION" in normalized:
        return "U.S. Bureau of Reclamation"

    if "AGRICULTURAL RESEARCH SERVICE" in normalized:
        return "USDA Agricultural Research Service"

    if has_any(
        normalized,
        (
            "US DEPARTMENT OF DEFENSE",
            "U S DEPARTMENT OF DEFENSE",
            "USA DEPT ARMY",
            "USA DEPT OF THE ARMY",
            "USA REAL ESTATE DIVISION",
        ),
    ):
        return "U.S. Department of Defense"

    if normalized in {"USDA", "DEPARTMENT OF AGRICULTURE"} or has_any(
        normalized,
        (
            "UNITED STATES DEPARTMENT OF AGRICULTURE",
            "UNITED STATES OF AMERICA DEPT OF AG",
            "USA DEPT OF AGRICULTURE",
        ),
    ):
        return "U.S. Department of Agriculture (agency unspecified)"

    if has_any(
        normalized,
        (
            "DEPARTMENT OF INTERIOR",
            "DEPARTMENT OF THE INTERIOR",
            "UNITED STATES DEPT OF THE INTERIOR",
            "UNITED STATES DEPT OF INTERIOR",
        ),
    ):
        return "U.S. Department of the Interior (agency unspecified)"

    exact_generic_names = {
        "USA",
        "U S A",
        "UNITED STATES",
        "UNITED STATES OF AMERICA",
        "THE UNITED STATES OF AMERICA",
        "USA GOVERNMENT",
        "UNITED STATES OF AMRERICA",
        "UNITED STATES OF AMERICA AND ITS ASSIGNS",
        "UNITED STATES OF AMERICA USDI",
        "USDI UNITED STATES OF AMERICA",
    }
    if normalized in exact_generic_names:
        return "United States (agency unspecified)"

    return None


def state_landholder(normalized: str, is_known_public: bool) -> str | None:
    """Consolidate Montana state agencies while retaining agency breakdowns."""
    governmental_prefix = bool(
        re.match(
            r"^(?:STATE|ST |MONTANA (?:DEPT|DEPARTMENT|DPT|FISH)|"
            r"MONTANA STATE (?:OF )?FISH|"
            r"MT (?:DEPT|DEPARTMENT)|DEPT|DEPARTMENT)\b",
            normalized,
        )
    )
    fwp_name = bool(
        re.search(r"\bFISH (?:AND )?(?:GAME|WILDLIFE)\b", normalized)
        or has_any(
            normalized,
            (
                "FISH WILDLIFE",
                "FISH GAME",
                "FISH WLDLF",
                "DEPARTMENT OF FISH",
                "DEPARTMENT OF FWP",
                "DEPT OF FWP",
                "DEPT OF FW P",
                "DEPT OF F W P",
                "DEPT F W P",
                "DEPT OF F W PARKS",
                "DEPT OF FW PARKS",
                " FWP",
            ),
        )
    )
    if (
        fwp_name
        and (is_known_public or governmental_prefix)
        and "CONSERVATION TRUST" not in normalized
        and "PRESERVE" not in normalized
    ):
        return "Montana Fish, Wildlife and Parks"

    if has_any(
        normalized,
        (
            "DNRC",
            "D N R C",
            "NATURAL RESOURCES",
            "STATE LANDS",
            "BOARD OF LAND COMMISSIONERS",
            "BOARD OF LAND COMMISIONERS",
            "LAND COMMISSIONERS",
            "TRUST LAND",
            "STATE OF MONTANA TRUST",
            "STATE OF MONTANA LANDS",
            "DEPT STATE LANDS",
            "DEPT OF STATE LANDS",
            "DEPARTMENT OF STATE LANDS",
        ),
    ):
        return "Montana DNRC / state trust lands"

    if has_any(normalized, ("TRANSPORTATION", "HIGHWAY", "HIGHWAYS")) and (
        is_known_public or governmental_prefix
    ):
        return "Montana Department of Transportation"

    if "MONTANA STATE PRISON" in normalized:
        return "Montana Department of Corrections"

    if "MONTANA STATE UNIVERSITY" in normalized:
        return "Montana State University"
    if "UNIVERSITY OF MONTANA" in normalized:
        return "University of Montana"

    state_prefix = bool(
        re.match(
            r"^(?:STATE OF MONTANA|STATE OF MT|ST OF MONTANA|ST OF MT|"
            r"MONTANA STATE OF|STATE MT|ST MT)\b",
            normalized,
        )
    )
    if normalized in {"STATE", "MONTANA STATE OF"} or state_prefix:
        return "State of Montana (agency unspecified)"

    # Every non-federal/non-tribal entry in the curated public list is a state
    # alias. This catches short historical spellings such as "DEPT STATE LANDS".
    if is_known_public:
        return "State of Montana (agency unspecified)"
    return None


def local_landholder(normalized: str) -> str | None:
    """Identify conservatively named counties, cities, towns, and districts."""
    if re.match(r"^(?:CITY|TOWN) OF [A-Z]", normalized):
        return normalized.title()

    if re.match(
        r"^[A-Z &-]+ COUNTY(?:$| (?:COMMISSIONERS|AIRPORT|GOVERNMENT)\b)",
        normalized,
    ):
        return normalized.title()

    if re.search(r"\bSCHOOL DIST(?:RICT)?\b", normalized):
        return normalized.title()

    if re.search(r"\b(?:AIRPORT AUTHORITY|IRRIGATION DISTRICT)\b", normalized):
        return normalized.title()
    return None


def classify_owner(
    owner_name: str, known_public_owners: set[str]
) -> Classification | None:
    normalized = normalize_owner_name(owner_name)

    # Trust ownership must be tested before federal ownership so land held by
    # the United States in trust is never included in the federal total.
    tribal = tribal_landholder(normalized)
    if tribal is not None:
        return Classification(TRIBAL_TRUST, tribal)

    federal = federal_landholder(normalized)
    if federal is not None:
        return Classification(FEDERAL, federal)

    state = state_landholder(normalized, owner_name in known_public_owners)
    if state is not None:
        return Classification(STATE, state)

    local = local_landholder(normalized)
    if local is not None:
        return Classification(LOCAL, local)
    return None


def collect_stats(
    chunk_size: int,
) -> tuple[dict[Classification, LandholderStats], float, int]:
    known_public_owners = public_landowners()
    unclassified_known = sorted(
        owner
        for owner in known_public_owners
        if classify_owner(owner, known_public_owners) is None
    )
    if unclassified_known:
        raise ValueError(
            "Curated public-owner names lack a classification: "
            + ", ".join(unclassified_known)
        )

    stats: dict[Classification, LandholderStats] = defaultdict(LandholderStats)
    statewide_acres = 0.0
    processed_count = 0

    print(f"Reading parcel attributes in {chunk_size:,}-row chunks...")
    for parcels in iter_parcel_attributes(
        ["PARCELID", "OwnerName", "TotalAcres"], chunk_size
    ):
        statewide_acres += float(parcels["TotalAcres"].sum())
        grouped = (
            parcels.loc[parcels["OwnerName"].notna()]
            .groupby("OwnerName", sort=False)
            .agg(
                parcel_count=("PARCELID", "count"),
                total_acres=("TotalAcres", "sum"),
            )
            .reset_index()
        )
        for owner_name, parcel_count, total_acres in grouped.itertuples(
            index=False, name=None
        ):
            classification = classify_owner(owner_name, known_public_owners)
            if classification is None:
                continue
            acres = float(total_acres)
            count = int(parcel_count)
            holder = stats[classification]
            holder.acres += acres
            holder.parcels += count
            holder.aliases[owner_name].acres += acres
            holder.aliases[owner_name].parcels += count

        processed_count += len(parcels)
        print(f"Processed {processed_count:,} records", end="\r")
    print()
    return stats, statewide_acres, processed_count


def ranked_records(
    stats: dict[Classification, LandholderStats], top_count: int
) -> list[dict]:
    ordered = sorted(
        stats.items(),
        key=lambda item: (-item[1].acres, item[0].landholder),
    )
    records = []
    for rank, (classification, holder) in enumerate(
        ordered[:top_count], start=1
    ):
        aliases = sorted(
            holder.aliases.items(),
            key=lambda item: (-item[1].acres, item[0]),
        )
        records.append(
            {
                "Rank": rank,
                "Landholder": classification.landholder,
                "Jurisdiction": classification.jurisdiction,
                "NumParcels": holder.parcels,
                "TotalAcres": holder.acres,
                "RecordedOwnerNames": [
                    {
                        "OwnerName": name,
                        "NumParcels": alias.parcels,
                        "TotalAcres": alias.acres,
                    }
                    for name, alias in aliases
                ],
            }
        )
    return records


def jurisdiction_totals(
    stats: dict[Classification, LandholderStats], statewide_acres: float
) -> list[dict]:
    totals = {
        jurisdiction: {"acres": 0.0, "parcels": 0}
        for jurisdiction in JURISDICTION_ORDER
    }
    for classification, holder in stats.items():
        totals[classification.jurisdiction]["acres"] += holder.acres
        totals[classification.jurisdiction]["parcels"] += holder.parcels

    return [
        {
            "Jurisdiction": jurisdiction,
            "NumParcels": totals[jurisdiction]["parcels"],
            "TotalAcres": totals[jurisdiction]["acres"],
            "PercentOfCadastralAcres": (
                totals[jurisdiction]["acres"] / statewide_acres * 100
                if statewide_acres
                else 0.0
            ),
        }
        for jurisdiction in JURISDICTION_ORDER
    ]


def format_report(
    records: list[dict],
    totals: list[dict],
    statewide_acres: float,
    parcel_count: int,
) -> str:
    lines = [
        "MONTANA'S MAJOR PUBLIC LANDHOLDERS",
        "===================================",
        "",
        "Calculated from Montana Cadastral parcel attributes. Acreage is summed",
        "from the source TotalAcres field. Agency spelling variants are grouped.",
        "Tribal governments and individual Indian trust records are reported",
        "separately and are excluded from the federal total.",
        "",
        "These results describe ownership names in the cadastral snapshot, not an",
        "authoritative surface-management or land-status inventory. Generic names",
        "such as USA or STATE OF MONTANA cannot be assigned to a specific agency.",
        "",
        f"Dataset: {parcel_count:,} parcels / {statewide_acres:,.0f} cadastral acres",
        "",
        "TOTALS BY JURISDICTION",
        "----------------------",
    ]

    for total in totals:
        exclusion = (
            " (tribal/trust land excluded)"
            if total["Jurisdiction"] == FEDERAL
            else ""
        )
        lines.append(
            f"{total['Jurisdiction']}{exclusion}: "
            f"{total['TotalAcres']:,.0f} acres across "
            f"{total['NumParcels']:,} parcels "
            f"({total['PercentOfCadastralAcres']:.1f}% of cadastral acres)"
        )

    lines.extend(["", "RANKING", "-------"])
    for record in records:
        lines.extend(
            [
                f"#{record['Rank']} {record['Landholder']}",
                f"  {record['Jurisdiction']} / {record['TotalAcres']:,.0f} acres / "
                f"{record['NumParcels']:,} parcels",
                "  Largest recorded owner-name components:",
            ]
        )
        for alias in record["RecordedOwnerNames"][:5]:
            lines.append(
                f"    - {alias['OwnerName']}: {alias['TotalAcres']:,.0f} acres / "
                f"{alias['NumParcels']:,} parcels"
            )
        remaining = len(record["RecordedOwnerNames"]) - 5
        if remaining > 0:
            lines.append(f"    - plus {remaining:,} smaller name variants")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="number of ranked landholders to report (default: 20)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="parcel records read at a time (default: 100,000)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"JSON output (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--text-output",
        type=Path,
        default=DEFAULT_TEXT_OUTPUT,
        help=f"text output (default: {DEFAULT_TEXT_OUTPUT})",
    )
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be at least 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.json_output.parent, args.text_output.parent)
    stats, statewide_acres, parcel_count = collect_stats(args.chunk_size)
    records = ranked_records(stats, args.top)
    totals = jurisdiction_totals(stats, statewide_acres)

    federal_total = next(
        total["TotalAcres"]
        for total in totals
        if total["Jurisdiction"] == FEDERAL
    )
    tribal_total = next(
        total["TotalAcres"]
        for total in totals
        if total["Jurisdiction"] == TRIBAL_TRUST
    )
    if not math.isfinite(federal_total) or not math.isfinite(tribal_total):
        raise ValueError("Jurisdiction totals must be finite")

    report = format_report(records, totals, statewide_acres, parcel_count)
    print(report, end="")
    args.text_output.write_text(report, encoding="utf-8")
    args.json_output.write_text(
        json.dumps(
            {
                "Methodology": {
                    "AcreageField": "TotalAcres",
                    "TribalTrustIncludedInFederal": False,
                    "StatewideCadastralAcres": statewide_acres,
                    "ParcelRecords": parcel_count,
                },
                "JurisdictionTotals": totals,
                "Ranking": records,
            },
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
