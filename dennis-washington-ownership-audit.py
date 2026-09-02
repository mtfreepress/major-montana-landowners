#!/usr/bin/env python3
"""Audit Montana cadastral surface parcels for Dennis Washington connections.

This is deliberately separate from the production owner-name grouping.  It reads
the same OWNERPARCEL layer and sums the same TotalAcres field as the main
landowner analysis, but writes only review artifacts under
``outputs/washington-audit``.

Matching is deliberately asymmetric: recognizable entity names can establish a
strong match, while a shared tax-bill address can only create a review candidate.
Fuzzy matches never establish ownership.  Edit the small, documented entity
registries below if later corporate-record research supports a different default.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyogrio

from analysis_common import OUTPUT_DIR, RAW_PARCELS, iter_parcel_attributes, require_file


TOP_20_CUTOFF = 95_281.0
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "washington-audit"

ADDRESS_COLUMNS = [
    "OwnerAddre",
    "OwnerAdd_1",
    "OwnerAdd_2",
    "OwnerCity",
    "OwnerState",
    "OwnerZipCo",
]
IDENTITY_COLUMNS = ["OwnerName", "DbaName", "CareOfTaxp"]
OPTIONAL_CONTEXT_COLUMNS = ["TaxYear", "PropertyID", "Assessment", "PropType"]
REQUIRED_COLUMNS = ["PARCELID", "CountyName", "OwnerName", "TotalAcres"]

# Exact names supported by a direct identity basis.  The Washington Companies'
# current portfolio page lists Envirocon, Modern Machinery, and Montana Resources;
# Montana Resources' own site describes its Washington Companies ownership.  The
# LLC spelling was omitted from the supplied seeds but is the current cadastral and
# operating-permit name, so it is intentionally included and reported.
#
# Sources checked when this registry was authored:
# https://www.washingtoncompanies.com/companies/
# https://www.montanaresources.com/about/
CONFIRMED_ENTITY_NAMES = {
    "DENNIS WASHINGTON",
    "DENNIS R WASHINGTON",
    "PHYLLIS WASHINGTON",
    "PHYLLIS J WASHINGTON",
    "DENNIS AND PHYLLIS WASHINGTON",
    "DENNIS R AND PHYLLIS J WASHINGTON",
    "WASHINGTON COMPANIES",
    "THE WASHINGTON COMPANIES",
    "WASHINGTON CORPORATIONS",
    "MONTANA RESOURCES",
    "MONTANA RESOURCES LLC",  # current name; omitted from supplied seed list
    "MONTANA RESOURCES LLP",
    "MODERN MACHINERY",
    "MODERN MACHINERY CO",
    "MODERN MACHINERY CO INC",
    "ENVIROCON",
    "ENVIROCON INC",
}

# These supplied search seeds are plausible but are not treated as confirmed by
# name alone.  Exact name + Washington headquarters address is strong; otherwise
# they remain weak candidates pending corporate/deed research.
CONDITIONAL_SEED_NAMES = {
    "GRANT CREEK RANCH",
    "GRANT CREEK RANCH LLC",
    "MONTANA LIMESTONE",
    "MONTANA LIMESTONE RESOURCES",
    "MONTANA LIMESTONE RESOURCES LLC",
    "WASHINGTON LIMESTONE",
    "WASHINGTON LIMESTONE LLC",
    "PIPESTONE QUARRY",
    "PIPESTONE QUARRY LLC",
}

# Current portfolio names not present in the supplied seeds.  They are searched
# and flagged rather than silently consolidated.  Exact matches are still review
# candidates, not automatically included in the defensible total.
OMITTED_PORTFOLIO_REVIEW_NAMES = {
    "AVIATION PARTNERS",
    "AVIATION PARTNERS INC",
    "BLUE WATER RAIL SERVICES",
    "BLUE WATER RAIL SERVICES LP",
    "SEASPAN CORPORATION",
    "SEASPAN MARINE",
    "SEASPAN SHIPYARDS",
    "SOUTHERN RAILWAY OF BC",
    "SPOKANE MACHINERY",
}

# Montana Rail Link ceased Washington operations in 2024.  These names, plus
# obvious BNSF/lessee artifacts, are exposed in the CSV if a broad/address rule
# finds them but excluded from all current Washington acreage scenarios.
HISTORICAL_OR_UNRELATED_NAMES = {
    "BURLINGTON NORTHERN COMMUNICATIONS",
    "MONT RAIL LINK",
    "MONTANA RAIL LINK",
    "MONTANA RAIL LINK INC",
    "MONTANA RAIL LINK MISSOULA ELECTIC CO OP LESSEE",
}

KNOWN_SEED_TEXT = {
    *CONFIRMED_ENTITY_NAMES,
    *CONDITIONAL_SEED_NAMES,
}
BROAD_KEYWORDS = [
    "WASHINGTON",
    "MONTANA RESOURCES",
    "GRANT CREEK",
    "PIPESTONE",
    "LIMESTONE",
    "ENVIROCON",
    "MODERN MACHINERY",
]
FUZZY_REFERENCES = sorted(CONFIRMED_ENTITY_NAMES | CONDITIONAL_SEED_NAMES)
FUZZY_GATE_TOKENS = {
    token
    for reference in FUZZY_REFERENCES
    for token in reference.split()
    if len(token) >= 5 and token not in {"COMPANY", "RESOURCES"}
}

CONFIDENCE_ORDER = {
    "unrelated": 0,
    "weak_candidate": 1,
    "address_candidate": 2,
    "strong_candidate": 3,
    "confirmed": 4,
}

OWNER_OUTPUT_COLUMNS = [
    "owner_name",
    "normalized_owner_name",
    "total_acres",
    "parcel_count",
    "counties",
    "county_acreage_breakdown",
    "owner_addresses",
    "match_reason",
    "confidence",
    "include_in_defensible_total",
]


def normalize_text(value: object) -> str:
    """Uppercase text, normalize common legal suffixes, punctuation and spaces."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().strip()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    replacements = {
        r"\bINCORPORATED\b": "INC",
        r"\bCORPORATION\b": "CORP",
        r"\bLIMITED LIABILITY COMPANY\b": "LLC",
        r"\bLIMITED PARTNERSHIP\b": "LP",
        r"\bCOMPANIES\b": "COMPANIES",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_address(value: object) -> str:
    """Normalize mailing text while retaining enough detail for exact sharing."""
    text = normalize_text(value)
    replacements = {
        r"\bP O BOX\b": "PO BOX",
        r"\bPOST OFFICE BOX\b": "PO BOX",
        r"\bDRIVE\b": "DR",
        r"\bAVENUE\b": "AVE",
        r"\bROAD\b": "RD",
        r"\bSTREET\b": "ST",
        r"\bBOULEVARD\b": "BLVD",
        r"\bHIGHWAY\b": "HWY",
        r"\bSUITE\b": "STE",
        r"\bINTERNATIONAL DRIVE\b": "INTERNATIONAL DR",
        r"\bINTL DR\b": "INTERNATIONAL DR",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series, *, address: bool = False) -> pd.Series:
    """Vectorized counterpart to normalize_text/normalize_address."""
    normalized = (
        series.astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
        .str.replace("&", " AND ", regex=False)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\bINCORPORATED\b", "INC", regex=True)
        .str.replace(r"\bCORPORATION\b", "CORP", regex=True)
        .str.replace(r"\bLIMITED LIABILITY COMPANY\b", "LLC", regex=True)
        .str.replace(r"\bLIMITED PARTNERSHIP\b", "LP", regex=True)
    )
    if address:
        normalized = (
            normalized.str.replace(r"\bP O BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bPOST OFFICE BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bDRIVE\b", "DR", regex=True)
            .str.replace(r"\bAVENUE\b", "AVE", regex=True)
            .str.replace(r"\bROAD\b", "RD", regex=True)
            .str.replace(r"\bSTREET\b", "ST", regex=True)
            .str.replace(r"\bBOULEVARD\b", "BLVD", regex=True)
            .str.replace(r"\bHIGHWAY\b", "HWY", regex=True)
            .str.replace(r"\bSUITE\b", "STE", regex=True)
            .str.replace(r"\bINTL DR\b", "INTERNATIONAL DR", regex=True)
        )
    return normalized.str.replace(r"\s+", " ", regex=True).str.strip()


def join_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns:
        return pd.Series("", index=frame.index, dtype="string")
    joined = frame[columns[0]].astype("string").fillna("")
    for column in columns[1:]:
        joined = joined.str.cat(frame[column].astype("string").fillna(""), sep=" ")
    return joined.str.replace(r"\s+", " ", regex=True).str.strip()


def is_headquarters_address(normalized_address: str) -> bool:
    if "MISSOULA" not in normalized_address or not re.search(
        r"\bMT\b", normalized_address
    ):
        return False
    return (
        "101 INTERNATIONAL DR" in normalized_address
        or "PO BOX 16630" in normalized_address
    )


def headquarters_mask(addresses: pd.Series) -> pd.Series:
    in_missoula = addresses.str.contains("MISSOULA", regex=False, na=False)
    in_montana = addresses.str.contains(r"\bMT\b", regex=True, na=False)
    at_location = addresses.str.contains(
        r"101 INTERNATIONAL DR|PO BOX 16630", regex=True, na=False
    )
    return in_missoula & in_montana & at_location


def prepare_chunk(
    parcels: pd.DataFrame, address_columns: list[str], identity_columns: list[str]
) -> pd.DataFrame:
    parcels = parcels.copy()
    for column in identity_columns:
        parcels[f"_norm_{column}"] = normalize_series(parcels[column])
    raw_address = join_columns(parcels, address_columns)
    parcels["_raw_owner_address"] = raw_address
    parcels["_normalized_owner_address"] = normalize_series(raw_address, address=True)
    return parcels


def fuzzy_candidates(names: set[str], cutoff: float) -> dict[str, tuple[str, float]]:
    """Return gated close-name variants; callers must keep these non-definitive."""
    matches: dict[str, tuple[str, float]] = {}
    exact_names = CONFIRMED_ENTITY_NAMES | CONDITIONAL_SEED_NAMES
    for name in sorted(names):
        if not name or name in exact_names:
            continue
        tokens = set(name.split())
        if not tokens.intersection(FUZZY_GATE_TOKENS):
            continue
        best_reference = ""
        best_ratio = 0.0
        for reference in FUZZY_REFERENCES:
            # Sequence similarity alone makes large, unrelated names such as
            # SAND CREEK RANCH LLC look close to GRANT CREEK RANCH LLC.  Require
            # the leading distinctive token as a conservative semantic anchor.
            # This retains GRANT CR RANCH and MONTANA RESOURCE variants.
            reference_anchor = reference.split()[0]
            if reference_anchor not in tokens:
                continue
            ratio = difflib.SequenceMatcher(None, name, reference).ratio()
            if ratio > best_ratio:
                best_reference = reference
                best_ratio = ratio
        if best_ratio >= cutoff:
            matches[name] = (best_reference, best_ratio)
    return matches


def discover_confirmed_addresses_and_fuzzy_names(
    columns: list[str],
    address_columns: list[str],
    identity_columns: list[str],
    chunk_size: int,
    fuzzy_cutoff: float,
    use_fuzzy: bool,
) -> tuple[set[str], dict[str, tuple[str, float]], int, set[int]]:
    confirmed_addresses: set[str] = set()
    fuzzy_name_pool: set[str] = set()
    tax_years: set[int] = set()
    processed = 0

    print("Pass 1/2: discovering confirmed-entity addresses and name variants...")
    for parcels in iter_parcel_attributes(columns, chunk_size):
        parcels = prepare_chunk(parcels, address_columns, identity_columns)
        processed += len(parcels)
        owner_names = parcels["_norm_OwnerName"]
        confirmed = owner_names.isin(CONFIRMED_ENTITY_NAMES)
        confirmed_addresses.update(
            address
            for address in parcels.loc[
                confirmed, "_normalized_owner_address"
            ].unique()
            if address
        )
        if use_fuzzy:
            fuzzy_name_pool.update(owner_names.loc[owner_names.ne("")].unique())
        if "TaxYear" in parcels:
            tax_years.update(
                int(value) for value in parcels["TaxYear"].dropna().unique()
            )
        print(f"  Scanned {processed:,} source records", end="\r", flush=True)
    print(f"  Scanned {processed:,} source records")

    fuzzy_matches = (
        fuzzy_candidates(fuzzy_name_pool, fuzzy_cutoff) if use_fuzzy else {}
    )
    print(f"  Confirmed-entity normalized addresses: {len(confirmed_addresses):,}")
    print(f"  Fuzzy owner-name variants retained: {len(fuzzy_matches):,}")
    return confirmed_addresses, fuzzy_matches, processed, tax_years


def identity_keyword_hits(
    row: pd.Series, identity_columns: list[str]
) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for column in identity_columns:
        value = row[f"_norm_{column}"]
        for keyword in BROAD_KEYWORDS:
            if keyword in value:
                hits.append((column, keyword))
        for portfolio_name in OMITTED_PORTFOLIO_REVIEW_NAMES:
            if portfolio_name in value:
                hits.append((column, f"portfolio:{portfolio_name}"))
    return hits


def classify_candidate(
    row: pd.Series,
    identity_columns: list[str],
    confirmed_addresses: set[str],
    fuzzy_matches: dict[str, tuple[str, float]],
) -> tuple[str, str] | None:
    owner_name = row["_norm_OwnerName"]
    address = row["_normalized_owner_address"]
    hq_address = is_headquarters_address(address)
    shared_address = bool(address) and address in confirmed_addresses
    reasons: list[str] = []
    confidence_options: list[str] = []

    if owner_name in CONFIRMED_ENTITY_NAMES:
        reasons.append("exact normalized owner-name match to confirmed registry")
        confidence_options.append("confirmed")
    elif owner_name in CONDITIONAL_SEED_NAMES:
        reasons.append("exact normalized owner-name match to supplied conditional seed")
        if hq_address or shared_address:
            confidence_options.append("strong_candidate")
        else:
            confidence_options.append("weak_candidate")

    for column in identity_columns:
        if column == "OwnerName":
            continue
        value = row[f"_norm_{column}"]
        if value in CONFIRMED_ENTITY_NAMES:
            reasons.append(f"exact confirmed entity match in {column}")
            confidence_options.append("strong_candidate")
        elif value in CONDITIONAL_SEED_NAMES:
            reasons.append(f"exact conditional-seed match in {column}")
            confidence_options.append("weak_candidate")

    keyword_hits = identity_keyword_hits(row, identity_columns)
    for column, keyword in keyword_hits:
        if keyword.startswith("portfolio:"):
            reasons.append(
                f"official-portfolio name omitted from supplied seeds in {column}: "
                f"{keyword.removeprefix('portfolio:')}"
            )
            confidence_options.append("weak_candidate")
        elif keyword == "WASHINGTON" and owner_name not in KNOWN_SEED_TEXT:
            reasons.append(
                f"broad WASHINGTON keyword in {column}; surname/term alone is not a link"
            )
            confidence_options.append("unrelated")
        else:
            reasons.append(f"broad keyword in {column}: {keyword}")
            confidence_options.append("weak_candidate")

    if hq_address:
        reasons.append("normalized mailing address matches Washington Companies HQ")
        confidence_options.append("address_candidate")
    if shared_address:
        reasons.append("exact normalized address shared with a confirmed entity")
        confidence_options.append("address_candidate")

    if owner_name in fuzzy_matches:
        reference, ratio = fuzzy_matches[owner_name]
        reasons.append(f"fuzzy owner-name match to {reference} ({ratio:.3f})")
        confidence_options.append("weak_candidate")

    if not reasons:
        return None

    # An exact conditional seed plus independent HQ/shared-address evidence is
    # stronger than address evidence alone.
    if owner_name in CONDITIONAL_SEED_NAMES and (hq_address or shared_address):
        confidence_options.append("strong_candidate")

    # Current-ownership exclusions override incidental address/name evidence.
    if owner_name in HISTORICAL_OR_UNRELATED_NAMES:
        reasons.append("known historical or unrelated record; excluded from current total")
        confidence = "unrelated"
    else:
        confidence = max(confidence_options, key=CONFIDENCE_ORDER.__getitem__)

    return "; ".join(dict.fromkeys(reasons)), confidence


def candidate_mask(
    parcels: pd.DataFrame,
    identity_columns: list[str],
    confirmed_addresses: set[str],
    fuzzy_matches: dict[str, tuple[str, float]],
) -> pd.Series:
    mask = pd.Series(False, index=parcels.index)
    exact_names = (
        CONFIRMED_ENTITY_NAMES
        | CONDITIONAL_SEED_NAMES
        | OMITTED_PORTFOLIO_REVIEW_NAMES
        | HISTORICAL_OR_UNRELATED_NAMES
    )
    broad_pattern = "|".join(re.escape(term) for term in BROAD_KEYWORDS)
    portfolio_pattern = "|".join(
        re.escape(term) for term in OMITTED_PORTFOLIO_REVIEW_NAMES
    )
    for column in identity_columns:
        values = parcels[f"_norm_{column}"]
        mask |= values.isin(exact_names)
        mask |= values.str.contains(broad_pattern, regex=True, na=False)
        mask |= values.str.contains(portfolio_pattern, regex=True, na=False)
    addresses = parcels["_normalized_owner_address"]
    mask |= headquarters_mask(addresses)
    mask |= addresses.isin(confirmed_addresses)
    mask |= parcels["_norm_OwnerName"].isin(fuzzy_matches)
    return mask


def collect_candidate_parcels(
    columns: list[str],
    address_columns: list[str],
    identity_columns: list[str],
    chunk_size: int,
    confirmed_addresses: set[str],
    fuzzy_matches: dict[str, tuple[str, float]],
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    processed = 0
    retained = 0

    print("Pass 2/2: collecting and classifying candidate parcels...")
    for parcels in iter_parcel_attributes(columns, chunk_size):
        parcels = prepare_chunk(parcels, address_columns, identity_columns)
        processed += len(parcels)
        selected = parcels.loc[
            candidate_mask(
                parcels, identity_columns, confirmed_addresses, fuzzy_matches
            )
        ].copy()
        if not selected.empty:
            classifications = selected.apply(
                lambda row: classify_candidate(
                    row, identity_columns, confirmed_addresses, fuzzy_matches
                ),
                axis=1,
            )
            selected["match_reason"] = classifications.map(
                lambda result: result[0] if result else ""
            )
            selected["confidence"] = classifications.map(
                lambda result: result[1] if result else "unrelated"
            )
            retained += len(selected)
            candidates.append(selected)
        print(
            f"  Scanned {processed:,}; retained {retained:,} candidate records",
            end="\r",
            flush=True,
        )
    print(f"  Scanned {processed:,}; retained {retained:,} candidate records")

    if not candidates:
        return pd.DataFrame(columns=columns)
    parcels = pd.concat(candidates, ignore_index=True)
    parcels["_TotalAcres_missing"] = parcels["TotalAcres"].isna()
    parcels["TotalAcres"] = pd.to_numeric(
        parcels["TotalAcres"], errors="coerce"
    ).fillna(0.0)
    return parcels


def raw_address(row: pd.Series, address_columns: list[str]) -> str:
    values = []
    for column in address_columns:
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            values.append(str(value).strip())
    return ", ".join(values)


def join_unique(values: Iterable[object], separator: str = " | ") -> str:
    return separator.join(
        sorted({str(value).strip() for value in values if str(value).strip()})
    )


def summarize_owners(
    parcels: pd.DataFrame, address_columns: list[str]
) -> pd.DataFrame:
    if parcels.empty:
        return pd.DataFrame(columns=OWNER_OUTPUT_COLUMNS)
    parcels = parcels.copy()
    parcels["_display_address"] = parcels.apply(
        lambda row: raw_address(row, address_columns), axis=1
    )
    records: list[dict[str, object]] = []
    for owner_name, rows in parcels.groupby("OwnerName", dropna=False, sort=False):
        county_totals = (
            rows.groupby("CountyName", dropna=False)["TotalAcres"]
            .sum()
            .sort_values(ascending=False)
        )
        confidence = max(
            rows["confidence"].unique(), key=CONFIDENCE_ORDER.__getitem__
        )
        normalized_names = rows["_norm_OwnerName"].unique()
        records.append(
            {
                "owner_name": "" if pd.isna(owner_name) else str(owner_name),
                "normalized_owner_name": join_unique(normalized_names),
                "total_acres": float(rows["TotalAcres"].sum()),
                "parcel_count": int(rows["PARCELID"].count()),
                "counties": join_unique(rows["CountyName"]),
                "county_acreage_breakdown": " | ".join(
                    f"{county}: {acres:.2f}"
                    for county, acres in county_totals.items()
                ),
                "owner_addresses": join_unique(rows["_display_address"]),
                "match_reason": join_unique(rows["match_reason"]),
                "confidence": confidence,
                "include_in_defensible_total": confidence
                in {"confirmed", "strong_candidate"},
            }
        )
    return (
        pd.DataFrame.from_records(records, columns=OWNER_OUTPUT_COLUMNS)
        .sort_values(["total_acres", "owner_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def scenario_totals(owners: pd.DataFrame) -> dict[str, float]:
    confirmed = owners.loc[owners["confidence"].eq("confirmed"), "total_acres"].sum()
    strong = owners.loc[
        owners["confidence"].isin({"confirmed", "strong_candidate"}),
        "total_acres",
    ].sum()
    plausible = owners.loc[
        owners["confidence"].ne("unrelated"), "total_acres"
    ].sum()
    return {
        "confirmed": float(confirmed),
        "confirmed_and_strong": float(strong),
        "all_plausible": float(plausible),
    }


def yn(condition: bool) -> str:
    return "YES" if condition else "NO"


def owner_detail_blocks(owners: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for row in owners.itertuples(index=False):
        lines.extend(
            [
                row.owner_name or "(blank owner name)",
                f"  Confidence: {row.confidence}",
                f"  Acres: {row.total_acres:,.2f}",
                f"  Parcels: {row.parcel_count:,}",
                f"  Counties: {row.counties or '(none)'}",
                f"  County acreage: {row.county_acreage_breakdown or '(none)'}",
                f"  Address: {row.owner_addresses or '(blank)'}",
                f"  Match basis: {row.match_reason}",
                "",
            ]
        )
    return lines


def duplicate_diagnostics(parcels: pd.DataFrame) -> dict[str, object]:
    nonblank = parcels.loc[parcels["PARCELID"].notna()].copy()
    duplicated = nonblank.loc[nonblank["PARCELID"].duplicated(keep=False)]
    duplicate_ids = int(duplicated["PARCELID"].nunique())
    extra_rows = int(len(duplicated) - duplicate_ids) if duplicate_ids else 0
    max_extra_acres = 0.0
    if duplicate_ids:
        summed = duplicated.groupby("PARCELID")["TotalAcres"].sum()
        largest = duplicated.groupby("PARCELID")["TotalAcres"].max()
        max_extra_acres = float((summed - largest).sum())
    normalized_collisions = (
        parcels.groupby("_norm_OwnerName")["OwnerName"].nunique().loc[lambda s: s > 1]
    )
    return {
        "duplicate_ids": duplicate_ids,
        "extra_rows": extra_rows,
        "max_extra_acres": max_extra_acres,
        "normalized_name_collisions": int(len(normalized_collisions)),
    }


def render_summary(
    owners: pd.DataFrame,
    parcels: pd.DataFrame,
    totals: dict[str, float],
    cutoff: float,
    source_records: int,
    tax_years: set[int],
    duplicate_info: dict[str, object],
) -> str:
    confirmed_and_strong = owners.loc[
        owners["confidence"].isin({"confirmed", "strong_candidate"})
    ]
    ambiguous = owners.loc[
        owners["confidence"].isin({"address_candidate", "weak_candidate"})
    ]
    largest_plausible = owners.loc[owners["confidence"].ne("unrelated")].head(1)
    largest_ambiguous = ambiguous.head(2)
    exact_portfolio_hits = owners.loc[
        owners["match_reason"].str.contains(
            "official-portfolio name omitted", regex=False, na=False
        )
    ]
    address_only_acres = float(
        owners.loc[
            owners["confidence"].eq("address_candidate"), "total_acres"
        ].sum()
    )
    weak_only_acres = float(
        owners.loc[owners["confidence"].eq("weak_candidate"), "total_acres"].sum()
    )
    missing_acres = int(parcels.get("_TotalAcres_missing", pd.Series(dtype=bool)).sum())

    lines = [
        "DENNIS WASHINGTON / WASHINGTON COMPANIES LANDOWNERSHIP AUDIT",
        "============================================================",
        "",
        f"Source: {RAW_PARCELS}",
        f"Source records scanned: {source_records:,}",
        f"Tax year values present: {', '.join(map(str, sorted(tax_years))) or 'not available'}",
        f"Top-20 cutoff: {cutoff:,.0f} acres",
        "Acreage basis: source TotalAcres, matching the production ranking.",
        "Scope: current cadastral surface parcels only; no historical acreage, leases,",
        "mineral-only rights, operating/permit boundaries, or non-cadastral estimates.",
        "",
        "CONFIDENCE RULES",
        "----------------",
        "confirmed: exact owner-name match to a directly supported person/company registry",
        "strong_candidate: conditional exact seed plus HQ/shared-address corroboration",
        "address_candidate: HQ or exact shared-address evidence only; not proof of ownership",
        "weak_candidate: broad/portfolio/fuzzy discovery requiring independent research",
        "unrelated: surname-only or known historical/unrelated match; excluded from totals",
        "",
        "CONFIRMED / STRONG WASHINGTON-CONTROLLED ENTITIES",
        "--------------------------------------------------",
        "",
    ]
    if confirmed_and_strong.empty:
        lines.extend(["None found.", ""])
    else:
        lines.extend(owner_detail_blocks(confirmed_and_strong))

    lines.extend(
        [
            f"CONFIRMED TOTAL: {totals['confirmed']:,.2f} acres",
            "",
            "INCLUDING STRONG CANDIDATES: "
            f"{totals['confirmed_and_strong']:,.2f} acres",
            "",
            "INCLUDING EVERY REMOTELY PLAUSIBLE CANDIDATE: "
            f"{totals['all_plausible']:,.2f} acres",
            "",
            "DISTANCE BELOW TOP-20 CUTOFF",
            "----------------------------",
            f"Confirmed only: {cutoff - totals['confirmed']:,.2f} acres",
            "Confirmed + strong candidates: "
            f"{cutoff - totals['confirmed_and_strong']:,.2f} acres",
            "Every remotely plausible candidate: "
            f"{cutoff - totals['all_plausible']:,.2f} acres",
            "",
            "SANITY CHECKS",
            "-------------",
        ]
    )
    any_over_10k = bool(
        (owners.loc[owners["confidence"].ne("unrelated"), "total_acres"] > 10_000).any()
    )
    lines.extend(
        [
            f"1. Any single plausible owner over 10,000 acres? {yn(any_over_10k)}",
            "2. Confirmed entities collectively exceed 25,000 acres? "
            f"{yn(totals['confirmed'] > 25_000)}",
            "3. Confirmed entities collectively exceed 50,000 acres? "
            f"{yn(totals['confirmed'] > 50_000)}",
            "4. Confirmed entities exceed the top-20 cutoff? "
            f"{yn(totals['confirmed'] >= cutoff)}",
            "5. Confirmed + strong candidates exceed the top-20 cutoff? "
            f"{yn(totals['confirmed_and_strong'] >= cutoff)}",
            "6. Every remotely plausible candidate exceeds the top-20 cutoff? "
            f"{yn(totals['all_plausible'] >= cutoff)}",
            "7. Address/weak candidates add: "
            f"{totals['all_plausible'] - totals['confirmed_and_strong']:,.2f} acres",
            f"   Address-only candidates add: {address_only_acres:,.2f} acres",
            f"   Other weak discovery candidates add: {weak_only_acres:,.2f} acres",
            "   Do all address-only candidates change the top-20 conclusion? "
            f"{yn(totals['confirmed_and_strong'] + address_only_acres >= cutoff)}",
            "",
        ]
    )

    if largest_plausible.empty:
        lines.append("Largest plausible entity: none")
    else:
        row = largest_plausible.iloc[0]
        lines.append(
            f"Largest plausible entity: {row['owner_name']} ({row['total_acres']:,.2f} acres)"
        )
    if largest_ambiguous.empty:
        lines.append("Largest ambiguous entities: none")
    else:
        lines.append("Largest ambiguous entities:")
        for row in largest_ambiguous.itertuples(index=False):
            lines.append(
                f"  {row.owner_name}: {row.total_acres:,.2f} acres ({row.confidence})"
            )
        two_largest_ambiguous = float(largest_ambiguous["total_acres"].sum())
        lines.append(
            "Would adding both to confirmed + strong reach the cutoff? "
            f"{yn(totals['confirmed_and_strong'] + two_largest_ambiguous >= cutoff)}"
        )
    lines.extend(
        [
            "",
            "DUPLICATION / DATA-QUALITY CHECKS",
            "---------------------------------",
            f"Candidate parcel IDs repeated in source results: {duplicate_info['duplicate_ids']:,}",
            f"Extra candidate rows among repeated IDs: {duplicate_info['extra_rows']:,}",
            "Maximum apparent acreage attributable to those extra rows: "
            f"{duplicate_info['max_extra_acres']:,.2f}",
            "Normalized owner names representing multiple original spellings: "
            f"{duplicate_info['normalized_name_collisions']:,}",
            f"Candidate rows with missing TotalAcres after loading: {missing_acres:,}",
            "Original owner spellings are aggregated separately; normalization is used for",
            "matching and does not itself duplicate parcels or acreage.",
            "",
            "OMITTED CURRENT-PORTFOLIO NAME CHECK",
            "------------------------------------",
            "Additional names searched: "
            + ", ".join(sorted(OMITTED_PORTFOLIO_REVIEW_NAMES)),
            "Matches: "
            + (
                ", ".join(exact_portfolio_hits["owner_name"].tolist())
                if not exact_portfolio_hits.empty
                else "none"
            ),
            "These names are flagged for review and are never silently consolidated.",
            "Supplied-seed variant added and disclosed by this audit: MONTANA RESOURCES LLC",
            "",
            "CONCLUSION",
            "----------",
            f"Confirmed only: {totals['confirmed']:,.2f} acres ({yn(totals['confirmed'] >= cutoff)} top 20)",
            "Confirmed + strong: "
            f"{totals['confirmed_and_strong']:,.2f} acres "
            f"({yn(totals['confirmed_and_strong'] >= cutoff)} top 20)",
            "All remotely plausible: "
            f"{totals['all_plausible']:,.2f} acres "
            f"({yn(totals['all_plausible'] >= cutoff)} top 20)",
        ]
    )
    if totals["all_plausible"] < cutoff:
        lines.append(
            "Even the deliberately generous upper-bound scenario is below the cutoff."
        )
    else:
        lines.append(
            "The generous upper bound reaches the cutoff; investigate ambiguous entities."
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    owners: pd.DataFrame,
    parcels: pd.DataFrame,
    summary: str,
    address_columns: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    owners.to_csv(output_dir / "washington-owner-candidates.csv", index=False)

    parcel_columns = [
        "PARCELID",
        *[column for column in OPTIONAL_CONTEXT_COLUMNS if column in parcels],
        "CountyName",
        "OwnerName",
        "TotalAcres",
        *address_columns,
        *[column for column in ["DbaName", "CareOfTaxp"] if column in parcels],
        "_norm_OwnerName",
        "_normalized_owner_address",
        "match_reason",
        "confidence",
    ]
    parcel_export = parcels.reindex(columns=parcel_columns).rename(
        columns={
            "_norm_OwnerName": "normalized_owner_name",
            "_normalized_owner_address": "normalized_owner_address",
        }
    )
    parcel_export = parcel_export.sort_values(
        ["confidence", "TotalAcres", "OwnerName"],
        ascending=[True, False, True],
    )
    parcel_export.to_csv(output_dir / "washington-parcels.csv", index=False)

    (output_dir / "washington-summary.txt").write_text(summary, encoding="utf-8")

    grouping_names = owners.loc[
        owners["include_in_defensible_total"], "owner_name"
    ].tolist()
    grouping = [
        {
            "owner": "WASHINGTON GROUP",
            "alternatesOwnershipNames": sorted(grouping_names),
        }
    ]
    (output_dir / "washington-grouping-suggestion.json").write_text(
        json.dumps(grouping, indent=4) + "\n", encoding="utf-8"
    )

    review = owners.loc[
        owners["confidence"].isin(
            {"strong_candidate", "address_candidate", "weak_candidate"}
        )
    ].copy()
    recommendations = {
        "strong_candidate": (
            "Verify current beneficial ownership in corporate filings/deeds before "
            "editing the production grouping."
        ),
        "address_candidate": (
            "Check Montana business filings and deeds; a shared tax address alone "
            "does not establish beneficial ownership."
        ),
        "weak_candidate": (
            "Determine whether the name variant/entity has an independent current "
            "Washington Companies connection."
        ),
    }
    review_output = pd.DataFrame(
        {
            "owner_name": review["owner_name"],
            "total_acres": review["total_acres"],
            "parcel_count": review["parcel_count"],
            "address": review["owner_addresses"],
            "reason_flagged": review["match_reason"],
            "recommended_manual_check": review["confidence"].map(recommendations),
        }
    )
    review_output.to_csv(output_dir / "washington-review-needed.csv", index=False)


def available_columns() -> list[str]:
    fields = list(pyogrio.read_info(require_file(RAW_PARCELS))["fields"])
    missing = sorted(set(REQUIRED_COLUMNS) - set(fields))
    if missing:
        raise ValueError(f"Cadastral layer is missing required columns: {missing}")
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"audit output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="source records per geometry-free read (default: 100000)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=TOP_20_CUTOFF,
        help="top-20 acreage threshold (default: 95281)",
    )
    parser.add_argument(
        "--fuzzy-cutoff",
        type=float,
        default=0.86,
        help="SequenceMatcher threshold for candidate-only fuzzy names (default: 0.86)",
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="skip optional fuzzy owner-name discovery",
    )
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if args.cutoff <= 0:
        parser.error("--cutoff must be positive")
    if not 0 <= args.fuzzy_cutoff <= 1:
        parser.error("--fuzzy-cutoff must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    fields = available_columns()
    address_columns = [column for column in ADDRESS_COLUMNS if column in fields]
    identity_columns = [column for column in IDENTITY_COLUMNS if column in fields]
    columns = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + OPTIONAL_CONTEXT_COLUMNS
            + address_columns
            + identity_columns
        )
    )
    columns = [column for column in columns if column in fields]

    print(f"Input: {RAW_PARCELS}")
    print(f"Output: {args.output_dir}")
    print(f"Acreage cutoff: {args.cutoff:,.0f}")
    print(f"Owner identity fields: {', '.join(identity_columns)}")
    print(f"Owner mailing fields: {', '.join(address_columns)}")

    confirmed_addresses, fuzzy_matches, source_records, tax_years = (
        discover_confirmed_addresses_and_fuzzy_names(
            columns,
            address_columns,
            identity_columns,
            args.chunk_size,
            args.fuzzy_cutoff,
            not args.no_fuzzy,
        )
    )
    parcels = collect_candidate_parcels(
        columns,
        address_columns,
        identity_columns,
        args.chunk_size,
        confirmed_addresses,
        fuzzy_matches,
    )
    owners = summarize_owners(parcels, address_columns)
    totals = scenario_totals(owners)
    duplicate_info = duplicate_diagnostics(parcels)
    summary = render_summary(
        owners,
        parcels,
        totals,
        args.cutoff,
        source_records,
        tax_years,
        duplicate_info,
    )
    write_outputs(args.output_dir, owners, parcels, summary, address_columns)

    print(f"Wrote {len(owners):,} owner candidates and {len(parcels):,} parcels.")
    print(f"Confirmed only: {totals['confirmed']:,.2f} acres")
    print(f"Confirmed + strong candidates: {totals['confirmed_and_strong']:,.2f} acres")
    print(f"All remotely plausible candidates: {totals['all_plausible']:,.2f} acres")
    print(f"Top-20 cutoff: {args.cutoff:,.2f} acres")
    print(
        "Conclusion: "
        + (
            "even the generous upper bound does not reach the top-20 cutoff."
            if totals["all_plausible"] < args.cutoff
            else "the generous upper bound reaches the cutoff; manual review is required."
        )
    )


if __name__ == "__main__":
    main()
