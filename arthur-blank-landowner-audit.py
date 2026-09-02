#!/usr/bin/env python3
"""Standalone Montana cadastral audit for Arthur Blank / AMB West.

The audit reads the same OWNERPARCEL layer, TotalAcres field, public-owner
exclusions, and production grouping map used by the published ranking.  It
does not edit production configuration.  Owner keywords, shared mailing
locations, fuzzy similarity, and proximity generate candidates; only a small
independently supported registry enters the confirmed acreage total.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely import union_all

from analysis_common import (
    OUTPUT_DIR,
    RAW_PARCELS,
    iter_parcel_attributes,
    owner_name_cleaner,
    public_landowners,
    require_file,
    sql_string,
)


AUDIT_GROUP = "ARTHUR BLANK / AMB WEST"
DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "blank-audit"
ANALYSIS_CRS = "EPSG:5070"
ONE_MILE_METERS = 1609.344

REQUIRED_COLUMNS = ["PARCELID", "CountyName", "OwnerName", "TotalAcres"]
ADDRESS_COLUMNS = [
    "OwnerAddre",
    "OwnerAdd_1",
    "OwnerAdd_2",
    "OwnerCity",
    "OwnerState",
    "OwnerZipCo",
]
CONTEXT_COLUMNS = [
    "TaxYear",
    "PropertyID",
    "Assessment",
    "PropType",
    "DbaName",
    "CareOfTaxp",
    "LegalDescr",
    "Township",
    "Range",
    "Section",
]

CONFIRMED_NAMES = {
    "MOUNTAIN SKY GUEST RANCH LLC",
    "MOUNTAIN SKY GUEST RANCH",
    "WEST CREEK RANCH LLC",
    "PARADISE VALLEY RANCH LLC",
    "RANCH AT DOME MOUNTAIN LLC",
    "BLANK ARTHUR M",
}

RANCH_COMPONENT = {
    "MOUNTAIN SKY GUEST RANCH LLC": "Mountain Sky Guest Ranch",
    "MOUNTAIN SKY GUEST RANCH": "Mountain Sky Guest Ranch",
    "WEST CREEK RANCH LLC": "West Creek Ranch",
    "PARADISE VALLEY RANCH LLC": "Paradise Valley Ranch",
    "RANCH AT DOME MOUNTAIN LLC": "Auster / Ranch at Dome Mountain",
    "BLANK ARTHUR M": "Auster / Ranch at Dome Mountain",
}

RELATIONSHIP_EVIDENCE = {
    "MOUNTAIN SKY GUEST RANCH LLC": (
        "Official AMB West page says Arthur Blank purchased Mountain Sky in 2001; "
        "current Montana Water Court records also jointly identify Arthur M. Blank "
        "and Mountain Sky Guest Ranch LLC."
    ),
    "MOUNTAIN SKY GUEST RANCH": (
        "Punctuation/suffix variant of Mountain Sky Guest Ranch at the identical "
        "PO Box 1219 Emigrant ranch-management address."
    ),
    "WEST CREEK RANCH LLC": (
        "Official AMB West page says Arthur Blank purchased West Creek in 2017; "
        "cadastral owner uses AMB West's shared PO Box 318 Emigrant address."
    ),
    "PARADISE VALLEY RANCH LLC": (
        "Official AMB West page says Arthur Blank purchased PVR in 2019; cadastral "
        "owner uses AMB West's shared PO Box 318 Emigrant address."
    ),
    "RANCH AT DOME MOUNTAIN LLC": (
        "Official AMB West pages identify Auster/Ranch at Dome Mountain as Blank's "
        "fourth property; cadastral owner uses the Blank organization Atlanta address."
    ),
    "BLANK ARTHUR M": (
        "Direct Arthur M. Blank cadastral ownership at the same 3223 Howell Mill Road "
        "Atlanta address as Ranch at Dome Mountain LLC."
    ),
}

EXACT_SEEDS = {
    "AMB WEST",
    "AMB WEST LLC",
    "MOUNTAIN SKY",
    "MOUNTAIN SKY GUEST RANCH",
    "MOUNTAIN SKY GUEST RANCH LLC",
    "WEST CREEK",
    "WEST CREEK RANCH",
    "WEST CREEK RANCH LLC",
    "PARADISE VALLEY RANCH",
    "PARADISE VALLEY RANCH LLC",
    "DOME MOUNTAIN",
    "THE RANCH AT DOME MOUNTAIN",
    "THE RANCH AT DOME MOUNTAIN LLC",
    "RANCH AT DOME MOUNTAIN LLC",
    "AUSTER",
    "ARTHUR BLANK",
    "ARTHUR M BLANK",
    "BLANK ARTHUR M",
}

SPECIFIC_KEYWORDS = [
    "AMB WEST",
    "MOUNTAIN SKY",
    "WEST CREEK",
    "PARADISE VALLEY RANCH",
    "DOME MOUNTAIN",
    "AUSTER",
    "ARTHUR BLANK",
    "ARTHUR M BLANK",
]
BROAD_FRAGMENTS = ["BLANK", "AMB", "MOUNTAIN SKY", "WEST CREEK", "PARADISE", "DOME", "AUSTER"]

KNOWN_ADDRESS_PATTERNS = [
    (r"PO BOX 318.*EMIGRANT.*MT.*59027", "PO BOX 318 EMIGRANT MT 59027"),
    (r"PO BOX 1219.*EMIGRANT.*MT.*59027", "PO BOX 1219 EMIGRANT MT 59027"),
    (r"3223 HOWELL MILL RD.*ATLANTA.*GA.*30327", "3223 HOWELL MILL RD NW ATLANTA GA 30327"),
]
KNOWN_ADDRESS_KEYS = {key for _, key in KNOWN_ADDRESS_PATTERNS}

# These are intentionally explicit so large false positives do not reappear as
# unresolved candidates on future runs.  They have distinct current cadastral
# owners/addresses, no official AMB portfolio link, and (for the ranches) merely
# neighbor one of the four official properties.
KNOWN_REJECTIONS = {
    "POINT OF ROCKS RANCH LLC": "independent adjoining Park County ranch; PO Box 79 Emigrant, no AMB evidence",
    "O HAIR RANCH COMPANY": "independent adjoining Park County ranch with separate Livingston address",
    "EIGHTMILE CREEK PV RANCH LLC": "independent adjoining Park County ranch with Dallas/Rockwall tax address",
    "DOME MOUNTAIN RANCH LLC": "residual/predecessor-name owner at Sinclair-related Maryland address; not the current AMB cadastral entity",
    "AMBER RANCH LLC": "AMB substring false positive; separate North Carolina address and unrelated Park/Gallatin ranch",
    "AMB ROGER MT DISCRETIONARY TRUST": "initials/substring false positive in Chouteau County with Pensacola address",
    "PARADISE VALLEY PROPERTIES LLC": "generic geographic-name owner with unrelated Oregon address",
    "BIG CREEK RANCH CO LLC": "adjoining ranch with separate Florida address; proximity alone is not ownership evidence",
}

FUZZY_REFERENCES = sorted(EXACT_SEEDS)
FUZZY_GATE = {"AMB", "BLANK", "MOUNTAIN", "SKY", "WEST", "CREEK", "PARADISE", "DOME", "AUSTER"}

CLASSIFICATION_ORDER = {
    "rejected_unrelated": 0,
    "plausible_candidate": 1,
    "strong_candidate": 2,
    "confirmed": 3,
}
CLASSIFICATION_LABELS = {
    "confirmed": "Confirmed",
    "strong_candidate": "Strong candidate",
    "plausible_candidate": "Plausible candidate",
    "rejected_unrelated": "Rejected / unrelated",
}

CANDIDATE_COLUMNS = [
    "owner_name",
    "normalized_owner_name",
    "acres",
    "parcel_count",
    "counties",
    "mailing_addresses",
    "discovery_method",
    "relationship_evidence",
    "classification",
    "confidence",
    "notes",
]


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().strip().replace("&", " AND ")
    text = re.sub(r"\bL\s*\.\s*L\s*\.\s*C\s*\.?", "LLC", text)
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    text = re.sub(r"\bLIMITED LIABILITY COMPANY\b", "LLC", text)
    text = re.sub(r"\bCORPORATION\b", "CORP", text)
    text = re.sub(r"\bINCORPORATED\b", "INC", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series, *, address: bool = False) -> pd.Series:
    result = (
        series.astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
        .str.replace("&", " AND ", regex=False)
        .str.replace(r"\bL\s*\.\s*L\s*\.\s*C\s*\.?", "LLC", regex=True)
        .str.replace(r"[^A-Z0-9]+", " ", regex=True)
        .str.replace(r"\bLIMITED LIABILITY COMPANY\b", "LLC", regex=True)
        .str.replace(r"\bCORPORATION\b", "CORP", regex=True)
        .str.replace(r"\bINCORPORATED\b", "INC", regex=True)
    )
    if address:
        result = (
            result.str.replace(r"\bP O BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bPOST OFFICE BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bROAD\b", "RD", regex=True)
            .str.replace(r"\bDRIVE\b", "DR", regex=True)
            .str.replace(r"\bAVENUE\b", "AVE", regex=True)
            .str.replace(r"\bSUITE\b", "STE", regex=True)
        )
    return result.str.replace(r"\s+", " ", regex=True).str.strip()


def join_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns:
        return pd.Series("", index=frame.index, dtype="string")
    result = frame[columns[0]].astype("string").fillna("")
    for column in columns[1:]:
        result = result.str.cat(frame[column].astype("string").fillna(""), sep=" ")
    return result.str.replace(r"\s+", " ", regex=True).str.strip()


def canonical_address_series(addresses: pd.Series) -> pd.Series:
    result = normalize_series(addresses, address=True)
    for pattern, key in KNOWN_ADDRESS_PATTERNS:
        result = result.mask(result.str.contains(pattern, regex=True, na=False), key)
    return result


def prepare_chunk(parcels: pd.DataFrame, address_columns: list[str]) -> pd.DataFrame:
    parcels = parcels.copy()
    parcels["_norm_owner"] = normalize_series(parcels["OwnerName"])
    raw_address = join_columns(parcels, address_columns)
    parcels["_raw_address"] = raw_address
    parcels["_address_key"] = canonical_address_series(raw_address)
    for column in ["DbaName", "CareOfTaxp"]:
        parcels[f"_norm_{column}"] = (
            normalize_series(parcels[column]) if column in parcels else ""
        )
    parcels["TotalAcres"] = pd.to_numeric(parcels["TotalAcres"], errors="coerce")
    return parcels


def specific_hits(text: str) -> list[str]:
    return [term for term in SPECIFIC_KEYWORDS if term in text]


def broad_hits(text: str) -> list[str]:
    return [term for term in BROAD_FRAGMENTS if term in text]


def fuzzy_matches(names: Iterable[str], cutoff: float) -> dict[str, tuple[str, float]]:
    matches: dict[str, tuple[str, float]] = {}
    for name in names:
        if not name or name in EXACT_SEEDS:
            continue
        if not set(name.split()).intersection(FUZZY_GATE):
            continue
        best_ref = ""
        best_ratio = 0.0
        for reference in FUZZY_REFERENCES:
            ratio = difflib.SequenceMatcher(None, name, reference).ratio()
            if ratio > best_ratio:
                best_ref, best_ratio = reference, ratio
        if best_ratio >= cutoff:
            matches[name] = (best_ref, best_ratio)
    return matches


def first_pass(
    columns: list[str], address_columns: list[str], chunk_size: int
) -> tuple[
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, float],
    set[int],
    int,
]:
    """Discover name/address candidates and build the production-comparable rank."""
    methods: dict[str, set[str]] = defaultdict(set)
    normalized_to_raw: dict[str, set[str]] = defaultdict(set)
    ranking_acres: dict[str, float] = defaultdict(float)
    cleaner = owner_name_cleaner()
    public = public_landowners()
    tax_years: set[int] = set()
    processed = 0

    print("Pass 1/2: owner/address discovery and statewide ranking...")
    for chunk in iter_parcel_attributes(columns, chunk_size):
        parcels = prepare_chunk(chunk, address_columns)
        processed += len(parcels)
        if "TaxYear" in parcels:
            tax_years.update(int(value) for value in parcels["TaxYear"].dropna().unique())

        valid = parcels.loc[parcels["OwnerName"].notna()].copy()
        for raw, normalized in valid[["OwnerName", "_norm_owner"]].drop_duplicates().itertuples(index=False):
            raw = str(raw)
            normalized_to_raw[normalized].add(raw)
            if normalized in EXACT_SEEDS:
                methods[raw].add("exact_normalized_seed")
            hits = specific_hits(normalized)
            if hits:
                methods[raw].add("specific_owner_keyword:" + "|".join(hits))
            broad = broad_hits(normalized)
            if broad:
                methods[raw].add("broad_owner_fragment:" + "|".join(broad))

        identity = valid["_norm_DbaName"].str.cat(valid["_norm_CareOfTaxp"], sep=" ")
        identity_mask = pd.Series(False, index=valid.index)
        for term in SPECIFIC_KEYWORDS:
            identity_mask |= identity.str.contains(term, regex=False, na=False)
        for raw in valid.loc[identity_mask, "OwnerName"].dropna().astype(str).unique():
            methods[raw].add("dba_or_careof_keyword")

        for raw in valid.loc[
            valid["_address_key"].isin(KNOWN_ADDRESS_KEYS), "OwnerName"
        ].dropna().astype(str).unique():
            methods[raw].add("exact_normalized_confirmed_address")

        valid["_production_group"] = valid["OwnerName"].replace(cleaner)
        valid = valid.loc[~valid["_production_group"].isin(public)]
        grouped = valid.groupby("_production_group")["TotalAcres"].sum(min_count=1)
        for name, acres in grouped.items():
            ranking_acres[str(name)] += float(0 if pd.isna(acres) else acres)
        print(f"  scanned {processed:,} records", end="\r", flush=True)
    print(f"  scanned {processed:,} records")
    return methods, normalized_to_raw, dict(ranking_acres), tax_years, processed


def geometry_discovery(
    methods: dict[str, set[str]],
) -> tuple[gpd.GeoDataFrame, dict[str, dict[str, float]]]:
    """Find owner names within one mile of confirmed parcels and direct contacts."""
    raw_names = sorted(CONFIRMED_NAMES)
    where = '"OwnerName" IN (' + ", ".join(map(sql_string, raw_names)) + ")"
    confirmed = gpd.read_file(
        require_file(RAW_PARCELS),
        where=where,
        columns=[
            "PARCELID",
            "PropertyID",
            "CountyName",
            "OwnerName",
            "TotalAcres",
            *ADDRESS_COLUMNS,
            "DbaName",
            "CareOfTaxp",
            "LegalDescr",
            "Township",
            "Range",
            "Section",
        ],
        engine="pyogrio",
    )
    confirmed["normalized_owner_name"] = confirmed["OwnerName"].map(normalize_text)
    confirmed["ranch_component"] = confirmed["normalized_owner_name"].map(RANCH_COMPONENT)
    confirmed["matched_grouped_entity"] = AUDIT_GROUP
    confirmed["match_reason"] = confirmed["normalized_owner_name"].map(
        RELATIONSHIP_EVIDENCE
    )

    source_bounds = confirmed.total_bounds
    source_bbox = (
        source_bounds[0] - 2_500,
        source_bounds[1] - 2_500,
        source_bounds[2] + 2_500,
        source_bounds[3] + 2_500,
    )
    nearby = gpd.read_file(
        require_file(RAW_PARCELS),
        bbox=source_bbox,
        columns=["PARCELID", "OwnerName", "TotalAcres", "CountyName"],
        engine="pyogrio",
    ).to_crs(ANALYSIS_CRS)
    projected = confirmed.to_crs(ANALYSIS_CRS)
    holding_union = union_all(projected.geometry)
    nearby = nearby.loc[
        nearby.geometry.intersects(holding_union.buffer(ONE_MILE_METERS))
        & ~nearby["OwnerName"].isin(CONFIRMED_NAMES)
    ].copy()
    nearby["_distance_m"] = nearby.geometry.distance(holding_union)
    nearby["_adjoining"] = nearby.geometry.intersects(holding_union.buffer(10.0))

    geo_stats: dict[str, dict[str, float]] = {}
    for raw_name, rows in nearby.groupby("OwnerName", dropna=True):
        raw_name = str(raw_name)
        methods[raw_name].add("geographic_within_one_mile")
        if rows["_adjoining"].any():
            methods[raw_name].add("geographic_adjoining_or_within_10m")
        geo_stats[raw_name] = {
            "nearby_parcels": int(len(rows)),
            "adjoining_parcels": int(rows["_adjoining"].sum()),
            "minimum_distance_meters": float(rows["_distance_m"].min()),
            "nearby_parcel_acres": float(rows["TotalAcres"].fillna(0).sum()),
        }
    return confirmed, geo_stats


def collect_candidate_rows(
    columns: list[str],
    address_columns: list[str],
    chunk_size: int,
    candidate_names: set[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    processed = retained = 0
    print("Pass 2/2: collecting all parcels for discovered owners...")
    for chunk in iter_parcel_attributes(columns, chunk_size):
        processed += len(chunk)
        selected = chunk.loc[chunk["OwnerName"].isin(candidate_names)].copy()
        if not selected.empty:
            selected = prepare_chunk(selected, address_columns)
            frames.append(selected)
            retained += len(selected)
        print(
            f"  scanned {processed:,}; retained {retained:,} candidate rows",
            end="\r",
            flush=True,
        )
    print(f"  scanned {processed:,}; retained {retained:,} candidate rows")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def join_unique(values: Iterable[object]) -> str:
    return " | ".join(
        sorted(
            {
                str(value).strip()
                for value in values
                if value is not None and not pd.isna(value) and str(value).strip()
            }
        )
    )


def classify_owner(
    owner_name: str,
    normalized: str,
    methods: set[str],
    geo: dict[str, float] | None,
) -> tuple[str, str, str, str]:
    if normalized in CONFIRMED_NAMES:
        return (
            "confirmed",
            "high",
            RELATIONSHIP_EVIDENCE[normalized],
            "Current owner-name, address and authoritative AMB/agency evidence agree.",
        )
    if normalized in KNOWN_REJECTIONS:
        return (
            "rejected_unrelated",
            "rejected",
            "No current Arthur Blank / AMB West ownership evidence.",
            KNOWN_REJECTIONS[normalized],
        )
    if "exact_normalized_confirmed_address" in methods:
        return (
            "plausible_candidate",
            "low",
            "Uses an exact normalized mailing location associated with a confirmed AMB entity.",
            "Obtain current entity managers/members or a deed chain before inclusion; address sharing alone is insufficient.",
        )
    if geo and geo.get("adjoining_parcels", 0):
        return (
            "rejected_unrelated",
            "rejected",
            "Cadastral geometry adjoins a confirmed ranch, but no name/address/corporate evidence links it to AMB West.",
            "Neighboring ownership is not beneficial ownership; distinct owner remains excluded.",
        )
    if any(method.startswith("fuzzy_name") for method in methods):
        return (
            "rejected_unrelated",
            "rejected",
            "Fuzzy name similarity only.",
            "Candidate discovery signal lacks address, geography and ownership corroboration.",
        )
    return (
        "rejected_unrelated",
        "rejected",
        "Keyword/fragment, identity-field, or proximity-only discovery signal.",
        "No authoritative current relationship to Arthur Blank / AMB West; excluded.",
    )


def summarize_candidates(
    rows: pd.DataFrame,
    methods: dict[str, set[str]],
    geo_stats: dict[str, dict[str, float]],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for owner_name, owner_rows in rows.groupby("OwnerName", dropna=False, sort=False):
        if pd.isna(owner_name):
            continue
        owner_name = str(owner_name)
        normalized_values = sorted(set(owner_rows["_norm_owner"]))
        normalized = normalized_values[0]
        classification, confidence, evidence, notes = classify_owner(
            owner_name, normalized, methods[owner_name], geo_stats.get(owner_name)
        )
        geo = geo_stats.get(owner_name, {})
        method_list = set(methods[owner_name])
        if geo:
            method_list.add(
                "geography_stats:"
                f"near={int(geo['nearby_parcels'])};"
                f"adjoining={int(geo['adjoining_parcels'])};"
                f"min_m={geo['minimum_distance_meters']:.1f}"
            )
        records.append(
            {
                "owner_name": owner_name,
                "normalized_owner_name": " | ".join(normalized_values),
                "acres": float(owner_rows["TotalAcres"].fillna(0).sum()),
                "parcel_count": int(owner_rows["PARCELID"].count()),
                "counties": join_unique(owner_rows["CountyName"]),
                "mailing_addresses": join_unique(owner_rows["_raw_address"]),
                "discovery_method": "; ".join(sorted(method_list)),
                "relationship_evidence": evidence,
                "classification": classification,
                "confidence": confidence,
                "notes": notes,
            }
        )
    return (
        pd.DataFrame(records, columns=CANDIDATE_COLUMNS)
        .sort_values(
            ["classification", "acres", "owner_name"],
            key=lambda s: s.map(CLASSIFICATION_ORDER) if s.name == "classification" else s,
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )


def scenario_rows(candidates: pd.DataFrame, scenario: str) -> pd.DataFrame:
    allowed = {
        "confirmed": {"confirmed"},
        "strong": {"confirmed", "strong_candidate"},
        "plausible": {"confirmed", "strong_candidate", "plausible_candidate"},
    }[scenario]
    return candidates.loc[candidates["classification"].isin(allowed)]


def scenario_total(candidates: pd.DataFrame, scenario: str) -> float:
    return float(scenario_rows(candidates, scenario)["acres"].sum())


def calculate_rank(
    ranking_acres: dict[str, float], candidates: pd.DataFrame, scenario: str
) -> int | None:
    included = scenario_rows(candidates, scenario)
    total = float(included["acres"].sum())
    if total <= 0:
        return None
    adjusted = dict(ranking_acres)
    cleaner = owner_name_cleaner()
    for row in included.itertuples(index=False):
        production_name = cleaner.get(row.owner_name, row.owner_name)
        adjusted[production_name] = adjusted.get(production_name, 0.0) - float(row.acres)
        if adjusted[production_name] <= 1e-6:
            adjusted.pop(production_name, None)
    adjusted[AUDIT_GROUP] = adjusted.get(AUDIT_GROUP, 0.0) + total
    grouped_total = adjusted[AUDIT_GROUP]
    return 1 + sum(
        1
        for name, acres in adjusted.items()
        if name != AUDIT_GROUP and acres > grouped_total + 1e-6
    )


def qa_diagnostics(confirmed: gpd.GeoDataFrame) -> dict[str, object]:
    duplicate_ids = confirmed.loc[
        confirmed["PARCELID"].notna()
        & confirmed["PARCELID"].duplicated(keep=False),
        "PARCELID",
    ]
    geometry_wkb = confirmed.geometry.to_wkb()
    return {
        "duplicate_parcel_ids": int(duplicate_ids.nunique()),
        "duplicate_parcel_rows": int(len(duplicate_ids)),
        "duplicate_geometries": int(geometry_wkb.duplicated().sum()),
        "null_acreage": int(confirmed["TotalAcres"].isna().sum()),
        "zero_acreage": int(confirmed["TotalAcres"].eq(0).sum()),
        "negative_acreage": int(confirmed["TotalAcres"].lt(0).sum()),
        "null_geometry": int(confirmed.geometry.isna().sum()),
        "empty_geometry": int(confirmed.geometry.is_empty.sum()),
        "invalid_geometry": int((~confirmed.geometry.is_valid).sum()),
    }


def format_rank(rank: int | None) -> str:
    return "n/a" if rank is None else f"#{rank:,}"


def markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def render_report(
    candidates: pd.DataFrame,
    confirmed: gpd.GeoDataFrame,
    geo_stats: dict[str, dict[str, float]],
    ranking_acres: dict[str, float],
    source_records: int,
    tax_years: set[int],
    qa: dict[str, object],
) -> str:
    totals = {scenario: scenario_total(candidates, scenario) for scenario in ["confirmed", "strong", "plausible"]}
    ranks = {scenario: calculate_rank(ranking_acres, candidates, scenario) for scenario in totals}
    confirmed_candidates = candidates.loc[candidates["classification"].eq("confirmed")]
    confirmed_acres = float(confirmed["TotalAcres"].sum())
    top20 = sorted(ranking_acres.values(), reverse=True)[19]

    lines = [
        "# Arthur Blank / AMB West Montana landownership audit",
        "",
        "## Bottom line",
        "",
        f"The 2026 Montana cadastral data ties **{confirmed_acres:,.2f} current fee-simple surface acres** "
        f"across **{len(confirmed):,} parcels**, all in **Park County**, to Arthur Blank / AMB West. "
        f"That produces an estimated statewide private-landowner rank of **{format_rank(ranks['confirmed'])}** "
        "under the project's existing grouping and public-owner methodology.",
        "",
        f"This is about {33_000 - confirmed_acres:,.0f} acres below the rough 33,000-acre portfolio "
        "reference. The audit does not fill that gap with leases, operational terrain, conservation "
        "easements, or neighboring ranches. The current top-20 cutoff is "
        f"{top20:,.2f} acres, so the result does not approach the published top 20.",
        "",
        "## Scenario totals",
        "",
    ]
    lines.extend(
        markdown_table(
            ["Scenario", "Acres", "Estimated statewide rank"],
            [
                ["A — confirmed only (primary)", f"{totals['confirmed']:,.2f}", format_rank(ranks["confirmed"])],
                ["B — confirmed + strong", f"{totals['strong']:,.2f}", format_rank(ranks["strong"])],
                ["C — broadest plausible", f"{totals['plausible']:,.2f}", format_rank(ranks["plausible"])],
            ],
        )
    )
    lines.extend(
        [
            "",
            "No non-confirmed entity survived the evidence review as strong or plausible, so the "
            "three scenario totals are identical. Address, fuzzy, and proximity-only matches remain "
            "visible in `blank-owner-candidates.csv` as rejected discoveries.",
            "",
            "## Confirmed component entities",
            "",
        ]
    )
    entity_rows = []
    for row in confirmed_candidates.itertuples(index=False):
        entity_rows.append(
            [
                row.owner_name.replace("|", "\\|"),
                f"{row.acres:,.2f}",
                f"{row.parcel_count:,}",
                row.counties,
                row.relationship_evidence.replace("|", "\\|"),
            ]
        )
    lines.extend(
        markdown_table(
            ["Cadastral owner", "Acres", "Parcels", "Counties", "Relationship evidence"],
            entity_rows,
        )
    )

    lines.extend(["", "### Ranch-level reconciliation", ""])
    component_totals = (
        confirmed.groupby("ranch_component").agg(
            acres=("TotalAcres", "sum"), parcels=("PARCELID", "count")
        )
    )
    published = {
        "Mountain Sky Guest Ranch": 10_767.0,
        "West Creek Ranch": 6_700.0,
        "Paradise Valley Ranch": 9_600.0,
        "Auster / Ranch at Dome Mountain": 6_278.0,
    }
    component_rows = []
    for component in published:
        cadastral = float(component_totals.loc[component, "acres"])
        component_rows.append(
            [
                component,
                f"{cadastral:,.2f}",
                f"{int(component_totals.loc[component, 'parcels']):,}",
                f"{published[component]:,.0f}",
                f"{cadastral - published[component]:+,.2f}",
            ]
        )
    lines.extend(
        markdown_table(
            ["Property", "Cadastral fee-simple acres", "Parcels", "Public footprint QA", "Difference"],
            component_rows,
        )
    )

    uncertain = candidates.loc[
        candidates["classification"].isin({"strong_candidate", "plausible_candidate"})
    ]
    lines.extend(["", "## Strong and plausible candidates", ""])
    if uncertain.empty:
        lines.append(
            "None. No shared-address, name-similarity, or adjoining entity had enough independent "
            "evidence to remain in an acreage scenario. Promotion would require a current deed, "
            "Secretary of State manager/member record, or authoritative AMB statement establishing control."
        )
    else:
        lines.extend(
            markdown_table(
                ["Owner", "Acres", "Class", "Needed evidence"],
                [
                    [row.owner_name, f"{row.acres:,.2f}", row.classification, row.notes]
                    for row in uncertain.itertuples(index=False)
                ],
            )
        )

    lines.extend(["", "## Rejected candidates", ""])
    rejected = candidates.loc[candidates["classification"].eq("rejected_unrelated")]
    material_names = set(KNOWN_REJECTIONS)
    material = rejected.loc[
        rejected["normalized_owner_name"].isin(material_names)
        | rejected["acres"].gt(5_000)
    ].head(25)
    lines.append(
        f"The broad search investigated {len(rejected):,} distinct rejected owner names. The most "
        "material or misleading results are below; the CSV preserves the complete list."
    )
    lines.append("")
    lines.extend(
        markdown_table(
            ["Owner", "Statewide acres", "Why rejected"],
            [
                [row.owner_name.replace("|", "\\|"), f"{row.acres:,.2f}", row.notes.replace("|", "\\|")]
                for row in material.itertuples(index=False)
            ],
        )
    )

    lines.extend(
        [
            "",
            "`POINT OF ROCKS RANCH LLC`, `O'HAIR RANCH COMPANY`, and "
            "`EIGHTMILE CREEK PV RANCH LLC` contain parcels adjoining or within one mile of the "
            "confirmed holdings, but each has a different owner name and mailing address and none "
            "appears among AMB West's official four-property portfolio. Their acreage is excluded.",
            "",
            "`DOME MOUNTAIN RANCH LLC` is especially misleading: it retains 35.30 cadastral acres "
            "under a Maryland address whose cadastral line says Sinclair Building, while Blank's current acquired acreage is "
            "titled to `RANCH AT DOME MOUNTAIN LLC` and directly to `BLANK ARTHUR M` at the shared "
            "Atlanta address. The predecessor-name parcel is not assigned to Blank.",
            "",
            "## County breakdown",
            "",
        ]
    )
    county = confirmed.groupby("CountyName")["TotalAcres"].agg(["sum", "count"])
    lines.extend(
        markdown_table(
            ["County", "Confirmed acres", "Parcels", "Share"],
            [
                [name, f"{row['sum']:,.2f}", f"{int(row['count']):,}", f"{row['sum'] / confirmed_acres:.1%}"]
                for name, row in county.iterrows()
            ],
        )
    )

    lines.extend(["", "## Largest parcels and concentrations", ""])
    largest = confirmed.sort_values("TotalAcres", ascending=False).head(15)
    lines.extend(
        markdown_table(
            ["Parcel ID", "Owner", "Ranch", "Acres", "Township/Range/Section"],
            [
                [
                    row.PARCELID,
                    row.OwnerName,
                    row.ranch_component,
                    f"{row.TotalAcres:,.2f}",
                    f"{row.Township}/{row.Range}/{row.Section}",
                ]
                for row in largest.itertuples(index=False)
            ],
        )
    )
    lines.extend(
        [
            "",
            "All confirmed acreage clusters in Park County's Paradise Valley. Geometry was used "
            "only to validate concentration and generate adjoining-owner candidates; acreage is "
            "always the cadastral `TotalAcres` value.",
            "",
            "## Published acreage comparison",
            "",
            "The public figures describe properties or operational footprints, not necessarily "
            "tax-title acreage. Mountain Sky's own guest-ranch site describes more than 17,000 "
            "*operational* acres, while AMB West lists 10,767 acres and the cadastral shows 7,860.77. "
            "AMB West separately states that its land program uses leased pasture. The audit does "
            "not infer ownership from either operational figure.",
            "",
            "The Dome acquisition announcement is more explicit: approximately 6,300 acres were "
            "described as **deeded and leased land**. The current cadastral fee-simple total is "
            "5,328.30 acres, so leased acreage is a documented likely contributor to that gap. "
            "Conservation easements at West Creek and Paradise Valley Ranch restrict land use but "
            "do not add acres and are not counted separately.",
            "",
            "## Specific findings",
            "",
            "- No current cadastral parcel is owned under the exact name `AMB WEST` or `AMB WEST LLC`. "
            "AMB West appears in current water-right/operational records, but those are not fee-simple parcels.",
            "- The only direct personal owner record is `BLANK ARTHUR M`, an 8.475-acre Park County "
            "parcel sharing the Dome Mountain entity's Atlanta mailing address.",
            "- PO Box 318 expands only to Paradise Valley Ranch and West Creek Ranch. PO Box 1219 "
            "expands only to the two Mountain Sky spelling/suffix variants. The Atlanta address "
            "expands only to Ranch at Dome Mountain and Arthur M. Blank.",
            "- No opaque address-linked LLC was found. The primary false-negative risk therefore "
            "comes from land not represented as fee-simple cadastral ownership—not an undiscovered "
            "large AMB-address LLC.",
            "",
            "## QA checks",
            "",
        ]
    )
    lines.extend(
        markdown_table(
            ["Check", "Result"],
            [
                ["Duplicate confirmed parcel IDs", qa["duplicate_parcel_ids"]],
                ["Duplicate confirmed source rows", qa["duplicate_parcel_rows"]],
                ["Duplicate confirmed geometries", qa["duplicate_geometries"]],
                ["Null / zero / negative acreage", f"{qa['null_acreage']} / {qa['zero_acreage']} / {qa['negative_acreage']}"],
                ["Null / empty / invalid geometry", f"{qa['null_geometry']} / {qa['empty_geometry']} / {qa['invalid_geometry']}"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Methodological caveats",
            "",
            f"- Source: `{RAW_PARCELS}`; {source_records:,} records scanned; TaxYear values "
            f"{', '.join(map(str, sorted(tax_years)))}.",
            "- Owner-name normalization changes case, punctuation, whitespace, ampersands and common "
            "LLC formatting for matching only. Raw cadastral names remain in the exports.",
            "- Shared addresses, fuzzy similarity and geometry never establish ownership on their own.",
            "- The ranking applies the existing production owner mappings and public-owner exclusions "
            "to the complete cadastral owner aggregation, then substitutes each audit scenario.",
            "- Cadastral omissions, timing lags, exempt/hidden records, or title held through an entity "
            "with no discoverable name/address connection remain possible.",
            "- `TotalAcres` is a tax-record field and may differ from GIS area. Geometry is not used "
            "to replace it.",
            "",
            "## Evidence sources",
            "",
            "- [AMB West: holding company and four current ranches](https://www.ambwest.com/about-amb-west)",
            "- [Mountain Sky Guest Ranch](https://www.ambwest.com/mountain-sky-guest-ranch)",
            "- [West Creek Ranch](https://www.ambwest.com/west-creek-ranch)",
            "- [Paradise Valley Ranch and Dome acquisition](https://www.ambwest.com/paradise-valley-ranch)",
            "- [Auster at Dome Mountain](https://www.ambwest.com/dome-project)",
            "- [2021 Dome acquisition: deeded and leased land](https://hallhall.com/resources/amb-west-holding-company-acquires-dome-mountain-ranch/)",
            "- [Montana Water Court: Arthur M. Blank and Mountain Sky Guest Ranch LLC](https://courts.mt.gov/external/Water/orders/2022/oct/MTWC_43B-0632-R-2022_10112022_kl.pdf)",
            "- [AMB West Land + Livestock: leased pasture](https://www.ambwest.com/land-livestock)",
            "",
            "## Reporting conclusion",
            "",
            f"**Arthur Blank / AMB West: {confirmed_acres:,.2f} confirmed current fee-simple acres, "
            f"{confirmed_acres:,.2f} broadest plausible acres, approximate statewide rank "
            f"{format_rank(ranks['confirmed'])}. This exploratory result would not alter the "
            "published top 20. No production grouping file was modified.**",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    candidates: pd.DataFrame,
    confirmed: gpd.GeoDataFrame,
    report: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_export = candidates.copy()
    candidate_export["acres"] = candidate_export["acres"].round(3)
    candidate_export["classification"] = candidate_export["classification"].map(
        CLASSIFICATION_LABELS
    )
    candidate_export.to_csv(output_dir / "blank-owner-candidates.csv", index=False)

    parcel_columns = [
        "PARCELID",
        "PropertyID",
        "OwnerName",
        "TotalAcres",
        "CountyName",
        *ADDRESS_COLUMNS,
        "DbaName",
        "CareOfTaxp",
        "LegalDescr",
        "Township",
        "Range",
        "Section",
        "matched_grouped_entity",
        "ranch_component",
        "match_reason",
    ]
    confirmed.drop(columns="geometry").reindex(columns=parcel_columns).sort_values(
        ["ranch_component", "TotalAcres"], ascending=[True, False]
    ).to_csv(output_dir / "blank-parcels.csv", index=False)
    confirmed.to_crs(4326).to_file(
        output_dir / "blank-confirmed-parcels.geojson", driver="GeoJSON"
    )
    (output_dir / "blank-audit.md").write_text(report, encoding="utf-8")


def available_columns() -> list[str]:
    fields = list(pyogrio.read_info(require_file(RAW_PARCELS))["fields"])
    missing = sorted(set(REQUIRED_COLUMNS) - set(fields))
    if missing:
        raise ValueError(f"Cadastral layer is missing required columns: {missing}")
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--fuzzy-cutoff", type=float, default=0.86)
    parser.add_argument("--no-fuzzy", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
    if not 0 <= args.fuzzy_cutoff <= 1:
        parser.error("--fuzzy-cutoff must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    fields = available_columns()
    address_columns = [column for column in ADDRESS_COLUMNS if column in fields]
    columns = list(
        dict.fromkeys(
            REQUIRED_COLUMNS
            + address_columns
            + [column for column in CONTEXT_COLUMNS if column in fields]
        )
    )
    print(f"Input: {RAW_PARCELS}")
    print(f"Output: {args.output_dir}")
    methods, normalized_to_raw, ranking_acres, tax_years, records = first_pass(
        columns, address_columns, args.chunk_size
    )
    if not args.no_fuzzy:
        for normalized, (reference, ratio) in fuzzy_matches(
            normalized_to_raw, args.fuzzy_cutoff
        ).items():
            for raw in normalized_to_raw[normalized]:
                methods[raw].add(f"fuzzy_name:{reference}:{ratio:.3f}")

    confirmed, geo_stats = geometry_discovery(methods)
    candidate_rows = collect_candidate_rows(
        columns, address_columns, args.chunk_size, set(methods)
    )
    candidates = summarize_candidates(candidate_rows, methods, geo_stats)
    qa = qa_diagnostics(confirmed)
    report = render_report(
        candidates, confirmed, geo_stats, ranking_acres, records, tax_years, qa
    )
    write_outputs(args.output_dir, candidates, confirmed, report)

    confirmed_total = scenario_total(candidates, "confirmed")
    strong_total = scenario_total(candidates, "strong")
    plausible_total = scenario_total(candidates, "plausible")
    print(f"Wrote {len(candidates):,} investigated owners and {len(confirmed):,} confirmed parcels.")
    print(f"Confirmed only: {confirmed_total:,.2f} acres")
    print(f"Confirmed + strong: {strong_total:,.2f} acres")
    print(f"Broadest plausible: {plausible_total:,.2f} acres")
    print(f"Estimated rank: {format_rank(calculate_rank(ranking_acres, candidates, 'confirmed'))}")


if __name__ == "__main__":
    main()
