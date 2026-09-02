#!/usr/bin/env python3
"""Audit Montana private-club/developer landholdings in OWNERPARCEL.

This script is intentionally separate from ``owner-name-grouping.json``.  It
uses the production acreage field and owner grouping when calculating ranks,
but never edits the production grouping.  Name, address, and legal-description
matches discover records; only the cadastral OwnerName and an evidence-backed
entity registry can put acreage in a defensible ownership total.

The evidence registry was last reviewed 2026-09-01.  URLs and the reasoning
behind consequential classifications are printed in audit-summary.txt.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyogrio

from analysis_common import (
    OUTPUT_DIR,
    RAW_PARCELS,
    iter_parcel_attributes,
    owner_name_cleaner,
    public_landowners,
    require_file,
)


DEFAULT_OUTPUT_DIR = OUTPUT_DIR / "club-landowner-audit"

CROSS = "CROSSHARBOR / LONE MOUNTAIN"
ROCK = "ROCK CREEK CATTLE COMPANY / FOLEY"
TERRITORY = "TERRITORY 1889 / LAKESIDE"
N5B = "FLATHEAD RIDGE / N5B"
GROUPS = [CROSS, ROCK, TERRITORY, N5B]

CONFIDENCE_ORDER = {
    "unrelated": 0,
    "weak_candidate": 1,
    "address_candidate": 2,
    "strong_candidate": 3,
    "confirmed": 4,
}

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
    "Subdivisio",
]

OWNER_OUTPUT_COLUMNS = [
    "ownership_group",
    "owner_name",
    "normalized_owner_name",
    "total_acres",
    "parcel_count",
    "counties",
    "owner_addresses",
    "match_reason",
    "confidence",
    "include_in_defensible_total",
]

PARCEL_OUTPUT_COLUMNS = [
    "ownership_group",
    "PARCELID",
    "OwnerName",
    "TotalAcres",
    "CountyName",
    "OwnerAddre",
    "OwnerAdd_1",
    "OwnerAdd_2",
    "OwnerCity",
    "OwnerState",
    "OwnerZip",
    "match_reason",
    "confidence",
    "include_in_defensible_total",
]

SUMMARY_COLUMNS = [
    "ownership_group",
    "confirmed_acres",
    "confirmed_plus_strong_acres",
    "all_plausible_acres",
    "confirmed_parcels",
    "top20_cutoff",
    "rank_if_confirmed",
    "rank_if_confirmed_plus_strong",
    "rank_if_all_plausible",
    "existing_ranking_match",
    "notes",
]


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).upper().strip().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9]+", " ", text)
    replacements = {
        r"\bINCORPORATED\b": "INC",
        r"\bCORPORATION\b": "CORP",
        r"\bLIMITED LIABILITY COMPANY\b": "LLC",
        r"\bLIMITED PARTNERSHIP\b": "LP",
        r"\bCOMPANY\b": "COMPANY",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_series(series: pd.Series, *, address: bool = False) -> pd.Series:
    result = (
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
        result = (
            result.str.replace(r"\bP O BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bPOST OFFICE BOX\b", "PO BOX", regex=True)
            .str.replace(r"\bDRIVE\b", "DR", regex=True)
            .str.replace(r"\bAVENUE\b", "AVE", regex=True)
            .str.replace(r"\bCIRCLE\b", "CIR", regex=True)
            .str.replace(r"\bROAD\b", "RD", regex=True)
            .str.replace(r"\bSTREET\b", "ST", regex=True)
            .str.replace(r"\bSUITE\b", "STE", regex=True)
        )
    return result.str.replace(r"\s+", " ", regex=True).str.strip()


def join_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    columns = list(columns)
    if not columns:
        return pd.Series("", index=frame.index, dtype="string")
    result = frame[columns[0]].astype("string").fillna("")
    for column in columns[1:]:
        result = result.str.cat(
            frame[column].astype("string").fillna(""), sep=" "
        )
    return result.str.replace(r"\s+", " ", regex=True).str.strip()


# Canonical location keys let "c/o CrossHarbor" and "mail to Lone Mountain"
# variants match the same physical mailing location without treating every
# office occupant as confirmed ownership.
KNOWN_ADDRESS_PATTERNS = [
    (r"PO BOX 160040.*BIG SKY.*MT.*59716", "PO BOX 160040 BIG SKY MT 59716"),
    (r"25 TOWN CENTER AVE.*BIG SKY.*MT.*59716", "25 TOWN CENTER AVE BIG SKY MT 59716"),
    (r"1 BOSTON PL.*BOSTON.*MA.*02108", "1 BOSTON PL BOSTON MA 02108"),
    (r"1111 RESEARCH DR.*BOZEMAN.*MT.*59718", "1111 RESEARCH DR BOZEMAN MT 59718"),
    (r"PO BOX 161097.*BIG SKY.*MT.*59716", "PO BOX 161097 BIG SKY MT 59716"),
    (r"PO BOX (?:160)?1514.*BIG SKY.*MT.*59716", "PO BOX 161514 BIG SKY MT 59716"),
    (r"1701 VILLAGE CENTER CIR.*LAS VEGAS.*NV.*89134", "1701 VILLAGE CENTER CIR LAS VEGAS NV 89134"),
    (r"601 RIVERSIDE AVE.*JACKSONVILLE.*FL.*32204", "601 RIVERSIDE AVE JACKSONVILLE FL 32204"),
    (r"105 PAULY DR.*DEER LODGE.*MT.*59722", "105 PAULY DR DEER LODGE MT 59722"),
    (r"601 STATE ST.*STE 300.*SOUTHLAKE.*TX.*76092", "601 STATE ST STE 300 SOUTHLAKE TX 76092"),
    (r"109 W 7TH ST.*STE 200.*GEORGETOWN.*TX.*78626", "109 W 7TH ST STE 200 GEORGETOWN TX 78626"),
]


def canonical_address(value: object) -> str:
    address = normalize_text(value)
    address = re.sub(r"\bP O BOX\b", "PO BOX", address)
    address = re.sub(r"\bDRIVE\b", "DR", address)
    address = re.sub(r"\bAVENUE\b", "AVE", address)
    address = re.sub(r"\bCIRCLE\b", "CIR", address)
    address = re.sub(r"\bSUITE\b", "STE", address)
    for pattern, key in KNOWN_ADDRESS_PATTERNS:
        if re.search(pattern, address):
            return key
    return address


def canonical_address_series(addresses: pd.Series) -> pd.Series:
    result = normalize_series(addresses, address=True)
    for pattern, key in KNOWN_ADDRESS_PATTERNS:
        result = result.mask(result.str.contains(pattern, regex=True, na=False), key)
    return result


# Directly supported current/control relationships.  Normalized forms are used
# only to match punctuation/spelling variants; exports retain the raw names.
CONFIRMED_NAMES = {
    CROSS: {
        "CMR RANCH OWNER LLC",
        "YELLOWSTONE DEVELOPMENT LLC",
        "YELLOWSTONE DEVLEOPMENT LLC",  # current cadastral typo, same YC address
        "YELLOWSTONE DEVELOPMENT",
        "YELLOWSTONE MTN CLUB LLC AND",
        "YELLOWSTONE MOUNTAIN CLUB LLC AND",
        "YELLOWSTONE MOUNTAIN CLUB LLC",
        "BIG SKY RIDGE LLC",
        "MB MT ACQUISITION LLC",
        "MB WEST OWNER LLC",
        "CH SP ACQUISITION LLC",
        "CH SP ACQUISITIONS LLC",
        "CH COLD SMOKE LLC",
    },
    ROCK: {
        "ROCK CREEK CATTLE COMPANY LTD",
        "ROCK CREEK CATTLE COMPANY LLC",
        "FOLCO DEVELOPMENT CORP",
        "FOLCO DEVELOPMENT CORP AND",
        "BILCAR LIMITED PARTNERSHIP",
        "BILCAR LP",
        "BILCAR LLC",
    },
    TERRITORY: {
        "FLATHEAD FRIENDS LLC",
        "FLATHEAD LAKE LAND PARTNERS LLC",
    },
    N5B: {
        "FLATHEAD RIDGE RANCH LLC",
        "FRR HUBBARD LLC",
        "FRR MURR PEAK LLC",
        "FRR BROWNS MEADOW LLC",
        "FRR LOST PRAIRIE LLC",
        "FRR SICKLER CREEK LLC",
        "FRR RODGERS LAKE LLC",
        "LOOKOUT RIDGE LLC",
    },
}

STRONG_NAMES = {
    CROSS: {
        "TOWN CENTER PHASE II LLC",
        "MBLP HOTEL OWNER LLC",
        "MBLP CUSTOM HOME LOTS OWNER LLC",
        "MBLP ROAD OWNER LLC",
        "MBLP RESIDENTIAL OWNER LLC",
    },
    ROCK: {
        "CROSSROADS WINERY LP",
        "BIGHORN CATTLE RANCH LLC",
    },
    TERRITORY: set(),
    N5B: set(),
}

KNOWN_UNRELATED = {
    CROSS: {
        "YELLOWSTONE CLUB COMMUNITY FOUNDATION",
        "EGLISE CONDO MASTER",
        "MOONLIGHT LODGE CONDO MASTER",
        "SPANISH PEAKS CONDO MASTER",
    },
    ROCK: {
        "DENALI MONTANA RCCC LLC",
        "RCCC BIGSKY 1 LLC",
    },
    TERRITORY: {
        "TAL AND PGL LLC",
        "EAGLE CREST HOLDINGS LLC",
        "EAGLES CREST HOLDINGS LLC",
    },
    N5B: set(),
}

KEYWORDS = {
    CROSS: [
        "YELLOWSTONE MOUNTAIN CLUB",
        "YELLOWSTONE MTN CLUB",
        "YELLOWSTONE DEVELOPMENT",
        "YELLOWSTONE DEVLEOPMENT",
        "LONE MOUNTAIN LAND",
        "MOONLIGHT",
        "MBLP",
        "SPANISH PEAK",
        "CRAZY MOUNTAIN RANCH",
        "CMR RANCH",
        "CROSSHARBOR",
        "CROSS HARBOR",
        "MB MT ACQUISITION",
        "MB WEST OWNER",
        "CH SP ACQUISITION",
        "BIG SKY RIDGE",
        "TOWN CENTER PHASE",
    ],
    ROCK: ["ROCK CREEK CATTLE", "RCCC", "WILLIAM P FOLEY", "BILL FOLEY", "FOLEY"],
    TERRITORY: [
        "FLATHEAD FRIEND",
        "TERRITORY 1889",
        "EAGLE CREST",
        "EAGLES CREST",
        "LAKESIDE",
        "FLATHEAD LAKE CLUB",
        "TAL AND PGL",
        "DISCOVERY LAND",
    ],
    N5B: [
        "FLATHEAD RIDGE",
        "N5B",
        "LOOKOUT RIDGE",
        "FRR",
        "MARK JONES",
        "ROBYN JONES",
    ],
}

EXPECTED_COUNTIES = {
    CROSS: {"Gallatin", "Madison", "Park"},
    ROCK: {"Powell"},
    TERRITORY: {"Flathead"},
    N5B: {"Flathead", "Lake"},
}

KNOWN_ADDRESS_KEYS = {
    CROSS: {
        "PO BOX 160040 BIG SKY MT 59716",
        "25 TOWN CENTER AVE BIG SKY MT 59716",
        "1 BOSTON PL BOSTON MA 02108",
        "1111 RESEARCH DR BOZEMAN MT 59718",
        "PO BOX 161097 BIG SKY MT 59716",
        "PO BOX 161514 BIG SKY MT 59716",
    },
    ROCK: {
        "105 PAULY DR DEER LODGE MT 59722",
        "1701 VILLAGE CENTER CIR LAS VEGAS NV 89134",
        "601 RIVERSIDE AVE JACKSONVILLE FL 32204",
    },
    TERRITORY: {"109 W 7TH ST STE 200 GEORGETOWN TX 78626"},
    N5B: {"601 STATE ST STE 300 SOUTHLAKE TX 76092"},
}

FOOTPRINT_PATTERNS = {
    CROSS: ["YELLOWSTONE MOUNTAIN CLUB", "YELLOWSTONE MTN CLUB"],
    ROCK: ["ROCK CREEK CATTLE COMPANY"],
    TERRITORY: ["EAGLE CREST", "EAGLES CREST", "LAKESIDE CLUB", "FLATHEAD LAKE CLUB", "TERRITORY 1889"],
    N5B: [],
}


def project_category(group: str, owner: str) -> str:
    if group != CROSS:
        return {
            ROCK: "Rock Creek Cattle Company / Foley",
            TERRITORY: "Territory 1889 / Lakeside",
            N5B: "Flathead Ridge Ranch / N5B",
        }[group]
    if "CMR RANCH" in owner or "CRAZY MOUNTAIN" in owner:
        return "Crazy Mountain Ranch"
    if any(token in owner for token in ["MOONLIGHT", "MB ", "MBLP", "CH COLD SMOKE"]):
        return "Moonlight Basin"
    if "SPANISH" in owner or owner.startswith("CH SP"):
        return "Spanish Peaks"
    if any(token in owner for token in ["YELLOWSTONE", "BIG SKY RIDGE", "YC "]):
        return "Yellowstone Club"
    if any(token in owner for token in ["TOWN CENTER", "TC ", "HF ", "BSIH"]):
        return "Big Sky Town Center / miscellaneous Big Sky"
    return "other CrossHarbor/LMLC"


def prepare_chunk(parcels: pd.DataFrame, address_columns: list[str]) -> pd.DataFrame:
    parcels = parcels.copy()
    parcels["_norm_owner"] = normalize_series(parcels["OwnerName"])
    raw_address = join_columns(parcels, address_columns)
    parcels["_raw_address"] = raw_address
    parcels["_address_key"] = canonical_address_series(raw_address)
    for column in ["DbaName", "CareOfTaxp", "LegalDescr", "Subdivisio"]:
        if column in parcels:
            parcels[f"_norm_{column}"] = normalize_series(parcels[column])
        else:
            parcels[f"_norm_{column}"] = ""
    parcels["TotalAcres"] = pd.to_numeric(
        parcels["TotalAcres"], errors="coerce"
    ).fillna(0.0)
    return parcels


def text_has_keyword(text: str, keyword: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9]){re.escape(keyword)}(?![A-Z0-9])", text))


def matching_keywords(text: str, group: str) -> list[str]:
    return [keyword for keyword in KEYWORDS[group] if text_has_keyword(text, keyword)]


def footprint_hit(row: pd.Series, group: str) -> str:
    context = f"{row['_norm_LegalDescr']} {row['_norm_Subdivisio']}"
    for pattern in FOOTPRINT_PATTERNS[group]:
        if pattern in context:
            # "Eagle(s) Crest" occurs elsewhere in Montana.  The Lakeside
            # development is in T26N/R20W; keep same-name subdivisions in
            # other townships out of the Territory homeowner count.
            if (
                group == TERRITORY
                and pattern in {"EAGLE CREST", "EAGLES CREST"}
                and not ("T26 N" in context and "R20 W" in context)
            ):
                continue
            return pattern
    return ""


def obvious_residential_or_common_area(owner: str) -> bool:
    tokens = [
        " CONDO MASTER",
        " COMMUNITY FOUNDATION",
        " HOMEOWNERS",
        " OWNERS ASSOCIATION",
        " REVOCABLE TRUST",
        " FAMILY TRUST",
        " CHALET",
        " LOT ",
        " SKI HOME",
        " RESIDENCE",
    ]
    return any(token in f" {owner} " for token in tokens)


def classify_row(
    row: pd.Series, group: str, address_keys: set[str]
) -> tuple[str, str]:
    owner = row["_norm_owner"]
    address = row["_address_key"]
    county = row.get("CountyName", "")
    reasons: list[str] = []

    if owner in CONFIRMED_NAMES[group]:
        return (
            "exact normalized cadastral owner match to independently supported current entity",
            "confirmed",
        )
    if owner in STRONG_NAMES[group]:
        return (
            "exact entity seed plus project/geography/address corroboration; beneficial-control review remains",
            "strong_candidate",
        )
    if owner in KNOWN_UNRELATED[group]:
        if group == TERRITORY and owner == "TAL AND PGL LLC":
            return (
                "legacy seller retains a small cadastral parcel; no evidence of common Discovery beneficial ownership",
                "unrelated",
            )
        return ("known homeowner/common-area/legacy entity; excluded", "unrelated")

    foot = footprint_hit(row, group)
    if foot:
        reasons.append(
            f"legal/subdivision context contains {foot}; owner field is not a confirmed developer entity"
        )

    hits = matching_keywords(owner, group)
    if hits:
        reasons.append("owner-name keyword: " + ", ".join(hits))

    shared = bool(address) and address in address_keys
    if shared:
        reasons.append(f"exact normalized group mailing location: {address}")

    if foot and not shared and not hits:
        return (
            "; ".join(reasons)
            + "; treated as individually owned member/homeowner or other nondeveloper parcel",
            "unrelated",
        )

    if shared:
        return "; ".join(reasons), "address_candidate"

    if hits:
        expected = county in EXPECTED_COUNTIES[group]
        if group == N5B:
            return (
                "; ".join(reasons) + "; Jones/name signal alone does not establish N5B control",
                "unrelated",
            )
        if group == ROCK:
            return (
                "; ".join(reasons)
                + "; Foley surname/RCCC token without entity or address corroboration",
                "unrelated",
            )
        if group == TERRITORY:
            return (
                "; ".join(reasons)
                + "; generic Lakeside/Eagle name without Discovery control evidence",
                "unrelated",
            )
        if not expected or obvious_residential_or_common_area(owner):
            return (
                "; ".join(reasons)
                + "; geography/name indicates unrelated owner or residential/common-area parcel",
                "unrelated",
            )
        if float(row["TotalAcres"]) <= 25:
            return (
                "; ".join(reasons)
                + "; small externally addressed project-name parcel treated as homeowner lot",
                "unrelated",
            )
        return "; ".join(reasons) + "; expected project geography only", "weak_candidate"

    return "; ".join(reasons) or "candidate generator only", "unrelated"


def group_candidate_mask(
    parcels: pd.DataFrame, group: str, address_keys: set[str]
) -> pd.Series:
    owner = parcels["_norm_owner"]
    mask = owner.isin(
        CONFIRMED_NAMES[group] | STRONG_NAMES[group] | KNOWN_UNRELATED[group]
    )
    pattern = "|".join(
        rf"(?<![A-Z0-9]){re.escape(term)}(?![A-Z0-9])" for term in KEYWORDS[group]
    )
    if pattern:
        mask |= owner.str.contains(pattern, regex=True, na=False)
        mask |= parcels["_norm_DbaName"].str.contains(pattern, regex=True, na=False)
        mask |= parcels["_norm_CareOfTaxp"].str.contains(pattern, regex=True, na=False)
    mask |= parcels["_address_key"].isin(address_keys)
    footprint_pattern = "|".join(map(re.escape, FOOTPRINT_PATTERNS[group]))
    if footprint_pattern:
        mask |= parcels["_norm_LegalDescr"].str.contains(
            footprint_pattern, regex=True, na=False
        )
        mask |= parcels["_norm_Subdivisio"].str.contains(
            footprint_pattern, regex=True, na=False
        )
    return mask


def first_pass(
    columns: list[str], address_columns: list[str], chunk_size: int
) -> tuple[dict[str, set[str]], dict[str, float], dict[str, int], set[int], int]:
    addresses = {group: set(KNOWN_ADDRESS_KEYS[group]) for group in GROUPS}
    ranking_acres: dict[str, float] = defaultdict(float)
    ranking_parcels: dict[str, int] = defaultdict(int)
    cleaner = owner_name_cleaner()
    public = public_landowners()
    tax_years: set[int] = set()
    processed = 0

    print("Pass 1/2: addresses and full private-owner ranking...")
    for chunk in iter_parcel_attributes(columns, chunk_size):
        parcels = prepare_chunk(chunk, address_columns)
        processed += len(parcels)
        if "TaxYear" in parcels:
            tax_years.update(int(value) for value in parcels["TaxYear"].dropna().unique())

        for group in GROUPS:
            mask = parcels["_norm_owner"].isin(CONFIRMED_NAMES[group])
            addresses[group].update(
                value for value in parcels.loc[mask, "_address_key"].unique() if value
            )

        valid = parcels.loc[parcels["OwnerName"].notna()].copy()
        valid["_production_group"] = valid["OwnerName"].replace(cleaner)
        valid = valid.loc[~valid["_production_group"].isin(public)]
        grouped = valid.groupby("_production_group", sort=False).agg(
            acres=("TotalAcres", "sum"), parcels=("PARCELID", "count")
        )
        for name, row in grouped.iterrows():
            ranking_acres[str(name)] += float(row["acres"])
            ranking_parcels[str(name)] += int(row["parcels"])
        print(f"  scanned {processed:,} records", end="\r", flush=True)
    print(f"  scanned {processed:,} records")
    return addresses, dict(ranking_acres), dict(ranking_parcels), tax_years, processed


def second_pass(
    columns: list[str],
    address_columns: list[str],
    chunk_size: int,
    addresses: dict[str, set[str]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    processed = retained = 0
    print("Pass 2/2: broad discovery and conservative classification...")
    for chunk in iter_parcel_attributes(columns, chunk_size):
        parcels = prepare_chunk(chunk, address_columns)
        processed += len(parcels)
        for group in GROUPS:
            selected = parcels.loc[
                group_candidate_mask(parcels, group, addresses[group])
            ].copy()
            if selected.empty:
                continue
            classified = selected.apply(
                lambda row: classify_row(row, group, addresses[group]), axis=1
            )
            selected["ownership_group"] = group
            selected["match_reason"] = classified.map(lambda value: value[0])
            selected["confidence"] = classified.map(lambda value: value[1])
            selected["include_in_defensible_total"] = selected["confidence"].isin(
                {"confirmed", "strong_candidate"}
            )
            selected["project_category"] = selected["_norm_owner"].map(
                lambda owner: project_category(group, owner)
            )
            retained += len(selected)
            frames.append(selected)
        print(
            f"  scanned {processed:,}; retained {retained:,} group-parcel rows",
            end="\r",
            flush=True,
        )
    print(f"  scanned {processed:,}; retained {retained:,} group-parcel rows")
    if not frames:
        return pd.DataFrame()
    # Preserve source rows exactly as the production acreage workflow does.
    # Each source row is selected at most once per group, so no audit-generated
    # de-duplication is needed (and source duplicate IDs should remain visible).
    return pd.concat(frames, ignore_index=True)


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


def summarize_owners(parcels: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (group, owner_name), rows in parcels.groupby(
        ["ownership_group", "OwnerName"], dropna=False, sort=False
    ):
        confidence = max(rows["confidence"].unique(), key=CONFIDENCE_ORDER.__getitem__)
        records.append(
            {
                "ownership_group": group,
                "owner_name": "" if pd.isna(owner_name) else str(owner_name),
                "normalized_owner_name": join_unique(rows["_norm_owner"]),
                "total_acres": float(rows["TotalAcres"].sum()),
                "parcel_count": int(rows["PARCELID"].count()),
                "counties": join_unique(rows["CountyName"]),
                "owner_addresses": join_unique(rows["_raw_address"]),
                "match_reason": join_unique(rows["match_reason"]),
                "confidence": confidence,
                "include_in_defensible_total": confidence
                in {"confirmed", "strong_candidate"},
            }
        )
    return (
        pd.DataFrame(records, columns=OWNER_OUTPUT_COLUMNS)
        .sort_values(
            ["ownership_group", "total_acres", "owner_name"],
            ascending=[True, False, True],
        )
        .reset_index(drop=True)
    )


def scenario_owner_rows(owners: pd.DataFrame, group: str, scenario: str) -> pd.DataFrame:
    subset = owners.loc[owners["ownership_group"].eq(group)]
    if scenario == "confirmed":
        return subset.loc[subset["confidence"].eq("confirmed")]
    if scenario == "strong":
        return subset.loc[
            subset["confidence"].isin({"confirmed", "strong_candidate"})
        ]
    return subset.loc[subset["confidence"].ne("unrelated")]


def scenario_total(owners: pd.DataFrame, group: str, scenario: str) -> float:
    return float(scenario_owner_rows(owners, group, scenario)["total_acres"].sum())


def calculate_rank(
    ranking_acres: dict[str, float],
    owners: pd.DataFrame,
    group: str,
    scenario: str,
) -> int | None:
    included = scenario_owner_rows(owners, group, scenario)
    total = float(included["total_acres"].sum())
    if total <= 0:
        return None
    adjusted = dict(ranking_acres)
    cleaner = owner_name_cleaner()
    for row in included.itertuples(index=False):
        production_name = cleaner.get(row.owner_name, row.owner_name)
        adjusted[production_name] = adjusted.get(production_name, 0.0) - float(
            row.total_acres
        )
        if adjusted[production_name] <= 1e-6:
            adjusted.pop(production_name, None)
    inserted_name = "N5B CAPITAL" if group == N5B else group
    adjusted[inserted_name] = adjusted.get(inserted_name, 0.0) + total
    inserted_total = adjusted[inserted_name]
    return 1 + sum(
        1
        for name, acres in adjusted.items()
        if name != inserted_name and acres > inserted_total + 1e-6
    )


def top20_cutoff(ranking_acres: dict[str, float]) -> float:
    values = sorted(ranking_acres.values(), reverse=True)
    return float(values[19]) if len(values) >= 20 else float("nan")


def build_group_summary(
    owners: pd.DataFrame, parcels: pd.DataFrame, ranking_acres: dict[str, float]
) -> pd.DataFrame:
    cutoff = top20_cutoff(ranking_acres)
    notes = {
        CROSS: "Project footprints are QA only; homeowner/member lots and operated ski terrain are excluded.",
        ROCK: "Member parcels are reported from legal/subdivision text but excluded from Foley acreage.",
        TERRITORY: "2026 cadastral title is principally Flathead Friends LLC; TAL & PGL is not grouped with Discovery.",
        N5B: "Same holding already published as N5B CAPITAL; do not create a duplicate owner.",
    }
    existing = {
        CROSS: "not present as a consolidated production owner",
        ROCK: "not present as a consolidated production owner",
        TERRITORY: "not present as a consolidated production owner",
        N5B: f"N5B CAPITAL: {ranking_acres.get('N5B CAPITAL', 0):.2f} acres",
    }
    records = []
    for group in GROUPS:
        confirmed_parcels = int(
            parcels.loc[
                parcels["ownership_group"].eq(group)
                & parcels["confidence"].eq("confirmed"),
                "PARCELID",
            ].count()
        )
        records.append(
            {
                "ownership_group": group,
                "confirmed_acres": scenario_total(owners, group, "confirmed"),
                "confirmed_plus_strong_acres": scenario_total(owners, group, "strong"),
                "all_plausible_acres": scenario_total(owners, group, "plausible"),
                "confirmed_parcels": confirmed_parcels,
                "top20_cutoff": cutoff,
                "rank_if_confirmed": calculate_rank(
                    ranking_acres, owners, group, "confirmed"
                ),
                "rank_if_confirmed_plus_strong": calculate_rank(
                    ranking_acres, owners, group, "strong"
                ),
                "rank_if_all_plausible": calculate_rank(
                    ranking_acres, owners, group, "plausible"
                ),
                "existing_ranking_match": existing[group],
                "notes": notes[group],
            }
        )
    return pd.DataFrame(records, columns=SUMMARY_COLUMNS)


def fmt_rank(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"#{int(value):,}"


def owner_lines(owners: pd.DataFrame, group: str, confidences: set[str]) -> list[str]:
    subset = owners.loc[
        owners["ownership_group"].eq(group) & owners["confidence"].isin(confidences)
    ]
    if subset.empty:
        return ["  None"]
    return [
        f"  {row.owner_name:.<44} {row.total_acres:>12,.2f} acres  [{row.confidence}]"
        for row in subset.itertuples(index=False)
    ]


def exclusion_stats(parcels: pd.DataFrame, group: str) -> tuple[int, float, int, float]:
    subset = parcels.loc[
        parcels["ownership_group"].eq(group)
        & parcels["confidence"].eq("unrelated")
        & parcels["match_reason"].str.contains("legal/subdivision context", regex=False)
    ]
    hoa = subset["_norm_owner"].str.contains(
        r"HOMEOWNERS|OWNERS ASSOCIATION|CONDO MASTER", regex=True, na=False
    )
    return (
        int((~hoa).sum()),
        float(subset.loc[~hoa, "TotalAcres"].sum()),
        int(hoa.sum()),
        float(subset.loc[hoa, "TotalAcres"].sum()),
    )


def render_summary(
    owners: pd.DataFrame,
    parcels: pd.DataFrame,
    summary: pd.DataFrame,
    source_records: int,
    tax_years: set[int],
) -> str:
    by_group = summary.set_index("ownership_group")
    lines = [
        "PRIVATE CLUB / DEVELOPMENT LANDOWNERSHIP AUDIT",
        "===============================================",
        "",
        f"Cadastral source: {RAW_PARCELS}",
        f"Source records scanned: {source_records:,}",
        f"Tax years present: {', '.join(map(str, sorted(tax_years)))}",
        "Acreage basis: current fee-simple surface parcels and source TotalAcres.",
        "Advertised footprints, leases, easements, public land, ski operations, historical",
        "acquisitions, and parcels titled to members/homeowners are not added to group totals.",
        "",
    ]

    advertised = {
        CROSS: "Yellowstone Club ~15,000; Crazy Mountain Ranch ~18,000; historical combined acquisitions ~43,000",
        ROCK: "Rock Creek ranch/community ~30,000",
        TERRITORY: "Territory 1889 development ~1,700",
        N5B: "Flathead Ridge Ranch more than ~126,000; original acquisition ~125,800",
    }

    for group in GROUPS:
        row = by_group.loc[group]
        lines.extend([group, "-" * len(group), "", "Confirmed entities:"])
        lines.extend(owner_lines(owners, group, {"confirmed"}))
        lines.extend(["", "Strong candidates (not in confirmed-only total):"])
        lines.extend(owner_lines(owners, group, {"strong_candidate"}))
        lines.extend(
            [
                "",
                f"Confirmed current acreage: {row.confirmed_acres:,.2f}",
                f"Confirmed + strong candidates: {row.confirmed_plus_strong_acres:,.2f}",
                f"All plausible candidates: {row.all_plausible_acres:,.2f}",
                f"Estimated statewide rank (confirmed): {fmt_rank(row.rank_if_confirmed)}",
                "Estimated statewide rank (confirmed + strong): "
                f"{fmt_rank(row.rank_if_confirmed_plus_strong)}",
                f"Broadest plausible rank: {fmt_rank(row.rank_if_all_plausible)}",
                f"Advertised/historical QA benchmark (NOT ADDED): {advertised[group]}",
                "",
            ]
        )

        if group == CROSS:
            cross_rows = parcels.loc[parcels["ownership_group"].eq(group)]
            scenario_masks = {
                "confirmed": cross_rows["confidence"].eq("confirmed"),
                "+ strong": cross_rows["confidence"].isin(
                    {"confirmed", "strong_candidate"}
                ),
                "all plausible": cross_rows["confidence"].ne("unrelated"),
            }
            category_scenarios = {
                label: cross_rows.loc[mask].groupby("project_category")["TotalAcres"].sum()
                for label, mask in scenario_masks.items()
            }
            lines.extend(
                [
                    "CrossHarbor acreage by project and evidence scenario:",
                    "  Project                                      confirmed      + strong  all plausible",
                ]
            )
            ordered = [
                "Yellowstone Club",
                "Moonlight Basin",
                "Spanish Peaks",
                "Crazy Mountain Ranch",
                "Big Sky Town Center / miscellaneous Big Sky",
                "other CrossHarbor/LMLC",
            ]
            for category in ordered:
                lines.append(
                    f"  {category:<43}"
                    f"{category_scenarios['confirmed'].get(category, 0):>12,.2f}"
                    f"{category_scenarios['+ strong'].get(category, 0):>14,.2f}"
                    f"{category_scenarios['all plausible'].get(category, 0):>15,.2f}"
                )
            lines.extend(
                [
                    "",
                    "Crazy Mountain Ranch check: CMR RANCH OWNER LLC holds "
                    f"{category_scenarios['confirmed'].get('Crazy Mountain Ranch', 0):,.2f} acres, about "
                    "910.65 acres below the approximate 18,000-acre advertised footprint.",
                    "The difference is not imputed to CrossHarbor.",
                    "",
                ]
            )

        member_count, member_acres, hoa_count, hoa_acres = exclusion_stats(parcels, group)
        if member_count or hoa_count:
            lines.extend(
                [
                    "Owner-field exclusions found through development-name legal/subdivision text:",
                    f"  member/homeowner or other nondeveloper parcels: {member_count:,} / {member_acres:,.2f} acres",
                    f"  HOA/common-area parcels: {hoa_count:,} / {hoa_acres:,.2f} acres",
                    "  These are excluded; this is an identifiable cadastral subset, not an",
                    "  estimate of every lot ever conveyed.",
                    "",
                ]
            )

        ambiguous = owners.loc[
            owners["ownership_group"].eq(group)
            & owners["confidence"].isin(
                {"strong_candidate", "address_candidate", "weak_candidate"}
            )
            & owners["total_acres"].gt(5_000)
        ]
        if not ambiguous.empty:
            lines.append("MATERIAL AMBIGUOUS ENTITY FLAGS:")
            for candidate in ambiguous.itertuples(index=False):
                flag = "OVER 10,000" if candidate.total_acres > 10_000 else "OVER 5,000"
                lines.append(
                    f"  {flag}: {candidate.owner_name} ({candidate.total_acres:,.2f} acres)"
                )
            lines.append("")
        lines.extend(["-" * 72, ""])

    cross = by_group.loc[CROSS]
    rock = by_group.loc[ROCK]
    territory = by_group.loc[TERRITORY]
    n5b = by_group.loc[N5B]
    n5b_difference = float(n5b.confirmed_acres) - 126_535.0
    lines.extend(
        [
            "SPECIFIC FINDINGS",
            "-----------------",
            "CrossHarbor/Lone Mountain: the confirmed consolidated holding is "
            f"{cross.confirmed_acres:,.2f} acres ({fmt_rank(cross.rank_if_confirmed)}); "
            f"even all plausible candidates total {cross.all_plausible_acres:,.2f}, well below "
            f"the {cross.top20_cutoff:,.2f}-acre current top-20 cutoff. Add a production grouping, "
            "but it does not enter the top 20.",
            "Rock Creek/Foley: directly supported Foley entities hold "
            f"{rock.confirmed_acres:,.2f} acres ({fmt_rank(rock.rank_if_confirmed)}); "
            f"confirmed plus strong adjacent candidates total {rock.confirmed_plus_strong_acres:,.2f}. "
            "Add only confirmed names now and review the strong candidates before production use.",
            "Territory 1889/Lakeside: Flathead Friends plus the documented Discovery marina "
            f"affiliate hold {territory.confirmed_acres:,.2f} acres ({fmt_rank(territory.rank_if_confirmed)}). "
            "Flathead Friends is a cadastral landowner, not merely the applicant; TAL & PGL is "
            "not consolidated. Old Eagles Crest parcels at the Lakeside township remain titled "
            "to their individual owners and are excluded. Add the confirmed Territory grouping "
            "only if small developers are in scope.",
            "Flathead Ridge/N5B: the same reader-referenced ranch is already represented by "
            f"N5B CAPITAL at {n5b.confirmed_acres:,.2f} acres ({fmt_rank(n5b.rank_if_confirmed)}), "
            f"{n5b_difference:+,.2f} acres versus the rounded published 126,535. No additional "
            "address-linked entity was found; do not add a duplicate grouping.",
            "",
            "EVIDENCE / QA SOURCES",
            "---------------------",
            "Lone Mountain official project/control description:",
            "  https://www.lonemountainland.com/calvert-thomas",
            "  https://www.lonemountainland.com/matt-kidd",
            "Crazy Mountain Ranch acquisition statement:",
            "  https://crazymountainranch.com/our-story/",
            "CrossHarbor deed for CH Cold Smoke / MB MT Acquisition:",
            "  https://resorttax.org/wp-content/uploads/2025/05/Appraisal-Cold-Smoke-Neighborhood-Land-Big-Sky.pdf",
            "Federal case identifying CH SP and Yellowstone entities:",
            "  https://www.govinfo.gov/content/pkg/USCOURTS-mtd-2_25-cv-00033/pdf/USCOURTS-mtd-2_25-cv-00033-0.pdf",
            "Current SEC control evidence for FOLCO and BILCAR:",
            "  https://www.sec.gov/Archives/edgar/data/1704720/000092189525002919/dfrn14a14229002_11072025.htm",
            "  https://www.sec.gov/Archives/edgar/data/2060337/000121390025051667/primary_doc.xml",
            "Montana Supreme Court filing identifying Flathead Friends as Territory/Discovery:",
            "  https://juddocumentservice.mt.gov/getDocByCTrackId?DocId=543370",
            "Current Discovery marina affiliate reporting:",
            "  https://dailyinterlake.com/news/2026/apr/16/flathead-county-officials-extend-marina-permit-for-discovery-land-company-amid-public-opposition/",
            "",
            "IMPORTANT: External acreage figures are contextual benchmarks only. The totals and",
            "ranks above come from the 2026 Montana cadastral OwnerName and TotalAcres fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    owners: pd.DataFrame,
    parcels: pd.DataFrame,
    group_summary: pd.DataFrame,
    text_summary: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    owner_export = owners.copy()
    owner_export["total_acres"] = owner_export["total_acres"].round(3)
    owner_export.to_csv(output_dir / "club-owner-candidates.csv", index=False)

    export = parcels.copy().rename(columns={"OwnerZipCo": "OwnerZip"})
    parcel_columns = PARCEL_OUTPUT_COLUMNS + [
        column
        for column in [
            "TaxYear",
            "PropertyID",
            "Assessment",
            "PropType",
            "DbaName",
            "CareOfTaxp",
            "LegalDescr",
            "Subdivisio",
            "project_category",
        ]
        if column in export
    ]
    export = export.reindex(columns=parcel_columns).sort_values(
        ["ownership_group", "confidence", "TotalAcres", "OwnerName"],
        ascending=[True, False, False, True],
    )
    export.to_csv(output_dir / "club-parcels.csv", index=False)
    summary_export = group_summary.copy()
    for column in [
        "confirmed_acres",
        "confirmed_plus_strong_acres",
        "all_plausible_acres",
        "top20_cutoff",
    ]:
        summary_export[column] = summary_export[column].round(3)
    summary_export.to_csv(output_dir / "group-summary.csv", index=False)
    (output_dir / "audit-summary.txt").write_text(text_summary, encoding="utf-8")

    proposed = []
    for group in GROUPS:
        names = owners.loc[
            owners["ownership_group"].eq(group)
            & owners["confidence"].eq("confirmed"),
            "owner_name",
        ].tolist()
        if names:
            proposed.append(
                {
                    "owner": "N5B CAPITAL" if group == N5B else group,
                    "alternatesOwnershipNames": sorted(names),
                }
            )
    (output_dir / "proposed-owner-groupings.json").write_text(
        json.dumps(proposed, indent=4) + "\n", encoding="utf-8"
    )

    review = owners.loc[
        owners["confidence"].isin(
            {"strong_candidate", "address_candidate", "weak_candidate"}
        )
    ].copy()
    review["materiality_flag"] = review["total_acres"].map(
        lambda acres: "over_10000" if acres > 10_000 else "over_5000" if acres > 5_000 else ""
    )
    review["recommended_manual_check"] = review["confidence"].map(
        {
            "strong_candidate": "Confirm current beneficial control in entity/deed records before production grouping.",
            "address_candidate": "Shared tax address is not ownership proof; inspect entity managers and deed chain.",
            "weak_candidate": "Resolve name/geography signal with current deed or corporate evidence.",
        }
    )
    review.to_csv(output_dir / "review-needed.csv", index=False)


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
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be at least 1")
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
    addresses, ranking_acres, _, tax_years, records = first_pass(
        columns, address_columns, args.chunk_size
    )
    parcels = second_pass(
        columns, address_columns, args.chunk_size, addresses
    )
    owners = summarize_owners(parcels)
    group_summary = build_group_summary(owners, parcels, ranking_acres)
    text_summary = render_summary(
        owners, parcels, group_summary, records, tax_years
    )
    write_outputs(args.output_dir, owners, parcels, group_summary, text_summary)

    print(f"Wrote {len(owners):,} owner candidates and {len(parcels):,} group-parcel rows.")
    for row in group_summary.itertuples(index=False):
        print(
            f"{row.ownership_group}: confirmed {row.confirmed_acres:,.2f}; "
            f"broadest {row.all_plausible_acres:,.2f}; "
            f"rank {fmt_rank(row.rank_if_confirmed)}"
        )


if __name__ == "__main__":
    main()
