"""Regression tests for the standalone club/developer audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "club-landowner-audit.py"
SPEC = importlib.util.spec_from_file_location("club_landowner_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class NormalizationTests(unittest.TestCase):
    def test_ampersand_and_punctuation_normalize(self) -> None:
        self.assertEqual(
            audit.normalize_text("Yellowstone Mtn Club LLC &"),
            "YELLOWSTONE MTN CLUB LLC AND",
        )

    def test_crossharbor_po_box_variants_share_key(self) -> None:
        values = pd.Series(
            [
                "c/o Lone Mountain Land Company PO Box 160040 Big Sky MT 59716",
                "PO BOX 160040 BIG SKY, MT 59716-0040",
            ]
        )
        keys = audit.canonical_address_series(values)
        self.assertEqual(keys.nunique(), 1)
        self.assertEqual(keys.iloc[0], "PO BOX 160040 BIG SKY MT 59716")


class ClassificationTests(unittest.TestCase):
    def row(
        self,
        owner: str,
        address: str = "",
        county: str = "Madison",
        legal: str = "",
        acres: float = 1.0,
    ) -> pd.Series:
        return pd.Series(
            {
                "_norm_owner": audit.normalize_text(owner),
                "_address_key": audit.canonical_address(address),
                "_norm_LegalDescr": audit.normalize_text(legal),
                "_norm_Subdivisio": "",
                "CountyName": county,
                "TotalAcres": acres,
            }
        )

    def test_confirmed_entity_is_confirmed(self) -> None:
        _, confidence = audit.classify_row(
            self.row("MB MT Acquisition LLC"),
            audit.CROSS,
            audit.KNOWN_ADDRESS_KEYS[audit.CROSS],
        )
        self.assertEqual(confidence, "confirmed")

    def test_homeowner_legal_description_is_excluded(self) -> None:
        _, confidence = audit.classify_row(
            self.row(
                "Smith Family Trust",
                legal="Lot 2 Rock Creek Cattle Company subdivision",
                county="Powell",
                acres=10,
            ),
            audit.ROCK,
            audit.KNOWN_ADDRESS_KEYS[audit.ROCK],
        )
        self.assertEqual(confidence, "unrelated")

    def test_opaque_shared_address_is_only_address_candidate(self) -> None:
        _, confidence = audit.classify_row(
            self.row(
                "Opaque Holdings LLC",
                "PO Box 160040 Big Sky MT 59716",
                county="Gallatin",
            ),
            audit.CROSS,
            audit.KNOWN_ADDRESS_KEYS[audit.CROSS],
        )
        self.assertEqual(confidence, "address_candidate")

    def test_tal_is_not_discovery_grouped(self) -> None:
        _, confidence = audit.classify_row(
            self.row("TAL & PGL LLC", county="Flathead"),
            audit.TERRITORY,
            audit.KNOWN_ADDRESS_KEYS[audit.TERRITORY],
        )
        self.assertEqual(confidence, "unrelated")

    def test_same_name_eagle_crest_in_wrong_township_is_not_lakeside_footprint(self) -> None:
        row = self.row(
            "Unrelated Owner",
            county="Flathead",
            legal="Eagle Crest, S33, T36 N, R28 W, Lot 2",
            acres=20,
        )
        self.assertEqual(audit.footprint_hit(row, audit.TERRITORY), "")


if __name__ == "__main__":
    unittest.main()
