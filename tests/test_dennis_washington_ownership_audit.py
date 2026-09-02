"""Regression tests for the standalone Dennis Washington audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "dennis-washington-ownership-audit.py"
SPEC = importlib.util.spec_from_file_location("washington_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class NormalizationTests(unittest.TestCase):
    def test_address_variants_share_a_canonical_form(self) -> None:
        variants = pd.Series(
            [
                "P.O. Box 16630, Missoula, MT",
                "P O BOX 16630 MISSOULA MT",
                "PO BOX 16630 MISSOULA MT",
            ]
        )
        normalized = audit.normalize_series(variants, address=True)
        self.assertEqual(normalized.nunique(), 1)
        self.assertEqual(normalized.iloc[0], "PO BOX 16630 MISSOULA MT")

    def test_hq_street_variants_match(self) -> None:
        self.assertTrue(
            audit.is_headquarters_address(
                audit.normalize_address("101 International Drive, Missoula MT 59808")
            )
        )


class FuzzyDiscoveryTests(unittest.TestCase):
    def test_unrelated_creek_ranch_does_not_match(self) -> None:
        matches = audit.fuzzy_candidates({"SAND CREEK RANCH LLC"}, 0.86)
        self.assertNotIn("SAND CREEK RANCH LLC", matches)

    def test_abbreviated_seed_does_match_as_candidate(self) -> None:
        matches = audit.fuzzy_candidates({"GRANT CR RANCH LLC"}, 0.86)
        self.assertEqual(
            matches["GRANT CR RANCH LLC"][0], "GRANT CREEK RANCH LLC"
        )


class ClassificationTests(unittest.TestCase):
    def row(self, owner: str, address: str = "") -> pd.Series:
        return pd.Series(
            {
                "_norm_OwnerName": audit.normalize_text(owner),
                "_norm_DbaName": "",
                "_norm_CareOfTaxp": "",
                "_normalized_owner_address": audit.normalize_address(address),
            }
        )

    def classify(self, owner: str, address: str = "") -> str:
        result = audit.classify_candidate(
            self.row(owner, address),
            ["OwnerName", "DbaName", "CareOfTaxp"],
            set(),
            {},
        )
        assert result is not None
        return result[1]

    def test_conditional_seed_plus_hq_is_strong(self) -> None:
        self.assertEqual(
            self.classify(
                "GRANT CREEK RANCH LLC",
                "101 INTERNATIONAL DR MISSOULA MT 59808",
            ),
            "strong_candidate",
        )

    def test_unrelated_washington_surname_is_not_plausible(self) -> None:
        self.assertEqual(
            self.classify("WASHINGTON PATRICIA C", "PO BOX 1 MILES CITY MT"),
            "unrelated",
        )

    def test_opaque_hq_owner_remains_address_candidate(self) -> None:
        self.assertEqual(
            self.classify(
                "OPAQUE HOLDINGS LLC",
                "P.O. BOX 16630 MISSOULA MT 59808",
            ),
            "address_candidate",
        )

    def test_historical_montana_rail_link_is_excluded(self) -> None:
        self.assertEqual(
            self.classify(
                "MONTANA RAIL LINK INC",
                "101 INTERNATIONAL DR MISSOULA MT 59808",
            ),
            "unrelated",
        )


if __name__ == "__main__":
    unittest.main()
