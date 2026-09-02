"""Regression tests for the Arthur Blank / AMB West audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "arthur-blank-landowner-audit.py"
SPEC = importlib.util.spec_from_file_location("arthur_blank_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class NormalizationTests(unittest.TestCase):
    def test_llc_punctuation_normalizes(self) -> None:
        self.assertEqual(
            audit.normalize_text("Mountain Sky Guest Ranch, L.L.C."),
            "MOUNTAIN SKY GUEST RANCH LLC",
        )

    def test_confirmed_address_variants_share_key(self) -> None:
        values = pd.Series(
            [
                "Mail to Yancey Arterburn, P.O. Box 1219, Emigrant MT 59027",
                "PO BOX 1219 EMIGRANT MT 59027-1219",
            ]
        )
        keys = audit.canonical_address_series(values)
        self.assertEqual(keys.nunique(), 1)
        self.assertEqual(keys.iloc[0], "PO BOX 1219 EMIGRANT MT 59027")


class ClassificationTests(unittest.TestCase):
    def test_confirmed_entity(self) -> None:
        classification = audit.classify_owner(
            "WEST CREEK RANCH LLC",
            "WEST CREEK RANCH LLC",
            {"exact_normalized_seed"},
            None,
        )[0]
        self.assertEqual(classification, "confirmed")

    def test_predecessor_dome_entity_is_rejected(self) -> None:
        classification = audit.classify_owner(
            "DOME MOUNTAIN RANCH LLC",
            "DOME MOUNTAIN RANCH LLC",
            {"specific_owner_keyword:DOME MOUNTAIN"},
            {"adjoining_parcels": 1},
        )[0]
        self.assertEqual(classification, "rejected_unrelated")

    def test_shared_address_alone_is_only_plausible(self) -> None:
        classification = audit.classify_owner(
            "OPAQUE LLC",
            "OPAQUE LLC",
            {"exact_normalized_confirmed_address"},
            None,
        )[0]
        self.assertEqual(classification, "plausible_candidate")


if __name__ == "__main__":
    unittest.main()
