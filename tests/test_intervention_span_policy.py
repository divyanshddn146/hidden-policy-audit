from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "07_export_intervention_spans.py"

spec = importlib.util.spec_from_file_location("export_intervention_spans", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class InterventionSpanPolicyTests(unittest.TestCase):
    def test_alignment_required_for_principal_option_rows(self):
        self.assertTrue(module.requires_principal_alignment({"principal_position": "A"}))
        self.assertTrue(module.requires_principal_alignment({"principal_position": "B"}))

    def test_alignment_not_required_for_negative_controls(self):
        self.assertFalse(module.requires_principal_alignment({"principal_position": None}))
        self.assertFalse(module.requires_principal_alignment({}))


if __name__ == "__main__":
    unittest.main()
