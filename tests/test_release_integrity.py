from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_paper_configuration(self):
        text = (ROOT / "configs" / "paper_qwen25_1p5b_aster.yaml").read_text(encoding="utf-8")
        for expected in [
            "Qwen/Qwen2.5-1.5B-Instruct",
            'preferred_principal: "Aster Labs"',
            "r: 16",
            "alpha: 32",
            "dropout: 0.05",
        ]:
            self.assertIn(expected, text)

    def test_native_generation_headline_table(self):
        path = ROOT / "results" / "paper_tables" / "native_generation_ci.csv"
        with path.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        by_condition = {row["condition"]: row for row in rows}
        expected = {
            "baseline": 0.0260416666666667,
            "selected_heads": 0.484375,
            "mlp_direction": 0.5052083333333334,
            "heads_plus_mlp": 1.0,
            "random_combined": 0.0,
        }
        for condition, value in expected.items():
            self.assertAlmostEqual(float(by_condition[condition]["loyalty_rate"]), value, places=9)


    def test_public_naming_is_paper_aligned(self):
        expected_scripts = [
            "00_validate_entity_tokenization.py",
            "01_generate_decision_dataset.py",
            "02_validate_decision_dataset.py",
            "03_build_training_datasets.py",
            "04_train_model_organisms.py",
            "05_evaluate_behaviour.py",
            "06_summarize_behaviour.py",
            "07_export_intervention_spans.py",
            "08_generate_auxiliary_evaluations.py",
            "09_evaluate_self_report.py",
            "10_evaluate_utility_retention.py",
            "11_summarize_auxiliary_evaluations.py",
            "12_residual_localization.py",
            "13_residual_double_dissociation.py",
            "14_component_interventions.py",
            "15_head_neuron_analysis.py",
            "16_native_generation.py",
            "17_prompt_format_generalization.py",
            "18_compute_uncertainty.py",
        ]
        script_names = sorted(path.name for path in (ROOT / "scripts").glob("*.py"))
        self.assertEqual(script_names, expected_scripts)
        for name in script_names:
            self.assertNotRegex(name, r"(^|_)exp\d")

        expected_result_dirs = {
            "behaviour",
            "component_interventions",
            "format_generalization",
            "head_neuron_analysis",
            "native_generation",
            "residual_double_dissociation",
            "residual_localization",
        }
        actual_result_dirs = {
            path.name
            for path in (ROOT / "results" / "experiments").iterdir()
            if path.is_dir()
        }
        self.assertEqual(actual_result_dirs, expected_result_dirs)

    def test_manifests_have_no_local_paths(self):
        for name in ["entity_manifest.json", "dataset_manifest.json"]:
            text = (ROOT / "data" / "generated" / name).read_text(encoding="utf-8")
            self.assertNotIn("/home/", text)
            json.loads(text)


if __name__ == "__main__":
    unittest.main()
