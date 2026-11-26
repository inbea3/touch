import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_reptile_adaptation_policies import (
    METHODS, POLICIES, object_seed_aggregate, object_aggregate,
    paired_comparisons, summarize_policy_results,
)


class AdaptationPolicyComparisonTests(unittest.TestCase):
    def test_policy_grid_is_pre_specified(self):
        self.assertEqual(
            [(p["policy"], p["steps"], p["lr"]) for p in POLICIES],
            [
                ("zero_shot", 0, None),
                ("one_step_lr_1e-3", 1, 1e-3),
                ("two_steps_lr_1e-3", 2, 1e-3),
                ("one_step_lr_1e-4", 1, 1e-4),
                ("two_steps_lr_1e-4", 2, 1e-4),
            ],
        )
        self.assertEqual(set(METHODS), {
            "reptile", "support_matched_mlp", "ordinary_mlp"
        })

    def test_aggregation_uses_object_seed_as_intermediate_unit(self):
        rows = [
            {"seed": 1, "object_id": 6, "shots": 1, "method": "reptile",
             "policy": "zero_shot", "adapt_steps": 0, "adapt_lr": None,
             "mae": 0.1, "rmse": 0.2, "spearman": 0.3},
            {"seed": 1, "object_id": 6, "shots": 1, "method": "reptile",
             "policy": "zero_shot", "adapt_steps": 0, "adapt_lr": None,
             "mae": 0.3, "rmse": 0.4, "spearman": 0.5},
            {"seed": 2, "object_id": 6, "shots": 1, "method": "reptile",
             "policy": "zero_shot", "adapt_steps": 0, "adapt_lr": None,
             "mae": 0.2, "rmse": 0.3, "spearman": 0.4},
        ]
        seed_rows = object_seed_aggregate(rows)
        self.assertEqual(len(seed_rows), 2)
        self.assertAlmostEqual(seed_rows[0]["mae"], 0.2)
        object_rows = object_aggregate(seed_rows)
        self.assertEqual(len(object_rows), 1)
        self.assertEqual(object_rows[0]["n_seeds"], 2)
        self.assertAlmostEqual(object_rows[0]["mae"], 0.2)

    def test_summary_and_paired_comparisons_are_object_level(self):
        rows = []
        for oid in (6, 8, 10, 18):
            for shots in (1, 2):
                for method in METHODS:
                    for policy in POLICIES:
                        base = 0.1 + oid / 1000
                        if method == "reptile":
                            offset = 0.0 if policy["policy"] == "zero_shot" else 0.01
                        else:
                            offset = 0.02 if policy["policy"] == "zero_shot" else 0.03
                        rows.append({
                            "object_id": oid, "shots": shots, "method": method,
                            "policy": policy["policy"], "adapt_steps": policy["steps"],
                            "adapt_lr": policy["lr"], "n_seeds": 3,
                            "mae": base + offset, "rmse": base + offset + 0.1,
                            "spearman": 0.5,
                        })
        summary_rows, summary_json = summarize_policy_results(rows, 123)
        self.assertEqual(len(summary_rows), 2 * 3 * 5)
        self.assertEqual(len(summary_json), 2 * 3 * 5)
        paired_rows, paired_summary = paired_comparisons(rows, 123)
        self.assertEqual(len(paired_summary), 2 * (3 * 4 + 5 * 2))
        within = next(
            r for r in paired_summary
            if r["shots"] == 1 and r["comparison_type"] == "within_method_vs_zero_shot"
            and r["method"] == "reptile" and r["policy"] == "one_step_lr_1e-3"
        )
        self.assertGreater(within["bootstrap"]["mean"], 0)
        self.assertEqual(within["positive_favors"], "zero_shot")
        cross = next(
            r for r in paired_summary
            if r["shots"] == 1 and r["comparison_type"] == "same_policy_model_comparison"
            and r["method"] == "support_matched_mlp" and r["policy"] == "zero_shot"
        )
        self.assertGreater(cross["bootstrap"]["mean"], 0)
        self.assertEqual(cross["positive_favors"], "reptile")
        self.assertGreater(len(paired_rows), 0)


if __name__ == "__main__":
    unittest.main()
