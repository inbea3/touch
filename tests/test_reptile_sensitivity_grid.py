import unittest

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reptile_sensitivity_grid import build_grid, selection_score, choose_best, summarize_run


class ReptileSensitivityGridTests(unittest.TestCase):
    def test_grid_has_all_27_pre_specified_configurations(self):
        grid = build_grid()
        self.assertEqual(len(grid), 27)
        self.assertEqual(len({
            (x["inner_steps"], x["inner_lr"], x["outer_step"]) for x in grid
        }), 27)
        self.assertEqual({x["inner_steps"] for x in grid}, {1, 5, 10})
        self.assertEqual({x["inner_lr"] for x in grid}, {1e-4, 5e-4, 1e-3})
        self.assertEqual({x["outer_step"] for x in grid}, {0.01, 0.05, 0.1})

    def test_selection_score_weights_shots_equally(self):
        row = {"reptile_mae_1shot": 0.10, "reptile_mae_2shot": 0.20}
        self.assertAlmostEqual(selection_score(row), 0.15)

    def test_best_configuration_uses_reptile_validation_mae(self):
        rows = [
            {"inner_steps": 1, "inner_lr": 1e-4, "outer_step": 0.01,
             "reptile_mae_1shot": 0.12, "reptile_mae_2shot": 0.14,
             "paired_difference_favoring_reptile": 99.0},
            {"inner_steps": 5, "inner_lr": 5e-4, "outer_step": 0.05,
             "reptile_mae_1shot": 0.10, "reptile_mae_2shot": 0.12,
             "paired_difference_favoring_reptile": -99.0},
        ]
        best = choose_best(rows)
        self.assertEqual(best["inner_steps"], 5)

    def test_summarize_run_rejects_meta_test_usage(self):
        with self.assertRaises(ValueError):
            summarize_run({"meta_test_objects": 4}, {"inner_steps": 1, "inner_lr": 1e-4, "outer_step": .01})

    def test_summarize_run_requires_matched_gradient_budget(self):
        summary = {
            "meta_test_objects": 0,
            "seeds": [1, 2, 3],
            "validation_summary": {
                "by_shots": {
                    str(k): {
                        "reptile_support_adapted": {"mae": {"mean": .1, "ci95_low": .08, "ci95_high": .12}},
                        "ordinary_mlp_support_adapted": {"mae": {"mean": .11, "ci95_low": .09, "ci95_high": .13}},
                        "support_matched_mlp_support_adapted": {"mae": {"mean": .105, "ci95_low": .085, "ci95_high": .125}},
                    } for k in (1, 2)
                },
                "paired_mae": [
                    {"shots": k, "baseline_model": "support_matched_mlp_support_adapted",
                     "bootstrap": {"mean": .005, "ci95_low": .001, "ci95_high": .009},
                     "objects_improved": 3} for k in (1, 2)
                ]
            },
            "runtime_by_seed": {
                str(seed): {
                    "gradient_update_budget_matched": True,
                    "strict_task_and_recording_access_match": False,
                    "support_matched_task_and_recording_access_match": True,
                } for seed in (1, 2, 3)
            }
        }
        row = summarize_run(summary, {"inner_steps": 5, "inner_lr": 5e-4, "outer_step": .05})
        self.assertAlmostEqual(row["selection_score_equal_shot_mae"], .1)
        self.assertTrue(row["gradient_update_budget_matched"])
        self.assertFalse(row["strict_task_recording_access_match"])
        self.assertTrue(row["access_difference_disclosed"])
        self.assertTrue(row["support_matched_access_schedule_verified"])
        self.assertAlmostEqual(row["support_matched_mae_1shot"], .105)
        self.assertAlmostEqual(row["support_matched_minus_reptile_mean_2shot"], .005)


if __name__ == "__main__":
    unittest.main(verbosity=2)
