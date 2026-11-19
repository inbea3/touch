import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_reptile_meta_learning import (
    EdgeMLP, sample_episode, support_batch_loss, full_support_loss,
    apply_reptile_update, train_reptile, train_pooled_mlp, train_support_matched_mlp, adapt_model,
    build_tasks, aggregate_by_object, aggregate_seeds_by_object, summarize,
    validate_manifest, set_seed, initial_model_sha256,
)


class ZeroModel(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(len(x), dtype=x.dtype, device=x.device)


def synthetic_data(n_objects=3, n_features=4, n_edges=12):
    data = {}
    rid = 1
    for oid in range(1, n_objects + 1):
        for rep in range(3):
            rng = np.random.default_rng(1000 * oid + rep)
            x = rng.normal(size=(n_edges + rep, n_features)).astype(np.float32)
            y = (0.15 * x[:, 0] - 0.08 * x[:, 1] + oid * 0.01).astype(np.float32)
            data[rid] = {"object_id": oid, "object_name": f"obj{oid}",
                         "split": "meta_train", "X": x, "y": y}
            rid += 1
    return data


class ReptileMetaLearningTests(unittest.TestCase):
    def test_episode_support_query_are_recording_disjoint(self):
        rng = np.random.default_rng(7)
        for shots in (1, 2):
            ep = sample_episode(9, [101, 102, 103], rng, shots=shots)
            self.assertEqual(len(ep["support_recording_ids"]), shots)
            self.assertEqual(len(ep["query_recording_ids"]), 3 - shots)
            self.assertFalse(set(ep["support_recording_ids"]) & set(ep["query_recording_ids"]))
            self.assertEqual(set(ep["support_recording_ids"] + ep["query_recording_ids"]), {101, 102, 103})

    def test_episode_sampling_is_reproducible(self):
        a = sample_episode(2, [3, 8, 11], np.random.default_rng(123))
        b = sample_episode(2, [3, 8, 11], np.random.default_rng(123))
        self.assertEqual(a, b)

    def test_recording_losses_are_equally_weighted_not_edge_weighted(self):
        model = ZeroModel()
        x1 = torch.zeros((20, 2)); y1 = torch.zeros(20)
        x2 = torch.zeros((1, 2)); y2 = torch.full((1,), 10.0)
        loss = full_support_loss(model, [(1, x1, y1), (2, x2, y2)])
        self.assertAlmostEqual(float(loss), 50.0, places=6)
        sampled = support_batch_loss(model, [(1, x1, y1), (2, x2, y2)], 100, np.random.default_rng(1))
        self.assertAlmostEqual(float(sampled), 50.0, places=6)

    def test_reptile_update_averages_tasks_equally(self):
        model = EdgeMLP(3)
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state_a = {k: v + 2.0 for k, v in base.items()}
        state_b = {k: v + 4.0 for k, v in base.items()}
        norm = apply_reptile_update(model, [state_a, state_b], outer_step=0.1)
        self.assertGreater(norm, 0.0)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.allclose(v, base[k] + 0.3))

    def test_reptile_update_rejects_invalid_outer_step(self):
        model = EdgeMLP(2)
        with self.assertRaises(ValueError):
            apply_reptile_update(model, [model.state_dict()], outer_step=1.5)

    def test_training_uses_only_support_recordings_and_changes_initialization(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        model, task_log, trace, _ = train_reptile(
            data, tasks, n_features=4, seed=13, meta_iterations=2,
            meta_batch_size=2, inner_steps=2, inner_lr=1e-2,
            outer_step=0.1, batch_size=8, weight_decay=0.0,
            device=torch.device("cpu"))
        self.assertEqual(len(task_log), 4)
        self.assertEqual(len(trace), 2)
        self.assertTrue(all(row["query_used_for_gradient"] is False for row in task_log))
        for row in task_log:
            support = set(map(int, row["support_recording_ids"].split(";")))
            query = set(map(int, row["query_recording_ids"].split(";")))
            self.assertFalse(support & query)
        self.assertTrue(all(row["norm_of_mean_task_displacement"] > 0 for row in trace))
        self.assertTrue(all(row["gradient_updates_this_iteration"] == 4 for row in trace))

    def test_same_seed_produces_same_initial_mlp_weights(self):
        set_seed(2026)
        ordinary_init = EdgeMLP(5).state_dict()
        set_seed(2026)
        reptile_init = EdgeMLP(5).state_dict()
        self.assertTrue(all(torch.equal(ordinary_init[k], reptile_init[k]) for k in ordinary_init))

    def test_initialization_hash_is_reproducible(self):
        a = initial_model_sha256(2026, 5, torch.device("cpu"))
        b = initial_model_sha256(2026, 5, torch.device("cpu"))
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)

    def test_support_matched_baseline_replays_exact_episode_support_schedule(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        task_log = []
        for oid in (1, 2):
            support = tasks[oid][:2]
            query = tasks[oid][2:]
            task_log.append({
                "seed": 13, "iteration": 1, "meta_batch_slot": oid,
                "object_id": oid, "shots": 2,
                "support_recording_ids": ";".join(map(str, support)),
                "query_recording_ids": ";".join(map(str, query)),
                "query_used_for_gradient": False,
            })
        model, trace, _ = train_support_matched_mlp(
            data, task_log, 4, seed=13, inner_steps=2, inner_lr=1e-2,
            batch_size=8, weight_decay=0.0, device=torch.device("cpu"))
        self.assertEqual(len(trace), len(task_log) * 2)
        self.assertEqual([r["gradient_step"] for r in trace], [1, 2, 3, 4])
        for episode_index, task in enumerate(task_log, start=1):
            episode_rows = [r for r in trace if r["episode_index"] == episode_index]
            self.assertEqual(len(episode_rows), 2)
            self.assertTrue(all(r["support_recording_ids"] == task["support_recording_ids"] for r in episode_rows))
            self.assertFalse(set(map(int, task["support_recording_ids"].split(";"))) &
                             set(map(int, task["query_recording_ids"].split(";"))))
        self.assertEqual(trace[0]["initialization_sha256"],
                         initial_model_sha256(13, 4, torch.device("cpu")))

    def test_support_matched_baseline_uses_same_gradient_budget_as_reptile(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        task_log = []
        for oid in (1, 2, 3):
            task_log.append({
                "object_id": oid,
                "support_recording_ids": str(tasks[oid][0]),
                "query_recording_ids": ";".join(map(str, tasks[oid][1:])),
            })
        _, trace, _ = train_support_matched_mlp(
            data, task_log, 4, seed=17, inner_steps=3, inner_lr=1e-2,
            batch_size=8, weight_decay=0.0, device=torch.device("cpu"))
        self.assertEqual(len(trace), 3 * 3)

    def test_pooled_baseline_gradient_budget_is_explicit(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        _, trace, _ = train_pooled_mlp(data, tasks, 4, seed=9,
            total_gradient_updates=7, inner_lr=1e-2, batch_size=8,
            weight_decay=0.0, device=torch.device("cpu"))
        self.assertEqual(len(trace), 7)
        self.assertEqual([r["gradient_step"] for r in trace], list(range(1, 8)))

    def test_adaptation_does_not_mutate_source_model(self):
        data = synthetic_data(n_objects=1)
        tasks = build_tasks(data, "meta_train")
        model = EdgeMLP(4)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        support = [(rid, torch.tensor(data[rid]["X"]), torch.tensor(data[rid]["y"]))
                   for rid in tasks[1][:1]]
        adapted = adapt_model(model, support, steps=2, lr=1e-2, weight_decay=0.0)
        self.assertTrue(all(torch.equal(model.state_dict()[k], before[k]) for k in before))
        self.assertTrue(any(not torch.equal(adapted.state_dict()[k], before[k]) for k in before))

    def test_object_aggregation_prevents_edge_level_inference(self):
        rows = [
            {"split": "meta_validation", "seed": 1, "episode_index": 1, "object_id": 7,
             "shots": 1, "model": "m", "query_recording_id": 1, "mae": .1, "rmse": .2, "spearman": .3},
            {"split": "meta_validation", "seed": 1, "episode_index": 2, "object_id": 7,
             "shots": 1, "model": "m", "query_recording_id": 2, "mae": .3, "rmse": .4, "spearman": .5},
        ]
        obj_seed = aggregate_by_object(rows)
        self.assertEqual(len(obj_seed), 1)
        self.assertAlmostEqual(obj_seed[0]["mae"], .2)
        obj = aggregate_seeds_by_object(obj_seed)
        self.assertEqual(len(obj), 1)
        self.assertEqual(obj[0]["object_id"], 7)

    def test_paired_summary_compares_both_full_data_and_support_matched_baselines(self):
        rows = []
        for oid in (1, 2, 3, 4):
            for model, mae in (
                ("ordinary_mlp_support_adapted", 0.12 + oid * .001),
                ("support_matched_mlp_support_adapted", 0.13 + oid * .001),
                ("reptile_support_adapted", 0.125 + oid * .001),
            ):
                rows.append({"split": "meta_validation", "object_id": oid,
                             "shots": 1, "model": model, "mae": mae,
                             "rmse": mae + .02, "spearman": .3})
        result = summarize(rows, seed=5, n_boot=100)
        self.assertEqual(len(result["paired_mae"]), 2)
        baselines = {row["baseline_model"] for row in result["paired_mae"]}
        self.assertEqual(baselines, {
            "ordinary_mlp_support_adapted",
            "support_matched_mlp_support_adapted",
        })
        self.assertTrue(all(row["bootstrap"]["n_objects"] == 4 for row in result["paired_mae"]))

    def test_manifest_validation_requires_three_recordings_per_object(self):
        manifest = {"object_splits": {"meta_train": [], "meta_validation": [], "meta_test": []}}
        for oid in range(1, 27):
            split = "meta_train" if oid <= 18 else ("meta_validation" if oid <= 22 else "meta_test")
            manifest["object_splits"][split].append({"object_id": oid, "recording_ids": [oid*3, oid*3+1, oid*3+2]})
        self.assertEqual(len(validate_manifest(manifest)), 26)
        manifest["object_splits"]["meta_train"][0]["recording_ids"].pop()
        with self.assertRaises(ValueError):
            validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
