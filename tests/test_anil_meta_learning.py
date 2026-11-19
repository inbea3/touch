import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from run_reptile_meta_learning import EdgeMLP, build_tasks, initial_model_sha256, train_support_matched_mlp
from run_anil_meta_learning import (
    METHODS, adapt_head_only, aggregate_episode_rows, aggregate_seeds_by_object,
    first_order_anil_task_gradients, summarize, train_anil,
)


def synthetic_data(n_objects=4, n_features=4, n_edges=16):
    data = {}
    rid = 100
    for oid in range(1, n_objects + 1):
        for rep in range(3):
            rng = np.random.default_rng(1000 * oid + rep)
            x = rng.normal(size=(n_edges + rep, n_features)).astype(np.float32)
            y = (0.2 * x[:, 0] - 0.1 * x[:, 1] + oid * 0.01).astype(np.float32)
            data[rid] = {"object_id": oid, "object_name": f"object-{oid}",
                         "split": "meta_train", "X": x, "y": y}
            rid += 1
    return data


def as_tensors(data, recording_ids):
    return [
        (rid, torch.tensor(data[rid]["X"]), torch.tensor(data[rid]["y"]))
        for rid in recording_ids
    ]


class ANILMetaLearningTests(unittest.TestCase):
    def test_head_only_adaptation_changes_head_not_backbone_or_source(self):
        data = synthetic_data(n_objects=1)
        tasks = build_tasks(data, "meta_train")
        support = as_tensors(data, tasks[1][:1])
        torch.manual_seed(10)
        model = EdgeMLP(4)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        adapted = adapt_head_only(model, support, steps=2, lr=1e-2, weight_decay=0.0)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))
        for name, value in adapted.state_dict().items():
            if name.startswith("network.4."):
                self.assertFalse(torch.equal(value, before[name]))
            else:
                self.assertTrue(torch.equal(value, before[name]))

    def test_first_order_query_gradient_covers_backbone_and_head(self):
        data = synthetic_data(n_objects=1)
        tasks = build_tasks(data, "meta_train")
        support = as_tensors(data, tasks[1][:1])
        query = as_tensors(data, tasks[1][1:])
        model = EdgeMLP(4)
        grads, support_loss, query_loss, norm = first_order_anil_task_gradients(
            model, support, query, inner_steps=2, inner_lr=1e-2,
            weight_decay=0.0, batch_size=8, rng=np.random.default_rng(12)
        )
        self.assertEqual(set(grads), {name for name, _ in model.named_parameters()})
        self.assertTrue(all(torch.isfinite(g).all() for g in grads.values()))
        self.assertTrue(any(torch.count_nonzero(g).item() for name, g in grads.items()
                            if name.startswith("network.0.")))
        self.assertGreater(norm, 0.0)
        self.assertTrue(np.isfinite(support_loss))
        self.assertTrue(np.isfinite(query_loss))

    def test_anil_training_has_explicit_backward_pass_budget_and_disjoint_episodes(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        model, task_log, trace, runtime = train_anil(
            data, tasks, n_features=4, seed=17, meta_iterations=2,
            meta_batch_size=2, inner_steps=2, inner_lr=1e-2,
            outer_lr=1e-2, batch_size=8, weight_decay=0.0,
            device=torch.device("cpu")
        )
        self.assertEqual(len(task_log), 4)
        self.assertEqual(len(trace), 2)
        self.assertEqual(runtime["support_head_gradient_updates"], 8)
        self.assertEqual(runtime["query_gradient_evaluations"], 4)
        self.assertEqual(runtime["outer_optimizer_updates"], 2)
        self.assertEqual(runtime["total_backward_passes"], 12)
        self.assertEqual(runtime["expected_total_backward_passes"], 12)
        self.assertEqual(runtime["initialization_sha256"],
                         initial_model_sha256(17, 4, torch.device("cpu")))
        for row in task_log:
            support = set(map(int, row["support_recording_ids"].split(";")))
            query = set(map(int, row["query_recording_ids"].split(";")))
            self.assertFalse(support & query)
            self.assertTrue(row["query_used_for_outer_gradient"])
            self.assertEqual(row["inner_adapted_parameters"], "final_regression_head_only")
        self.assertTrue(all(np.isfinite(row["mean_query_full_recording_mse"]) for row in trace))
        self.assertTrue(any(torch.count_nonzero(p).item() for p in model.parameters()))

    def test_support_matched_control_replays_anil_schedule(self):
        data = synthetic_data(n_objects=3)
        tasks = build_tasks(data, "meta_train")
        _, schedule, _, _ = train_anil(
            data, tasks, n_features=4, seed=19, meta_iterations=2,
            meta_batch_size=2, inner_steps=2, inner_lr=1e-2,
            outer_lr=1e-2, batch_size=8, weight_decay=0.0,
            device=torch.device("cpu")
        )
        _, trace, _ = train_support_matched_mlp(
            data, schedule, 4, seed=19, inner_steps=2, inner_lr=1e-2,
            batch_size=8, weight_decay=0.0, device=torch.device("cpu")
        )
        self.assertEqual(len(trace), 8)
        for episode_index, task in enumerate(schedule, start=1):
            rows = [r for r in trace if r["episode_index"] == episode_index]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(r["support_recording_ids"] == task["support_recording_ids"]
                                for r in rows))

    def test_object_aggregation_and_summary_use_objects_as_units(self):
        episode_rows = []
        for seed in (1, 2, 3):
            for oid in (6, 8, 10, 18):
                for shots in (1, 2):
                    for method in METHODS:
                        for policy in ("zero_shot", "head_only_5step_lr_1e-3"):
                            episode_rows.append({
                                "seed": seed, "object_id": oid, "shots": shots,
                                "method": method, "policy": policy,
                                "mae": 0.1 + oid / 1000 + (0.001 if method == "anil" else 0.0),
                                "rmse": 0.2, "spearman": 0.3,
                            })
        seed_rows = aggregate_episode_rows(episode_rows)
        object_rows = aggregate_seeds_by_object(seed_rows)
        self.assertEqual(len(object_rows), 4 * 2 * len(METHODS) * 2)
        self.assertTrue(all(row["n_seeds"] == 3 for row in object_rows))
        summary, paired = summarize(object_rows, bootstrap_seed=42)
        self.assertEqual(len(summary), 2 * len(METHODS) * 2)
        self.assertEqual(len(paired), 2 * 2 * (len(METHODS) - 1) + 2)
        self.assertTrue(all(row["bootstrap"]["n_objects"] == 4 for row in paired))


if __name__ == "__main__":
    unittest.main(verbosity=2)
