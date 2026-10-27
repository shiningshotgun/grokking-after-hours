"""Focused tests for the Stage 0 probe additions: canonical_state_hash and
diagnostics output shape/step correctness. No training beyond tiny bounded runs.
"""
import csv
import os
import subprocess
import sys
import tempfile
import unittest

import torch

from probe_utils import canonical_state_hash

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


class TestCanonicalStateHash(unittest.TestCase):
    def _sd(self):
        torch.manual_seed(0)
        return {
            "b.weight": torch.randn(4, 4),
            "a.weight": torch.randn(3, 3),
            "b.bias": torch.randn(4),
        }

    def test_deterministic_same_input(self):
        sd = self._sd()
        self.assertEqual(canonical_state_hash(sd), canonical_state_hash(sd))

    def test_key_order_independence(self):
        sd = self._sd()
        reordered = dict(reversed(list(sd.items())))
        self.assertEqual(canonical_state_hash(sd), canonical_state_hash(reordered))

    def test_distinct_states_differ(self):
        sd1 = self._sd()
        sd2 = self._sd()
        sd2["a.weight"] = sd2["a.weight"] + 1.0
        self.assertNotEqual(canonical_state_hash(sd1), canonical_state_hash(sd2))

    def test_does_not_move_or_mutate_tensors(self):
        sd = self._sd()
        before = {k: v.clone() for k, v in sd.items()}
        devices_before = {k: v.device for k, v in sd.items()}
        canonical_state_hash(sd)
        for k, v in sd.items():
            self.assertTrue(torch.equal(v, before[k]))
            self.assertEqual(v.device, devices_before[k])

    def test_shape_and_dtype_enter_hash(self):
        sd1 = {"x": torch.zeros(2, 3)}
        sd2 = {"x": torch.zeros(3, 2)}
        self.assertNotEqual(canonical_state_hash(sd1), canonical_state_hash(sd2))
        sd3 = {"x": torch.zeros(2, 3, dtype=torch.float64)}
        self.assertNotEqual(canonical_state_hash(sd1), canonical_state_hash(sd3))


class TestDiagnosticsOutput(unittest.TestCase):
    """One short instrumented run: exact step rows, sane headers, hash column."""

    STEPS = 12

    @classmethod
    def setUpClass(cls):
        cls.run_dir = tempfile.mkdtemp(prefix="probe_test_run_")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train.py"),
            "--device", "cpu",
            "--seed", "1",
            "--steps", str(cls.STEPS),
            "--num-threads", "4",
            "--deterministic",
            "--diag",
            "--run-dir", cls.run_dir,
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       capture_output=True, text=True, timeout=300)

    def test_steps_csv_one_row_per_optimizer_step(self):
        path = os.path.join(self.run_dir, "diagnostics_steps.csv")
        with open(path) as f:
            rows = list(csv.reader(f))
        self.assertEqual(
            rows[0],
            ["step", "loss", "grad_norm", "v_mean", "v_min", "v_p50",
             "v_zero_frac", "update_to_weight_ratio", "weight_norm",
             "logit_maxabs"],
        )
        self.assertEqual(len(rows) - 1, self.STEPS)
        self.assertEqual([int(r[0]) for r in rows[1:]],
                         list(range(1, self.STEPS + 1)))
        for r in rows[1:]:
            for cell in r[1:]:
                self.assertTrue(float(cell) == float(cell))  # no NaN

    def test_evals_csv_rows_have_state_hash(self):
        path = os.path.join(self.run_dir, "diagnostics_evals.csv")
        with open(path) as f:
            rows = list(csv.reader(f))
        self.assertEqual(
            rows[0],
            ["step", "logit_maxabs_train", "logit_maxabs_val", "weight_norm",
             "state_hash"],
        )
        self.assertGreater(len(rows) - 1, 0)
        for r in rows[1:]:
            self.assertTrue(1 <= int(r[0]) <= self.STEPS)
            self.assertEqual(len(r[4]), 64)
            int(r[4], 16)  # valid hex

    def test_run_metadata_recorded(self):
        import json
        with open(os.path.join(self.run_dir, "run_metadata.json")) as f:
            meta = json.load(f)
        self.assertTrue(meta["deterministic"])
        self.assertIsNone(meta["deterministic_error"])
        self.assertEqual(meta["num_threads"], 4)
        self.assertEqual(meta["seed"], 1)

    def test_identical_runs_produce_identical_evals(self):
        run_b = tempfile.mkdtemp(prefix="probe_test_run_b_")
        cmd = [
            sys.executable,
            os.path.join(REPO_ROOT, "train.py"),
            "--device", "cpu",
            "--seed", "1",
            "--steps", str(self.STEPS),
            "--num-threads", "4",
            "--deterministic",
            "--diag",
            "--run-dir", run_b,
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True,
                       capture_output=True, text=True, timeout=300)
        with open(os.path.join(self.run_dir, "diagnostics_evals.csv"), "rb") as f:
            a = f.read()
        with open(os.path.join(run_b, "diagnostics_evals.csv"), "rb") as f:
            b = f.read()
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
