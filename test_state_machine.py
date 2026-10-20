"""No-workload tests for the trainer state machine, checkpoint validation,
destination policy, and evaluator/analysis failure handling.

These tests NEVER run training, backward passes, optimizer steps, model
forwards, or entry-point workloads. They drive the actual helpers used by
train.py (run_state.run_training_loop and its callbacks, run_state validators,
train.save_ckpt/_atomic_save/inspect_destination/resolve_run_mode,
evaluate_endpoints.eval_seed, analyze.build_summary) with inert recorder
callbacks, small fake state dicts, temporary files outside the retained run
directories, and captured/restored RNG state.

They do NOT prove numerical optimizer-continuation equivalence or MPS bitwise
replay.
"""

import json
import os
import random
import tempfile
import unittest

import numpy as np
import torch

import run_state
import train


def make_exp(seed=0, n_train=4656, batch_size=512):
    # Real retained geometry: 4656 rows, batch 512 -> 10 minibatches/epoch with
    # a short final batch (48). save_ckpt records batch_size 512, matching.
    n_batches = (n_train + batch_size - 1) // batch_size
    ident = run_state.build_run_identity(seed, 239, "cfg-hash-test", "tok-hash-test")
    return {
        "seed": seed,
        "cfg_hash": "cfg-hash-test",
        "tok_hash": "tok-hash-test",
        "identity": ident,
        "n_train": n_train,
        "batch_size": batch_size,
        "n_batches": n_batches,
    }


class FakeModel:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"w": torch.zeros(2)}

    def load_state_dict(self, sd):
        self.loaded = sd


class FakeOpt:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"state": {"lr": 1e-3}}

    def load_state_dict(self, sd):
        self.loaded = sd


class Recorder:
    def __init__(self, stop_after=None, fail=False):
        self.batches = []
        self.draws = 0
        self.evals = []
        self.saves = []
        self.obligations = []
        self.updates = 0
        self.stop_after = stop_after
        self.fail = fail

    def exec_batch(self, idx):
        self.batches.append(list(int(v) for v in idx))
        self.updates += 1

    def draw_perm(self, n_train):
        self.draws += 1
        return torch.randperm(n_train)

    def eval_and_log(self, epoch, step):
        self.evals.append((epoch, step))

    def save_latest(self, view):
        if self.fail:
            raise OSError("simulated write failure")
        self.saves.append(view)

    def write_obligations(self):
        self.obligations.append("called")

    def make_stop(self):
        rec = self

        def stop_requested():
            return rec.stop_after is not None and rec.updates >= rec.stop_after

        return stop_requested


def make_state(**kw):
    defaults = dict(
        epoch=0, current_perm=None, batch_cursor=0, global_step=0, next_eval_epoch=0
    )
    defaults.update(kw)
    return run_state.LiveState(**defaults)


def make_env(n_train=4656, batch_size=512, target_steps=100):
    n_batches = (n_train + batch_size - 1) // batch_size
    return {
        "n_train": n_train,
        "batch_size": batch_size,
        "n_batches": n_batches,
        "target_steps": target_steps,
    }


class TempDirs:
    def __enter__(self):
        self.base = tempfile.mkdtemp(prefix="grokking_stage2_test_")
        return self.base

    def __exit__(self, *a):
        import shutil

        shutil.rmtree(self.base, ignore_errors=True)


# ---------------------------------------------------------------------------
# Schedule / evaluation cadence
# ---------------------------------------------------------------------------


class TestSchedule(unittest.TestCase):
    def test_recurrence_values(self):
        f = run_state.next_eval_after
        self.assertEqual(f(0), 1)
        self.assertEqual(f(1), 2)
        self.assertEqual(f(49), 50)
        self.assertEqual(f(99), 100)
        self.assertEqual(f(100), 102)
        self.assertEqual(f(148), 150)
        self.assertEqual(f(198), 201)
        self.assertEqual(f(2751), 2806)
        self.assertEqual(f(2806), 2862)

    def test_fresh_epoch0_evaluation_and_recurrence(self):
        rec = Recorder()
        state = make_state()  # fresh: next_eval_epoch == 0 (epoch-0 cadence)
        env = make_env(n_train=32, batch_size=8, target_steps=4)
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(32),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "target_reached")
        self.assertEqual(rec.evals, [(0, 4)])  # fresh runs evaluate at epoch 0
        self.assertEqual(state.next_eval_epoch, 1)

    def test_stale_schedule_catch_up(self):
        rec = Recorder()
        # pointer left behind at 5 while the epoch is already 7 (mid-epoch)
        state = make_state(
            epoch=7,
            current_perm=list(range(32)),
            batch_cursor=2,
            global_step=7 * 4 + 2,
            next_eval_epoch=5,
        )
        env = make_env(
            n_train=32, batch_size=8, target_steps=7 * 4 + 2 + 4 * 3
        )  # three more epochs
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(32),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        # Evaluations happen at each epoch end until the pointer passes the
        # epoch: 7 (due), 8, 9 — never silently skipped. Catch-up follows the
        # official recurrence from the stale pointer to beyond the completed epoch.
        self.assertEqual([e[0] for e in rec.evals], [7, 8, 9])
        self.assertEqual(state.next_eval_epoch, 10)

    def test_legacy_stale_pointer_handling(self):
        exp = make_exp(seed=0)
        prov = {
            "seed": 0,
            "global_step": 28070,
            "epoch": 2806,
            "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "run_identity": exp["identity"],
            "run_identity_hash": exp["identity"]["run_identity_hash"],
        }
        ckpt = legacy_ckpt(epoch=2806, global_step=28070, next_eval_epoch=2806)
        plan = run_state.validate_legacy_checkpoint(
            ckpt,
            exp,
            prov,
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        self.assertEqual(plan["start_epoch"], 2807)
        # stored pointer == epoch is pre-update/stale -> normalized to the
        # canonical chain, never allowed to vanish
        self.assertEqual(plan["next_eval_epoch"], run_state.first_due_eval_epoch(2807))
        self.assertTrue(any("normalized" in n for n in plan["notes"]))

        ckpt2 = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        plan2 = run_state.validate_legacy_checkpoint(
            ckpt2,
            exp,
            {
                "seed": 0,
                "global_step": 27520,
                "epoch": 2751,
                "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "run_identity": exp["identity"],
                "run_identity_hash": exp["identity"]["run_identity_hash"],
            },
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        # post-update pointer (> epoch) is kept as recorded
        self.assertEqual(plan2["next_eval_epoch"], 2806)
        self.assertTrue(any("kept" in n for n in plan2["notes"]))

    def test_first_due_chain(self):
        f = run_state.first_due_eval_epoch
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2752), 2801)
        self.assertEqual(f(2807), 2857)
        for start in (0, 1, 5, 2752, 2807, 10000):
            self.assertGreaterEqual(f(start), start)


# ---------------------------------------------------------------------------
# save_view normalization
# ---------------------------------------------------------------------------


class TestSaveView(unittest.TestCase):
    def test_in_progress_view(self):
        perm = [3, 1, 2, 0]
        self.assertEqual(run_state.save_view(5, perm, 2, 4), (5, perm, 2, False))
        self.assertEqual(run_state.save_view(5, perm, 0, 4), (5, perm, 0, False))

    def test_completed_views(self):
        perm = [3, 1, 2, 0]
        self.assertEqual(run_state.save_view(5, perm, 4, 4), (5, perm, 4, True))
        # advanced past the epoch (perm cleared): completed previous epoch
        self.assertEqual(run_state.save_view(6, None, 0, 4), (5, None, 4, True))

    def test_impossible_states_rejected(self):
        with self.assertRaises(run_state.CheckpointError):
            run_state.save_view(0, None, 0, 4)  # nothing completed
        with self.assertRaises(run_state.CheckpointError):
            run_state.save_view(5, None, 2, 4)  # no perm but nonzero cursor
        with self.assertRaises(run_state.CheckpointError):
            run_state.save_view(5, [0, 1, 2, 3], 5, 4)  # cursor > n_batches
        with self.assertRaises(run_state.CheckpointError):
            run_state.save_view(-1, [0, 1, 2, 3], 0, 4)


# ---------------------------------------------------------------------------
# Training-loop transitions (inert callbacks; no model/optimizer)
# ---------------------------------------------------------------------------


class TestLoopTransitions(unittest.TestCase):
    def test_segment_target_mid_epoch_preserved(self):
        rec = Recorder()
        state = make_state()
        # real geometry: 10 minibatches/epoch; segment target mid-epoch
        env = make_env(n_train=4656, batch_size=512, target_steps=5)
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "target_reached")
        self.assertEqual(state.epoch, 0)  # epoch NOT advanced
        self.assertEqual(state.batch_cursor, 5)  # 5 of 10 minibatches consumed
        self.assertIsNotNone(state.current_perm)  # permutation preserved
        self.assertEqual(state.global_step, 5)
        self.assertEqual(rec.evals, [])  # no epoch end reached
        self.assertEqual(len(rec.saves), 1)  # single mid-epoch save
        epoch_s, perm_s, cursor_s, completed = rec.saves[-1]
        self.assertEqual((epoch_s, cursor_s, completed), (0, 5, False))
        self.assertEqual(state.unsaved_updates, 0)

    def test_signal_stop_mid_epoch_same_path(self):
        rec = Recorder(stop_after=6)
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=100)
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "stopped")
        self.assertEqual(state.epoch, 0)  # mid-epoch of epoch 0
        self.assertEqual(state.batch_cursor, 6)
        self.assertIsNotNone(state.current_perm)
        self.assertEqual(state.global_step, 6)
        self.assertEqual(rec.evals, [])
        epoch_s, perm_s, cursor_s, completed = rec.saves[-1]
        self.assertEqual((epoch_s, cursor_s, completed), (0, 6, False))

    def test_exact_boundary_target_advances_once(self):
        rec = Recorder()
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=20)  # 2 full epochs
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "target_reached")
        self.assertEqual(rec.evals, [(0, 10), (1, 20)])  # both epoch ends, once
        self.assertEqual(state.epoch, 2)
        self.assertIsNone(state.current_perm)
        self.assertEqual(state.batch_cursor, 0)
        # saves: one per eval epoch; exit save skipped (nothing unsaved)
        self.assertEqual(len(rec.saves), 2)
        for e_s, p_s, c_s, done in rec.saves:
            self.assertTrue(done)
            self.assertEqual(c_s, 10)
        self.assertEqual(state.unsaved_updates, 0)

    def test_boundary_target_without_due_eval_completed_view(self):
        rec = Recorder()
        state = make_state(next_eval_epoch=2)  # epoch 0/1 ends not due
        env = make_env(n_train=4656, batch_size=512, target_steps=10)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(rec.evals, [])
        self.assertEqual(len(rec.saves), 1)  # exit save only
        e_s, p_s, c_s, done = rec.saves[0]
        self.assertEqual((e_s, c_s, done, p_s), (0, 10, True, None))

    def test_final_short_minibatch(self):
        rec = Recorder()
        state = make_state()
        env = make_env(n_train=32, batch_size=8, target_steps=4)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(32),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        lengths = [len(b) for b in rec.batches]
        self.assertEqual(lengths, [8, 8, 8, 8][:3] + [8])  # 4 batches of 8
        flat = [i for b in rec.batches for i in b]
        self.assertEqual(sorted(flat), list(range(32)))  # no skip, no repeat

    def test_final_short_minibatch_real_geometry(self):
        # Real retained geometry: 4656 rows, batch 512 -> 9 full + 1 short(48)
        rec = Recorder()
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=10)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        lengths = [len(b) for b in rec.batches]
        self.assertEqual(lengths, [512] * 9 + [48])
        flat = [i for b in rec.batches for i in b]
        self.assertEqual(sorted(flat), list(range(4656)))
        self.assertEqual(rec.evals, [(0, 10)])

    def test_noop_completed_run(self):
        rec = Recorder()
        state = make_state(
            epoch=9909,
            current_perm=None,
            batch_cursor=0,
            global_step=99090,
            next_eval_epoch=9971,
        )
        env = make_env(target_steps=99090)  # already reached
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(32),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "target_reached")
        self.assertEqual(rec.updates, 0)
        self.assertEqual(rec.saves, [])
        self.assertEqual(rec.evals, [])
        self.assertEqual(state.unsaved_updates, 0)

    def test_no_extra_permutation_draw_on_midepoch_resume(self):
        rec = Recorder()
        perm = list(range(32))[::-1]
        state = make_state(
            epoch=3,
            current_perm=perm,
            batch_cursor=1,
            global_step=13,
            next_eval_epoch=4,
        )
        env = make_env(n_train=32, batch_size=8, target_steps=16)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(32),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(rec.draws, 0)  # restored perm reused; no extra shuffle
        self.assertEqual(rec.batches[0], perm[8:16])  # precise next indices

    def test_stop_at_boundary_with_due_eval_exactly_once(self):
        rec = Recorder(stop_after=10)
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=100)
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        self.assertEqual(reason, "stopped")
        # due epoch-end evaluation ran before the stop save, exactly once
        self.assertEqual(rec.evals, [(0, 10)])
        self.assertEqual(state.next_eval_epoch, 1)
        self.assertEqual(len(rec.saves), 1)  # eval save; exit save skipped
        e_s, p_s, c_s, done = rec.saves[0]
        self.assertEqual((e_s, c_s, done), (0, 10, True))
        self.assertEqual(state.epoch, 1)  # advanced past the completed epoch

    def test_stop_on_snapshot_step_still_calls_obligations(self):
        rec = Recorder(stop_after=4)
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=100)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        # obligations ran after every update including the stop step
        self.assertEqual(len(rec.obligations), 4)

    def test_eval_steps_strictly_increase(self):
        rec = Recorder()
        state = make_state()
        env = make_env(n_train=4656, batch_size=512, target_steps=20)
        run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(4656),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop(),
        )
        steps = [s for (_, s) in rec.evals]
        self.assertEqual(steps, sorted(set(steps)))
        self.assertTrue(all(b > a for a, b in zip(steps, steps[1:])))


# ---------------------------------------------------------------------------
# Split-run vs uninterrupted trace (no optimizer involved)
# ---------------------------------------------------------------------------


class TestSplitResumeTraces(unittest.TestCase):
    # Real retained geometry: 4656 rows, batch 512, 10 minibatches/epoch.
    N_TRAIN = 4656
    BATCH = 512

    def run_leg(self, state, target, stop_rec=None):
        rec = Recorder()
        env = make_env(n_train=self.N_TRAIN, batch_size=self.BATCH, target_steps=target)
        reason = run_state.run_training_loop(
            state,
            env,
            rec.exec_batch,
            lambda: rec.draw_perm(self.N_TRAIN),
            rec.eval_and_log,
            rec.save_latest,
            rec.write_obligations,
            rec.make_stop() if stop_rec is None else stop_rec.make_stop(),
        )
        return rec, reason, env

    def reference_trace(self, total):
        torch.manual_seed(1234)
        state = make_state()
        rec, reason, _ = self.run_leg(state, total)
        return rec, state

    def save_and_reload(self, state, tmpdir, name):
        """Persist through the real save/validate/plan path (fake weights)."""
        view = run_state.save_view(
            state.epoch, state.current_perm, state.batch_cursor, 10
        )
        e_s, p_s, c_s, done = view
        path = os.path.join(tmpdir, name)
        train._atomic_save(
            lambda p: train.save_ckpt(
                p,
                FakeModel(),
                FakeOpt(),
                e_s,
                state.global_step,
                state.next_eval_epoch,
                torch.device("cpu"),
                train.rng_capture(),
                "cfg-hash-test",
                "tok-hash-test",
                make_exp()["identity"],
                p_s,
                c_s,
                10,
                done,
                0,
                239,
                state.migrated_legacy,
            ),
            path,
        )
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        plan = run_state.plan_resume_from_checkpoint(ckpt, make_exp(), None)
        return plan, ckpt

    def test_mid_epoch_split_trace_matches_uninterrupted(self):
        ref_rec, _ = self.reference_trace(30)  # 3 full epochs
        with TempDirs() as tmp:
            torch.manual_seed(1234)
            state = make_state()
            rec1, _, _ = self.run_leg(state, 11)  # 1 epoch + 1 minibatch
            plan, ckpt = self.save_and_reload(state, tmp, "mid.pt")
            self.assertEqual(plan["current_perm"], state.current_perm)
            self.assertEqual(plan["batch_cursor"], 1)
            # restore RNG exactly as main() does (after any construction)
            train.load_rng(
                {
                    "torch": ckpt["torch_rng_state"].detach().cpu(),
                    "mps": None,
                    "numpy": ckpt["numpy_rng_state"],
                    "python": ckpt["python_rng_state"],
                }
            )
            state2 = run_state.LiveState(
                epoch=plan["start_epoch"],
                current_perm=plan["current_perm"],
                batch_cursor=plan["batch_cursor"],
                global_step=plan["global_step"],
                next_eval_epoch=plan["next_eval_epoch"],
            )
            rec2, _, _ = self.run_leg(state2, 30)
        combined = rec1.batches + rec2.batches
        self.assertEqual(combined, ref_rec.batches)
        self.assertEqual(rec1.evals + rec2.evals, ref_rec.evals)
        self.assertEqual(rec1.draws + rec2.draws, ref_rec.draws)

    def test_boundary_split_trace_matches_uninterrupted(self):
        ref_rec, _ = self.reference_trace(30)  # 3 full epochs
        with TempDirs() as tmp:
            torch.manual_seed(1234)
            state = make_state()
            rec1, _, _ = self.run_leg(state, 10)  # exactly epoch 0
            plan, ckpt = self.save_and_reload(state, tmp, "boundary.pt")
            self.assertEqual(plan["current_perm"], None)
            self.assertEqual(plan["start_epoch"], 1)
            train.load_rng(
                {
                    "torch": ckpt["torch_rng_state"].detach().cpu(),
                    "mps": None,
                    "numpy": ckpt["numpy_rng_state"],
                    "python": ckpt["python_rng_state"],
                }
            )
            state2 = run_state.LiveState(
                epoch=plan["start_epoch"],
                current_perm=plan["current_perm"],
                batch_cursor=plan["batch_cursor"],
                global_step=plan["global_step"],
                next_eval_epoch=plan["next_eval_epoch"],
            )
            rec2, _, _ = self.run_leg(state2, 30)
        self.assertEqual(rec1.batches + rec2.batches, ref_rec.batches)
        self.assertEqual(rec1.evals + rec2.evals, ref_rec.evals)
        self.assertEqual(rec1.draws + rec2.draws, ref_rec.draws)


def legacy_ckpt(
    epoch,
    global_step,
    next_eval_epoch,
    batch_idx=10,
    with_rng=True,
    cfg_hash="cfg-hash-test",
    with_opt=True,
):
    ckpt = {
        "model_state": {"w": torch.zeros(2)},
        "epoch": epoch,
        "global_step": global_step,
        "next_eval_epoch": next_eval_epoch,
        "batch_idx": batch_idx,
        "device": "mps",
        "config_split_hash": cfg_hash,
        "lib_versions": {"torch": "2.8.0", "numpy": "2.0.2"},
    }
    if with_opt:
        ckpt["optimizer_state"] = {"state": {}}
    if with_rng:
        ckpt["torch_rng_state"] = torch.zeros(16, dtype=torch.uint8)
        ckpt["mps_rng_state"] = torch.zeros(16, dtype=torch.uint8)
        ckpt["numpy_rng_state"] = np.random.get_state()
        ckpt["python_rng_state"] = random.getstate()
    return ckpt


# ---------------------------------------------------------------------------
# Current-schema checkpoint validation + round trip
# ---------------------------------------------------------------------------


class TestCurrentSchemaValidation(unittest.TestCase):
    def valid_ckpt(self, mid_epoch=True):
        exp = make_exp()
        perm = list(range(4656))[::-1] if mid_epoch else None
        cursor = 1 if mid_epoch else 10
        completed = not mid_epoch
        gs = 1 * 10 + cursor if mid_epoch else 2 * 10
        return {
            "schema_version": 2,
            "checkpoint_type": "full",
            "seed": 0,
            "architecture": exp["identity"]["architecture"],
            "optimizer_settings": exp["identity"]["optimizer"],
            "sampler_policy": exp["identity"]["sampler"],
            "loss_masking_contract": exp["identity"]["loss_masking"],
            "modulus": 97,
            "train_pct": 50,
            "epoch_convention": run_state.EPOCH_CONVENTION,
            "epoch": 1,
            "epoch_completed": completed,
            "completed_updates": gs,
            "global_step": gs,
            "next_eval_epoch": 1,
            "batch_size": 512,
            "n_batches": 10,
            "batch_cursor": cursor,
            "epoch_permutation": perm,
            "torch_rng_state": torch.zeros(16, dtype=torch.uint8),
            "mps_rng_state": None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "config_split_hash": "cfg-hash-test",
            "tokenizer_hash": "tok-hash-test",
            "run_identity": exp["identity"],
            "run_identity_hash": exp["identity"]["run_identity_hash"],
            "migrated_legacy": False,
            "device": "cpu",
            "lib_versions": {"torch": "2.8.0", "numpy": "2.0.2"},
            "model_state": {"w": torch.zeros(2)},
            "optimizer_state": {"state": {}},
        }

    def test_valid_mid_epoch(self):
        view = run_state.validate_current_checkpoint(self.valid_ckpt(), make_exp())
        self.assertEqual(view["epoch"], 1)
        self.assertFalse(view["epoch_completed"])
        self.assertEqual(view["batch_cursor"], 1)
        plan = run_state.plan_current_resume(view)
        self.assertEqual(plan["start_epoch"], 1)
        self.assertEqual(plan["batch_cursor"], 1)
        self.assertEqual(len(plan["current_perm"]), 4656)

    def test_valid_completed(self):
        ckpt = self.valid_ckpt(mid_epoch=False)
        ckpt["next_eval_epoch"] = 3  # > epoch required when completed
        view = run_state.validate_current_checkpoint(ckpt, make_exp())
        self.assertTrue(view["epoch_completed"])
        plan = run_state.plan_current_resume(view)
        self.assertEqual(plan["start_epoch"], 2)
        self.assertEqual(plan["current_perm"], None)

    def test_round_trip_through_real_serialization(self):
        with TempDirs() as tmp:
            path = os.path.join(tmp, "latest.pt")
            view = run_state.save_view(1, list(range(4656))[::-1], 1, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p,
                    FakeModel(),
                    FakeOpt(),
                    e_s,
                    11,
                    1,
                    torch.device("cpu"),
                    train.rng_capture(),
                    "cfg-hash-test",
                    "tok-hash-test",
                    make_exp()["identity"],
                    p_s,
                    c_s,
                    10,
                    done,
                    0,
                    239,
                ),
                path,
            )
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            plan = run_state.plan_resume_from_checkpoint(ckpt, make_exp(), None)
            self.assertEqual(plan["start_epoch"], 1)
            self.assertEqual(plan["batch_cursor"], 1)
            self.assertEqual(plan["global_step"], 11)
            self.assertEqual(plan["current_perm"], list(range(4656))[::-1])

    def mutate_reject(self, mutator, expectation=run_state.CheckpointError):
        ckpt = self.valid_ckpt()
        mutator(ckpt)
        with self.assertRaises(expectation):
            run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_missing_required_fields_rejected(self):
        for field in (
            "schema_version",
            "checkpoint_type",
            "seed",
            "architecture",
            "optimizer_settings",
            "sampler_policy",
            "loss_masking_contract",
            "epoch_convention",
            "epoch",
            "epoch_completed",
            "global_step",
            "completed_updates",
            "next_eval_epoch",
            "batch_size",
            "n_batches",
            "batch_cursor",
            "config_split_hash",
            "tokenizer_hash",
            "run_identity",
            "run_identity_hash",
            "model_state",
            "optimizer_state",
            "torch_rng_state",
            "numpy_rng_state",
            "python_rng_state",
        ):
            self.mutate_reject(lambda c, f=field: c.pop(f, None))

    def test_missing_permutation_mid_epoch_rejected(self):
        self.mutate_reject(lambda c: c.__setitem__("epoch_permutation", None))

    def test_unknown_future_schema_rejected(self):
        ckpt = self.valid_ckpt()
        ckpt["schema_version"] = 3
        with self.assertRaises(run_state.UnsupportedCheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_nonint_and_bool_schema_rejected(self):
        ckpt = self.valid_ckpt()
        ckpt["schema_version"] = "2"
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())
        ckpt2 = self.valid_ckpt()
        ckpt2["schema_version"] = True
        with self.assertRaises(run_state.CheckpointError):
            run_state.plan_resume_from_checkpoint(ckpt2, make_exp(), None)

    def test_snapshot_type_rejected_as_resume_source(self):
        ckpt = self.valid_ckpt()
        ckpt["checkpoint_type"] = "snapshot"
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_invalid_permutations_rejected(self):
        # duplicate value (perm[0] set equal to perm[1])
        self.mutate_reject(
            lambda c: c["epoch_permutation"].__setitem__(0, c["epoch_permutation"][1])
        )
        self.mutate_reject(
            lambda c: c.__setitem__("epoch_permutation", list(range(4655)))
        )  # wrong length
        self.mutate_reject(
            lambda c: c.__setitem__("epoch_permutation", [0] * 4656)
        )  # duplicates
        self.mutate_reject(
            lambda c: c.__setitem__("epoch_permutation", [-1] + list(range(4655)))
        )  # out of range
        self.mutate_reject(
            lambda c: c.__setitem__("epoch_permutation", [4656] + list(range(1, 4656)))
        )  # out of range

    def test_invalid_cursors_and_counters(self):
        self.mutate_reject(lambda c: c.__setitem__("batch_cursor", 11))
        self.mutate_reject(lambda c: c.__setitem__("batch_cursor", -1))
        self.mutate_reject(lambda c: c.__setitem__("completed_updates", 4))
        self.mutate_reject(lambda c: c.__setitem__("global_step", -1))
        self.mutate_reject(lambda c: c.__setitem__("epoch", True))
        self.mutate_reject(lambda c: c.__setitem__("epoch_completed", 1))

    def test_inconsistent_completion_flags(self):
        # mid-epoch view flipped to completed while cursor < n_batches
        self.mutate_reject(lambda c: c.__setitem__("epoch_completed", True))
        # completed view with cursor < n_batches
        ckpt = self.valid_ckpt(mid_epoch=False)
        ckpt["batch_cursor"] = 2
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_inconsistent_schedule_pointer(self):
        # completed with pointer == epoch -> invalid
        ckpt = self.valid_ckpt(mid_epoch=False)
        ckpt["next_eval_epoch"] = 1
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())
        # in-progress with pointer < epoch -> invalid
        ckpt2 = self.valid_ckpt()
        ckpt2["epoch"] = 3
        ckpt2["next_eval_epoch"] = 2
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt2, make_exp())
        # missing / negative / non-int pointer -> invalid (evaluation must not
        # silently disappear)
        for bad in (-1, None, "x", 1.5):
            ckpt3 = self.valid_ckpt()
            ckpt3["next_eval_epoch"] = bad
            with self.assertRaises(run_state.CheckpointError):
                run_state.validate_current_checkpoint(ckpt3, make_exp())

    def test_identity_and_hash_mismatch(self):
        self.mutate_reject(
            lambda c: c.__setitem__("architecture", dict(c["architecture"], d_model=64))
        )
        self.mutate_reject(
            lambda c: c.__setitem__(
                "sampler_policy", dict(c["sampler_policy"], batch_size=256)
            )
        )
        self.mutate_reject(lambda c: c.__setitem__("config_split_hash", "different"))
        self.mutate_reject(lambda c: c.__setitem__("tokenizer_hash", "different"))
        self.mutate_reject(lambda c: c.__setitem__("run_identity_hash", "0" * 64))
        # stored hash not matching recomputed identity contents
        self.mutate_reject(
            lambda c: c["run_identity"].__setitem__("run_identity_hash", "1" * 64)
        )

    def test_seed_mismatch(self):
        self.mutate_reject(lambda c: c.__setitem__("seed", 1))


# ---------------------------------------------------------------------------
# Legacy checkpoint validation
# ---------------------------------------------------------------------------


class TestLegacyValidation(unittest.TestCase):
    EXP = None

    @classmethod
    def setUpClass(cls):
        cls.EXP = make_exp(seed=0)

    def prov(self, epoch, gs, seed=0):
        exp = self.EXP
        return {
            "seed": seed,
            "global_step": gs,
            "epoch": epoch,
            "checkpoint_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "run_identity": exp["identity"],
            "run_identity_hash": exp["identity"]["run_identity_hash"],
        }

    def test_synthetic_valid_future_pointer_27520_accepted(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        plan = run_state.validate_legacy_checkpoint(
            ckpt,
            self.EXP,
            self.prov(2751, 27520),
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        self.assertEqual(plan["start_epoch"], 2752)
        self.assertEqual(plan["global_step"], 27520)
        self.assertEqual(plan["next_eval_epoch"], 2806)
        self.assertTrue(plan["migrated_legacy"])
        self.assertEqual(plan["current_perm"], None)

    def test_known_pause_28070_stale_pointer_normalized(self):
        ckpt = legacy_ckpt(epoch=2806, global_step=28070, next_eval_epoch=2806)
        plan = run_state.validate_legacy_checkpoint(
            ckpt,
            self.EXP,
            self.prov(2806, 28070),
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        self.assertEqual(plan["start_epoch"], 2807)
        self.assertEqual(plan["next_eval_epoch"], run_state.first_due_eval_epoch(2807))
        self.assertTrue(any("normalized" in n for n in plan["notes"]))

    def test_provenance_required(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(ckpt, self.EXP, None)
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(ckpt, self.EXP, {})

    def test_canonical_digest_aliases_rejected(self):
        h = "a" * 64
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        for alias in ("checkpoint_hash", "sha256", "file_sha256"):
            prov = {
                "seed": 0,
                "global_step": 27520,
                "epoch": 2751,
                alias: h,
                "run_identity": self.EXP["identity"],
                "run_identity_hash": self.EXP["identity"]["run_identity_hash"],
            }
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                run_state.validate_legacy_checkpoint(
                    ckpt, self.EXP, prov, checkpoint_sha256=h
                )
        # conflicting alias alongside canonical key also rejected
        prov2 = {
            "seed": 0,
            "global_step": 27520,
            "epoch": 2751,
            "checkpoint_sha256": h,
            "checkpoint_hash": h,
            "run_identity": self.EXP["identity"],
            "run_identity_hash": self.EXP["identity"]["run_identity_hash"],
        }
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt, self.EXP, prov2, checkpoint_sha256=h
            )

    def test_provenance_mismatch_refused(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520, seed=1),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27510),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2750, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_midepoch_legacy_refused(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27515, next_eval_epoch=2751)
        with self.assertRaises(run_state.LegacyCheckpointRefused) as ctx:
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27515),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        self.assertIn("mid-epoch", str(ctx.exception))

    def test_zero_batches_done_refused(self):
        ckpt = legacy_ckpt(epoch=2752, global_step=27520, next_eval_epoch=2801)
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2752, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_missing_identity_refused(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        del ckpt["epoch"]
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        ckpt2 = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        del ckpt2["global_step"]
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt2,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_batch_idx_contradiction_refused_missing_accepted(self):
        ckpt = legacy_ckpt(
            epoch=2751, global_step=27520, next_eval_epoch=2806, batch_idx=7
        )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )
        ckpt2 = legacy_ckpt(
            epoch=2751, global_step=27520, next_eval_epoch=2806, batch_idx=None
        )
        ckpt2.pop("batch_idx")
        plan = run_state.validate_legacy_checkpoint(
            ckpt2,
            self.EXP,
            self.prov(2751, 27520),
            checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        self.assertEqual(plan["start_epoch"], 2752)

    def test_model_only_legacy_refused(self):
        ckpt = legacy_ckpt(
            epoch=2751, global_step=27520, next_eval_epoch=2806, with_opt=False
        )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_missing_rng_refused(self):
        ckpt = legacy_ckpt(
            epoch=2751, global_step=27520, next_eval_epoch=2806, with_rng=False
        )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_cfg_hash_mismatch_refused(self):
        ckpt = legacy_ckpt(
            epoch=2751, global_step=27520, next_eval_epoch=2806, cfg_hash="other"
        )
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 27520),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_inconsistent_step_epoch_arithmetic_refused(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=99999, next_eval_epoch=2806)
        with self.assertRaises(run_state.LegacyCheckpointRefused):
            run_state.validate_legacy_checkpoint(
                ckpt,
                self.EXP,
                self.prov(2751, 99999),
                checkpoint_sha256="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            )

    def test_future_schema_not_treated_as_legacy(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        ckpt["schema_version"] = 99
        with self.assertRaises(run_state.UnsupportedCheckpointError):
            run_state.plan_resume_from_checkpoint(ckpt, self.EXP, None)
        # a schema-2 label forces the strict current path (never legacy)
        ckpt2 = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2806)
        ckpt2["schema_version"] = 2
        with self.assertRaises(run_state.CheckpointError):
            run_state.plan_resume_from_checkpoint(ckpt2, self.EXP, None)


# ---------------------------------------------------------------------------
# Destination policy (pure + filesystem)
# ---------------------------------------------------------------------------


class TestDestinationPolicy(unittest.TestCase):
    def test_fresh_refuses_logged_metrics_without_checkpoint(self):
        insp = {
            "latest_exists": False,
            "csv_exists": True,
            "csv_header": run_state.EXPECTED_CSV_HEADER,
            "csv_data_rows": 355,
            "csv_last_step": 99090,
            "snapshot_files": [],
            "other_pt_files": [],
        }
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_fresh_destination(insp)

    def test_fresh_refuses_latest_and_conflicting_artifacts(self):
        base = {
            "csv_exists": False,
            "csv_header": None,
            "csv_data_rows": 0,
            "csv_last_step": None,
            "snapshot_files": [],
            "other_pt_files": [],
        }
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_fresh_destination(dict(base, latest_exists=True))
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_fresh_destination(
                dict(base, snapshot_files=["snapshot_step100000.pt"])
            )
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_fresh_destination(
                dict(base, other_pt_files=["seed0_step2000.pt"])
            )
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_fresh_destination(
                dict(
                    base,
                    csv_exists=True,
                    csv_header=["step", "wrong"],
                    csv_data_rows=0,
                    csv_last_step=None,
                )
            )

    def test_fresh_allows_clean_or_header_only(self):
        base = {
            "latest_exists": False,
            "csv_exists": False,
            "csv_header": None,
            "csv_data_rows": 0,
            "csv_last_step": None,
            "snapshot_files": [],
            "other_pt_files": [],
        }
        run_state.validate_fresh_destination(dict(base))
        run_state.validate_fresh_destination(
            dict(base, csv_exists=True, csv_header=run_state.EXPECTED_CSV_HEADER)
        )

    def test_resume_refuses_checkpoint_behind_last_logged_row(self):
        insp = {
            "latest_exists": True,
            "csv_exists": True,
            "csv_header": run_state.EXPECTED_CSV_HEADER,
            "csv_data_rows": 355,
            "csv_last_step": 99090,
            "snapshot_files": [],
            "other_pt_files": [],
        }
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_resume_destination(insp, 98000)
        run_state.validate_resume_destination(insp, 99090)  # equal is OK
        run_state.validate_resume_destination(insp, 99500)

    def test_resume_requires_latest(self):
        insp = {
            "latest_exists": False,
            "csv_exists": False,
            "csv_header": None,
            "csv_data_rows": 0,
            "csv_last_step": None,
            "snapshot_files": [],
            "other_pt_files": [],
        }
        with self.assertRaises(run_state.DestinationError):
            run_state.validate_resume_destination(insp, 10)

    def test_reject_existing_outputs_leaves_files_unchanged(self):
        with TempDirs() as tmp:
            p = os.path.join(tmp, "out.json")
            with open(p, "w") as f:
                f.write("KEEP ME")
            with open(p) as f:
                before = f.read()
            with self.assertRaises(run_state.DestinationError):
                run_state.reject_existing_outputs([p])
            with open(p) as f:
                self.assertEqual(f.read(), before)
            free = os.path.join(tmp, "new.json")
            run_state.reject_existing_outputs([free])  # absent -> fine

    def test_invocation_args(self):
        run_state.validate_invocation_args(100000, None)
        run_state.validate_invocation_args(100000, 2000)
        for bad in (0, -5):
            with self.assertRaises(run_state.DestinationError):
                run_state.validate_invocation_args(bad, None)
            with self.assertRaises(run_state.DestinationError):
                run_state.validate_invocation_args(1000, bad)


class TestDestinationFilesystem(unittest.TestCase):
    def write_csv(self, path, rows):
        import csv as _csv

        with open(path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(run_state.EXPECTED_CSV_HEADER)
            for r in rows:
                w.writerow(r)

    def test_inspect_and_fresh_refusal_mirrors_retained_tree(self):
        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(os.path.join(rd, "snapshots"))
            self.write_csv(
                os.path.join(rd, "metrics.csv"),
                [(10, 0, 3.1, 0.0, 3.2, 0.0), (99090, 9908, 0.7, 35.9, 0.7, 29.1)],
            )
            listing_before = sorted(os.listdir(tmp))
            insp = train.inspect_destination(
                rd,
                os.path.join(rd, "metrics.csv"),
                os.path.join(rd, "latest.pt"),
                os.path.join(rd, "snapshots"),
            )
            self.assertEqual(insp["csv_data_rows"], 2)
            self.assertEqual(insp["csv_last_step"], 99090)
            self.assertFalse(insp["latest_exists"])
            with self.assertRaises(run_state.DestinationError):
                run_state.validate_fresh_destination(insp)
            # refusal happened without any file change
            self.assertEqual(sorted(os.listdir(tmp)), listing_before)

    def test_resolve_fresh_succeeds_in_unused_dir(self):
        with TempDirs() as tmp:
            rd = os.path.join(tmp, "future_run_seed0")
            mode, plan, insp, ckpt = train.resolve_run_mode(
                rd,
                os.path.join(rd, "metrics.csv"),
                os.path.join(rd, "latest.pt"),
                os.path.join(rd, "snapshots"),
                make_exp(),
                None,
            )
            self.assertEqual(mode, "fresh")
            self.assertEqual(plan["start_epoch"], 0)
            self.assertEqual(plan["next_eval_epoch"], 0)

    def test_resolve_resume_consistent_and_behind(self):
        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(rd)
            csv_path = os.path.join(rd, "metrics.csv")
            latest_path = os.path.join(rd, "latest.pt")
            self.write_csv(csv_path, [(10, 0, 3.0, 0.0, 3.1, 0.0)])
            view = run_state.save_view(0, list(range(4656)), 10, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p,
                    FakeModel(),
                    FakeOpt(),
                    e_s,
                    10,
                    1,
                    torch.device("cpu"),
                    train.rng_capture(),
                    "cfg-hash-test",
                    "tok-hash-test",
                    make_exp()["identity"],
                    p_s,
                    c_s,
                    10,
                    done,
                    0,
                    239,
                ),
                latest_path,
            )
            mode, plan, insp, ckpt = train.resolve_run_mode(
                rd,
                csv_path,
                latest_path,
                os.path.join(rd, "snapshots"),
                make_exp(),
                None,
            )
            self.assertEqual(mode, "resume")
            self.assertEqual(plan["global_step"], 10)
            self.assertEqual(plan["start_epoch"], 1)
            # checkpoint behind the last logged row -> fail closed
            self.write_csv(
                csv_path, [(10, 0, 3.0, 0.0, 3.1, 0.0), (20, 1, 2.0, 1.0, 2.1, 1.0)]
            )
            with self.assertRaises(run_state.DestinationError):
                train.resolve_run_mode(
                    rd,
                    csv_path,
                    latest_path,
                    os.path.join(rd, "snapshots"),
                    make_exp(),
                    None,
                )

    def test_snapshot_at_latest_refused_not_fresh(self):
        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(rd)
            csv_path = os.path.join(rd, "metrics.csv")
            self.write_csv(csv_path, [(10, 0, 3.0, 0.0, 3.1, 0.0)])
            latest = os.path.join(rd, "latest.pt")
            torch.save(
                {"state_dict": {"w": torch.zeros(2)}, "step": 10, "seed": 0}, latest
            )
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                train.resolve_run_mode(
                    rd,
                    csv_path,
                    latest,
                    os.path.join(rd, "snapshots"),
                    make_exp(),
                    None,
                )


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------


class TestAtomicWrites(unittest.TestCase):
    def test_success_replaces(self):
        with TempDirs() as tmp:
            p = os.path.join(tmp, "f.pt")
            torch.save({"v": 1}, p)
            train._atomic_save(lambda q: torch.save({"v": 2}, q), p)
            self.assertEqual(torch.load(p, weights_only=True)["v"], 2)

    def test_interrupted_write_preserves_prior_good_file(self):
        with TempDirs() as tmp:
            p = os.path.join(tmp, "f.pt")
            torch.save({"v": 1}, p)

            def failing(q):
                torch.save({"v": 2}, q)
                raise OSError("simulated interruption")

            with self.assertRaises(OSError):
                train._atomic_save(failing, p)
            self.assertEqual(torch.load(p, weights_only=True)["v"], 1)
            self.assertFalse(os.path.exists(p + ".tmp"))


# ---------------------------------------------------------------------------
# Evaluator + analysis failure handling
# ---------------------------------------------------------------------------


class TestEvaluatorSafety(unittest.TestCase):
    def test_missing_snapshot_is_explicit_unmeasured(self):
        import evaluate_endpoints as ee

        with TempDirs() as tmp:
            missing = os.path.join(tmp, "does_not_exist.pt")
            r = ee.eval_seed(0, missing, None, None, 239, torch.device("cpu"))
            self.assertEqual(r["status"], "unmeasured")
            self.assertIn("file not found", r["reason"])

    def test_wrong_identity_snapshot_unmeasured(self):
        import evaluate_endpoints as ee

        with TempDirs() as tmp:
            p = os.path.join(tmp, "snap.pt")
            torch.save({"state_dict": {}, "step": 999, "seed": 0}, p)
            r = ee.eval_seed(0, p, None, None, 239, torch.device("cpu"))
            self.assertEqual(r["status"], "unmeasured")
            self.assertIn("step", r["reason"])
            p2 = os.path.join(tmp, "snap2.pt")
            torch.save({"state_dict": {}, "step": 100000, "seed": 1}, p2)
            r2 = ee.eval_seed(0, p2, None, None, 239, torch.device("cpu"))
            self.assertEqual(r2["status"], "unmeasured")
            self.assertIn("seed", r2["reason"])

    def test_out_collision_guards(self):
        import evaluate_endpoints as ee  # noqa: F401 (import must stay inert)

        with TempDirs() as tmp:
            existing = os.path.join(tmp, "endpoint_metrics.json")
            with open(existing, "w") as f:
                f.write("HISTORICAL")
            # the exact guard evaluate_endpoints.main() applies to --out
            with self.assertRaises(run_state.DestinationError):
                run_state.reject_existing_outputs([existing])
            with open(existing) as f:
                self.assertEqual(f.read(), "HISTORICAL")


class TestAnalysisSafety(unittest.TestCase):
    def endpoint(self, seed, acc=90.0, step=100000, status="measured"):
        return {
            "seed": seed,
            "snapshot_step": step,
            "snapshot_seed": seed,
            "status": status,
            "train": {
                "loss": 1.0,
                "accuracy": acc,
                "n_equations": 4656,
                "n_tokens": 9312,
                "correct_equations": 4190,
            },
            "val": {
                "loss": 1.0,
                "accuracy": acc,
                "n_equations": 4656,
                "n_tokens": 9312,
                "correct_equations": 4190,
            },
        }

    def rows(self):
        def row(step, epoch, v, t=99.0):
            return {
                "step": step,
                "epoch": epoch,
                "train_loss": 1.0,
                "train_acc": t,
                "val_loss": 1.0,
                "val_acc": v,
            }

        return [row(10, 0, 0.0), row(3760, 375, 99.5), row(4300, 429, 100.0)]

    def test_validate_endpoint_entry(self):
        import analyze

        entry, reason = analyze.validate_endpoint_entry(self.endpoint(0), 0)
        self.assertIsNotNone(entry)
        for bad, why in [
            (self.endpoint(0, status="unmeasured"), "status"),
            (self.endpoint(1), "seed"),
            (self.endpoint(0, step=99000), "step"),
        ]:
            out, r = analyze.validate_endpoint_entry(bad, 0)
            self.assertIsNone(out)
        broken = self.endpoint(0)
        del broken["val"]["accuracy"]
        out, r = analyze.validate_endpoint_entry(broken, 0)
        self.assertIsNone(out)
        self.assertIn("accuracy", r)

    def test_partial_endpoint_set_never_fabricates_three_seed_mean(self):
        import analyze

        endpoints = {
            "endpoints_100000": {
                "0": self.endpoint(0, 36.0180412371134),
                "1": self.endpoint(1, 100.0),
                "2": self.endpoint(2, 99.97852233676976, status="unmeasured"),
            }
        }
        values, coverage = analyze.endpoint_val_summary(endpoints)
        self.assertEqual(coverage["n"], 2)
        self.assertFalse(coverage["complete"])
        summary = analyze.build_summary(
            {0: self.rows(), 1: self.rows(), 2: self.rows()}, endpoints
        )
        self.assertIsNone(summary["mean_endpoint_val_acc"])
        self.assertIsNotNone(summary["mean_endpoint_val_acc_partial"])
        self.assertAlmostEqual(
            summary["mean_endpoint_val_acc_partial"]["value"],
            (36.0180412371134 + 100.0) / 2.0,
        )
        self.assertIn(
            "NOT the three-seed mean", summary["mean_endpoint_val_acc_partial"]["note"]
        )

    def test_full_endpoint_set_mean(self):
        import analyze

        endpoints = {
            "endpoints_100000": {
                "0": self.endpoint(0, 36.0180412371134),
                "1": self.endpoint(1, 100.0),
                "2": self.endpoint(2, 99.97852233676976),
            }
        }
        summary = analyze.build_summary(
            {0: self.rows(), 1: self.rows(), 2: self.rows()}, endpoints
        )
        self.assertTrue(summary["endpoint_val_coverage"]["complete"])
        self.assertAlmostEqual(
            summary["mean_endpoint_val_acc"],
            (36.0180412371134 + 100.0 + 99.97852233676976) / 3.0,
        )
        self.assertIsNone(summary["mean_endpoint_val_acc_partial"])

    def test_missing_endpoints_file(self):
        import analyze

        summary = analyze.build_summary(
            {0: self.rows(), 1: self.rows(), 2: self.rows()}, None
        )
        self.assertIsNone(summary["mean_endpoint_val_acc"])
        self.assertEqual(summary["endpoint_val_coverage"]["n"], 0)
        for s in "012":
            self.assertIsNone(summary["seeds"][s]["endpoint_100k"])
            self.assertIsNotNone(summary["seeds"][s]["endpoint_100k_unmeasured_reason"])

    def test_thresholds_are_first_observed_rows(self):
        import analyze

        summary = analyze.build_summary(
            {0: self.rows(), 1: self.rows(), 2: self.rows()}, None
        )
        s0 = summary["seeds"]["0"]
        self.assertEqual(s0["first_val_ge_99"], 3760)
        self.assertEqual(s0["first_val_ge_100"], 4300)
        self.assertEqual(s0["best_val_acc_logged"], 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Real production artifact-obligation writer (not recorder call-counts)
# ---------------------------------------------------------------------------


class TestObligationWriterReal(unittest.TestCase):
    def test_snapshot_and_seed0_backup_written_and_idempotent(self):
        with TempDirs() as tmp:
            run_dir = os.path.join(tmp, "seed0")
            snap_dir = os.path.join(run_dir, "snapshots")
            os.makedirs(snap_dir)
            state = make_state(
                epoch=200,
                current_perm=None,
                batch_cursor=0,
                global_step=2000,
                next_eval_epoch=201,
            )
            model = FakeModel()
            opt = FakeOpt()
            exp = make_exp()

            def full_saver_for_view(view):
                rngs = train.rng_capture()
                return train.make_full_saver(
                    state,
                    model,
                    opt,
                    view,
                    torch.device("cpu"),
                    rngs,
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    10,
                    0,
                    239,
                )

            writer = train.make_obligation_writer(
                state, model, run_dir, snap_dir, 0, 10, full_saver_for_view
            )
            writer()
            snap = os.path.join(snap_dir, "snapshot_step2000.pt")
            backup = os.path.join(run_dir, "seed0_step2000.pt")
            self.assertTrue(os.path.exists(snap))
            self.assertTrue(os.path.exists(backup))
            snap_payload = torch.load(snap, weights_only=True)
            self.assertEqual(snap_payload["step"], 2000)
            self.assertEqual(snap_payload["seed"], 0)
            self.assertEqual(set(snap_payload["state_dict"].keys()), {"w"})
            # idempotent: existing files are not rewritten
            snap_size = os.path.getsize(snap)
            backup_size = os.path.getsize(backup)
            writer()
            self.assertEqual(os.path.getsize(snap), snap_size)
            self.assertEqual(os.path.getsize(backup), backup_size)


# ---------------------------------------------------------------------------
# Actual main() no-op orchestration (read-only destination)
# ---------------------------------------------------------------------------


class TestMainNoOpOrchestration(unittest.TestCase):
    def test_completed_target_exits_without_mutation_or_workload(self):
        import hashlib
        import sys
        from unittest.mock import patch

        with TempDirs() as tmp:
            run_dir = os.path.join(tmp, "seed0")
            os.makedirs(run_dir)
            exp = make_exp()
            ckpt_path = os.path.join(run_dir, "latest.pt")
            st = make_state(
                epoch=9909,
                current_perm=None,
                batch_cursor=0,
                global_step=99090,
                next_eval_epoch=9971,
            )
            view = run_state.save_view(st.epoch, st.current_perm, st.batch_cursor, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p,
                    FakeModel(),
                    FakeOpt(),
                    e_s,
                    99090,
                    9971,
                    torch.device("cpu"),
                    train.rng_capture(),
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    p_s,
                    c_s,
                    10,
                    done,
                    0,
                    239,
                ),
                ckpt_path,
            )

            def inventory(root):
                out = {}
                for dp, _, files in os.walk(root):
                    for f in files:
                        fp = os.path.join(dp, f)
                        out[os.path.relpath(fp, root)] = hashlib.sha256(
                            open(fp, "rb").read()
                        ).hexdigest()
                return out

            before = inventory(run_dir)
            fake_data = (torch.zeros(4656, 17), torch.zeros(100, 17), ["t"] * 239)

            def sentinel(*a, **k):
                raise AssertionError("workload constructor must not be called")

            old_argv = sys.argv
            sys.argv = [
                "train.py",
                "--seed",
                "0",
                "--steps",
                "99090",
                "--run-dir",
                run_dir,
            ]
            try:
                with (
                    patch("train.build_split", return_value=fake_data),
                    patch("train.hash_config_split", return_value="cfg-hash-test"),
                    patch("train.tokenizer_hash", return_value="tok-hash-test"),
                    patch("train.rng_capture", side_effect=sentinel),
                    patch("train.load_rng", side_effect=sentinel),
                    patch("train.make_obligation_writer", side_effect=sentinel),
                    patch("run_state.run_training_loop", side_effect=sentinel),
                    patch("train.Transformer", side_effect=sentinel),
                    patch("torch.optim.AdamW", side_effect=sentinel),
                ):
                    train.main()
            finally:
                sys.argv = old_argv

            after = inventory(run_dir)
            self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# R1: real stop path through shared loop + real obligation writer
# ---------------------------------------------------------------------------


class TestRealStopPath(unittest.TestCase):
    def test_stop_at_boundary_writes_all_three_artifacts(self):
        import os

        with TempDirs() as tmp:
            run_dir = os.path.join(tmp, "seed0")
            snap_dir = os.path.join(run_dir, "snapshots")
            os.makedirs(snap_dir)
            perm = list(range(4656))
            state = make_state(
                epoch=199,
                current_perm=perm,
                batch_cursor=9,
                global_step=1999,
                next_eval_epoch=199,
            )
            model = FakeModel()
            opt = FakeOpt()
            exp = make_exp()

            def full_saver_for_view(view):
                rngs = train.rng_capture()
                return train.make_full_saver(
                    state,
                    model,
                    opt,
                    view,
                    torch.device("cpu"),
                    rngs,
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    10,
                    0,
                    239,
                )

            def save_latest(view):
                train._atomic_save(
                    full_saver_for_view(view), os.path.join(run_dir, "latest.pt")
                )

            writer = train.make_obligation_writer(
                state, model, run_dir, snap_dir, 0, 10, full_saver_for_view
            )
            rec = Recorder(stop_after=1)

            def stop_requested():
                return rec.updates >= 1

            env = make_env(n_train=4656, batch_size=512, target_steps=100000)
            reason = run_state.run_training_loop(
                state,
                env,
                rec.exec_batch,
                lambda: rec.draw_perm(4656),
                rec.eval_and_log,
                save_latest,
                writer,
                lambda: rec.updates >= 1,
            )
            self.assertEqual(reason, "stopped")
            self.assertEqual(rec.updates, 1)
            self.assertEqual(rec.evals, [(199, 2000)])
            # artifacts exist
            snap = os.path.join(snap_dir, "snapshot_step2000.pt")
            backup = os.path.join(run_dir, "seed0_step2000.pt")
            latest = os.path.join(run_dir, "latest.pt")
            self.assertTrue(os.path.exists(snap))
            self.assertTrue(os.path.exists(backup))
            self.assertTrue(os.path.exists(latest))
            sp = torch.load(snap, weights_only=True)
            self.assertEqual(sp["step"], 2000)
            self.assertEqual(sp["seed"], 0)
            # idempotence by bytes
            import hashlib

            def h(p):
                return hashlib.sha256(open(p, "rb").read()).hexdigest()

            before = {p: h(p) for p in (snap, backup, latest)}
            old_stop = train._stop_requested
            train._stop_requested = True
            try:
                writer()
            finally:
                train._stop_requested = old_stop
            after = {p: h(p) for p in (snap, backup, latest)}
            self.assertEqual(before, after)


# ---------------------------------------------------------------------------
# R2: continuing recovery before first update
# ---------------------------------------------------------------------------


class TestContinuingRecovery(unittest.TestCase):
    def test_recovery_artifact_exists_before_first_update(self):
        import os

        with TempDirs() as tmp:
            run_dir = os.path.join(tmp, "seed0")
            snap_dir = os.path.join(run_dir, "snapshots")
            os.makedirs(snap_dir)
            perm = list(range(4656))
            state = make_state(
                epoch=200,
                current_perm=None,
                batch_cursor=0,
                global_step=2000,
                next_eval_epoch=200,
            )
            model = FakeModel()
            opt = FakeOpt()
            exp = make_exp()

            def full_saver_for_view(view):
                rngs = train.rng_capture()
                return train.make_full_saver(
                    state,
                    model,
                    opt,
                    view,
                    torch.device("cpu"),
                    rngs,
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    10,
                    0,
                    239,
                )

            writer = train.make_obligation_writer(
                state, model, run_dir, snap_dir, 0, 10, full_saver_for_view
            )
            snap = os.path.join(snap_dir, "snapshot_step2000.pt")
            self.assertFalse(os.path.exists(snap))
            # production recovery before loop
            writer()
            self.assertTrue(os.path.exists(snap))

            seen = {"value": False}

            def exec_batch(idx):
                seen["value"] = os.path.exists(snap)
                rec.exec_batch(idx)

            rec = Recorder()
            env = make_env(n_train=4656, batch_size=512, target_steps=2001)
            run_state.run_training_loop(
                state,
                env,
                exec_batch,
                lambda: rec.draw_perm(4656),
                rec.eval_and_log,
                lambda v: None,
                writer,
                lambda: False,
            )
            self.assertTrue(seen["value"])


class TestResolveSuppliedDigest(unittest.TestCase):
    def _setup(self, tmp):
        import os, hashlib

        rd = os.path.join(tmp, "seed0")
        os.makedirs(rd)
        csv_path = os.path.join(rd, "metrics.csv")
        latest_path = os.path.join(rd, "latest.pt")
        snap_dir = os.path.join(rd, "snapshots")
        os.makedirs(snap_dir)
        with open(latest_path, "wb") as f:
            f.write(b"checkpoint-bytes")
        real_hash = hashlib.sha256(open(latest_path, "rb").read()).hexdigest()
        # close file handle by recompute via context
        with open(latest_path, "rb") as f:
            real_hash = hashlib.sha256(f.read()).hexdigest()
        return rd, csv_path, latest_path, snap_dir, real_hash

    def _prov(self, rd, real_hash, overrides=None):
        exp = make_exp()
        prov = {
            "seed": 0,
            "global_step": 1,
            "epoch": 0,
            "checkpoint_sha256": real_hash,
            "run_identity": exp["identity"],
            "run_identity_hash": exp["identity"]["run_identity_hash"],
        }
        if overrides:
            prov.update(overrides)
        return prov, exp

    def test_bad_supplied_sidecars_block_load(self):
        from unittest.mock import patch

        bad_cases = [
            ({}, "missing"),
            ({"checkpoint_sha256": None}, "null"),
            ({"checkpoint_sha256": "not-hex"}, "malformed"),
            ({"checkpoint_sha256": "b" * 64}, "wrong"),
            ({"checkpoint_hash": "a" * 64}, "alias-only"),
            ({"checkpoint_sha256": "a" * 64, "sha256": "a" * 64}, "conflicting-alias"),
            ("not-a-dict", "invalid-type"),
        ]
        for overrides, label in bad_cases:
            with TempDirs() as tmp:
                rd, csv_path, latest_path, snap_dir, real_hash = self._setup(tmp)
                if overrides == "not-a-dict":
                    prov = "not-a-dict"
                else:
                    prov, exp = self._prov(rd, real_hash, overrides)
                    if label == "missing":
                        prov.pop("checkpoint_sha256", None)
                mock_load = patch("torch.load")
                with mock_load as ml:
                    ml.side_effect = RuntimeError("must not load")
                    with self.assertRaises(run_state.LegacyCheckpointRefused):
                        train.resolve_run_mode(
                            rd, csv_path, latest_path, snap_dir, make_exp(), prov
                        )
                    ml.assert_not_called()

    def test_correct_supplied_digest_resolves(self):
        from unittest.mock import patch
        import torch

        with TempDirs() as tmp:
            rd, csv_path, latest_path, snap_dir, real_hash = self._setup(tmp)
            # write a valid current-schema checkpoint
            exp = make_exp()
            ckpt_path = latest_path
            st = make_state(
                epoch=0,
                current_perm=list(range(4656)),
                batch_cursor=1,
                global_step=1,
                next_eval_epoch=1,
            )
            view = run_state.save_view(st.epoch, st.current_perm, st.batch_cursor, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p,
                    FakeModel(),
                    FakeOpt(),
                    e_s,
                    1,
                    1,
                    torch.device("cpu"),
                    train.rng_capture(),
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    p_s,
                    c_s,
                    10,
                    done,
                    0,
                    239,
                ),
                ckpt_path,
            )
            real_hash = (
                __import__("hashlib").sha256(open(ckpt_path, "rb").read()).hexdigest()
            )
            prov = {
                "seed": 0,
                "global_step": 1,
                "epoch": 0,
                "checkpoint_sha256": real_hash,
                "run_identity": exp["identity"],
                "run_identity_hash": exp["identity"]["run_identity_hash"],
            }
            mode, plan, insp, ckpt = train.resolve_run_mode(
                rd, csv_path, latest_path, snap_dir, make_exp(), prov
            )
            self.assertEqual(mode, "resume")

    def test_no_sidecar_trusted_local_still_works(self):
        import os, torch

        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(rd)
            csv_path = os.path.join(rd, "metrics.csv")
            latest_path = os.path.join(rd, "latest.pt")
            snap_dir = os.path.join(rd, "snapshots")
            os.makedirs(snap_dir)
            exp = make_exp()
            st = make_state(
                epoch=0,
                current_perm=list(range(4656)),
                batch_cursor=1,
                global_step=1,
                next_eval_epoch=1,
            )
            view = run_state.save_view(st.epoch, st.current_perm, st.batch_cursor, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p,
                    FakeModel(),
                    FakeOpt(),
                    e_s,
                    1,
                    1,
                    torch.device("cpu"),
                    train.rng_capture(),
                    "cfg-hash-test",
                    "tok-hash-test",
                    exp["identity"],
                    p_s,
                    c_s,
                    10,
                    done,
                    0,
                    239,
                ),
                latest_path,
            )
            mode, plan, insp, ckpt = train.resolve_run_mode(
                rd, csv_path, latest_path, snap_dir, make_exp(), None
            )
            self.assertEqual(mode, "resume")

    def test_identity_negatives(self):
        import os, torch, hashlib

        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(rd)
            csv_path = os.path.join(rd, "metrics.csv")
            latest_path = os.path.join(rd, "latest.pt")
            snap_dir = os.path.join(rd, "snapshots")
            os.makedirs(snap_dir)
            exp = make_exp()
            ckpt = legacy_ckpt(epoch=0, global_step=10, next_eval_epoch=1)
            # make boundary-valid: epoch0 global_step10 => 1*10 boundary, but legacy requires >=1 and bd_implied=10 ok
            torch.save(ckpt, latest_path)
            real_hash = hashlib.sha256(open(latest_path, "rb").read()).hexdigest()
            base_prov = {
                "seed": 0,
                "global_step": 10,
                "epoch": 0,
                "checkpoint_sha256": real_hash,
                "run_identity": exp["identity"],
                "run_identity_hash": exp["identity"]["run_identity_hash"],
            }
            # missing identity
            p1 = dict(base_prov)
            p1.pop("run_identity")
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                train.resolve_run_mode(
                    rd, csv_path, latest_path, snap_dir, make_exp(), p1
                )
            # missing hash
            p2 = dict(base_prov)
            p2.pop("run_identity_hash")
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                train.resolve_run_mode(
                    rd, csv_path, latest_path, snap_dir, make_exp(), p2
                )
            # wrong identity
            p3 = dict(base_prov)
            p3["run_identity"] = dict(exp["identity"], seed=1)
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                train.resolve_run_mode(
                    rd, csv_path, latest_path, snap_dir, make_exp(), p3
                )
            # wrong hash
            p4 = dict(base_prov)
            p4["run_identity_hash"] = "b" * 64
            with self.assertRaises(run_state.LegacyCheckpointRefused):
                train.resolve_run_mode(
                    rd, csv_path, latest_path, snap_dir, make_exp(), p4
                )


class TestStrictArithmeticCoverage(unittest.TestCase):
    def test_inprogress_arithmetic_mismatch_equal_counters(self):
        ckpt = TestCurrentSchemaValidation().valid_ckpt() if False else None
        # reuse helper from existing class
        t = TestCurrentSchemaValidation()
        base = t.valid_ckpt()
        # equal global/completed counters but wrong step arithmetic
        base["global_step"] = 5
        base["completed_updates"] = 5
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(base, make_exp())

    def test_completed_arithmetic_mismatch_equal_counters(self):
        t = TestCurrentSchemaValidation()
        ckpt = t.valid_ckpt(mid_epoch=False)
        ckpt["next_eval_epoch"] = 3
        ckpt["global_step"] = 15
        ckpt["completed_updates"] = 15
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_zero_step_inprogress_coherent(self):
        t = TestCurrentSchemaValidation()
        ckpt = t.valid_ckpt()
        # build coherent zero-step state
        ckpt["epoch"] = 0
        ckpt["epoch_completed"] = False
        ckpt["global_step"] = 0
        ckpt["completed_updates"] = 0
        ckpt["batch_cursor"] = 0
        ckpt["next_eval_epoch"] = 0
        ckpt["epoch_permutation"] = list(range(4656))
        # arithmetic: 0*10+0=0 ok
        run_state.validate_current_checkpoint(ckpt, make_exp())

    def test_mps_missing_and_empty_rejected_cpu_allowed(self):
        t = TestCurrentSchemaValidation()
        ckpt = t.valid_ckpt()
        ckpt["device"] = "mps"
        ckpt["mps_rng_state"] = None
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt, make_exp())
        ckpt2 = t.valid_ckpt()
        ckpt2["device"] = "mps"
        ckpt2["mps_rng_state"] = torch.zeros(0, dtype=torch.uint8)
        with self.assertRaises(run_state.CheckpointError):
            run_state.validate_current_checkpoint(ckpt2, make_exp())
        # CPU may omit
        ckpt3 = t.valid_ckpt()
        ckpt3["device"] = "cpu"
        ckpt3["mps_rng_state"] = None
        run_state.validate_current_checkpoint(ckpt3, make_exp())


# ---------------------------------------------------------------------------
# R5: synthetic historical-field fixtures
# ---------------------------------------------------------------------------


class TestSyntheticHistoricalFields(unittest.TestCase):
    def prov(self, epoch, gs):
        exp = make_exp(seed=0)
        return {
            "seed": 0,
            "global_step": gs,
            "epoch": epoch,
            "checkpoint_sha256": "a" * 64,
            "run_identity": exp["identity"],
            "run_identity_hash": exp["identity"]["run_identity_hash"],
        }

    def test_synthetic_27520_fields(self):
        ckpt = legacy_ckpt(epoch=2751, global_step=27520, next_eval_epoch=2751)
        plan = run_state.validate_legacy_checkpoint(
            ckpt,
            make_exp(seed=0),
            self.prov(2751, 27520),
            checkpoint_sha256="a" * 64,
        )
        self.assertEqual(plan["start_epoch"], 2752)
        self.assertEqual(plan["next_eval_epoch"], run_state.first_due_eval_epoch(2752))

    def test_synthetic_28070_fields(self):
        ckpt = legacy_ckpt(epoch=2806, global_step=28070, next_eval_epoch=2806)
        plan = run_state.validate_legacy_checkpoint(
            ckpt,
            make_exp(seed=0),
            self.prov(2806, 28070),
            checkpoint_sha256="a" * 64,
        )
        self.assertEqual(plan["start_epoch"], 2807)
        self.assertEqual(plan["next_eval_epoch"], run_state.first_due_eval_epoch(2807))


# ---------------------------------------------------------------------------
# B4: stale eval pointer -> serialization -> validation -> resume
# ---------------------------------------------------------------------------


class TestCatchupRoundtrip(unittest.TestCase):
    def test_stale_pointer_serializes_validates_resumes(self):
        import os, torch
        with TempDirs() as tmp:
            rd = os.path.join(tmp, "seed0")
            os.makedirs(rd)
            latest_path = os.path.join(rd, "latest.pt")
            exp = make_exp()
            # in-progress epoch 2, cursor 3, stale pointer 1
            perm = list(range(4656))[::-1]
            st = make_state(epoch=2, current_perm=perm, batch_cursor=3,
                            global_step=2*10+3, next_eval_epoch=1)
            view = run_state.save_view(st.epoch, st.current_perm, st.batch_cursor, 10)
            e_s, p_s, c_s, done = view
            train._atomic_save(
                lambda p: train.save_ckpt(
                    p, FakeModel(), FakeOpt(), e_s, st.global_step, 1,
                    torch.device("cpu"), train.rng_capture(), "cfg-hash-test",
                    "tok-hash-test", exp["identity"], p_s, c_s, 10, done, 0, 239,
                ),
                latest_path,
            )
            ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
            # fix pointer to stale 1 in ckpt
            ckpt["next_eval_epoch"] = 1
            torch.save(ckpt, latest_path)
            # validation should reject stale pointer
            with self.assertRaises(run_state.CheckpointError):
                run_state.validate_current_checkpoint(ckpt, exp)
            # now set coherent pointer and re-save
            ckpt["next_eval_epoch"] = run_state.first_due_eval_epoch(2)
            torch.save(ckpt, latest_path)
            ckpt2 = torch.load(latest_path, map_location="cpu", weights_only=False)
            v = run_state.validate_current_checkpoint(ckpt2, exp)
            plan = run_state.plan_current_resume(v)
            self.assertGreaterEqual(plan["next_eval_epoch"], 2)
