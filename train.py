"""Grokking trainer with explicit pause/resume state semantics.

State machine invariants (implemented in run_state.py, shared with tests):
- `run_state.save_view` maps live state onto exactly two unambiguous checkpoint
  states: epoch-completed (cursor == n_batches) or epoch-in-progress
  (0 <= cursor < n_batches with the epoch permutation stored). An epoch is
  advanced and its permutation discarded only after all its minibatches and
  required bookkeeping are complete.
- Fresh runs start at the epoch-0 evaluation cadence (next_eval_epoch = 0);
  due evaluations use catch-up comparison (epoch >= next_eval_epoch), so a
  stale pointer can never silently end evaluation.
- Segment/target limits and signal stops preserve an unfinished epoch exactly
  like a mid-epoch state; post-update snapshot obligations run before a stop
  can break, so a stop exactly on a snapshot step still writes the snapshot.
- Full checkpoints and snapshots are written atomically.
- Resume validation is strict: identity, counters, permutation, cursor, RNG
  and schedule-pointer consistency are all required. Legacy checkpoints are a
  separate explicit path gated by global-step/epoch arithmetic and a trusted
  provenance record.
- Destination checks run BEFORE any directory/CSV/checkpoint mutation:
  retained metrics without a checkpoint are never appended to by a fresh run,
  and a checkpoint behind the last logged row fails closed.
"""

import argparse, csv, hashlib, json, os, random, signal, time, sys
import numpy as np
import torch
import torch.nn.functional as F

import run_state
from data import build_split, ArithmeticTokenizer
from model import Transformer

SNAPSHOT_STEPS = [100, 200, 500, 1000, 2000, 5000] + list(range(10000, 100001, 10000))

_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def full_eval(model, data, device):
    model.eval()
    try:
        with torch.no_grad():
            x = data[:, :-1].to(device)
            y = data[:, 1:].to(device)
            logits = model(x).transpose(-2, -1)
            eq_token_index = ArithmeticTokenizer.get_tokens().index("=")
            eq_pos = int(
                torch.nonzero(y[0] == eq_token_index, as_tuple=False).squeeze()
            )
            y_rhs = y[:, eq_pos + 1 :]
            logits_rhs = logits[:, :, eq_pos + 1 :]
            loss = F.cross_entropy(logits_rhs, y_rhs, reduction="mean")
            pred = logits_rhs.argmax(dim=1)
            row_acc = (pred == y_rhs).all(dim=1).float().mean() * 100.0
        return float(loss.item()), float(row_acc.item())
    finally:
        model.train()


def tokenizer_hash(tokens):
    h = hashlib.sha256()
    h.update(repr(list(tokens)).encode())
    return h.hexdigest()


def hash_config_split(train_data, val_data):
    # Single definition shared with the correction supplement and the evaluator
    # (run_state.config_prefix_five_field): 5-field config prefix + int64
    # train bytes + int64 val bytes.
    h = hashlib.sha256()
    h.update(run_state.config_prefix_five_field().encode())
    h.update(train_data.numpy().tobytes())
    h.update(val_data.numpy().tobytes())
    return h.hexdigest()


def rng_capture():
    return {
        "torch": torch.get_rng_state(),
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def load_rng(rng_states):
    torch.set_rng_state(rng_states["torch"].detach().cpu().byte().contiguous())
    if rng_states.get("mps") is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng_states["mps"].detach().cpu().byte().contiguous())
    np.random.set_state(rng_states["numpy"])
    random.setstate(rng_states["python"])


def save_ckpt(
    path,
    model,
    optimizer,
    epoch,
    global_step,
    next_eval_epoch,
    device,
    rng_states,
    cfg_hash,
    tok_hash,
    identity,
    current_perm,
    batch_cursor,
    n_batches,
    epoch_completed,
    seed,
    vocab,
    migrated_legacy=False,
):
    """Serialize one full checkpoint from an explicit, already-normalized view.

    `epoch`/`current_perm`/`batch_cursor`/`epoch_completed` must come from
    run_state.save_view so only the two unambiguous states can be written.
    """
    ckpt = {
        "schema_version": run_state.CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_type": "full",
        "seed": seed,
        "architecture": identity["architecture"],
        "optimizer_settings": identity["optimizer"],
        "sampler_policy": identity["sampler"],
        "loss_masking_contract": identity["loss_masking"],
        "modulus": identity["modulus"],
        "train_pct": identity["train_pct"],
        "epoch_convention": run_state.EPOCH_CONVENTION,
        "epoch": epoch,
        "epoch_completed": bool(epoch_completed),
        "completed_updates": global_step,
        "global_step": global_step,
        "next_eval_epoch": next_eval_epoch,
        "batch_size": run_state.EXPECTED_BATCH_SIZE,
        "n_batches": n_batches,
        "batch_cursor": batch_cursor,
        "epoch_permutation": (list(current_perm) if current_perm is not None else None),
        "torch_rng_state": rng_states["torch"].byte().contiguous(),
        "mps_rng_state": (
            rng_states.get("mps").byte().contiguous()
            if rng_states.get("mps") is not None
            else None
        ),
        "numpy_rng_state": rng_states["numpy"],
        "python_rng_state": rng_states["python"],
        "config_split_hash": cfg_hash,
        "tokenizer_hash": tok_hash,
        "run_identity": identity,
        "run_identity_hash": identity["run_identity_hash"],
        "migrated_legacy": bool(migrated_legacy),
        "device": str(device),
        "lib_versions": {"torch": torch.__version__, "numpy": np.__version__},
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    torch.save(ckpt, path)


def _atomic_save(save_fn, path):
    """Write via a temp file in the same directory, then atomically replace.

    If serialization fails, the previous file (if any) is preserved and the
    temp file is removed, so an interrupted write never corrupts an existing
    checkpoint or snapshot.
    """
    tmp = path + ".tmp"
    try:
        save_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise


def read_metrics_summary(csv_path):
    """Return (header, data_row_count, last_step) for a metrics CSV."""
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, 0, None
    header = rows[0]
    data = [r for r in rows[1:] if r]
    last = None
    if data:
        try:
            last = int(float(data[-1][0]))
        except (ValueError, IndexError):
            raise run_state.DestinationError(
                "metrics.csv has an unparseable last row: %r" % (data[-1],)
            )
    return header, len(data), last


def inspect_destination(run_dir, csv_path, latest_path, snap_dir):
    """Inventory a run destination WITHOUT creating or modifying anything."""
    insp = {
        "run_dir_exists": os.path.isdir(run_dir),
        "latest_exists": os.path.exists(latest_path),
        "csv_exists": os.path.exists(csv_path),
        "csv_header": None,
        "csv_data_rows": 0,
        "csv_last_step": None,
        "snapshot_files": [],
        "other_pt_files": [],
    }
    if insp["csv_exists"]:
        header, nrows, last = read_metrics_summary(csv_path)
        insp["csv_header"] = header
        insp["csv_data_rows"] = nrows
        insp["csv_last_step"] = last
    if insp["run_dir_exists"]:
        for name in sorted(os.listdir(run_dir)):
            if name.endswith(".pt") and name != "latest.pt":
                insp["other_pt_files"].append(name)
        if os.path.isdir(snap_dir):
            insp["snapshot_files"] = sorted(
                n for n in os.listdir(snap_dir) if n.endswith(".pt")
            )
    return insp


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_run_mode(run_dir, csv_path, latest_path, snap_dir, exp, provenance):
    """Decide fresh-vs-resume and validate BEFORE any output mutation.

    Never infers a fresh restart from the absence of latest.pt when metrics
    already exist: validate_fresh_destination refuses such destinations.
    """
    insp = inspect_destination(run_dir, csv_path, latest_path, snap_dir)
    if insp["latest_exists"]:
        # Full checkpoints embed RNG tuples/ndarrays that the restricted
        # (weights_only=True) loader rejects; weights_only=False here is
        # therefore scoped to trusted local full checkpoints only and must not
        # be generalized to unknown files. Model-only snapshots use restricted
        # loading in evaluate_endpoints.py.
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        ckpt_hash = file_sha256(latest_path)
        plan = run_state.plan_resume_from_checkpoint(
            ckpt, exp, provenance, checkpoint_sha256=ckpt_hash
        )
        run_state.validate_resume_destination(insp, plan["global_step"])
        return "resume", plan, insp, ckpt
    run_state.validate_fresh_destination(insp)
    return "fresh", run_state.fresh_plan(), insp, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--segment-steps", type=int, default=None)
    ap.add_argument("--run-dir", type=str, default=None)
    ap.add_argument(
        "--legacy-provenance",
        type=str,
        default=None,
        help="JSON file with a trusted {seed, global_step, epoch} "
        "record for a legacy-checkpoint boundary resume",
    )
    args = ap.parse_args()

    run_state.validate_invocation_args(args.steps, args.segment_steps)

    device = get_device()
    seed = args.seed
    run_dir = args.run_dir or os.path.join("runs", "seed%d" % seed)
    csv_path = os.path.join(run_dir, "metrics.csv")
    latest_path = os.path.join(run_dir, "latest.pt")
    snap_dir = os.path.join(run_dir, "snapshots")

    provenance = None
    if args.legacy_provenance:
        with open(args.legacy_provenance) as f:
            provenance = json.load(f)

    train_data, val_data, tokens = build_split()
    vocab = len(tokens)
    cfg_hash = hash_config_split(train_data, val_data)
    tok_hash = tokenizer_hash(tokens)
    identity = run_state.build_run_identity(seed, vocab, cfg_hash, tok_hash)
    n_train = len(train_data)
    batch_size = 512
    n_batches = (n_train + batch_size - 1) // batch_size
    exp = {
        "seed": seed,
        "cfg_hash": cfg_hash,
        "tok_hash": tok_hash,
        "identity": identity,
        "n_train": n_train,
        "batch_size": batch_size,
        "n_batches": n_batches,
    }

    # --- resolve and validate the destination BEFORE touching any output ---
    mode, plan, insp, ckpt = resolve_run_mode(
        run_dir, csv_path, latest_path, snap_dir, exp, provenance
    )
    for note in plan["notes"]:
        print("note: %s" % note, flush=True)
    if mode == "resume" and not insp["csv_exists"]:
        print(
            "note: destination has no metrics.csv; future eval rows will "
            "start a new file (previous history is not available here)",
            flush=True,
        )

    # --- target calculation and honest no-op before any output mutation ---
    if args.segment_steps is not None:
        _target_steps = min(args.steps, plan["global_step"] + args.segment_steps)
    else:
        _target_steps = args.steps
    if plan["global_step"] >= _target_steps:
        print(
            "no-op: global_step %d already >= target %d; nothing was written"
            % (plan["global_step"], _target_steps),
            flush=True,
        )
        return

    # --- outputs are only mutated from here on ---
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(snap_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(run_state.EXPECTED_CSV_HEADER)

    # Model/optimizer construction consumes RNG; RNG from a resumed checkpoint
    # is restored AFTER construction and before any further sampling.
    torch.manual_seed(seed)
    model = Transformer(
        n_layers=2,
        n_heads=4,
        d_model=128,
        dropout=0.0,
        max_context_len=16,
        vocab_len=vocab,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98), eps=1e-8
    )

    if mode == "resume":
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        load_rng(
            {
                "torch": ckpt["torch_rng_state"],
                "mps": ckpt.get("mps_rng_state"),
                "numpy": ckpt["numpy_rng_state"],
                "python": ckpt["python_rng_state"],
            }
        )
        print(
            "Resumed (%s) seed %d: start_epoch %d, global_step %d, "
            "next_eval_epoch %d%s"
            % (
                "legacy-migrated" if plan["migrated_legacy"] else "current",
                seed,
                plan["start_epoch"],
                plan["global_step"],
                plan["next_eval_epoch"],
                ", mid-epoch cursor %d/%d" % (plan["batch_cursor"], n_batches)
                if plan["current_perm"] is not None
                else "",
            ),
            flush=True,
        )

    state = run_state.LiveState(
        epoch=plan["start_epoch"],
        current_perm=plan["current_perm"],
        batch_cursor=plan["batch_cursor"],
        global_step=plan["global_step"],
        next_eval_epoch=plan["next_eval_epoch"],
        migrated_legacy=plan["migrated_legacy"],
    )
    env = {"n_train": n_train, "batch_size": batch_size, "n_batches": n_batches}

    eq_token_index = ArithmeticTokenizer.get_tokens().index("=")

    def exec_batch(idx):
        batch_x = train_data[idx, :-1].to(device)
        batch_y = train_data[idx, 1:].to(device)
        u = state.global_step + 1
        lr = (u / 10.0) * 1e-3 if u <= 10 else 1e-3
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.zero_grad()
        logits = model(batch_x).transpose(-2, -1)
        eq_pos = int(
            torch.nonzero(batch_y[0] == eq_token_index, as_tuple=False).squeeze()
        )
        y_rhs = batch_y[:, eq_pos + 1 :]
        logits_rhs = logits[:, :, eq_pos + 1 :]
        loss = F.cross_entropy(logits_rhs, y_rhs, reduction="mean")
        loss.backward()
        optimizer.step()

    def eval_and_log(epoch, step):
        tr_loss, tr_acc = full_eval(model, train_data, device)
        va_loss, va_acc = full_eval(model, val_data, device)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, epoch, tr_loss, tr_acc, va_loss, va_acc])
            f.flush()
        print(
            "epoch %d step %d train_acc %.2f val_acc %.2f"
            % (epoch, step, tr_acc, va_acc),
            flush=True,
        )

    def make_full_saver(view):
        epoch_s, perm_s, cursor_s, completed = view
        rngs = rng_capture()

        def _save(p):
            save_ckpt(
                p,
                model,
                optimizer,
                epoch_s,
                state.global_step,
                state.next_eval_epoch,
                device,
                rngs,
                cfg_hash,
                tok_hash,
                identity,
                perm_s,
                cursor_s,
                n_batches,
                completed,
                seed,
                vocab,
                state.migrated_legacy,
            )

        return _save

    def save_latest(view):
        _atomic_save(make_full_saver(view), latest_path)
        print(
            "checkpoint saved: step %d epoch %s completed=%s"
            % (state.global_step, view[0], view[3]),
            flush=True,
        )

    def write_obligations():
        """Write due-but-missing artifacts for exactly the current step.

        Idempotent (existence-checked) and atomic, so bookkeeping can never be
        skipped at a stop step nor emitted twice. Model-only snapshots keep
        their historical model-only type; no optimizer state is invented.
        """
        gs = state.global_step
        if gs in SNAPSHOT_STEPS:
            sp = os.path.join(snap_dir, "snapshot_step%d.pt" % gs)
            if not os.path.exists(sp):

                def _snap(p):
                    torch.save(
                        {"state_dict": model.state_dict(), "step": gs, "seed": seed}, p
                    )

                _atomic_save(_snap, sp)
                print("saved snapshot %s" % sp, flush=True)
        if seed == 0 and gs == 2000:
            p2000 = os.path.join(run_dir, "seed0_step2000.pt")
            if not os.path.exists(p2000):
                view = run_state.save_view(
                    state.epoch, state.current_perm, state.batch_cursor, n_batches
                )
                _atomic_save(make_full_saver(view), p2000)
                print("saved %s" % p2000, flush=True)

    if args.segment_steps is not None:
        target_steps = min(args.steps, state.global_step + args.segment_steps)
    else:
        target_steps = args.steps
    env["target_steps"] = target_steps

    # Due-but-missing artifacts at the resumed step (e.g. a crash between an
    # update and its snapshot write) are written before any new minibatch.
    # (Only reached for genuinely continuing runs; completed targets exited
    # read-only above.)
    write_obligations()

    def draw_perm():
        return torch.randperm(n_train)

    model.train()
    t0 = time.time()
    reason = run_state.run_training_loop(
        state,
        env,
        exec_batch,
        draw_perm,
        eval_and_log,
        save_latest,
        write_obligations,
        stop_requested=lambda: _stop_requested,
    )

    elapsed = time.time() - t0
    print(
        "seed %d done (%s). steps_this_run=%d elapsed=%.1f global_step=%d"
        % (seed, reason, state.steps_this_run, elapsed, state.global_step),
        flush=True,
    )


if __name__ == "__main__":
    main()
