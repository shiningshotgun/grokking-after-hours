#!/usr/bin/env python
"""Inference-only 100,000-step endpoint measurement from model-only snapshots.

Failure handling: every per-seed artifact access (existence, hash, load,
identity validation) happens inside that seed's error handling, so an absent
or unreadable archive yields an explicit "unmeasured" entry instead of a crash.
Restricted loading (weights_only=True) is used for model-only snapshots; the
restricted full-checkpoint path in train.py remains the only
weights_only=False user and is scoped to trusted local files.

Output safety: --out is required and must not already exist; the collision is
checked before any dataset/model work, so existing evidence outputs are never
overwritten and no partial writes happen into them.
"""

import argparse, datetime, hashlib, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

import run_state
from data import build_split, ArithmeticTokenizer
from model import Transformer

OUTPUT_SCHEMA_VERSION = 2


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_split_hash(train_data, val_data):
    # Training-time definition (5-field prefix), shared with train.py via
    # run_state. The retained endpoint_metrics.json recorded the older
    # 2-field-prefix digest c4625d6b... under this field name; the two
    # definitions share the same split bytes but differ in the config prefix,
    # so their digests differ. See endpoint_metadata_correction.json.
    h = hashlib.sha256()
    h.update(run_state.config_prefix_five_field().encode())
    h.update(train_data.numpy().tobytes())
    h.update(val_data.numpy().tobytes())
    return h.hexdigest()


def make_model(vocab):
    return Transformer(
        n_layers=2,
        n_heads=4,
        d_model=128,
        dropout=0.0,
        max_context_len=16,
        vocab_len=vocab,
    )


def validate_state_dict(sd, vocab):
    expected = make_model(vocab).state_dict()
    if set(sd.keys()) != set(expected.keys()):
        raise ValueError("tensor name mismatch")
    for k, v in expected.items():
        if tuple(sd[k].shape) != tuple(v.shape):
            raise ValueError("shape mismatch %s" % k)
    return True


def evaluate_split(model, data, device, batch_size=512):
    model.eval()
    eq_token_index = ArithmeticTokenizer.get_tokens().index("=")
    total_ce_sum = 0.0
    total_tokens = 0
    correct_eqs = 0
    total_eqs = 0
    with torch.no_grad():
        n = data.shape[0]
        for start in range(0, n, batch_size):
            chunk = data[start : start + batch_size]
            x = chunk[:, :-1].to(device)
            y = chunk[:, 1:].to(device)
            logits = model(x).transpose(-2, -1)
            eq_pos = int(
                torch.nonzero(y[0] == eq_token_index, as_tuple=False).squeeze().item()
            )
            y_rhs = y[:, eq_pos + 1 :]
            logits_rhs = logits[:, :, eq_pos + 1 :]
            ce = F.cross_entropy(logits_rhs, y_rhs, reduction="sum")
            total_ce_sum += float(ce.item())
            total_tokens += int(y_rhs.numel())
            pred = logits_rhs.argmax(dim=1)
            row_correct = (pred == y_rhs).all(dim=1)
            correct_eqs += int(row_correct.sum().item())
            total_eqs += int(row_correct.numel())
    mean_ce = total_ce_sum / total_tokens if total_tokens else float("nan")
    acc = 100.0 * correct_eqs / total_eqs if total_eqs else float("nan")
    return {
        "loss": mean_ce,
        "accuracy": acc,
        "n_equations": total_eqs,
        "n_tokens": total_tokens,
        "correct_equations": correct_eqs,
    }


def eval_seed(seed, snapshot_path, train_data, val_data, vocab, device):
    """Evaluate one seed's model-only snapshot with full per-seed error handling.

    Every fallible step (existence, hashing, loading, identity checks) is
    inside the per-seed try block: an absent archive becomes an explicit
    "unmeasured" result, never a crash and never a fabricated measurement.
    """
    result = {"seed": seed, "snapshot_path": snapshot_path}
    try:
        if not os.path.exists(snapshot_path):
            result["status"] = "unmeasured"
            result["reason"] = "file not found: %s" % snapshot_path
            return result
        result["source_sha256"] = file_sha256(snapshot_path)
        # Restricted loading for model-only snapshots: the file holds only a
        # plain dict of tensors and ints, which weights_only=True accepts.
        ckpt = torch.load(snapshot_path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
            raise ValueError("missing state_dict")
        if ckpt.get("step") != 100000:
            raise ValueError("unexpected step %s" % ckpt.get("step"))
        if ckpt.get("seed") != seed:
            raise ValueError("seed mismatch")
        validate_state_dict(ckpt["state_dict"], vocab)
        model = make_model(vocab)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device)
        result["snapshot_step"] = ckpt["step"]
        result["snapshot_seed"] = ckpt["seed"]
        result["status"] = "measured"
        result["train"] = evaluate_split(model, train_data, device)
        result["val"] = evaluate_split(model, val_data, device)
    except Exception as e:
        result["status"] = "unmeasured"
        result["reason"] = repr(e)
    return result


def eval_latest(seed, latest_path, train_data, val_data, vocab, device):
    """Full-checkpoint cross-check.

    weights_only=False is required for full checkpoints (they embed RNG state
    tuples/ndarrays) and is scoped to trusted local files recorded by this
    project's own trainer; it must not be generalized to unknown files.
    """
    result = {"seed": seed, "path": latest_path}
    try:
        if not os.path.exists(latest_path):
            result["status"] = "unmeasured"
            result["reason"] = "file not found: %s" % latest_path
            return result
        result["source_sha256"] = file_sha256(latest_path)
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise ValueError("not a full checkpoint")
        validate_state_dict(ckpt["model_state"], vocab)
        model = make_model(vocab)
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        result["status"] = "measured"
        result["global_step"] = ckpt.get("global_step")
        result["epoch"] = ckpt.get("epoch")
        result["train"] = evaluate_split(model, train_data, device)
        result["val"] = evaluate_split(model, val_data, device)
    except Exception as e:
        result["status"] = "unmeasured"
        result["reason"] = repr(e)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        required=True,
        help="New, unused output JSON path. Existing evidence outputs are "
        "never overwritten (endpoint_metrics.json is the retained record).",
    )
    ap.add_argument("--device", default=None)
    ap.add_argument(
        "--artifact-root",
        default=".",
        help="Root directory containing runs/seedN/{snapshots,latest.pt}; "
        "use the external verified archive root to re-evaluate archived checkpoints.",
    )
    args = ap.parse_args()

    # Collision protection before any expensive work or partial write.
    run_state.reject_existing_outputs([args.out])

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    )
    train_data, val_data, tokens = build_split()
    vocab = len(tokens)
    cfg_hash = config_split_hash(train_data, val_data)
    try:
        script_sha = file_sha256(os.path.abspath(__file__))
    except Exception:
        script_sha = None
    results = {
        "metadata": {
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "generator": {
                "script": "evaluate_endpoints.py",
                "script_sha256": script_sha,
            },
            "device": str(device),
            "dtype": "float32",
            "library_versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
            "split": {
                "modulus": 97,
                "train_count": int(train_data.shape[0]),
                "val_count": int(val_data.shape[0]),
                "vocab": vocab,
                "config_split_hash": cfg_hash,
                "config_split_hash_definition": (
                    "sha256(str({'modulus': 97, 'train_pct': 50, 'bs': 512, "
                    "'lr': 0.001, 'wd': 1.0}) + train_int64_bytes + "
                    "val_int64_bytes); see endpoint_metadata_correction.json"
                ),
            },
            "evaluation": {
                "mode": "eval/no_grad",
                "shuffling": False,
                "masking": "answer_plus_EOS_after_=",
                "accuracy": "equation_level_both_tokens",
                "aggregation": "cross_entropy_by_token_count, accuracy_by_equation_count",
            },
            "note": "MPS not bitwise-deterministic; comparability caveat. Endpoint measurements separate from last_logged CSV rows. Missing/unreadable inputs are reported as explicit 'unmeasured' entries.",
        },
        "endpoints_100000": {},
        "stale_latest_checkpoints": {},
    }
    for seed in (0, 1, 2):
        snap = os.path.join(
            args.artifact_root,
            "runs",
            "seed%d" % seed,
            "snapshots",
            "snapshot_step100000.pt",
        )
        results["endpoints_100000"][str(seed)] = eval_seed(
            seed, snap, train_data, val_data, vocab, device
        )
        latest = os.path.join(args.artifact_root, "runs", "seed%d" % seed, "latest.pt")
        results["stale_latest_checkpoints"][str(seed)] = eval_latest(
            seed, latest, train_data, val_data, vocab, device
        )
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
