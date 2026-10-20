#!/usr/bin/env python
"""Reproducible analysis: plots and summaries from original CSVs + measured endpoints.

Safety rules:
- Endpoint entries are validated (status, seed identity, step identity, numeric
  fields) before use; missing/unmeasured seeds are handled explicitly. A subset
  average is never presented as the three-seed mean and never divided by 3.
- Existing output files are never overwritten: all intended output paths are
  checked for collisions before any expensive work or partial write, so future
  runs must pass a new, unused --out-dir/--summary.
"""

import argparse, csv, json, os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run_state

SEEDS = [0, 1, 2]
ENDPOINT_STEP = 100000


def load_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(
                {
                    k: (float(v) if k not in ("step", "epoch") else int(float(v)))
                    for k, v in row.items()
                }
            )
    return rows


def first_cross(rows, col, thresh):
    """First observed evaluation row reaching the threshold (not interpolated)."""
    for row in rows:
        if row[col] >= thresh:
            return row["step"]
    return None


def validate_endpoint_entry(entry, seed, require_step=ENDPOINT_STEP):
    """Return (entry_or_None, reason). Only fully valid measured entries pass."""
    if not isinstance(entry, dict):
        return None, "entry missing"
    if entry.get("status") != "measured":
        return None, "status=%r" % entry.get("status")
    if entry.get("seed") != seed:
        return None, "seed identity mismatch (%r)" % entry.get("seed")
    if require_step is not None and entry.get("snapshot_step") != require_step:
        return None, "step identity mismatch (%r)" % entry.get("snapshot_step")
    for split in ("train", "val"):
        block = entry.get(split)
        if not isinstance(block, dict):
            return None, "missing %r block" % split
        for field in (
            "loss",
            "accuracy",
            "n_equations",
            "n_tokens",
            "correct_equations",
        ):
            v = block.get(field)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return None, "missing/invalid %s.%s" % (split, field)
    return entry, None


def endpoint_val_summary(endpoints, seeds=SEEDS):
    """Validated per-seed endpoint val accuracies + explicit coverage.

    Returns (values: {seed: acc}, coverage: dict). The three-seed mean is only
    produced when every seed is measured; a subset mean is labeled partial.
    """
    values = {}
    missing = []
    for s in seeds:
        entry = None
        if isinstance(endpoints, dict):
            entry = endpoints.get("endpoints_100000", {}).get(str(s))
        entry, reason = validate_endpoint_entry(entry, s)
        if entry is None:
            missing.append({"seed": s, "reason": reason})
        else:
            values[s] = float(entry["val"]["accuracy"])
    n = len(values)
    coverage = {
        "measured_seeds": sorted(values.keys()),
        "n": n,
        "total": len(seeds),
        "complete": n == len(seeds),
        "unmeasured": missing,
    }
    return values, coverage


def build_summary(data, endpoints):
    """Pure summary builder over CSV rows {seed: rows} and the endpoints JSON."""
    summary = {"seeds": {}}
    for s in SEEDS:
        rows = data[s]
        last = rows[-1]
        entry, reason = (None, "endpoints unavailable")
        if isinstance(endpoints, dict):
            entry, reason = validate_endpoint_entry(
                endpoints.get("endpoints_100000", {}).get(str(s)), s
            )
        summary["seeds"][str(s)] = {
            "first_train_ge_99": first_cross(rows, "train_acc", 99.0),
            "first_val_ge_50": first_cross(rows, "val_acc", 50.0),
            "first_val_ge_95": first_cross(rows, "val_acc", 95.0),
            "first_val_ge_99": first_cross(rows, "val_acc", 99.0),
            "first_val_ge_100": first_cross(rows, "val_acc", 100.0),
            "best_val_acc_logged": max(r["val_acc"] for r in rows),
            "last_logged_step": last["step"],
            "last_logged_train_acc": last["train_acc"],
            "last_logged_val_acc": last["val_acc"],
            "endpoint_100k": entry,
            "endpoint_100k_unmeasured_reason": (reason if entry is None else None),
        }

    # Figure-2-style statistic: the MEAN of each seed's best observed
    # validation accuracy (max of that seed's logged rows), not a maximum over
    # seeds and distinct from mean 100k endpoint performance.
    maxima = [summary["seeds"][str(s)]["best_val_acc_logged"] for s in SEEDS]
    summary["best_val_acc_mean_over_seeds"] = float(np.mean(maxima))
    summary["best_val_acc_definition"] = (
        "mean of each seed's best observed validation accuracy (per-seed maxima); "
        "Figure-2-style statistic, distinct from mean 100k endpoint validation accuracy"
    )

    values, coverage = endpoint_val_summary(endpoints)
    summary["endpoint_val_coverage"] = coverage
    summary["mean_endpoint_val_acc"] = (
        float(np.mean([values[s] for s in SEEDS])) if coverage["complete"] else None
    )
    if 0 < coverage["n"] < coverage["total"]:
        summary["mean_endpoint_val_acc_partial"] = {
            "value": float(np.mean([values[s] for s in sorted(values)])),
            "over_seeds": sorted(values.keys()),
            "note": "partial-coverage mean; NOT the three-seed mean",
        }
    else:
        summary["mean_endpoint_val_acc_partial"] = None
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--endpoints", default="endpoint_metrics.json")
    ap.add_argument("--out-dir", default="plots")
    ap.add_argument("--summary", default="analysis_summary.json")
    args = ap.parse_args()

    plot_paths = [
        os.path.join(args.out_dir, "accuracy_curves.png"),
        os.path.join(args.out_dir, "loss_curves.png"),
    ]
    # Collision protection before any expensive work or partial write.
    run_state.reject_existing_outputs(plot_paths + [args.summary])

    seeds = SEEDS
    data = {}
    for s in seeds:
        csv_path = os.path.join(args.runs_dir, "seed%d" % s, "metrics.csv")
        if not os.path.exists(csv_path):
            raise SystemExit("missing metrics CSV for seed %d: %s" % (s, csv_path))
        data[s] = load_csv(csv_path)

    endpoints = None
    if os.path.exists(args.endpoints):
        with open(args.endpoints) as f:
            endpoints = json.load(f)
    else:
        print(
            "note: endpoints file %s not found; endpoint fields will be "
            "reported as unmeasured" % args.endpoints
        )

    summary = build_summary(data, endpoints)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)

    os.makedirs(args.out_dir, exist_ok=True)
    # Plot accuracy (only validated measured endpoint points are drawn)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for i, s in enumerate(seeds):
        rows = data[s]
        steps = [r["step"] for r in rows]
        tr = [r["train_acc"] for r in rows]
        va = [r["val_acc"] for r in rows]
        ax.plot(
            steps,
            tr,
            color=colors[i],
            linestyle="-",
            linewidth=1.5,
            label="seed %d train" % s,
        )
        ax.plot(
            steps,
            va,
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
            label="seed %d val" % s,
        )
        entry, _ = (None, None)
        if isinstance(endpoints, dict):
            entry, _ = validate_endpoint_entry(
                endpoints.get("endpoints_100000", {}).get(str(s)), s
            )
        if entry is not None:
            ax.plot(
                [ENDPOINT_STEP],
                [entry["train"]["accuracy"]],
                marker="o",
                color=colors[i],
                markersize=6,
            )
            ax.plot(
                [ENDPOINT_STEP],
                [entry["val"]["accuracy"]],
                marker="s",
                color=colors[i],
                markersize=6,
            )
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Modular division mod 97: train/val accuracy per seed")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_paths[0], dpi=150)
    plt.close(fig)

    # Plot loss
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, s in enumerate(seeds):
        rows = data[s]
        steps = [r["step"] for r in rows]
        tr = [r["train_loss"] for r in rows]
        va = [r["val_loss"] for r in rows]
        ax.plot(
            steps,
            tr,
            color=colors[i],
            linestyle="-",
            linewidth=1.5,
            label="seed %d train" % s,
        )
        ax.plot(
            steps,
            va,
            color=colors[i],
            linestyle="--",
            linewidth=1.5,
            label="seed %d val" % s,
        )
    ax.set_xscale("log")
    ax.set_xlabel("step (log scale)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Modular division mod 97: train/val loss per seed")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_paths[1], dpi=150)
    plt.close(fig)
    print("wrote plots and", args.summary)


if __name__ == "__main__":
    main()
