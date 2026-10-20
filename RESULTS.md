# Results — grokking reimplementation (modular division, x/y mod 97)

Reimplementation of one experiment from Power et al., 2022
(https://arxiv.org/abs/2201.02177), default configuration: AdamW, lr 1e-3,
weight decay 1.0, 50% data, 100,000 updates, 3 seeds.

## Threshold crossings (first observed evaluation rows, not interpolated)
| seed | first train>=99% | first val>=50% | first val>=95% | first val>=99% | first val=100% | best val (logged) |
|------|------------------|----------------|----------------|----------------|----------------|-------------------|
| 0    | 1,210            | 2,960          | 3,490          | 3,760          | 4,300          | 100.0             |
| 1    | 1,570            | 2,590          | 2,940          | 3,340          | 5,020          | 100.0             |
| 2    | 1,350            | 2,890          | 3,460          | 3,730          | 11,430         | 100.0             |

"first X" values are the step of the first evaluation row in the raw CSV
reaching the threshold (eval cadence `next=max(int(1.02*e), e+1)`). The
earlier table mislabeled 4,300 / 5,020 / 11,430 as "first val>=99%"; those
steps are the first rows at exactly 100% validation accuracy, and the true
first val>=99% rows are 3,760 / 3,340 / 3,730. Both sets are retained above.

## Measured 100,000-step endpoints (inference-only, from snapshots)
| seed | train acc        | train loss | val acc           | val loss |
|------|------------------|------------|-------------------|----------|
| 0    | 36.0180412371134 | 0.6984     | 28.737113402061855| 0.7166   |
| 1    | 100.0            | 0.0875     | 100.0             | 0.0941   |
| 2    | 99.97852233676976| 0.0198     | 99.97852233676976 | 0.0239   |

Mean validation accuracy over the three measured seeds: **76.2385452462772%**.
These are the retained local measurements from `endpoint_metrics.json`; they
were not rerun in this correction pass. Seed 0's low endpoint is a measured
fact of the retained run, not a conclusion about causes.

## Last logged rows (~98,950-99,090 steps; NOT the 100k endpoint)
| seed | step   | train acc | val acc |
|------|--------|-----------|---------|
| 0    | 99,090 | 35.91     | 29.08   |
| 1    | 98,950 | 8.23      | 6.64    |
| 2    | 98,950 | 100.00    | 99.96   |

## Reporting definitions
- "first X" columns: first observed evaluation row in the raw CSV reaching the
  threshold; observed crossings, not interpolated.
- Scoring: after the "=", exactly TWO tokens are scored per equation — the
  answer residue and the EOS token (one answer token plus EOS). "endpoint":
  inference-only forward evaluation of the step-100000 model-only snapshot over
  the full train and validation splits (equation-level accuracy over both
  scored tokens; token-mean cross-entropy after "=").
- "best val (logged)" / Figure-2-style statistic: the MEAN of each seed's best
  observed validation accuracy (i.e. the mean of the three per-seed maxima,
  each 100.0 here → 100.0). It is NOT a maximum over seeds and NOT "a maximum
  when averaged"; and it is distinct from the mean 100k endpoint validation
  accuracy (76.2385452462772%), which summarizes the end-of-training state.
- "last logged": final CSV row (~98.9–99.1k steps), not the 100k endpoint.
  `results.json` and the CSVs describe last-logged semantics throughout.

## Historical cadence deviations (recorded facts from the retained CSVs)
- Seeds 1 and 2 follow the canonical eval chain exactly (every eval epoch in
  their CSVs matches `next=max(int(1.02*e), e+1)` from epoch 0).
- Seed 0's cadence deviates from that canonical chain after its step-2000
  pilot pause: CSV rows exist at epochs 199, 202, 206, ... where the canonical
  chain has 198, 201, 205, ..., and at epoch 2807 (row 28,080) after the
  28,070-step pause — consistent with the legacy resume path clamping a stale
  schedule pointer to the resume epoch. These are recorded observations about
  the eval schedule only; no causal claim about the accuracy dips is made or
  implied. The rewritten trainer normalizes stale pointers deterministically
  (canonical-chain recomputation) and reports the normalization explicitly.

## Limitations and instability
- Training did not stay perfect: repeated joint train/validation dips and
  recoveries occurred in all seeds. Their cause is not established; MPS,
  optimizer choice, heat, and pause history are not demonstrated causes.
- Seed 1 returned to 100% train and validation at step 97,010 before its last
  logged dip; endpoint at 100k is 100%.
- Neither permanent forgetting nor uninterrupted seed-2 retention is
  established.
- Figure 2 describes best validation accuracy averaged over 3 seeds (mean of
  per-seed maxima), not selection of the best seed. Our per-seed maxima are
  100%, but maxima are not used to hide later failures. Mean endpoint
  validation accuracy is distinct from that statistic.
- Default setting scope: AdamW weight decay 1, 100k updates, 50% split. This
  is not literal Figure 1 left (Adam/no weight decay/1e6 updates); do not say
  validation rise was "on schedule" for that different setup.
- Built-in AdamW vs official custom code is an implementation difference, not
  a demonstrated cause.
- Vocab 239 matches official construction.
- MPS is not bitwise-deterministic.
- The stage-2 state-machine corrections (train.py/run_state.py) are
  future-behavior fixes verified by no-workload unit tests; they did not
  produce the original runs, and no historical checkpoint migration was
  executed. Optimizer-continuation equivalence at 100k (archived snapshots
  are model-only) remains out of reach and untested; no "resume verified
  end-to-end" claim is made beyond the tested state transitions.

## Artifacts
- Plots: `plots/accuracy_curves.png`, `plots/loss_curves.png` (matches the
  retained CSVs; regeneration is collision-protected and requires new output
  paths)
- Raw metrics: `runs/seed0/metrics.csv`, `runs/seed1/metrics.csv`,
  `runs/seed2/metrics.csv` (unchanged bytes; last-logged semantics)
- Summaries: `results.json`, `analysis_summary.json`
- Endpoints: `endpoint_metrics.json` (unchanged; includes per-snapshot
  SHA-256 and the stale `latest.pt` cross-check) — its `hash_matches_expected:
  false` flag is explained and reconciled by
  `endpoint_metadata_correction.json` (both hash definitions, byte layout,
  independent recomputation; the original JSON was not modified)
- Environment record: `requirements.txt` (recorded versions, not a tested
  fresh-install lockfile)
- State tests: `test_state_machine.py` (no-workload; see README for scope)
- Checkpoints/snapshots: not kept in-tree; preserved in the private verified
  archive (see README "Endpoint re-evaluation from external checkpoints").
  In-tree `latest.pt` copies were stale (~98.9–99.1k steps), never final 100k
  full-state checkpoints.
