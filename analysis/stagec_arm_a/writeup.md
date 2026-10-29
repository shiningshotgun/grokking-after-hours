# Stage C - epsilon arm (eps 1e-4) vs the deterministic CPU control

Seed 1, 100000 steps, CPU, 4 threads, deterministic, weight_decay 1.0 both sides. The intended configuration change is epsilon alone (control 1e-8, arm 1e-4), and the recorded run metadata confirms the intended flags; other runtime variables are not independently controlled beyond what the metadata records.

## What was verified about the replay itself

- The tail replay's metrics.csv, diagnostics_steps.csv, and diagnostics_evals.csv are byte-identical to the original control export (sha256 match against the independently retained control archive, not just the manifest inside the new bundle).
- Full step telemetry 1..100000 on both runs, no gaps. Dense tail: 106 evals, every 10 steps, 98950..100000, joining the recorded logs exactly at the shared seam step 98950 (all overlapping fields incl. state_hash match).
- Endpoint weights equality is asserted by the replay pipeline (42 tensors, snapshot/latest/optimizer). Not independently re-verified outside that pipeline; the .pt tensors were not part of the analysis export.

## The reviewer's last-1050-steps question

The control's final open episode does not close. val_acc 38.53% at step 98950, ranging 10.91-45.81 across the tail, ending at 25.24% at step 100000. The collapse persists to the end of training. (This window was previously unobserved; the dense tail now covers it.)

## Pre-registered scoring (frozen definition, applied identically to both runs)

Episode = maximal run of consecutive evals at least X accuracy points below the running max after the first >=99% val crossing; depth = max drop. Metric: val_acc.

| | Control (eps 1e-8) | Arm A (eps 1e-4) |
|---|---|---|
| First >=99% val crossing | step 4570 | step 2300 |
| Episodes at 2pt | 26 | 25 |
| Episodes at 5pt | 25 (median depth 31.27) | 20 (median depth 36.29) |
| Episodes at 10pt | 23 | 18 |
| Baseline ranges (5pt) | count [14,25], median depth [13.4,49.7] | count and median inside range |
| Terminal episode (onset at the 5pt definition) | open from 86170 (onset 82840 at 2pt), depth 75.3, never closes | open from 19330, depth 99.1, never closes |

By the letter of the pre-registered yardstick, Arm A's episode count and median depth sit inside the baseline ranges. But the counts hide a qualitative difference: the arm's last episode opened at step 19330 - about a fifth of the way through training - and never closed through its last recorded evaluation at step 98950 (depth 99.1 points, val_acc ~1%). The control's dips were recurrent and transient until its own late collapse at 86170. Under eps 1e-4 the dips were not damped; the run instead collapsed, far earlier, and stayed collapsed for the rest of its recorded evaluations. Coverage differs between the runs: only the control has a dense tail through step 100000; Arm A's behavior in 98950..100000 is unobserved.

## Probes (descriptive, not hypothesis tests)

- float32 CE saturation: not observed in either run. Loss floor 9.88e-3 (control), 1.61e-2 (arm); zero steps below 1e-6 in both.
- update_to_weight_ratio around onsets: the control showed elevation in the 50 steps before episode onsets (median 2.17e-3 vs 1.43e-3 overall). The arm shows the reverse (1.30e-3 pre-onset vs 2.45e-3 overall). The control's pre-onset elevation does not replicate in the arm.
- grad_norm spike-precedes-dip: descriptive counts only, no pre-specified null: control 17/25 onsets had a >p95 spike in the prior 50 steps, arm 1/20. Gradient spikes are serially correlated and no null model was pre-registered for this, so these counts are observations, not evidence for or against the spike-precedes-dip prediction. The arm's grad_norm p95 is much higher overall (257.6 vs 51.5).

## Limits

- Single seed (seed 1) on each side. Nothing here generalizes across seeds.
- This is one paired-seed comparison (one arm vs one control, seed 1). No mechanism conclusion and no cross-seed generalization follows from it.
- Probe numbers are descriptive windows, not pre-registered tests; no null model was specified for them.
- Tensor-level endpoint equality is pipeline-reported, not externally re-verified.

## Analysis provenance

Numbers recomputed from the exported artifacts by an independent analysis script (rebuilt against the frozen plan after the original script was lost; the rebuild reproduces every published control number exactly - crossing 4570, 26/25/23 episodes, median 31.27, deepest 93.9 @30320 - before being applied to the arm).
