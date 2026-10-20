# Stage 2 pilot report — seed 0, 2,000 steps

## Environment
- Device: MPS (Apple Silicon) — `torch.backends.mps.is_available() == True`
- torch 2.8.0, numpy 2.0.2, matplotlib 3.9.4, Python 3.9.6
- MPS does not guarantee bitwise-deterministic ops; results are not claimed bitwise-exact.

## Measurements
- Steps completed: 2,000 (200 epochs × 10 batches)
- Wall time: 117.7 s
- Training throughput: 17.0 steps/sec
- Peak RSS (max resident): 356.6 MB (well under 6 GB; no swap)
- Added time per eval point: ~0.21 s (full val + full train forward)
- Eval points in pilot: ~150 (epoch 0..199 with schedule)
- Full checkpoint (`latest.pt` / `seed0_step2000.pt`): ~5.5 MB each
- Model-only snapshot: ~1.85 MB each

## Projections for full 3-seed plan (100,000 steps / seed)
- Wall time per seed: 100,000 / 17.0 ≈ 5,882 s ≈ 1.63 h, plus eval overhead ~320 × 0.21 s ≈ 67 s and checkpoint I/O ≈ 160 s → ≈ 1.7 h/seed; ≈ 5.1 h total.
- Storage per seed: 1 latest.pt (5.5 MB) + 16 snapshots (16 × 1.85 ≈ 29.6 MB) = 35.1 MB; seed 0 also keeps seed0_step2000.pt (5.5 MB) → 40.6 MB.
- Total storage: ≈ 40.6 + 35.1 + 35.1 ≈ 110.8 MB (under 200 MB cap).

## Notes
- Vocab size is 239 (official openai/grok data.py has 20 operator keys; the original spec text said 21/240). Param breakdown: blocks 394,240; embedding 30,592; head 30,592; total 455,424.
- Pilot saved seed0_step2000.pt, latest.pt, and snapshots for steps 100, 200, 500, 1000, 2000 (and 100/200/500 as scheduled).
- Pilot completed without environment or software failure.