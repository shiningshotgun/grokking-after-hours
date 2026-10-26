# What this taught me

I trained a small transformer on modular division (mod 97) with three seeds, 100,000 steps each, on my Mac. All three runs showed the pattern from the paper: train accuracy reached 100% early, validation stayed low for a long stretch, then climbed late.

Seed 1's last logged validation accuracy was 6.64%, around step 99,000, so the summary I built from the logs said two of the three seeds had lost generalization by the end. When I loaded the 100,000-step snapshots and evaluated them directly, seed 1 scored 100%. Validation oscillates late in training and the summary had picked up a dip. I wrote the endpoint evaluation script only after the numbers looked wrong. It should have existed before the long runs.

Averaged over the three seeds, final validation came out near 76%. One seed actually finished near 29% and the other two near 100%, so I report per-seed numbers first now.

Seed 0 was paused and resumed partway through, and its evaluation schedule drifted from the other two after the resume. Comparisons between seeds after that point carry that difference. My snapshots also don't store optimizer state, so I can't claim the resumed run matches an uninterrupted one.

In all three runs, train and validation accuracy dip together repeatedly, long after grokking. I can't explain the dips. The hardware backend, the optimizer, and the pause in seed 0 are candidates, and nothing I ran separates them.

If I ran this again: write the endpoint evaluator first, keep the evaluation schedule identical across seeds, plan pauses before starting, save optimizer state in the checkpoints I intend to report on, and measure the endpoint at that exact step.
