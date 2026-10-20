# What this taught me

Running this project changed how I think about reproducing research. The training was the easy half: three seeds of modular-division grokking on my Mac, and the phenomenon from the paper showed up. Train accuracy saturated early, validation followed much later. The hard half started once the runs finished, when I had to say what they actually showed.

Seed 1 is the clearest example. The Stage 4 summary built from the training logs said two of the three seeds had lost generalization by the end, and seed 1's last logged validation was 6.64%, near step 99,000. When the actual 100,000-step snapshots were evaluated with no training updates, seed 1 was at 100%. A nearby logged point in a series that oscillates is not an endpoint. If a number is going to be called final, it has to come from a dedicated evaluation at exactly that step, which means the endpoint evaluator needs to exist before the long runs start, not get built after the numbers look strange.

The mean has the same problem from the other side. Averaged over three seeds, final validation comes out near 76%, which makes the setup look like it partially works. One seed ends near 29%, the other two near 100%. Those are different stories and the average hides both of them. Per-seed numbers first, the mean second, if at all.

Checkpoints and evaluation schedules decide what can be claimed afterwards. Seed 0 was paused and resumed, and its evaluation schedule drifted away from the other two partway through, so every seed-to-seed comparison after that point carries that difference. Because the archived snapshots don't include optimizer state, I also can't claim a resumed run is numerically the same as an uninterrupted one. None of this is visible while the run is printing accuracy. It becomes visible when someone asks a harder question about the result.

Tests work the same way. A passing suite says the assertions that exist pass, nothing more: the behaviours someone wrote assertions for behave as asserted. When a result looks surprising, the first thing to check is whether any assertion even covers it.

The paper's headline figure averages each seed's best validation accuracy; it does not say what the seeds do at the end. In my runs, train and validation accuracy dip together, repeatedly, long after grokking. I can't explain the dips: the hardware backend, the optimizer and the pause in seed 0 are all candidates and nothing I ran separates them. That is a statement about these three seeds on this machine, not about grokking in general.

If I ran this again, the order would change. The endpoint evaluator gets written before the long runs. Eval cadence stays identical across seeds from step one, and pauses get planned before the run starts. Every checkpoint at an endpoint I intend to report includes optimizer state. Per-seed numbers go first, and anything called final gets evaluated at that exact step.
