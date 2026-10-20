# Paper versus this study

This page compares our reproduction with the original paper, so the scope of the results is not read wider than it is.

Paper: Power et al., "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets", https://arxiv.org/abs/2201.02177 (text read at https://ar5iv.labs.arxiv.org/html/2201.02177). Section and appendix references below are to that text.
This study: `README.md` and `RESULTS.md` in this repository.

This is the Grokking paper only. The sibling Grokfast study is a different paper with its own comparison page.

## Side by side

| Item | Paper | This study |
|---|---|---|
| Task | Binary operation tables; the headline example (Figure 1, left) is division mod 97 with 50% of the data in training. Section A.1 lists many other operations. | One task: modular division `x / y mod 97`, 50% train and 50% validation, fixed split shared by all seeds (`README.md`, `RESULTS.md`). |
| Model | Decoder-only transformer, 2 layers, width 128, 4 heads, about 4e5 non-embedding parameters (Appendix A.1.2). | Two-layer decoder-only transformer, width 128, 4 heads, 455,424 parameters in total including embeddings (`PILOT.md`). The paper's 4e5 counts non-embedding parameters and ours is a total, so the two figures are not like for like. The non-embedding blocks in ours are 394,240 (`PILOT.md`). Vocab size is 239 (`README.md`, `PILOT.md`; see Implementation fidelity below). |
| Optimizer | For most experiments, AdamW, learning rate 1e-3, weight decay 1, betas (0.9, 0.98), linear warmup over the first 10 updates, minibatch 512 or half the training set, whichever is smaller (A.1.2). | PyTorch built-in AdamW at the same settings (`README.md`, `RESULTS.md`). The authors' released code uses a custom AdamW class instead; see Implementation fidelity below. That is an implementation difference, not a demonstrated cause of any result (`RESULTS.md`, limitations). |
| Budget | 1e5 updates by default. 5e5 for the learning-time curves in Section 3.1.1. 1e6, with Adam and no weight decay, for Section 3.1 (A.1.2). Figure 1 left shows validation accuracy near 1e6 steps. | 1e5 updates, the paper's default budget. This is not the Figure 1 left setup (Adam, no weight decay, 1e6 updates). `RESULTS.md` states this, so validation rise here is not compared to that figure. |
| Seeds | 3 random seeds for each experiment and dataset size, except Section 3.1.1, which aggregates 7 (A.1.2). | 3 seeds, one task, one fraction, one optimizer setting. That is 3 runs in total, not 3 per cell of a sweep. |
| Data fraction | Varied. Figure 1 center and Section 3.1.1 study how training time grows as the training fraction shrinks. | Fixed at 50%. Our study says nothing about the data-fraction curve. |
| Main reported metric | Figure 2 right: best validation accuracy achieved after 1e5 steps, averaged over 3 seeds. Figure 1 center: steps until validation first reaches 99%. | Per-seed first-observed threshold crossings, per-seed best logged validation, measured 100k-step endpoints, and the mean endpoint. `RESULTS.md` keeps these apart. |
| Hardware | Not compared here. | Apple Silicon GPU (MPS). The reproducibility evidence in this repository does not establish bitwise-identical reruns on this setup (`RESULTS.md`, `README.md`). I make no claim about other hardware. |

## Implementation fidelity

- Vocab size. The paper text I read does not state one. The authors' released code builds the vocabulary from EOS and `=`, 20 operator keys, residues 0 to 96 and 120 S5 permutations, which is 239 tokens (https://raw.githubusercontent.com/openai/grok/main/grok/data.py, `ArithmeticTokenizer.get_tokens`). The original task spec for this study said 240 (21 operators). `PILOT.md` records that as a known difference from the spec, and `RESULTS.md` states that 239 matches the official construction.
- Optimizer. The authors' code configures a class `CustomAdamW` with betas (0.9, 0.98), defined in https://raw.githubusercontent.com/openai/grok/main/grok/training.py. The paper text does not describe it as custom. This study used PyTorch built-in AdamW.
- Version. Both files were read on 2026-10-21 from the main branch, which was at commit 3d64b1d8c1d595dd8ebdb7771998823f1b14c7b3 (repository https://github.com/openai/grok). That is the current code, not necessarily the code at the time of the paper.
- Parameters. The paper's about 4e5 counts non-embedding parameters. Ours is a total of 455,424, of which the non-embedding blocks are 394,240 (`PILOT.md`). The two are not like for like.

## Where the numbers must not be mixed

- The paper's Figure 2 statistic is a mean over seeds of each seed's best validation accuracy. Ours matches that definition (the mean of the three per-seed maxima, each 100.0). It does not describe where the seeds end. The measured 100k endpoints differ: seed 0 ends at 28.74%, seed 1 at 100%, seed 2 at 99.98%, so the mean endpoint is 76.24% (`RESULTS.md`, "Measured 100,000-step endpoints"). The two statistics answer different questions.
- Our curves come from logged rows. The last logged rows sit near 98.9k to 99.1k steps, not at the 100k endpoint. Seed 1's last logged validation is 6.64% while its measured endpoint is 100%. Plotted logged curves are not endpoint measurements.
- First crossings are the first evaluation rows that met the threshold. They are not interpolated and not equivalent to the paper's median time-to-99% curves, which are computed across data fractions and many seeds.
- Three seeds match the paper's seed count per setting. They do not match the paper's breadth: one task, one fraction, one optimizer here against many operations and sizes there. The 7-seed learning-time experiment has no counterpart in this study.

## What this study supports and does not

Supports: in this setup, all three seeds reached high validation accuracy long after fitting the training data, with training and validation accuracy dipping repeatedly afterwards in every seed. The cause of those dips is not established (`RESULTS.md`, "Limitations and instability").
Does not support: a claim that this reproduces the whole paper, the data-fraction trend, the effect of weight decay or other regularizers, or any statement about operations other than modular division. More seeds alone would not replicate the paper's broader experimental grid (many operations and data fractions). This page does not set a required seed count.

## Unknown or not checked

- Whether the paper's per-seed results match ours seed by seed. The paper reports aggregates, not our run-level numbers.
- The effect of the built-in optimizer versus the authors' optimizer. We recorded it as a difference, not a measured one.
- Whether the paper's own runs used the 239-token construction in the current official code. The paper text does not state a vocab size.
- Anything the paper reports about its hardware or exact data split.
