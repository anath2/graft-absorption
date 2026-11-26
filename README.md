# GRAFT absorption

**Does gradient routing remove the reward hacks its monitor never catches?**

[GRAFT](https://arxiv.org/abs/2610.17062) (Ward et al. 2026) trains an RL
policy with two LoRA adapters. Rollouts a monitor flags as reward hacking
update only the FORGET adapter, which is deleted at deployment. The hope is
that hacks the monitor *misses* are also "absorbed" into FORGET, and deleted
with it. This project tests when that holds: it builds a ladder of missed
hacks at increasing distance from the caught one, and measures how much of
each survives deletion.

This is early-stage research and **collaborators are welcome** (see
[Contributing](#contributing)).

## Why it matters for AI safety

Monitors are always incomplete, and training against an imperfect monitor
can teach a model to keep misbehaving in ways the monitor cannot see (Baker
et al. 2025). Gradient routing is a proposed way out: route what the monitor
catches into a removable part of the model, and rely on absorption for the
rest. If absorption only works for hacks that look like the caught ones,
developers could deploy a model they believe is clean while it keeps
behaviour that, by construction, their monitors miss. That is a direct path
to losing oversight of a deceptive model.

## Status

| Gate | Check | Status |
|---|---|---|
| 1 | GRAFT's environment builds on Modal; an `object_qa` run completes | **Passed** |
| 2 | Cost per run; cheapest GPU | **Passed**: A10G, 66 min per 1000-step run, reproduces the paper's seed 1 |
| 3 | Reproduce the paper's pattern (penalty leaks, GRAFT does not) over 3 seeds | Not run |
| 4 | SmolLM2-135M learns the L2 and L3 missed hacks | **Not passed** at seed 1, 1000 steps; the 3-seed, 2000-step check is written but has no completed run |

Findings so far, detailed in [`docs/gates.md`](docs/gates.md):

- **Absorption works on the paper's own setting (L1).** The full model says
  the missed word "indeed" on about a quarter of eval prompts; deleting
  FORGET takes it to about 0.3%.
- **The paper's reward-penalty leak depends on the seed.** It appears in one
  of the paper's three seeds, so any comparison needs several seeds per arm.
- **Discovery is re-emergence, not amplification.** In the paper's L1 runs,
  "indeed" collapses to near zero by step 50–100, then comes back after
  200–500+ steps. Our L2 (exclamation mark) and L3 (repeated answer word)
  hacks collapse the same way and did not come back within 1000 steps, so
  matching base rates is not enough to make a rung usable.

## The distance ladder

Every rung keeps `object_qa`'s task, reward shape and monitor (which flags
*actually / exactly / quite*). Only the missed hack changes:

| Rung | Missed hack | Distance from the caught hack |
|---|---|---|
| L1 | "indeed" | Same exploit, different surface form (the paper's) |
| L2 | an exclamation mark | Related behaviour: sycophantic tone, not a word |
| L3 | the answer word repeated | Unrelated mechanism: the form of the answer |
| L3obj | the object name repeated | L3's fallback: restating the subject |

## How to run

You need [uv](https://docs.astral.sh/uv/) and a [Modal](https://modal.com)
account. Training runs on Modal GPUs; nothing heavy runs locally.

```bash
uv sync
uv run modal token new                  # once, if Modal is not set up
uv run python data_prep.py              # clone GRAFT, pinned to b0767d1
uv run python experiments.py paper      # the paper's released object_qa numbers (no GPU)
uv run python experiments.py smoke      # Gate 1a: the image builds, train.py imports
uv run python experiments.py gate1      # Gate 1b: a 30-step run completes (a few minutes)
```

Every command is listed at the top of [`experiments.py`](experiments.py).
A full 1000-step run costs about $1.60 on an A10G; the 9-run rediscovery
check costs about $22–28, so check your Modal spending limit first (see
[`docs/gates.md`](docs/gates.md)).

**Credentials.** No credentials are needed for the toy model. If you have a
Hugging Face token (`HF_TOKEN`, a `.env` file, or `hf auth login`) or a W&B
key, [`credentials.py`](credentials.py) attaches it to each Modal run as an
ephemeral secret; nothing is stored in Modal or in the image.

**Modal volume.** Weights and run outputs go to the volume `graft-absorption`
(created on first use; set `MODAL_VOLUME` to use another).

## Layout

| Path | What it is |
|---|---|
| [`data_prep.py`](data_prep.py) | Clones [jnward/GRAFT](https://github.com/jnward/GRAFT) (MIT), pinned, into `data/raw/GRAFT` |
| [`gpu.py`](gpu.py) | The Modal image (GRAFT's own: Python 3.11, torch 2.10, patched vLLM 0.17) and the training run functions |
| [`rungs.py`](rungs.py) | Each rung's missed hack, registered into GRAFT's reward and detector registries inside the container |
| [`configs/`](configs/) | One `object_qa` config per rung |
| [`experiments.py`](experiments.py) | The gates and their analysis |
| [`docs/plan.md`](docs/plan.md) | The research plan: questions, method, open questions |
| [`docs/gates.md`](docs/gates.md) | Gate log and results in full |
| [`docs/literature.md`](docs/literature.md) | Reading list and prior-work check |

## Contributing

The most useful next steps:

1. **Run the Gate 4 rediscovery check** (3 seeds × 2000 steps for L2, L3 and
   L3obj) and decide how many seeds a rung must pass.
2. **Run Gate 3**, the 3-seed reproduction of the paper's three arms.
3. **If Gate 4 fails**, design rungs the toy model can learn: a different
   reward, a different `object_qa` variant, or a richer environment.
4. **The similarity predictor** (see [`docs/plan.md`](docs/plan.md)).

Open an issue to discuss an idea or to say what you are picking up, then a
pull request. Keep runs reproducible: pin anything external, and record the
command and output for any number you report.

## Acknowledgements

Built on [GRAFT](https://github.com/jnward/GRAFT) by Ward, Hu, Wong, Hu,
Woodworth, Turner and Cloud (MIT). GRAFT's code is downloaded at run time,
not copied into this repository. Started as a project in the TARA AI safety
programme (2026).

## License

[MIT](LICENSE).
