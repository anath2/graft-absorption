# Research plan: predicting absorption in gradient routing

Builds on Ward et al.'s [GRAFT](https://arxiv.org/abs/2610.17062) (MIT code,
[jnward/GRAFT](https://github.com/jnward/GRAFT)). Reading list and prior-work
check: [`literature.md`](literature.md). Gate results so far:
[`gates.md`](gates.md).

## Problem

**Risk:** loss of human oversight through deceptive AI.

- Gradient routing removes unwanted behaviour from a model. It trains
  labelled bad data into a separate part of the model, then deletes that
  part. It has been proposed for removing dangerous knowledge (Shilov et al.
  2025) and for RL against imperfect monitors (GRAFT, Ward et al. 2026).
- Labels are always incomplete: monitors miss new hacks, and filters miss
  dangerous knowledge in harmless-looking text. These methods rely on
  *absorption*: related, unlabelled data also goes into the part that is
  deleted. If absorption fails, the model keeps behaviour its developers
  believe is gone, and their monitors cannot see it, because those monitors
  missed it in the first place.
- Nobody knows when absorption works. In GRAFT's toy tests, each missed hack
  is the same exploit as the caught one, in a different form. No routing
  method has measured absorption against how different the unlabelled data
  is.

## Goal

- An analysis of how much unlabelled behaviour gradient routing removes, as
  that behaviour moves further from the labelled behaviour.
- A test of whether similarity inside the model (activations or gradients),
  measured before training, predicts what routing will absorb.
- A test of whether the model can still do the unlabelled behaviour when
  asked: did routing remove the skill, or only make it rarer?
- Public code and results.

## Research questions

- **RQ1:** After FORGET is removed, how does the rate of missed hacks depend
  on how different they are from the caught hacks (distance rungs L1–L3)?
- **RQ2:** Does GRAFT remove the *capability* for a missed hack, or only
  lower its rate?
- **RQ3:** How do monitor false positives change leakage and target-task
  performance?
- **RQ4 (stretch):** Does FORGET-adapter activation flag missed hacks?

## Plan

1. **Reproduce GRAFT.** Run GRAFT, a reward penalty and no intervention on
   one toy environment (`object_qa`) with SmolLM2-135M, 3 seeds per arm at
   matched learning rates. Measure the cost per run.
2. **Build a ladder of missed hacks** at three distances from the caught
   hack: the same exploit in another form (L1, the paper's), a related
   behaviour (L2), and an unrelated mechanism (L3). Match their starting
   rates to the caught hack's where possible.
3. **Measure similarity before training.** For each missed hack, compute its
   activation and gradient similarity to the caught hack in the starting
   model. The base model rarely produces some hacks, so similarity may need
   to be computed on hand-written hack completions.
4. **Run the sweep.** Train with routing at each rung. Measure how much of
   each missed hack remains after deletion, next to the task score.
5. **Test the predictor.** Check whether similarity predicts absorption
   better than the distance labels. This needs enough missed hacks per rung
   for some statistical power.
6. **Test capability after deletion** (RQ2).

## How this differs from the closest work

[jacobcd52/graft](https://github.com/jacobcd52/graft) routes one kind of hack
and tests on another in GRAFT's *SFT* setting on Qwen3-32B, with one seed per
arm. This project is in *RL*, where the policy is pushed toward whatever the
monitor misses; its toy completions have no narration (the regime that leaked
in their SFT runs); it uses 3 seeds; and it adds the capability check and
false positives. See [`literature.md`](literature.md) for the full check.

## Open questions

- **Gate 4 has not passed.** The L2 and L3 hacks collapse early in training
  and were not rediscovered within 1000 steps at seed 1. If the 3-seed,
  2000-step check also fails, the toy environment cannot carry the distance
  sweep without changing its reward or environment, or moving to a richer
  environment (for example Countdown-Code).
- How many seeds must rediscover a rung's hack for it to count.
- What similarity is computed on, for hacks the base model rarely emits.
