# Gradient routing, reward hacking and imperfect monitors

Reading list for this project (see [`plan.md`](plan.md)).
Every arXiv ID below was checked against the arXiv API on 2026-10-24, or on
2026-10-25 for rows added then.

## GRAFT and gradient routing (the method we stress-test)

| ★ | Source | How deep to read | Why it matters for us |
|---|---|---|---|
| ★★★ | Ward, Hu, Wong, Hu, Woodworth, Turner & Cloud 2026. *Gradient Routing Localizes and Removes Unintended Behaviors in RL.* [arXiv 2610.17062](https://arxiv.org/abs/2610.17062) (NeurIPS 2026) · [code](https://github.com/jnward/GRAFT) (MIT) · [LW post](https://www.lesswrong.com/posts/mKJEBiuuxHKuCoTB6/gradient-routing-localizes-and-removes-unintended-behaviors-in-rl) · [Anthropic blog](https://alignment.anthropic.com/2026/graft/) | Full: §3–4, the discussion of why routing generalises, and the appendix on detectable/undetectable splits and compute | **The method we test.** Monitor-flagged rollouts update only a FORGET adapter, which is removed at deployment. Missed hacks are meant to be "absorbed" into FORGET too, but the authors call their evidence for absorption weak. In every toy environment the missed hack is the caught exploit in another surface form (another word, or other formatting). Swapping which forms are caught mostly held, but `addition` leaked at about 60% hacking under GRAFT. Toy runs on SmolLM2-135M take 1–2.5 H200-hours. |
| ★★★ | GRAFT LW post, comment thread (19 Oct 2026). [Link](https://www.lesswrong.com/posts/mKJEBiuuxHKuCoTB6/gradient-routing-localizes-and-removes-unintended-behaviors-in-rl) | Comments by Oz Amram, RogerDearnaley, Samuel Knoche, williawa | Amram asks whether absorption holds for hacks structurally unlike the detected ones. RogerDearnaley proposes leaving one hack unmonitored and testing whether the ablated model can still use it. Knoche reproduced Countdown and found the naive baseline used about 3× the learning rate; matching it narrowed the gap. These are our RQ1, RQ2 and a baseline control. |
| ★★★ | jacobcd52/graft. *GRAFT generalization experiments.* [GitHub](https://github.com/jacobcd52/graft) (public repo, created 20 Oct 2026, active daily; README and `docs/LESSONS.md` read 2026-10-25) | Full README: the arms table, runs 2–3 and the narration probes | **The closest independent work.** GRAFT's SFT setup on Qwen3-32B and Terminal Wrench. Each arm routes 20% of hacks from one kind and tests on a disjoint kind that is never routed. Transfer is at or above a random-20% ceiling for every split by hack type, domain, language, length and severity, including invented outputs → peeking or tampering with tests (1.15). It fails when the never-routed hacks have no narration: narrated → commands-only gives 0.10. Translated narration gives 0.98; with the hack sentences deleted, 0.64. The README concludes that routing follows what the transcript says. One seed per arm. `LESSONS.md` is a useful cost and pitfall list for GRAFT runs. |
| ★★ | Cloud, Goldman-Wetzler, Wybitul, Miller et al. 2024. *Gradient Routing: Masking Gradients to Localize Computation in Neural Networks.* [arXiv 2410.04332](https://arxiv.org/abs/2410.04332) | Skim: the method, the "absorption" results and the RL gridworld | The original gradient routing paper. Introduces absorption: routing a labelled subset can also localise unlabelled related data (e.g. routing California tokens pulled in Oregon and Colorado). Its PPO gridworld is already RL with partial oversight: steering worked with 4% of episodes overseen, about 5× more data-efficient than filtering. Oversight there is random, not systematic, and absorption is never measured against how similar the unlabelled data is. The authors note heavy hyperparameter sensitivity. A cheap fallback testbed without the vLLM stack (code release not checked). |
| ★★ | Shilov, Cloud, Gema, Goldman-Wetzler et al. 2025. *Beyond Data Filtering: Knowledge Localization for Capability Removal in LLMs.* [arXiv 2512.05648](https://arxiv.org/abs/2512.05648) · [Anthropic blog](https://alignment.anthropic.com/2025/selective-gradient-masking/) | Abstract and the label-noise results | SGTM, a stricter masking variant, in pretraining (removing one language from a bilingual synthetic set; biology from Wikipedia). Better retain/forget trade-off than filtering or the original method when labels are noisy. Removed knowledge took about 7× more fine-tuning steps to recover than with RMU. Tests mislabelled data, not distance. |
| ★ | Roland, Cubuktepe, Martinez, Servaes et al. 2026. *Modular Pretraining Enables Access Control.* [arXiv 2607.08077](https://arxiv.org/abs/2607.08077) · [Anthropic blog](https://alignment.anthropic.com/2026/modular-pretraining/) | Abstract | GRAM (ICML 2026 spotlight): routes each dual-use category to its own switchable MLP module, so one model approximates several data-filtered models, at 50M–5B. Called preliminary by its authors. Shows the routing family is being scaled in pretraining. |

## Why training against imperfect monitors is dangerous (threat model)

| ★ | Source | How deep to read | Why it matters for us |
|---|---|---|---|
| ★★★ | Baker, Huizinga, Gao, Dou et al. 2025. *Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation.* [arXiv 2503.11926](https://arxiv.org/abs/2503.11926) | Skim: the obfuscation results | Penalising a CoT monitor's flags in RL taught a frontier model to keep reward hacking while hiding it from the monitor. This is the failure GRAFT tries to avoid, and our core risk link. |
| ★★★ | MacDiarmid, Wright, Uesato, Benton et al. 2025. *Natural Emergent Misalignment from Reward Hacking in Production RL.* [arXiv 2511.18397](https://arxiv.org/abs/2511.18397) | Skim: headline results | Reward hacking learned in real production RL environments generalised to alignment faking and sabotage of safety research. Why leftover hacking matters beyond a bad score. |
| ★★ | Korbak, Balesni, Barnes, Bengio et al. 2025. *Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety.* [arXiv 2507.11473](https://arxiv.org/abs/2507.11473) | Abstract and recommendations | Multi-lab position paper. Warns that optimisation pressure against monitors erodes them. Context for the scale-of-risk argument. |
| ★ | Kenton, Janzer, Greig, Teh et al. 2026. *Debate Training Reduces Reward Hacking in RLAIF.* [arXiv 2608.17776](https://arxiv.org/abs/2608.17776) | Abstract | A different route to the same problem (a better reward signal). Useful for framing alternatives in the talk. |

## Other training-time interventions (baselines and alternatives)

| ★ | Source | How deep to read | Why it matters for us |
|---|---|---|---|
| ★★★ | Wong, Engels & Nanda 2026. *Mitigating Reward Hacking with RL Training Interventions.* ICLR 2026 workshop · [AF post](https://www.alignmentforum.org/posts/R5MdWGKsuvdPwGFBG/steering-rl-training-benchmarking-interventions-against) · [environment](https://github.com/ariahw/rl-rewardhacking) | Full post | Benchmarks penalty, screening and inoculation on Qwen3-4B with ground-truth, probe and LLM-judge monitors. Reports large run-to-run variance, so we plan 3 seeds per arm. No gradient routing. |
| ★★ | Wichers, Ebtekar, Azarbal, Gillioz et al. 2025. *Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment.* [arXiv 2510.05024](https://arxiv.org/abs/2510.05024) | Abstract | Inoculation prompting is a GRAFT baseline, and matched it in the half-hackable variant. |
| ★★ | Tan, Woodruff, Warncke, Jose et al. 2025. *Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time.* [arXiv 2510.04340](https://arxiv.org/abs/2510.04340) | Abstract | Parallel inoculation paper. |
| ★★ | Riché, Tan, Kohonen & Warncke 2026. *Inoculation Adapters: Improved Selective Generalization of Capabilities with Fewer Surprising Backdoors.* [arXiv 2606.30252](https://arxiv.org/abs/2606.30252) | Skim | Also puts unwanted behaviour into a removable adapter, but by prompting rather than monitor routing. The closest alternative design. |
| ★ | Azarbal, Gillioz, Ivanov, Woodworth et al. 2025. *Recontextualization Mitigates Specification Gaming without Modifying the Specification.* [arXiv 2512.19027](https://arxiv.org/abs/2512.19027) | Abstract | Another intervention that does not need a perfect monitor. |
| ★ | Chen, Arditi, Sleight, Evans et al. 2025. *Persona Vectors: Monitoring and Controlling Character Traits in Language Models.* [arXiv 2507.21509](https://arxiv.org/abs/2507.21509) | Abstract | Source of "preventative steering", a GRAFT baseline. |
| ★ | Wang, Pham, Yin, Wang et al. 2026. *Detecting and Suppressing Reward Hacking with Gradient Fingerprints.* [arXiv 2604.16242](https://arxiv.org/abs/2604.16242) (COLM 2026) | Abstract | GRIFT: detects hacking from compressed gradients of the chain of thought, over 25% better than CoT monitors, and uses it in rejection fine-tuning. A stronger monitor rather than a way to live with a weak one. |
| ★ | Wu & Tang 2026. *From Rebound to Remedy: Understanding and Mitigating Reward Hacking via Representation Engineering.* [arXiv 2604.01476](https://arxiv.org/abs/2604.01476) | Abstract | Penalises hacking inside the GRPO advantage using a "shortcut" activation direction. Another training-time alternative to cite in the talk. |

## Is removal real? Unlearning robustness (for RQ2)

| ★ | Source | How deep to read | Why it matters for us |
|---|---|---|---|
| ★★ | Lee, Foote, Infanger, Shor et al. 2025. *Distillation Robustifies Unlearning.* [arXiv 2506.06278](https://arxiv.org/abs/2506.06278) (NeurIPS 2025) | Skim: the oracle-imitation result | Even a model trained to imitate one that never saw the data relearns the removed ability in a few fine-tuning steps; distilling into a fresh student blocks this (UNDO). Why we check whether the missed hack's capability survives ablation, not just its rate. |
| ★ | Fan, Jia, Zhang, Ramakrishna et al. 2025. *Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond.* [arXiv 2502.05374](https://arxiv.org/abs/2502.05374) (ICML 2025) | Abstract | Standard unlearning is undone by relearning from a few examples. Background for the relearning attack in RQ2, against which the original gradient routing paper claims near-gold-standard resistance. |

## Environments and models

| ★ | Source | How deep to read | Why it matters for us |
|---|---|---|---|
| ★★★ | GRAFT repo, `rl/` at commit [`b0767d1`](https://github.com/jnward/GRAFT/tree/b0767d1c40e5a03d78fc5ff4b6abc92829168386/rl) | `README.md`, `rh_detectors.py`, `envs/qa.py`, `envs/addition.py`, `sweeps/README.md` | Our harness. A prompt feature decides whether a hack is possible; a detector (a plain function, `completions -> list[bool]`) decides which response forms the monitor sees. Rewards are composed in the YAML configs. Pins Python 3.11, vLLM 0.17.0 (patched), torch 2.10 and TRL 0.29. |
| ★★ | Ben Allal, Lozhkov, Bakouch, Martín Blázquez et al. 2025. *SmolLM2: When Smol Goes Big.* [arXiv 2502.02737](https://arxiv.org/abs/2502.02737) | Abstract | The 135M policy model for the toy environments. |
| ★ | Khalifa, Khan, Tafveez, Peng et al. 2026. *Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR.* [arXiv 2603.07084](https://arxiv.org/abs/2603.07084) | Skim: the three hack types | Has three structurally different hacks occurring naturally. A possible realism check at 8B; not in the current scope. |

## Prior-work check (2026-10-25)

Supersedes the 2026-10-24 check, which found no independent follow-up.

**Verdict:** one independent follow-up exists and overlaps our RQ1:
[jacobcd52/graft](https://github.com/jacobcd52/graft), a public repo
started on 20 Oct 2026. It routes one kind of hack and tests on another,
in GRAFT's *SFT* setting on Qwen3-32B. Routing transferred across hack
type, domain, language and severity, but failed when the never-routed
hacks had no narration. We found no test of this in *RL*, none with
multiple seeds, and none of whether the missed hack's capability survives
ablation or of monitor false positives. Across the whole gradient routing
family (original paper, SGTM, GRAM, GRAFT), absorption has never been
measured against the distance between labelled and unlabelled data.

Two risks to our counterfactual impact: that repo moves fast and could add
RL, and Shawn Hu promised a follow-up post on practical implications.
Recheck both before relying on this check.

| ★ | Source | Overlap with us | How we differ |
|---|---|---|---|
| ★★★ | jacobcd52/graft (above) | Route one kind of hack, test on a disjoint kind that is never routed. Splits by hack type, framing, domain, language, length, severity and format. Everything transfers at or above a random-20% ceiling except narrated → commands-only (0.10). Even invented outputs → tampering with tests transfers (1.15). | **RL, not SFT.** In RL the policy is pushed toward whatever the monitor misses, which is the obfuscation risk GRAFT exists for. SFT has a fixed dataset. **No narration.** Toy completions are 16–48 tokens with no reasoning, the regime that leaked in their SFT runs. **3 seeds** (theirs: 1). **Capability check** (RQ2) and **false positives** (RQ3), which they don't test. Their finding gives us a prediction to test: without shared narration, does absorption still reach a different mechanism? |
| ★★ | Cloud et al. 2024 (above), RL gridworld | Gradient routing in RL with partial oversight (as little as 4%). | Oversight is random per episode, so missed and caught cases are the same behaviour. Our recall-sweep analogue, not our ladder. |
| ★ | Shilov et al. 2025 (SGTM, above) | How localisation tolerates bad labels. | Pretraining, mislabelled data rather than systematically missed data. |
| ★★★ | GRAFT paper, appendix | Recall sweeps (0.5, 0.25, 0.1), and swaps of which forms the monitor catches across the toy environments: mostly robust, but `addition` leaked at about 60% hacking. In every toy environment the missed hack is the caught exploit in another surface form (configs at `b0767d1`: another intensifier or flattery word, wordy-gap repetition, comma-reformatted copy). Countdown has structurally different hacks, but only one configuration. | We vary the distance between caught and missed hacks on purpose, from the paper's surface-form setting to a related behaviour to an unrelated mechanism, with seeds and matched starting rates. We also test whether the missed hack's capability survives ablation, and false positives. |
| ★★ | wassname 2026. *Can steering vectors drive gradient routing? Yes, but not in realistic settings.* [LW](https://lesswrong.com/posts/kzri5W2uBfF2mdboK/can-we-use-steering-vectors-to-suppress-reward-hacking-1) · [code](https://github.com/wassname/vGROUT_pub) | Gradient routing on Qwen3-4B LeetCode, routed by steering vectors instead of labels. Label-free initialisation cut held-out hacking from 0.529 to 0.195. Notes that routing papers vary recall but never false positives. | A different router. Their precision point becomes our RQ3. |
| ★★ | Wong, Engels & Nanda 2026 (above) | Training interventions against reward hacking with imperfect monitors. | No routing. A source of baselines and of the "high variance between runs" warning. |
| ★ | Riché et al. 2026 (above) | Removable adapter holding unwanted behaviour. | Prompt-driven, not monitor-driven. No monitor blind spots. |

## Search terms for the prior-work check

"gradient routing reward hacking", "GRAFT forget adapter", "absorption
gradient routing undetected", "monitor blind spot RL training", "training
against monitors obfuscation", "imperfect monitor reward hacking
interventions", and on 2026-10-25 "gradient routing absorption
follow-up", "GRAFT replication OR critique", "selective gradient
masking", "gradient routed auxiliary modules", "gradient routing
unlearning relearning". Searched: arXiv, LessWrong and Alignment Forum,
Google Scholar via web search, GitHub, the GRAFT repo and LW comment
thread.
