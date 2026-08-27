# Jina-side social-mixture sweep — armed design proposal (owner-approved prep, 2026-08-27)

Mirrors the MiniLM instrument in jina-v5-nano **document-prompted** D768 space.
Propose-before-GPU per the standing discipline: this doc is the armed design + HEADs;
the sweep proper does NOT take the GPU until signed off. Prereq embeds/truths are
minutes-scale and ride GPU gaps in the drain window.

## Space + prompt (non-negotiable)
- Model `jinaai/jina-embeddings-v5-text-nano-retrieval`, document prompt `"Document: "`
  applied EXACTLY as the champion pipeline (a raw embed silently measures prompt
  mismatch, not register coverage — see p2_jina_embed.py). Reuse p2_jina_embed.py.
- All substrates + probes normalized (`_norm`) before knn/fuzzy/train, as the champion.

## Prerequisites (GPU, minutes-scale, run in drain gaps BEFORE the sweep)

**P-A. Jina social pools (4 corpora), document-prompted, holdout-disjoint.**
reddit + CA jina 250K probes already exist (reddit-jina-250k, ca-jina-250k). Need:
- twitter + bluesky jina 250K probes (front-300k reserved slice, same as MiniLM).
- mixture-draw pools for ALL 4: at 1M base × up to 50% social = 125k/corpus max draw;
  + 250k probe ⇒ embed ~500k/corpus (front 300k = probe-reserved, draws from ≥300k).
- Output: /data/latent-basemap/substrates/{reddit,ca,twitter,bluesky}-jina-pool/ (f16 N×768)
  + the 4 probe registers {reddit,ca,twitter,bluesky}-jina-250k (reddit/CA exist).

**P-B. Per-language jina probe truths (the standing blocker) — 20 languages.**
fineweb2 holdout slices DISJOINT from every jina substrate span: the 2M used first
50k/lang, the 6.25M topup through 156,250/lang ⇒ draw holdouts from global row
**≥ 156,250 per language** (verify against jina-prompted manifest.json/manifest-6250k.json
ml_spans). Document-prompted jina embeds + knn/fuzzy truth. **~100k/lang** (keeps the
20-lang build inside ~30-40 min GPU; bump to 250k only if time allows).
- Source: /data/chunks/fineweb2-<lang>-chunked-500 (all 20 langs present).
- Output: substrates/probe-lang-<lang>-jina/ + sandbox/probe-lang-<lang>-jina/edges-k15-fuzzy.npz.

**P-C. EN base-register holdouts (fineweb / RedPajama / pile), document-prompted.**
The jina base used EN spans 666667/corpus (2M). Draw EN holdout probes DISJOINT from
those spans, document-prompted jina embed + knn/fuzzy. These are the in-distribution
EN registers (the language floor's EN anchor). ~100-250k each.
- Output: substrates/probe-{fineweb,rpj,pile}-jina/ + their truths.

## Sweep proper (queues after the MiniLM ceiling arms + prereqs)

**Base:** 1M jina multilingual base = the existing champion mixture (EN fw/RPJ/pile +
20 langs × 50k), i.e. `multi-1m.f16.npy` composition. NOTE: the jina prompted set has
NO code corpus (unlike MiniLM's starcoder 10%), so there is no code register on the
jina side — the code-displacement question is MiniLM-specific.

**Share ladder (balanced family):** {0, 10, 20, 30} (+40/50 IFF the MiniLM ceiling arms
show worst-register still rising past 30%). balanced share s = (1−s) multi-base +
s/4 each of reddit/CA/twitter/bluesky (holdout-disjoint ≥300k).

**Transfer check:** ONE reddit-only point at 20% (rmix-jina-20) — tests whether the
MiniLM "balance beats volume" finding transfers to jina space.

**Recipe:** champion-bs16k (jina D768; = the jina-multi-2m champion recipe: md000, dose4,
rankneg 500k=25% of... at 1M rankneg 250k=25%; fneg1.0, tanh4.0, pos0.10, bs16k), seed 42.
Same recipe across all arms so the mixture is the sole variable.

**Probe suite (maximin, worst-register FIRST):** 20 languages + 4 social holdouts +
3 EN base holdouts = 27 registers. **Languages in the maximin is the whole point** — a
social-heavy mix must not sink the language floor. Maximin winner = the mixture whose
WORST register (likely a low-resource language or a social corpus) is least bad.
Interior-optimum + per-register delta-vs-0% reported as in the MiniLM sweep.

## HEADs / paths
- Substrates: substrates/jina-bmix{10,20,30[,40,50]}-1m/, substrates/jina-rmix20-1m/
- Base 0%: reuse the existing jina-multi... 1M champion map if one exists at this exact
  recipe+seed; else train jina-multi-1m/champion-bs16k as the 0% point (mirrors how the
  MiniLM 0% = minilm-mix-1m/rankfrac-25). CONFIRM which before the sweep.
- Builders (to write, all CPU except embeds/truths): a jina variant of
  build_mixture_substrates.py (jina pools + multi base, f16), a jina lang/EN probe
  builder (reuse p2_jina_embed.py + image_map_pipeline knn/fuzzy), a jina mixture_probe
  (27 registers, maximin worst-first, full-matrix assert like the MiniLM scorer).
- Scorer: adapt mixture_probe.py for the jina maps + 27-register suite.

## Sequencing
1. Prereq embeds/truths (P-A/P-B/P-C) in drain GPU gaps — minutes-scale, no sealed claim.
2. Sweep proper queues AFTER the MiniLM bmix40/50 ceiling arms (so the +40/50 decision
   is informed) and after prereqs exist. Propose armed HEADs (this doc) → owner signoff →
   build substrates (CPU) → knn/fuzzy/train/score (GPU) → maximin.

## Open items for signoff
- jina base 0% map: reuse existing champion vs train jina-multi-1m/champion-bs16k? (confirm)
- per-lang probe size 100k vs 250k (time budget).
- +40/50 jina arms gated on the MiniLM ceiling result.
- social pool embed size (500k/corpus proposed) — enough for 50% share + probe.
