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

**Base (REDESIGNED — LANGUAGE-PRESERVING, delegate URGENT 2026-08-28):** the jina-multi-2m
composition = **1M EN (fw/RPJ/pile) + 20 languages × 50k (1M ML) = 2M**. NOT the 1M
`multi-1m` (which is 20×50k = ALL ML, NO EN — a 1M pure-ML base has no EN to displace, so
it would FORCE language-displacement). The 2M base gives a full 1M EN block to absorb the
social displacement while holding every language block fixed.

**CODE-PRESERVING = LANGUAGE-PRESERVING (the bmix10cp mechanism transfers).** MiniLM proved
that displacement KILLS whatever SMALL register's budget it eats: code died because social
displaced its 10% budget across the whole base, and the ENTIRE bmix30/40/50 ladder lost the
maximin monotonically to code collapse — while bmix10cp (displace ONLY the large non-code
corpora, code held identical) WON the maximin. On the jina side the SMALL registers are the
20 per-language blocks (50k each), and LANGUAGES ARE THE CORE OF THE JINA MAXIMIN. So social
MUST displace ONLY the three large EN corpora; every per-language block is held IDENTICAL to
the 0% baseline across every arm. A proportional-across-the-whole-base displacement would
re-learn the code lesson at ~6 arms' cost (language floors collapse exactly like code did).

**MATCHED, LANGUAGE-PRESERVING ladder.** Build the 2M base ONCE. Every mixed arm = the SAME
base rows with a seed-42 share of the EN 1M displaced by social — the 20 language blocks
(50k each) BIT-IDENTICAL across all arms, EN rows a matched subset, social share the sole
variable (the bmix10cp construction: displace only the large corpora, preserve the small ones).

**Share ladder (balanced family):** {0, 10, 20, 30} (+40/50 gated as before). balanced share
s displaces s×2M rows drawn ONLY from the EN 1M, PROPORTIONALLY across fw/RPJ/pile, replaced
by s/4 each of reddit/CA/twitter/bluesky (holdout-disjoint ≥300k). ALL 20 language blocks
(50k each) UNTOUCHED — count identical to the 0% baseline in every arm. The 0% arm = the
undisplaced base. **Manifest MUST verify: per-language row counts identical across all arms**
(the language-preservation proof, analogous to bmix10cp's starcoder==200k proof). At the max
share the EN block must not be over-drawn: 30% of 2M = 600k social ≤ the 1M EN — OK to ~50%.

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
- Substrates: substrates/jina-bmix{10,20,30[,40,50]}-2m/, substrates/jina-rmix20-2m/ (2M base now).
- Base 0% = the jina-multi-2m champion map (the undisplaced 2M base: 1M EN + 20×50k langs). This is
  the EXISTING jina-multi-2m/champion-bs16k (0.6426 / v2 0.6830) — reuse it as the 0% arm (its
  substrate IS the undisplaced base), so no fresh 0% train needed. The mixed arms displace only its
  EN 1M. (SUPERSEDES the 2026-08-27 ruling (a) "train jina-multi-1m 0%": the 1M pure-ML base is
  incompatible with language-preserving displacement, so the base moves to the 2M EN+lang composition
  whose 0% map already exists.) Per-lang probe size (b) 100k, gating (c), pool size (d) unchanged.
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

## Signoff (delegate 2026-08-27 — all four ruled)
- (a) TRAIN a fresh jina-multi-1m/champion-bs16k as the 0% point (~45 min; no jina 1M rung
  exists to reuse). AND adopt matched-displacement across the WHOLE ladder (above).
- (b) per-lang probe size = **100k/lang** (keep the 20-lang build ~35 min).
- (c) +40/50 jina arms GATED on the MiniLM ceiling arms showing continued worst-register
  rise; if MiniLM turns over at 40, jina stops at 30.
- (d) social pool embed size = **500k/corpus** approved.

## Addition (delegate 2026-08-27): language-floor backfill of EXISTING jina maps
When P-B's 20-language truths land, ALSO score the maps we already own — jina-multi-2m
champion-bs16k, champion-x8-h3072, the 6.25M champion, and the md010 arm (when sealed) —
through the 20-language suite. Cheap (transform + FFR, no train). We have never had
per-language OOD numbers for these maps; the sweep's 0% interpretation needs to know
whether the current maps even have a language floor. Report per-map × per-language FFR +
worst-language, alongside the sweep.
