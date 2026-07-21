# Weighted fuzzy graph build and validation

Status: corrected implementation prepared 2026-07-21. The artifact built on
2026-07-20 is useful exploratory evidence, but it is not the canonical training
input. A clean rebuild and production-path canary remain required before merge.

This work implements Path A from
`latent-labs/guides/spec-weighted-graph-build.md`: recompute exact cosine
distances on the existing IVF-PQ neighbor topology, calculate UMAP directed
membership strengths, and apply the probabilistic t-conorm. It does not yet
establish that weighted sampling improves a map; that requires the matched
uniform-versus-fuzzy training experiment described below.

## Decision summary

| Item | Current conclusion |
| --- | --- |
| Fuzzy membership math | V1 passes against `fuzzy_simplicial_set` on an identical 100k topology. |
| Existing 30M artifact | Full structural scan passes: 738,221,242 unique, sorted, in-range, non-self directed edges with finite positive weights. File size is 4,586,950,577 bytes (4.59 GB decimal / 4.27 GiB). |
| Existing manifest | Not trainer-admissible: it lacks the production `graph_sha`/`graph_sha256` contract and full ordered embedding identities. Builder provenance is dirty and points at the parent commit. |
| Existing V2 | Directionally shows poor IVF-PQ recall (~0.254–0.258), but used unsafe rank-0 self removal under exact-vector ties. Rerun with explicit query IDs. |
| Existing V3 | Proves only isolated `DeviceEdgeSampler` CDF behavior. It bypasses admission and does not instantiate the 30M production hybrid path. Its 9.51 GB RSS was sampled before the CDF and draw histogram, not at peak. |
| Existing V4 | Directed weight monotonicity passes. The old report called all 60 probes mutual; only 29 were mutual. The corrected validator recomputes reverse membership and exact t-conorm values. |
| 30M production readiness | Pending a fresh, versioned rebuild from a clean commit, CPU admission, corrected V4, and one no-update GPU canary that stamps `hybrid` / `HostStreamEdgeSampler` / weighted sampling / device-fp16 X. |
| 150M/405M readiness | Builder interchange path exists, but there is no streaming sharded trainer consumer. Current gather is sparse-page-I/O bound. Do not describe these paths as scale-ready. |

Do not overwrite the exploratory file at
`/data/checkpoints/pumap/edges_30m_k15_fuzzy.npz`. Publish the corrected build at
a new versioned path and retain both receipts.

## Corrected builder contract

`experiments/build_weighted_graph.py` has four stages:

1. Phase A validates every topology row, gathers stored embeddings without a
   float32-to-float16 narrowing conversion, computes cosine distance in float32,
   and emits directed membership chunks.
2. Phase B partitions directed memberships by the unordered endpoint pair.
3. Phase C coalesces duplicate directed keys with the same sum-before-t-conorm
   semantics as scipy sparse matrices, then emits both directions.
4. Phase D performs a full output scan, publishes a single sorted NPZ, and
   writes a trainer-compatible content manifest.

The work directory is admitted by `build-contract.json`. Its identity binds:

- the full input topology SHA-256, byte count, node count, `k`, `nprobe`, and
  full-scan topology result;
- every ordered embedding member's canonical path, full SHA-256, bytes, dtype,
  full row count, and consumed row count;
- all result-affecting parameters, output path/mode, and distance precision;
- the builder commit, dirty state, and builder-source SHA-256.

Every phase-A chunk, phase-B bucket closure, and phase-C output has a write-once
receipt binding the contract and full output hash. Resume rehashes the staged
files. A changed input, shard order, parameter, builder byte, or staged byte
requires a fresh work directory. Legacy `.done` markers are intentionally not
accepted.

The input topology is fully scanned, rather than sampled, for:

- exact `source = row_id` repeated `k` times;
- in-range integer targets;
- distinct targets within each row;
- self slots, reported with both the edge-slot and node denominators.

The accepted 30M topology contains 192,940 self slots: 0.042876% of its 450M
slots and, because targets are distinct per row, 0.643133% of nodes. These are
removed by UMAP membership semantics. The earlier memo incorrectly described
0.043% as a percentage of nodes.

## Trainer admission and expected execution path

The corrected single-file manifest uses the production graph-manifest schema and
contains the output SHA-256/bytes, data fingerprint, and ordered canonical
embedding records. Ordered records are necessary because the accepted 30M pack
has thirty files all named `embeddings.npy`; a basename-keyed dictionary is
ambiguous.

`Round0014MaterializedArray` now admits its accepted feature pack independently
of the graph treatment. The historical `device_uniform` capability remains
restricted to its exact accepted uniform graph. Any new graph must pass its own
sibling manifest and full ordered-shard checks. This avoids weakening the old
seal while allowing a content-bound weighted treatment over the same rows.

At 30M, the expected weighted training footprint is approximately:

- device fp16 X: 23.04 GB;
- host int32 sources and targets: 5.91 GB total;
- host float64 weighted CDF: 5.91 GB;
- weights and transient/prefetch allocations in host memory.

The graph and CDF do not fit beside X on a 32 GB GPU. The expected production
selection is therefore `hybrid`, sampler class `HostStreamEdgeSampler`, positive
sampling `weighted_with_replacement`, and X residency `device_fp16`. A
`DeviceEdgeSampler` test with a one-column stub is not execution evidence for
that path.

The corrected V3 is deliberately CPU-bounded. It verifies the sibling manifest,
all ordered input identities, graph/data pairing, full weight domain, and both
host/device sampler CDF math on a deterministic edge sample without allocating a
second 5.9 GB per-edge histogram. It records that the production GPU canary has
not run. The release canary must use the real trainer admission path, perform no
optimizer update, and capture the actual `train_accounting` pipeline stamp.

## Validation status

### V1: membership and symmetrization oracle

The historical 100k result remains valid:

| Metric | Value |
| --- | --- |
| GPU exact kNN vs sklearn mean Jaccard@15 | 1.0000 |
| Builder / oracle directed edges | 2,130,704 / 2,130,704 |
| Edge-set differences | 0 / 0 |
| Maximum weight absolute difference | 1.19e-07 |
| Maximum sigma / rho difference | 0 / 0 |

The regression suite now additionally covers offset chunks, duplicate directed
keys, full topology scans, fp32 preservation, explicit self IDs under tied
vectors, content-bound resume, and a CPU end-to-end build whose manifest passes
production content validation.

### V2: topology honesty and Path-B candidate rerank

Historical IVF-PQ recall@15 was 0.2536, 0.2568, and 0.2580 for nprobe 32, 64,
and 128. This is strong evidence that increasing nprobe alone is not the answer,
but the validator assumed a separate-base query's rank-0 result was self. Exact
ties make that assumption false.

The corrected validator masks the explicit query row before top-k and also
measures Path B:

- exact top-15 coverage inside ANN top-64 (and a configurable candidate width);
- recall after exact fp32 reranking of those candidates;
- ANN, exact-reference, and rerank wall times;
- p10 as well as mean recall.

Do not repeat the earlier “top-64 should reach at least 0.9” estimate as a
result. Candidate coverage has not yet been measured. This short diagnostic can
run after the current GPU owner releases the card and need not block the matched
Path-A training comparison.

### V3: consumer contract

The JSON under `docs/weighted_graph_validation/v3_30m.json` is retained as the
historical isolated-device-sampler output. It is superseded as a production
admission claim. Regenerate V3 against the clean artifact, then run the separate
no-update GPU canary described above.

### V4: physical spot check

The historical check established monotone directed weights for 20/20 sampled
nodes and the weak inequality `symmetrized >= forward` for 60/60 probes. It did
not establish mutuality; independent inspection found only 29/60 reverse
memberships. The corrected V4 recomputes the neighbor's directed row, labels
actual mutual and one-way pairs separately, and compares the artifact to
`w_ij + w_ji - w_ij*w_ji` within tolerance.

## Scale and performance, corrected

Phase A is storage-bound above page-cache scale. A measured 2.4M-row random
gather from the 84M-row RedPajama source took 71.1 seconds, or 0.0338M rows/s.
`ShardedEmbeddings.gather` already sorts local row IDs within each shard; that
measurement is the sorted implementation. Sorting sparse page requests does not
turn them into sequential I/O.

Current-order estimates therefore remain approximately:

- 150M: about 22 hours for Phase A;
- 405M: about 60 hours for Phase A.

The previous 1.4-hour 405M estimate from “a small sorting addition” was not
supported. Reaching sequential bandwidth requires a different dataflow: an
external reorder/join, topology traversal organized by embedding blocks, or a
fused topology-and-weight build that consumes candidate vectors while resident.
That is substantial implementation work and needs its own benchmark.

The sharded output is also not yet a scale consumer. `load_sharded_edges`
concatenates every part into memory and now requires an explicit
`allow_materialize=True` acknowledgement for bounded validation. A 150M/405M
claim requires a streaming trainer loader and its own admission/performance
tests.

The local 150M FineWeb source remains unusable after roughly row 35.9M because
its shard history is not aligned to the topology. Do not build a partially
misaligned fuzzy graph, and do not re-embed FineWeb merely to rescue this lossy
topology before the 30M topology decision is made.

## Research sequence

After the current GPU-owned round finishes:

1. From a clean commit, rebuild the 30M weighted graph into a new artifact and
   fresh work directory. Confirm full output scan, CPU V3 admission, and corrected
   V4.
2. Run the no-update real-trainer canary and require the actual pipeline stamp:
   `hybrid`, `HostStreamEdgeSampler`, `weighted_with_replacement`,
   `device_fp16`. Do not infer this from requested configuration.
3. Run a matched 30M A/B on the same symmetrized 738,221,242-edge endpoint
   universe: uniform-with-replacement versus fuzzy-proportional sampling. Use the
   same accepted Round-0020 duplicate cap, model, seed, update horizon, scoring
   panel, and accepted OOD rider. Comparing fuzzy only to the historical k15
   graph would confound endpoint universe and sampler semantics.
4. Use otherwise idle GPU time for the corrected 3M top-64/top-128 candidate and
   exact-rerank benchmark. It does not gate step 3.
5. If fuzzy sampling wins, or candidate reranking materially improves recall,
   prioritize a higher-recall 30M topology build and repeat a controlled
   comparison before attempting 150M/405M scale machinery.

Duplicate-family mass does not need to block the matched A/B when both arms use
the accepted global cap. Before the cap, same-family edges are 0.253% of fuzzy
edges but 0.896% of fuzzy weight mass; after applying the retained-source cap,
they are about 0.0277% of edges and 0.0986% of weight mass. The cap should be
identical across arms and recorded in both execution receipts.

The old cap sampler cannot be reused unchanged: it assumes exactly 15 edge slots
per source and is restricted to `device_uniform`. The symmetrized fuzzy endpoint
universe has variable source degree and selects the hybrid path. The host sampler
now supports a retained-node universe without materializing a 732M-element edge
index: uniform draws use exact rejection conditioning over retained sources,
weighted draws zero excluded-source CDF intervals, and negative endpoints are
uniform over retained nodes. Focused distribution tests cover both arms. The
follow-up round still needs to wire the accepted R0020 exclusion artifact into
this path and record the resulting retained edge count and weight mass.

## Release checklist

- [x] Full topology validation and duplicate-target rejection.
- [x] Float32-preserving exact-distance input path.
- [x] Duplicate directed-key coalescing matches scipy.
- [x] Content-bound, hash-verified resume receipts.
- [x] Trainer-compatible manifest with duplicate-basename-safe ordered records.
- [x] Accepted feature-pack reuse without broadening `device_uniform`.
- [x] Corrected V2 self exclusion plus candidate/rerank metrics.
- [x] Corrected CPU V3 scope and peak-RSS measurement point.
- [x] Corrected V4 mutuality/t-conorm check.
- [x] Variable-degree hybrid retained-node sampling for the matched duplicate cap.
- [ ] Clean-commit 30M rebuild at a versioned path.
- [ ] Corrected real-artifact V3/V4 results.
- [ ] Real production-path no-update GPU canary and accounting stamp.
- [ ] Matched uniform-versus-fuzzy training round.

Merge the code after the CPU suite and synthetic end-to-end checks pass. Treat
the clean artifact rebuild and GPU canary as experiment-input release gates, not
as reasons to hold unrelated CPU implementation or the Path-B benchmark code.
