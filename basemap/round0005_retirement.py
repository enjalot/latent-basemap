"""Fail-closed registry for launchers superseded by the Round 0005 contract."""
from __future__ import annotations


class RetiredLauncherError(RuntimeError):
    """Raised before a retired path can parse outputs, spawn, or touch CUDA."""


RETIRED_LAUNCHERS = {
    "experiments/run_r1_kernel.py":
        "the generic kernel launcher can train outside the exact six-node program",
    "experiments/run_r1_ablation.py":
        "the generic ablation launcher can train outside the exact six-node program",
    "experiments/run_experiment.py":
        "the generic training CLI is not one of the six issued Round 0005 nodes",
    "experiments/run_canary.py":
        "the generic canary spawns the retired training CLI outside queue admission",
    "experiments/build_prompted_graph.py":
        "the prompted graph builder is not a staged input or an admitted queue node",
    "experiments/embed_prompted_200k.py":
        "the prompted embedder can load a GPU model outside queue admission",
    "experiments/golden_validate.py":
        "the legacy golden scorer can invoke CUDA outside the exact nine-map program",
    "experiments/run_round0001_gpu_canary.py":
        "Round 0001 is superseded and its GPU canary has no Round 0005 admission",
    "experiments/bench_input_pipeline.py":
        "the direct GPU pipeline benchmark is outside the issued no-training program",
    "experiments/profile_pipeline.py":
        "the direct GPU training profiler is outside the issued no-training program",
    "experiments/measure_knn_cost.py":
        "the direct GPU kNN benchmark is outside the issued no-training program",
    "experiments/build_testbed.py":
        "the direct GPU testbed builder is superseded by sealed staged Round 0005 data",
    "experiments/score_8m_bridge.py":
        "the historical 8M scorer is outside the issued six-node Round 0005 program",
    "basemap/eval.py":
        "the legacy evaluator CLI can call GPU kNN outside queue admission",
    "experiments/space_passport.py":
        "the direct GPU space probe is outside queue admission",
    "experiments/dag_template.py":
        "the legacy G1 DAG consumes pre-gate references and uses a non-manifest 8M path",
    "experiments/run_o2_frontier.py":
        "the historical O2 frontier invokes the removed shared-reference scorer interface",
    "experiments/run_o2_4m.py":
        "the historical O2 controls invoke the removed shared-reference scorer interface",
    "experiments/run_g0.py":
        "the historical G0 rescore invokes the removed shared-reference scorer interface",
    "experiments/run_backfill_2m_s44.py":
        "the historical 2M backfill invokes the removed shared-reference scorer interface",
    "experiments/run_o1_prompted.py":
        "the prompted study was superseded and invokes the removed scorer interface",
    "experiments/run_overnight_program.py":
        "same-process 8M training/scoring has no exact release-bound scale admission",
    "experiments/score_a3_rescore.py":
        "the direct 8M rescore has no exact release-bound performance certificate",
    "experiments/build_reference.py":
        "pre-gate shared high-D references are forbidden; each scorer builds a private sibling",
    "experiments/run_8m_canary.py":
        "the standalone canary uses the legacy controller instead of a signed scale queue",
    "experiments/score_complete_panel.py":
        "the generic scorer is limited below 8M; use the exact-gated 8M scorer",
    "basemap/panel_v2.py":
        "the generic evaluator CLI is limited below 8M and has no scale admission inputs",
    "train_dedicated_modal.py":
        "remote 15M/30M training bypasses the release-bound scale queue and GPU-hour cap",
    "build_150m_index_modal.py":
        "remote 30M/150M GPU indexing bypasses release-bound scale admission",
    "train_15m_modal.py":
        "remote 15M training bypasses the release-bound scale queue and GPU-hour cap",
    "train_and_project_modal.py":
        "remote 15M training/projection bypasses release-bound scale admission",
    "sweep_global_modal.py":
        "the remote 15M sweep bypasses the release-bound scale queue and GPU-hour cap",
    "sweep_structure_modal.py":
        "the remote 15M sweep bypasses the release-bound scale queue and GPU-hour cap",
    "sweep_v3_modal.py":
        "the remote 15M sweep bypasses the release-bound scale queue and GPU-hour cap",
    "bench_knn_modal.py":
        "the remote GPU kNN benchmark bypasses exact admission and runtime telemetry",
    "bench_query_a100.py":
        "the direct A10G/A100 query benchmark bypasses queue admission and the lease",
    "bench_scale_modal.py":
        "the remote scale/training benchmark bypasses release and scale admission",
    "bench_throughput_modal.py":
        "the remote GPU throughput benchmark bypasses queue admission and caps",
    "bench_train_gpu_modal.py":
        "the remote training benchmark bypasses the issued no-training program",
    "build_faiss_index_modal.py":
        "the remote GPU index builder bypasses the sealed Round 0005 input program",
    "debug_faiss_modal.py":
        "the direct remote FAISS GPU diagnostic bypasses the controller lease",
    "train_combined_modal.py":
        "the remote combined trainer bypasses the issued no-training program",
    "train_modal.py":
        "the generic remote trainer bypasses exact release-bound admission",
    "autoresearch/train.py":
        "the autoresearch trainer is outside the issued no-training program",
    "autoresearch/prepare.py":
        "the autoresearch executable evaluation harness is outside Round 0005",
    "project_local.py":
        "the direct model projector can select CUDA outside controller admission",
    "scale_experiment.py":
        "the generic scaling trainer can select CUDA outside the exact program",
    "train_local.py":
        "the generic local trainer can select CUDA outside the exact program",
    "validate_umap.py":
        "the generic validation trainer can select CUDA outside queue admission",
}


def retirement_message(launcher: str) -> str:
    reason = RETIRED_LAUNCHERS.get(launcher)
    if reason is None:
        raise ValueError(f"unknown retired launcher: {launcher}")
    return (
        f"{launcher} is RETIRED for Round 0005: {reason}. "
        "Use experiments/prepare_round0005_queue.py and basemap.run_controller with "
        "an exact signed manifest; >=8M work additionally requires the release-bound "
        "round0005_performance_certificate.v3 and a reopened-input row derivation."
    )


def refuse_retired_launcher(launcher: str) -> None:
    raise RetiredLauncherError(retirement_message(launcher))


# Complete executable GPU/training/scorer inventory.  A production path is
# either one exact admitted node or is explicitly retired above; there is no
# unclassified generic executable lane.
ADMITTED_GPU_ENTRYPOINTS = {
    "basemap/run_controller.py",
    "experiments/score_complete_panel.py",
    "experiments/compare_panel_cache.py",
    "experiments/round0005_performance_gate.py",
    "experiments/calibrate_jina_embedding.py",
    "experiments/run_round0005_seal_canary.py",
    # The sole additive successor is admitted only through the Round-0014
    # target-specific queue/controller contract.
    "experiments/run_round0014_node.py",
}
EXECUTABLE_GPU_ENTRYPOINTS = ADMITTED_GPU_ENTRYPOINTS | {
    "basemap/eval.py",
    "basemap/panel_v2.py",
    "experiments/bench_input_pipeline.py",
    "experiments/build_prompted_graph.py",
    "experiments/build_reference.py",
    "experiments/build_testbed.py",
    "experiments/embed_prompted_200k.py",
    "experiments/golden_validate.py",
    "experiments/measure_knn_cost.py",
    "experiments/profile_pipeline.py",
    "experiments/run_8m_canary.py",
    "experiments/run_canary.py",
    "experiments/run_experiment.py",
    "experiments/run_overnight_program.py",
    "experiments/run_r1_ablation.py",
    "experiments/run_r1_kernel.py",
    "experiments/run_round0001_gpu_canary.py",
    "experiments/score_8m_bridge.py",
    "experiments/score_a3_rescore.py",
    "experiments/space_passport.py",
    "bench_knn_modal.py",
    "bench_query_a100.py",
    "bench_scale_modal.py",
    "bench_throughput_modal.py",
    "bench_train_gpu_modal.py",
    "build_faiss_index_modal.py",
    "debug_faiss_modal.py",
    "train_combined_modal.py",
    "train_modal.py",
    "autoresearch/train.py",
    "autoresearch/prepare.py",
    "project_local.py",
    "scale_experiment.py",
    "train_local.py",
    "validate_umap.py",
    "build_150m_index_modal.py",
    "train_15m_modal.py",
    "train_and_project_modal.py",
    "train_dedicated_modal.py",
    "sweep_global_modal.py",
    "sweep_structure_modal.py",
    "sweep_v3_modal.py",
}
if not EXECUTABLE_GPU_ENTRYPOINTS - ADMITTED_GPU_ENTRYPOINTS <= set(RETIRED_LAUNCHERS):
    raise AssertionError("Round 0005 executable GPU inventory has an unclassified path")
