"""O1 — prompted-vs-unprompted 200k transfer, end-to-end via the fail-stop DAG.

embed_prompted (faithfulness-gated) -> build_prompted_graph -> perf_canary ->
train 3 prompted legacy_lp maps (verdict-gated) -> score prompted maps through one
shared reference -> compare vs the unprompted G0 legacy maps + the prompt-shift report.

The prompted train configs (_o1_prompted_s{42,43,44}.yaml) are byte-faithful to the
unprompted 200k legacy runs except the substrate paths (fair comparison). The kernel
is legacy_lp (finalized by G1).
"""
from __future__ import annotations
import os, sys, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import Job, run_jobs, known_service_pids
from basemap.round0005_retirement import refuse_retired_launcher

PY = ".venv/bin/python"
W = "/data/latent-basemap/closure/o1"
P = "/data/latent-basemap/jina-en-200k-prompted"
UNP = "/data/latent-basemap/jina-en-200k"
TEXT = "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train"
EMB = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
SRC = EMB
EVID = "experiments/evidence/r1_o1"


def _one(pat):
    d = sorted(glob.glob(os.path.join("experiments/results", pat)))
    if not d:
        raise SystemExit(f"o1: no run dir for {pat}")
    return d[-1]


def main():
    refuse_retired_launcher("experiments/run_o1_prompted.py")
    os.makedirs(W, exist_ok=True); os.makedirs(EVID, exist_ok=True)
    d = lambda p: os.path.join(W, p)
    verdict = d("perf_canary_verdict.json")
    emb_manifest = f"{P}/train/data-00000.npy.manifest.json"
    graph = f"{P}/edges_k50_fuzzy.npz"
    prompted_scores = f"{EVID}/complete_200k_prompted_v22.json"
    compare_out = f"{EVID}/o1_prompted_vs_unprompted.json"

    jobs = []
    # 1) embed prompted 200k (faithfulness-gated; aborts <0.98 mean / <0.95 min)
    jobs.append(Job(
        name="embed_prompted", certifying=True, outputs=[emb_manifest],
        argv=[PY, "experiments/embed_prompted_200k.py", "--testbed", UNP, "--out", P,
              "--text-dir", TEXT, "--embed-dir", EMB, "--batch-size", "512",
              "--n-faithfulness-probe", "40"],
        done_marker=d("embed.done.json"), log=d("embed.log"), manifest=d("embed.manifest.json"),
        cwd=os.getcwd(), required_free_gb=8.0,
        input_paths=["experiments/embed_prompted_200k.py"]))
    # 2) build prompted graph + centroids + held-out + shift report
    jobs.append(Job(
        name="build_graph", certifying=True, outputs=[graph, f"{graph}.manifest.json"],
        argv=[PY, "experiments/build_prompted_graph.py", "--prompted", P, "--unprompted", UNP,
              "--text-dir", TEXT, "--embed-dir", EMB, "--k", "50", "--n-holdout", "5000",
              "--n-anchors", "2000"],
        done_marker=d("graph.done.json"), log=d("graph.log"), manifest=d("graph.manifest.json"),
        cwd=os.getcwd(), required_free_gb=8.0, deps=["embed_prompted"],
        input_paths=["experiments/build_prompted_graph.py", emb_manifest]))
    # 3) perf canary from the prompted s42 train config
    cfg42 = "experiments/configs/_o1_prompted_s42.yaml"
    jobs.append(Job(
        name="perf_canary", certifying=True, outputs=[verdict],
        argv=[PY, "experiments/run_canary.py", "--train-config", cfg42,
              "--run-dir", d("canary_run"), "--out", verdict,
              "--floor", "200", "--warn", "250", "--max-steps", "1200", "--warmup", "200"],
        done_marker=d("canary.done.json"), log=d("canary.log"), manifest=d("canary.manifest.json"),
        cwd=os.getcwd(), required_free_gb=8.0, deps=["build_graph"],
        input_paths=[cfg42, "experiments/run_canary.py", "experiments/run_experiment.py",
                     "basemap/pumap/parametric_umap/core.py", graph]))
    # 4) train 3 prompted maps (verdict-gated)
    train_dirs = {}
    for seed in (42, 43, 44):
        cfg = f"experiments/configs/_o1_prompted_s{seed}.yaml"
        rd = d(f"train_prompted_s{seed}")
        train_dirs[seed] = rd
        jobs.append(Job(
            name=f"train_prompted_s{seed}", certifying=True,
            outputs=[os.path.join(rd, "coords.parquet"), os.path.join(rd, "model.pt"),
                     os.path.join(rd, "results.json")],
            argv=[PY, "experiments/run_experiment.py", cfg,
                  "--override", f"logging.run_dir_override={rd}"],
            done_marker=d(f"train_s{seed}.done.json"), log=d(f"train_s{seed}.log"),
            manifest=d(f"train_s{seed}.manifest.json"), cwd=os.getcwd(), required_free_gb=8.0,
            deps=["perf_canary"], canary_dep="perf_canary", require_passing_verdict=verdict,
            predicted_wall_s=1750.0,
            input_paths=[cfg, "experiments/run_experiment.py",
                         "basemap/pumap/parametric_umap/core.py", verdict, graph]))
    # 5) score the 3 prompted maps through ONE shared prompted reference
    runs = [f"prompted_s{s}={train_dirs[s]}" for s in (42, 43, 44)]
    jobs.append(Job(
        name="score_prompted", certifying=True, outputs=[prompted_scores],
        argv=[PY, "experiments/score_complete_panel.py", "--runs", *runs,
              "--testbed", P, "--source", SRC, "--reference", d("ref_prompted.npz"),
              "--out", prompted_scores],
        done_marker=d("score.done.json"), log=d("score.log"), manifest=d("score.manifest.json"),
        cwd=os.getcwd(), required_free_gb=12.0,
        deps=[f"train_prompted_s{s}" for s in (42, 43, 44)],
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py"]))
    # 6) compare vs unprompted G0 legacy + shift report (CPU)
    jobs.append(Job(
        name="compare", certifying=True, outputs=[compare_out],
        argv=[PY, "experiments/o1_compare.py", "--prompted", prompted_scores,
              "--unprompted", "experiments/evidence/r1_kernel/complete_200k_v22.json",
              "--shift", f"{P}/prompt_shift_report.json", "--out", compare_out],
        done_marker=d("compare.done.json"), log=d("compare.log"), manifest=d("compare.manifest.json"),
        cwd=os.getcwd(), required_free_gb=0.0, deps=["score_prompted"],
        input_paths=["experiments/o1_compare.py", prompted_scores]))

    summary = run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=d("o1_ctl.json"))
    print(json.dumps({j["name"]: j["status"] for j in summary["jobs"]}, indent=1))
    bad = [j for j in summary["jobs"] if j["status"] not in ("ok", "skipped_done")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
