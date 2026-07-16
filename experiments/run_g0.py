"""G0 — build shared v2.2 references and rescore the persisted kernel maps.

No training. Runs through the controller (one scoring process per corpus, under a
held lease), builds ONE verified reference per corpus, rescores every kernel map
through it (each stamps hiD_reference_reused=true), then a CPU decision node emits
kernel_decision_v22.json (legacy vs umap(a=b=1) and legacy vs fitted std curve as
separate questions). Finally the 2M rescore telemetry drives the wall/peak
regression gate — a fresh performance test, not a static reread.
"""
from __future__ import annotations
import argparse, os, sys, glob, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from basemap.run_controller import Job, run_jobs, known_service_pids

PY = ".venv/bin/python"
RES = "experiments/results"
SRC = "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
EVID = "experiments/evidence/r1_kernel"
CLO = "/data/latent-basemap/closure/g0"
ARMS = ["legacy_a1b1", "umap_a1b1", "umap_stdcurve"]


def _one(pat):
    ds = sorted(glob.glob(os.path.join(RES, pat)))
    if not ds:
        raise SystemExit(f"G0: no run dir for {pat}")
    return ds[-1]


def runs_for(corpus, seeds):
    pref = "r1_kernel_2m_" if corpus == "2m" else "r1_kernel_"
    out = []
    for arm in ARMS:
        for s in seeds:
            out.append(f"{arm}_s{s}={_one(f'{pref}{arm}_s{s}_*')}")
    return out


def score_job(corpus, testbed, seeds, out_path, ref_path):
    return Job(
        name=f"rescore_{corpus}", certifying=True, outputs=[out_path],
        argv=[PY, "experiments/score_complete_panel.py", "--runs", *runs_for(corpus, seeds),
              "--testbed", testbed, "--source", SRC, "--reference", ref_path, "--out", out_path],
        done_marker=f"{CLO}/rescore_{corpus}.done.json", log=f"{CLO}/rescore_{corpus}.log",
        manifest=f"{CLO}/rescore_{corpus}.manifest.json", cwd=os.getcwd(),
        required_free_gb=24.0,
        input_paths=["experiments/score_complete_panel.py", "basemap/panel_v2.py"])


def main():
    os.makedirs(CLO, exist_ok=True); os.makedirs(EVID, exist_ok=True)
    out200 = f"{EVID}/complete_200k_v22.json"
    out2m = f"{EVID}/complete_2m_v22.json"
    decision = f"{EVID}/kernel_decision_v22.json"
    jobs = [
        score_job("200k", "/data/latent-basemap/jina-en-200k", (42, 43, 44), out200,
                  f"{CLO}/ref_200k.npz"),
        score_job("2m", "/data/latent-basemap/jina-en-2m", (42, 43), out2m,
                  f"{CLO}/ref_2m.npz"),
        # decision node also runs the fresh-2M wall/peak envelope check (G0.3).
        Job(name="kernel_decision", certifying=True, outputs=[decision],
            argv=[PY, "experiments/kernel_decision.py", "--p200", out200, "--p2m", out2m,
                  "--out", decision],
            done_marker=f"{CLO}/decision.done.json", log=f"{CLO}/decision.log",
            manifest=f"{CLO}/decision.manifest.json", cwd=os.getcwd(), required_free_gb=0.0,
            deps=["rescore_200k", "rescore_2m"],
            input_paths=["experiments/kernel_decision.py", out200, out2m]),
    ]
    summary = run_jobs(jobs, allowed_pids=known_service_pids(), summary_path=f"{CLO}/g0_ctl.json")
    print(json.dumps({j["name"]: j["status"] for j in summary["jobs"]}, indent=1))
    bad = [j for j in summary["jobs"] if j["status"] not in ("ok", "skipped_done")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
