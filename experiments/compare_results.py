#!/usr/bin/env python3
"""
Compare results across experiments.

Reads results.json files from experiment run directories and produces
comparison tables. Useful for analyzing sweeps and scaling experiments.

Usage:
    # Compare all runs in results directory
    python -m experiments.compare_results experiments/results/

    # Compare specific runs
    python -m experiments.compare_results experiments/results/run_1 experiments/results/run_2

    # Filter by name pattern
    python -m experiments.compare_results experiments/results/ --filter "scaling_*"

    # Output as CSV for further analysis
    python -m experiments.compare_results experiments/results/ --csv results.csv
"""

import argparse
import json
import sys
import os
import fnmatch
from pathlib import Path


def find_result_files(paths, pattern=None):
    """Find all results.json files in given paths."""
    results = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.name == "results.json":
            results.append(p)
        elif p.is_dir():
            for rfile in sorted(p.rglob("results.json")):
                if pattern:
                    run_name = rfile.parent.name
                    if not fnmatch.fnmatch(run_name, pattern):
                        continue
                results.append(rfile)
    return results


def load_results(paths):
    """Load and return list of (path, results_dict)."""
    loaded = []
    for p in paths:
        with open(p) as f:
            loaded.append((str(p.parent.name), json.load(f)))
    return loaded


def print_table(results, csv_path=None):
    """Print comparison table."""
    if not results:
        print("No results found.")
        return

    rows = []
    for name, r in results:
        cfg = r.get("config", {})
        model_cfg = cfg.get("model", {})
        train_cfg = cfg.get("train", {})
        data_cfg = cfg.get("data", {})

        row = {
            "name": cfg.get("name", name)[:35],
            "n_samples": r.get("data", {}).get("n_train", "?"),
            "input_dim": r.get("data", {}).get("n_features", "?"),
            "eval_mode": r.get("run_manifest", {}).get("eval_contract", {}).get("mode", "?"),
            "hidden_dim": model_cfg.get("hidden_dim", "?"),
            "n_layers": model_cfg.get("n_layers", "?"),
            "n_params": r.get("model", {}).get("n_params", "?"),
            "batch_size": train_cfg.get("batch_size", "?"),
            "n_epochs": train_cfg.get("n_epochs", "?"),
            "train_s": r.get("timing", {}).get("train_s", 0),
            "samp_per_sec": r.get("timing", {}).get("samples_per_sec", 0),
        }

        # Prefer test metrics; fall back to train metrics for transductive/precomputed runs.
        mt = r.get("metrics_test") or r.get("metrics_train", {})
        row["trust"] = mt.get("trustworthiness", None)
        row["dist_corr"] = mt.get("distance_correlation", None)
        row["knn_pres"] = mt.get("knn_preservation", None)
        row["silhouette"] = mt.get("silhouette", None)

        # UMAP baseline if available
        ub = r.get("umap_baseline", {}).get("metrics_test", {})
        row["umap_dist_corr"] = ub.get("distance_correlation", None)
        row["umap_knn_pres"] = ub.get("knn_preservation", None)

        rows.append(row)

    # Print table
    fmt = "{:<35} {:>8} {:>6} {:>6} {:>8} {:>9} {:>9} {:>9} {:>9}"
    header = fmt.format(
        "Name", "N_train", "H_dim", "Layers", "Params",
        "Train(s)", "Dist Corr", "KNN Pres", "Trust"
    )
    print(header)
    print("─" * len(header))
    for r in rows:
        def _fmt(v, dp=4):
            if v is None:
                return "—"
            if isinstance(v, float):
                return f"{v:.{dp}f}"
            return str(v)

        print(fmt.format(
            r["name"],
            _fmt(r["n_samples"], 0),
            _fmt(r["hidden_dim"], 0),
            _fmt(r["n_layers"], 0),
            _fmt(r["n_params"], 0),
            _fmt(r["train_s"], 1),
            _fmt(r["dist_corr"]),
            _fmt(r["knn_pres"]),
            _fmt(r["trust"]),
        ))

    # CSV output
    if csv_path:
        import csv
        keys = list(rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results")
    parser.add_argument("paths", nargs="+", help="Paths to result directories or files")
    parser.add_argument("--filter", type=str, default=None, help="Filter by name pattern (glob)")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    args = parser.parse_args()

    files = find_result_files([Path(p) for p in args.paths], args.filter)
    if not files:
        print("No results.json files found.")
        sys.exit(1)

    results = load_results(files)
    print_table(results, args.csv)


if __name__ == "__main__":
    main()
