#!/usr/bin/env python3
"""Create a wandb report for the parametric UMAP sweep experiments."""

import wandb_workspaces.reports.v2 as wr

ENTITY = "latent-interfaces"
PROJECT = "latent-basemap-sweep"

report = wr.Report(
    project=PROJECT,
    entity=ENTITY,
    title="Parametric UMAP — ls-squad Sweep",
    description="Parameter sweep over n_neighbors, hidden_dim, n_layers on SQuAD 20k embeddings (768d, nomic-embed-text-v1.5)",
)

report.blocks = [
    wr.TableOfContents(),

    # ── Section 1: Overview ──
    wr.H1("Sweep Progress & Overview"),
    wr.P("Training loss curves across all runs, colored by configuration."),
    wr.PanelGrid(
        runsets=[wr.Runset(project=PROJECT, entity=ENTITY, groupby=[])],
        panels=[
            wr.LinePlot(
                title="Training Loss (per batch)",
                x="global_step",
                y=["batch_loss"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="UMAP Loss",
                x="global_step",
                y=["umap_loss"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="Correlation Loss",
                x="global_step",
                y=["corr_loss"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="Learning Rate",
                x="global_step",
                y=["learning_rate"],
                groupby="run",
            ),
        ],
    ),

    # ── Section 2: Final Metrics Comparison ──
    wr.H1("Final Evaluation Metrics"),
    wr.P("Metrics computed on the held-out test set after training completes."),
    wr.PanelGrid(
        runsets=[wr.Runset(project=PROJECT, entity=ENTITY, groupby=[])],
        panels=[
            wr.BarPlot(
                title="Trustworthiness (test)",
                metrics=[wr.Metric(name="test/trustworthiness")],
                groupby="run",
            ),
            wr.BarPlot(
                title="Distance Correlation (test)",
                metrics=[wr.Metric(name="test/distance_correlation")],
                groupby="run",
            ),
            wr.BarPlot(
                title="KNN Preservation (test)",
                metrics=[wr.Metric(name="test/knn_preservation")],
                groupby="run",
            ),
            wr.BarPlot(
                title="Train Time (s)",
                metrics=[wr.Metric(name="train_time_s")],
                groupby="run",
            ),
        ],
    ),

    # ── Section 3: Reference UMAP Comparison ──
    wr.H1("vs Reference UMAP"),
    wr.P("How well does the parametric UMAP match the precomputed standard UMAP layout?"),
    wr.PanelGrid(
        runsets=[wr.Runset(project=PROJECT, entity=ENTITY, groupby=[])],
        panels=[
            wr.BarPlot(
                title="Procrustes Disparity (lower=better)",
                metrics=[wr.Metric(name="ref_test/procrustes_disparity")],
                groupby="run",
            ),
            wr.BarPlot(
                title="2D Distance Correlation vs Ref",
                metrics=[wr.Metric(name="ref_test/dist_correlation_2d")],
                groupby="run",
            ),
            wr.BarPlot(
                title="2D KNN Overlap vs Ref",
                metrics=[wr.Metric(name="ref_test/knn_overlap_2d")],
                groupby="run",
            ),
        ],
    ),

    # ── Section 4: Parameter Importance ──
    wr.H1("Parameter Relationships"),
    wr.P("Scatter plots to see which hyperparameters drive quality."),
    wr.PanelGrid(
        runsets=[wr.Runset(project=PROJECT, entity=ENTITY, groupby=[])],
        panels=[
            wr.ScatterPlot(
                title="n_neighbors vs Trustworthiness",
                x=wr.SummaryMetric(name="n_neighbors"),
                y=wr.SummaryMetric(name="test/trustworthiness"),
            ),
            wr.ScatterPlot(
                title="hidden_dim vs Distance Correlation",
                x=wr.SummaryMetric(name="hidden_dim"),
                y=wr.SummaryMetric(name="test/distance_correlation"),
            ),
            wr.ScatterPlot(
                title="n_layers vs KNN Preservation",
                x=wr.SummaryMetric(name="n_layers"),
                y=wr.SummaryMetric(name="test/knn_preservation"),
            ),
            wr.ScatterPlot(
                title="n_params vs Train Time",
                x=wr.SummaryMetric(name="n_params"),
                y=wr.SummaryMetric(name="train_time_s"),
            ),
            wr.ParallelCoordinatesPlot(
                columns=[
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="n_neighbors")),
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="hidden_dim")),
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="n_layers")),
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="test/trustworthiness")),
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="test/distance_correlation")),
                    wr.ParallelCoordinatesPlotColumn(metric=wr.Metric(name="ref_test/procrustes_disparity")),
                ],
            ),
        ],
    ),

    # ── Section 5: Training Dynamics ──
    wr.H1("Training Dynamics"),
    wr.P("Gradient norms, distance stats, and edge quality metrics during training."),
    wr.PanelGrid(
        runsets=[wr.Runset(project=PROJECT, entity=ENTITY, groupby=[])],
        panels=[
            wr.LinePlot(
                title="Gradient Norm (post-clip)",
                x="global_step",
                y=["grad_norm_post"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="Clipping Ratio",
                x="global_step",
                y=["clipping_ratio"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="Mean Embedding Distance",
                x="global_step",
                y=["mean_distance"],
                groupby="run",
                smoothing_factor=0.8,
            ),
            wr.LinePlot(
                title="Positive vs Negative Q values",
                x="global_step",
                y=["pos_qs_mean", "neg_qs_mean"],
                groupby="run",
                smoothing_factor=0.8,
            ),
        ],
    ),
]

report.save()
print(f"Report URL: {report.url}")
