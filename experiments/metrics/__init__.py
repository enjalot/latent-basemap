"""Production, CPU-only map-quality metrics for the basemap program.

This directory is shared by several metric proposals; each ships its own
module, driver, and tests, and none of them registers anything.

- ``collapse_fog.py``       COLLAPSE and FOG (Metric Option A), with
                            ``null_calibration.py`` for the Gaussian-null
                            ``median -/+ k * MAD_n`` multiplier and
                            ``calibrate_collapse_fog.py`` as the driver.
                            Numbers and bands in ``REPORT.md``.
- ``k256_two_sided.py``     the restored TWO-SIDED ``purity_fidelity_k256``
                            criterion (Metric Option B), with its own Gaussian
                            -null recalibration of ``k2`` at ``n = 29`` and a
                            would-be-verdict driver. Numbers, band and verdict
                            table in ``REPORT-k256.md``.
- ``density_v3.py``         the repaired density metric (Metric Option C):
                            degenerate-anchor exclusion, an 8,000-anchor seeded
                            draw, and Spearman as the primary statistic, with
                            ``density_v3_validation.py`` as the driver.
                            Numbers, stability sweep and caveats in
                            ``REPORT-density.md``.
"""
