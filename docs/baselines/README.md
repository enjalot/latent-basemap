# Publication baselines

Two queues fill the main comparison gap in the paper.

| study | scale | question | default run |
|---|---:|---|---|
| [50M fneg-off control](50m-fneg-off.md) | 50M | What does fneg change when every other training choice stays fixed? | seeds 42, 43, 44 |
| [Upstream ParamRepulsor](paramrepulsor-2m.md) | 2M | How does the released external method score on the same substrate and instruments? | seed 42 pilot |

Both queues are descriptive. They do not register a gate, select a method, or
change the promoted 100M recipe.

## After the GPU is free

Use clean detached worktrees at the baseline commit. The 50M control uses the
standard latent-basemap environment. Prepare its full family with:

```bash
release=$(git -C /home/enjalot/code/latent-basemap-baselines rev-parse HEAD)
git -C /home/enjalot/code/latent-basemap-baselines worktree add --detach \
  /home/enjalot/code/latent-basemap-baselines-run "$release"
ln -s /home/enjalot/code/latent-basemap-run/.venv \
  /home/enjalot/code/latent-basemap-baselines-run/.venv
cd /home/enjalot/code/latent-basemap-baselines-run
.venv/bin/python \
  -m experiments.prepare_baseline_50m_fneg_off_queue \
  --release-sha "$release"
```

For an initial paired run, add `--seeds 43 --queue-root
/data/latent-basemap/runs/round-0269/queue-seed43-pilot`.

ParamRepulsor needs its own checkout because its published package requires
Python below 3.12 and pins NumPy 2.0.2 and Numba 0.60.0:

```bash
release=$(git -C /home/enjalot/code/latent-basemap-baselines rev-parse HEAD)
git -C /home/enjalot/code/latent-basemap-baselines worktree add --detach \
  /home/enjalot/code/latent-basemap-paramrepulsor-run "$release"
/home/enjalot/code/latent-basemap-paramrepulsor-run/experiments/setup_paramrepulsor_env.sh \
  /home/enjalot/code/latent-basemap-paramrepulsor-run
cd /home/enjalot/code/latent-basemap-paramrepulsor-run
.venv/bin/python \
  -m experiments.prepare_paramrepulsor_2m_queue \
  --release-sha "$release"
```

The setup script refuses an existing `.venv`; this prevents it from replacing
the production environment by accident.

Once the queues have been prepared, inspect them without launching a node:

```bash
roundrun /data/latent-basemap/runs/round-0269/queue/queue.json --preflight-only
roundrun /data/latent-basemap/runs/round-0270/queue/queue.json --preflight-only
```

Launch with the same commands after removing `--preflight-only`.
