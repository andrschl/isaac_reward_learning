# isaac_irl
IRL workflow for Isaac Lab: train RL policies, collect synthetic demos, and train a learned reward.

## Setup
Install Isaac Lab first: https://isaac-sim.github.io/IsaacLab/

Use your existing Isaac Lab environment:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Or create a local Conda environment:

```bash
conda env create -f environment.yml
conda activate isaac_irl
```

## Key Commands
Run all commands from repository root.

### Train RL Policy
```bash
python scripts/rsl_rl/train_rl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless
```

### Evaluate / Export Policy
```bash
python scripts/rsl_rl/play.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --load_run "<run_folder_name>" \
  --checkpoint "model_2000.pt"
```

Livestream instead of headless: replace `--headless` with `--livestream 2` and
connect via the Isaac Sim WebRTC Streaming Client at `<host-ip>:8211`.

### Collect Synthetic Demos
```bash
python scripts/recording/record_synthetic_demos.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --num_demos 1000 \
  --load_run "<run_folder_name>" \
  --checkpoint "model_2000.pt"
```

By default, demo rollout length matches the environment's built-in max episode length.
Use `--demo_length <steps>` only if you want to override it.

### Train IRL
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5
```

Optional IRL discount override:
- `--irl_discount_gamma <0..1>` (default: uses PPO `gamma`)
- IRL return targets are normalized by episode length by default (`irl.normalize_returns_by_episode_length: true`).

### Mix Behavioral Cloning into the Policy Loss
The PPO actor loss can be convex-combined with a BC term on the same expert
demos:

```
L_policy = (1 - α) · L_PPO  +  α · L_BC
```

Set `bc.alpha` in `configs/franka_lift/experiment.yaml` or pass `--bc_alpha`.
The BC dataset is the same HDF5 written by `record_synthetic_demos.py` — the
loader reads the `obs/<group>/<leaf>` and `actions` channels alongside the
`features` channel already consumed by IRL, so no extra files are needed.

```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --bc_alpha 0.3
```

- `--bc_alpha 0` → pure PPO+IRL (default; identical to the IRL-only path).
- `--bc_alpha 1` → pure offline BC: normal PPO/IRL passes are skipped;
  periodic stochastic + inference validation rollouts may still run for logging,
  while the policy is updated from expert (obs, action) pairs each iteration.
- `--bc_loss_type nll|mse` selects the BC loss (default `nll` — log-likelihood
  of the expert action under the actor's Gaussian).

Logged metrics: `BC/loss`, `BC/alpha`.

### Sweep BC `alpha` Across Multiple Runs
`scripts/irl/bc_alpha_sweep.sh` runs `train_irl.py` sequentially across a list
of `bc.alpha` values (and optionally seeds). With the default `LOGGER=wandb`,
each run is logged with proper metadata so you can compare runs cleanly in the UI:

- `name`   = `alpha_<value>` (or `alpha_<value>_seed_<s>` when sweeping seeds)
- `group`  = the sweep's `EXPERIMENT_NAME` (so the runs cluster together)
- `tags`   = `alpha=...`, `seed=...`, `loss=...` (for filtering)
- `config` = key hyperparams (`bc/alpha`, `bc/loss_type`, `ppo/learning_rate`,
  `irl/discount_gamma`, ...) so you can plot any metric vs. any hyperparam

```bash
# Defaults: alphas = 0.0 0.1 0.3 0.5 0.7 1.0; seed = 42; logger = wandb.
scripts/irl/bc_alpha_sweep.sh

# Customize via env vars (see the script header for the full list):
ALPHAS="0 0.5 1" SEEDS="1 2 3" \
  EXPERIMENT_NAME=bc_ablation \
  LOG_PROJECT_NAME=isaac_irl_bc_ablation \
  scripts/irl/bc_alpha_sweep.sh
```

For W&B runs, make sure `WANDB_API_KEY` is set (put it in `.env` at the
repo root — it's auto-loaded — or run `wandb login` once).

Per-run local artifacts also land under
`logs/irl/<EXPERIMENT_NAME>/<timestamp>_alpha_<value>/` (checkpoints, configs,
videos when rollouts are active), so you keep a local copy alongside W&B.
`alpha=1` is BC-only, so the sweep skips final-video recording for that run.

### Train IRL + Record Video Every 50 Iterations
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --video \
  --video_interval_iterations 50 \
  --video_length 1500
```

Videos are saved under `logs/irl/<experiment>/<run>/videos/train/`.
When `--video_interval_iterations` is used, video filenames use learning-iteration indices:
`rl-video-iter-0.mp4`, `rl-video-iter-50.mp4`, `rl-video-iter-100.mp4`, ...

### Resume IRL
```bash
python scripts/irl/train_irl.py \
  --task Isaac-Lift-Cube-Franka-v0 \
  --headless \
  --expert_data_path logs/demos/franka_lift/demos.hdf5 \
  --resume \
  --load_run "<previous_irl_run_folder>" \
  --checkpoint "model_*.pt"
```

### Logging
IRL training logs to Weights & Biases by default. Use `--logger noop` for local smoke runs without external logging.

## End-to-End Example
```bash
python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless
python scripts/rsl_rl/play.py --task Isaac-Lift-Cube-Franka-v0 --headless --load_run "<run>" --checkpoint "<ckpt>"
python scripts/recording/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --headless --num_demos 1000 --load_run "<run>" --checkpoint "<ckpt>"
python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path logs/demos/franka_lift/demos.hdf5
```

## Tests
```bash
PYTHONPATH=src pytest -q
```
