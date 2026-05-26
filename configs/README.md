# Config Layout

- `configs/franka_lift/experiment.yaml`: single experiment config (env, reward, policy, algo, irl, runner, bc, etc.).

## Reward Types

- `dense`: MLP reward model. Uses `hidden_dims`, `activation`. Default: L2 regularization.
- `linear`: linear reward model (r = w^T x). Uses `linear_projection`. Default: L2 ball projection.

Field applicability:
- Dense: `hidden_dims`, `activation` apply; `linear_projection` invalid when not linear.
- Linear: `linear_projection`, `linear_projection_radius` apply; `hidden_dims`, `activation` ignored.
- Both: `regularization`, `regularization_strength`, `elastic_alpha`.

## Hard-Break Changes

The train parser rejects stale keys:
- `runner.reward_update_interval`
- `runner.num_steps_per_env_rl`
- `runner.policy_updates_per_cycle`
- `runner.imitator_buffer.store_discounted_feature_returns`
- `runner.expert_buffer.store_discounted_feature_returns`

Reward config keys must match `RewardModelCfg` directly:
- `reward.hidden_dims` (not `reward_hidden_dims`)
- `reward.is_linear` (not `reward_is_linear`)

IRL return settings (`irl`):
- `discount_gamma`: optional discount override for IRL reward updates (`null` = use PPO `algo.gamma`).
- `normalize_returns_by_episode_length`: divide discounted returns by episode length (default `true`).

Runner update settings (`runner`):
- `use_learned_reward`: train the agent on the learned reward model instead of the raw env reward.
- `steps_per_env_per_cycle`: vectorized env steps collected before an agent update.
- `rl_updates_per_cycle`: RL-algorithm update calls per learning iteration.

Runner validation settings (`runner`):
- `validation_interval`: run stochastic + inference validation every N learning iterations (`0` disables).
- `validation_steps_per_env`: env steps per validation mode (`null` = `5 * steps_per_env_per_cycle`, `0` disables).
- Validation emits feature returns only: `ValidationFeatures/<mode>/{expert,imitator,gap}/<feature>`; old reward-update `EvalFeatures/*` logs are intentionally disabled.

## Behavioral Cloning (`bc`)

Convex-combines a BC term into the policy loss: `L = (1 - α) · L_PPO + α · L_BC`.
The BC dataset is read from `irl.expert_data_path` (same HDF5 as IRL features) —
the loader pulls `data/demo_*/obs/<group>/<leaf>` and `data/demo_*/actions`
alongside the `features` channel already consumed by IRL.

Fields (under `bc:`):
- `alpha` (float in `[0, 1]`, default `0.0`): mixing weight. `0` = pure PPO+IRL
  (no BC overhead), `1` = pure offline BC (normal PPO/IRL passes are skipped;
  periodic stochastic + inference validation rollouts may still run for logging).
- `loss_type` (`nll` | `mse`, default `nll`): `nll` maximizes log-likelihood of
  the expert action under the actor's Gaussian; `mse` regresses the actor's
  mean action to the expert action.
- `batch_size` (int, default `256`): expert minibatch size sampled each PPO
  optimizer step.
- `val_fraction` (float in `[0, 1)`, default `0.0`): fraction of expert
  *episodes* held out (never trained on) to report a validation BC loss.
  The split is at the episode level, so no trajectory leaks between train and
  val. `0` disables it; the held-out loss is computed during validation passes
  (see `runner.validation_interval`).

CLI overrides: `--bc_alpha`, `--bc_loss_type`. Logged metrics: `BC/loss`,
`BC/alpha`, and (when `val_fraction > 0`) `BC/val_loss`.
