# Config Layout

- `configs/franka_lift/experiment.yaml`: single experiment config (env, reward, policy, algo, irl, runner, etc.).

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
- `runner.imitator_buffer.store_discounted_feature_returns`
- `runner.expert_buffer.store_discounted_feature_returns`

Reward config keys must match `RewardModelCfg` directly:
- `reward.hidden_dims` (not `reward_hidden_dims`)
- `reward.is_linear` (not `reward_is_linear`)

IRL return settings (`irl`):
- `discount_gamma`: optional discount override for IRL reward updates (`null` = use PPO `algo.gamma`).
- `normalize_returns_by_episode_length`: divide discounted returns by episode length (default `true`).
