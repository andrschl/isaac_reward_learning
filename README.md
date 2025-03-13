# isaac_reward_learning
Library for inverse reinforcement learning and RLHF for robotic manipulation.

## Collecting trajectories from existing policy
- `python scripts/rsl_rl/train_rl.py --task Isaac-Lift-Cube-Franka-v0 --headless`
- `python scripts/utils/record_synthetic_demos.py --task Isaac-Lift-Cube-Franka-v0 --num_demos 1000`
- `python scripts/irl/train_irl.py --task Isaac-Lift-Cube-Franka-v0 --headless --expert_data_path 'logs/rsl_rl/franka_lift/demos/hdf_dataset.hdf5'`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/tools/inspect_demonstrations.py logs/rsl_rl/franka_lift/DATASET`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/tools/split_train_val.py PATH_TO_DATASET --ratio 0.2`
- `~/IsaacLab/isaaclab.sh -p scripts/imitation_learning/train_bc.py --task Isaac-Lift-Cube-Franka-v0 --algo bc --dataset scripts/imitation_learning/logs/demos/hdf_dataset.hdf5`
- ``

