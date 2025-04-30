export ISAACLAB_PATH=/workspace/isaaclab &&
cd /workspace/isaaclab &&
/isaac-sim/python.sh -m pip uninstall -y toml &&
/isaac-sim/python.sh -m pip install toml &&
/isaac-sim/python.sh -m pip install --upgrade pip &&
/isaac-sim/python.sh -m pip install --upgrade pip setuptools wheel &&
/isaac-sim/python.sh -m pip install -e source/isaac_reward_learning &&
/isaac-sim/python.sh -m pip install -e source/isaaclab_tasks &&
export WANDB_USERNAME=sebastien-epfl-epfl &&
export WANDB_API_KEY=377aa0f3fcfdedeeed6ad9a746b76bb67204b0e9 &&
echo $TMPDIR &&