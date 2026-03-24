from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Add project root so storage, utils are importable
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _load_train_irl_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "scripts" / "irl" / "train_irl.py"
    module_name = "train_irl_module_expert_loader"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_load_feature_episodes_accepts_pt_payload(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.pt"

    torch.save(
        {
            "episodes": [
                torch.ones(3, 2, dtype=torch.float32),
                torch.zeros(5, 2, dtype=torch.float32),
            ]
        },
        payload_path,
    )

    episodes = module.load_feature_episodes(str(payload_path), expected_feature_dim=2)
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (5, 2)


def test_load_feature_episodes_accepts_named_hdf5_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")

        demo_0 = data_group.create_group("demo_0")
        demo_0_features = demo_0.create_group("features")
        demo_0_features.create_dataset("reach", data=torch.ones(3).numpy())
        demo_0_features.create_dataset("lift", data=(2.0 * torch.ones(3)).numpy())

        demo_1 = data_group.create_group("demo_1")
        demo_1_features = demo_1.create_group("features")
        demo_1_features.create_dataset("reach", data=torch.zeros(4).numpy())
        demo_1_features.create_dataset("lift", data=torch.ones(4).numpy())

    episodes = module.load_feature_episodes(str(payload_path), expected_feature_dim=2)
    assert len(episodes) == 2
    assert episodes[0].shape == (3, 2)
    assert episodes[1].shape == (4, 2)
    # Feature columns are sorted by name: "lift" < "reach"
    assert torch.allclose(episodes[0][:, 0], 2.0 * torch.ones(3))  # lift
    assert torch.allclose(episodes[0][:, 1], torch.ones(3))  # reach
    assert torch.allclose(episodes[1][:, 0], torch.ones(4))  # lift
    assert torch.allclose(episodes[1][:, 1], torch.zeros(4))  # reach


def test_load_feature_episodes_rejects_legacy_hdf5_matrix_payload(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    module = _load_train_irl_module()
    payload_path = tmp_path / "legacy.hdf5"

    with h5py.File(payload_path, "w") as file_handle:
        data_group = file_handle.create_group("data")
        demo_0 = data_group.create_group("demo_0")
        demo_0.create_dataset("features", data=torch.ones(3, 2).numpy())

    with pytest.raises(ValueError, match="legacy feature format|Regenerate demos"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_non_feature_torch_payload(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad.pt"
    torch.save({"obs": torch.randn(3, 2)}, payload_path)

    with pytest.raises(ValueError, match="must contain one of keys"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_load_feature_episodes_rejects_feature_dim_mismatch(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "bad_dim.pt"
    torch.save([torch.randn(3, 3)], payload_path)

    with pytest.raises(ValueError, match="feature dim mismatch"):
        module.load_feature_episodes(str(payload_path), expected_feature_dim=2)


def test_subset_episodes_first_returns_first_n():
    module = _load_train_irl_module()
    episodes = [
        torch.ones(2, 2),
        torch.ones(3, 2) * 2,
        torch.ones(4, 2) * 3,
    ]
    out = module._subset_episodes(episodes, max_num=2, strategy="first", seed=42)
    assert len(out) == 2
    assert torch.allclose(out[0], episodes[0])
    assert torch.allclose(out[1], episodes[1])


def test_subset_episodes_random_same_seed_returns_same_subset():
    module = _load_train_irl_module()
    episodes = [torch.ones(i + 1, 2) * i for i in range(10)]
    out_a = module._subset_episodes(episodes, max_num=3, strategy="random", seed=123)
    out_b = module._subset_episodes(episodes, max_num=3, strategy="random", seed=123)
    assert len(out_a) == 3
    assert len(out_b) == 3
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b)


def test_make_expert_buffer_loader_with_max_num_trajectories_loads_subset(tmp_path: Path):
    module = _load_train_irl_module()
    payload_path = tmp_path / "expert.pt"
    torch.save(
        {"episodes": [torch.ones(2, 2), torch.ones(3, 2), torch.ones(4, 2)]},
        payload_path,
    )

    from utils.runtime_context import RuntimeContext
    from storage.feature_storage import FeatureBufCfg, FeatureTrajectoryBuffer

    loader = module._make_expert_buffer_loader(
        str(payload_path),
        expected_feature_dim=2,
        max_num_trajectories=1,
        subset_strategy="first",
        seed=42,
    )
    ctx = RuntimeContext(num_envs=1, feature_dim=2)
    buffer = FeatureTrajectoryBuffer(cfg=FeatureBufCfg(min_ep_len=1), ctx=ctx, gamma=0.99)
    loader(buffer)
    assert len(buffer) == 1
