from types import SimpleNamespace

from datasets.build import available_datasets, _landmark_regression_transform
from datasets.download import normalize_download_mode, should_download
from datasets.preprocessing import current_preprocessing_profile
from datasets.registry import get_dataset_spec


def test_registry_and_router_stay_aligned():
    names = available_datasets()
    assert len(names) >= 54
    for name in ("dtd", "eurosat", "pcam", "pathmnist", "tissuemnist"):
        assert name in names
        assert get_dataset_spec(name).name == name


def test_download_policy_is_backward_compatible():
    assert normalize_download_mode(True) == "yes"
    assert normalize_download_mode(False) == "no"
    assert should_download("auto", "safe_auto")
    assert not should_download("auto", "large_auto")
    assert should_download("yes", "large_auto")
    assert not should_download("yes", "manual")


def test_landmark_transform_no_longer_depends_on_private_structured_helper():
    args = SimpleNamespace(input_size=64, train_interpolation="bicubic", imagenet_norm=True)
    transform = _landmark_regression_transform(args)
    assert transform is not None


def test_preprocessing_profile_is_serializable():
    args = SimpleNamespace(input_size=224, train_interpolation="bicubic", crop_ratio=0.875, imagenet_norm=True)
    profile = current_preprocessing_profile(args).to_dict()
    assert profile["input_size"] == 224
    assert len(profile["mean"]) == 3
