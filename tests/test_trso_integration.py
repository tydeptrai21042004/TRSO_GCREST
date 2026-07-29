from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from main import calibrate_trso_model
from models.tuning_modules.mdl_tangent_core import (
    mdl_tangent_parameter_count,
    merge_mdl_tangent_cores_,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


def _loader():
    generator = torch.Generator().manual_seed(17)
    images = torch.randn(16, 3, 8, 8, generator=generator)
    labels = torch.arange(16) % 3
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def _args(tmp_path):
    return SimpleNamespace(
        task_type="single_label",
        smoothing=0.0,
        output_dir=str(tmp_path),
    )


def test_end_to_end_mdl_calibration_writes_report(tmp_path):
    model = TinyClassifier()
    running_mean = model.features[1].running_mean.clone()
    selected = calibrate_trso_model(model, _loader(), torch.device("cpu"), _args(tmp_path))

    report_path = tmp_path / "mdl_tangent_calibration.json"
    assert report_path.is_file()
    assert isinstance(selected, list)
    assert any(parameter.requires_grad for parameter in model.parameters())
    assert mdl_tangent_parameter_count(model) >= 0
    torch.testing.assert_close(model.features[1].running_mean, running_mean)


def test_calibrated_model_trains_and_merges_without_output_change(tmp_path):
    model = TinyClassifier()
    calibrate_trso_model(model, _loader(), torch.device("cpu"), _args(tmp_path))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable

    optimizer = torch.optim.SGD(trainable, lr=1e-2)
    images, labels = next(iter(_loader()))
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        before = model(images)
    merged = merge_mdl_tangent_cores_(model)
    with torch.no_grad():
        after = model(images)

    assert merged >= 0
    assert mdl_tangent_parameter_count(model) == 0
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    assert not any(hasattr(module, "parametrizations") for module in model.modules())
