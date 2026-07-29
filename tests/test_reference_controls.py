from types import SimpleNamespace

import torch
from torch import nn

from main import set_trainability_policy
from models.model_support import method_category


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 6), nn.ReLU(), nn.BatchNorm1d(6))
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        return self.head(self.backbone(x))


def test_full_fine_tuning_trains_every_parameter_and_receives_gradients():
    model = TinyClassifier().train()
    set_trainability_policy(model, SimpleNamespace(tuning_method="full"))
    assert method_category("full") == "reference_control"
    assert all(parameter.requires_grad for parameter in model.parameters())

    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 4)), torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_linear_probing_freezes_backbone_and_trains_only_task_head():
    model = TinyClassifier().train()
    set_trainability_policy(model, SimpleNamespace(tuning_method="linear"))
    assert method_category("linear") == "reference_control"

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"head.weight", "head.bias"}
    assert not model.backbone[2].training

    loss = nn.CrossEntropyLoss()(model(torch.randn(4, 4)), torch.tensor([0, 1, 2, 1]))
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.head.bias.grad is not None
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
