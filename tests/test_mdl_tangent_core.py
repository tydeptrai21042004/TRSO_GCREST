import copy
import inspect

import torch
from torch import nn
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

from models.tuning_modules.mdl_tangent_core import (
    MDLTangentCoreParametrization,
    calibrate_mdl_tangent_core,
    iter_mdl_tangent_parametrizations,
    mdl_tangent_basis_value_count,
    mdl_tangent_parameter_count,
    merge_mdl_tangent_cores_,
)


class TinyTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.norm = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.norm(self.proj(x))))


def loader():
    features = torch.eye(4).repeat(8, 1)
    labels = torch.tensor([0, 1, 0, 1]).repeat(8)
    return DataLoader(TensorDataset(features, labels), batch_size=4, shuffle=False)


def test_proposal_api_has_no_manual_model_selection_controls():
    signature = inspect.signature(calibrate_mdl_tangent_core)
    forbidden = {
        "rank", "max_rank", "budget", "parameter_budget", "threshold",
        "layers", "batches", "calibration_batches", "head_policy",
        "dense_core", "diagonal_core", "gain",
    }
    assert forbidden.isdisjoint(signature.parameters)


def test_calibration_is_batchnorm_safe_and_identity_preserving():
    torch.manual_seed(0)
    model = TinyTransferNet()
    model.train()
    probe = torch.randn(6, 4)
    running_mean = model.norm.running_mean.clone()
    running_var = model.norm.running_var.clone()
    model.eval()
    with torch.no_grad():
        before = model(probe).clone()
    model.train()

    report = calibrate_mdl_tangent_core(
        model, loader(), nn.CrossEntropyLoss(), device="cpu"
    )

    model.eval()
    with torch.no_grad():
        after = model(probe)
    torch.testing.assert_close(before, after, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(model.norm.running_mean, running_mean)
    torch.testing.assert_close(model.norm.running_var, running_var)
    assert report.calibration_batches == len(loader())
    assert report.method == "reliability_weighted_spectral_allocation_full_core"
    payload = getattr(model, "_mdl_tangent_report")
    assert payload["legacy_method_id"] == "global_cross_fitted_reproducibility_entropy_spectral_tangent_core"
    assert "partition consistency" in payload["terminology_note"].lower()
    assert mdl_tangent_parameter_count(model) == report.adapter_parameters
    assert mdl_tangent_basis_value_count(model) == report.frozen_basis_values
    assert report.head_policy == "full"
    assert report.head_trainable_parameters == sum(p.numel() for p in model.head.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())
    assert any(parameter.requires_grad for parameter in model.parameters())


def test_selected_update_trains_and_exactly_merges():
    torch.manual_seed(1)
    model = TinyTransferNet()
    report = calibrate_mdl_tangent_core(
        model, loader(), nn.CrossEntropyLoss(), device="cpu"
    )
    assert report.selected_tensors >= 1
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.05)
    features, labels = next(iter(loader()))
    loss = nn.CrossEntropyLoss()(model(features), labels)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        before = model(features)
    attached = list(iter_mdl_tangent_parametrizations(model))
    merged = merge_mdl_tangent_cores_(model)
    with torch.no_grad():
        after = model(features)
    assert merged == len(attached)
    assert not list(iter_mdl_tangent_parametrizations(model))
    assert mdl_tangent_parameter_count(model) == 0
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)


def test_diagonal_core_uses_linear_not_quadratic_coordinates():
    left = torch.eye(5)[:, :3]
    right = torch.eye(7)[:, :3]
    diagonal = MDLTangentCoreParametrization(left, right, (5, 7), dense_core=False)
    dense = MDLTangentCoreParametrization(left, right, (5, 7), dense_core=True)
    assert diagonal.trainable_parameter_count == 3
    assert dense.trainable_parameter_count == 9
    assert diagonal.basis_value_count == dense.basis_value_count == 36


def test_calibration_is_deterministic():
    torch.manual_seed(7)
    first = TinyTransferNet()
    second = copy.deepcopy(first)
    report_a = calibrate_mdl_tangent_core(
        first, loader(), nn.CrossEntropyLoss(), device="cpu"
    )
    report_b = calibrate_mdl_tangent_core(
        second, loader(), nn.CrossEntropyLoss(), device="cpu"
    )
    assert report_a.to_dict() == report_b.to_dict()


class TinyTokenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(6)
        self.attn = nn.Linear(6, 6)
        self.norm2 = nn.LayerNorm(6)
        self.mlp = nn.Sequential(nn.Linear(6, 12), nn.GELU(), nn.Linear(12, 6))
        self.head = nn.Linear(6, 3)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return self.head(x.mean(dim=1))


def test_transformer_like_model_uses_same_architecture_neutral_rule():
    torch.manual_seed(11)
    inputs = torch.randn(24, 5, 6)
    labels = torch.arange(24) % 3
    data = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    model = TinyTokenNet()
    report = calibrate_mdl_tangent_core(
        model, data, nn.CrossEntropyLoss(), device="cpu"
    )
    names = {record.name for record in report.records}
    assert "attn.weight" in names
    assert "mlp.0.weight" in names
    assert "mlp.2.weight" in names
    assert not any("norm" in name for name in names)
    assert any(parameter.requires_grad for parameter in model.parameters())



def test_full_rule_uses_global_geometric_allocation_without_rescue():
    torch.manual_seed(19)
    model = TinyTransferNet()
    report = calibrate_mdl_tangent_core(
        model, loader(), nn.CrossEntropyLoss(), device="cpu", ablation="full"
    )
    assert report.dense_rescue_activated is False
    payload = getattr(model, "_mdl_tangent_report")
    assert payload["selection_rule"] == "global_geometric_partition_consistency_allocation_full_core"
    assert payload["legacy_selection_rule"] == "global_geometric_information_dimension_allocation_full_core"
    assert all(record.core_mode in {"global_geometric_evidence_full_core", "global_geometric_unallocated"} for record in report.records)


def test_sampling_variance_ablation_is_explicit_and_not_a_capacity_control():
    torch.manual_seed(23)
    model = TinyTransferNet()
    report = calibrate_mdl_tangent_core(
        model, loader(), nn.CrossEntropyLoss(), device="cpu",
        ablation="no_sampling_variance",
    )
    assert report.ablation == "no_sampling_variance"
    payload = getattr(model, "_mdl_tangent_report")
    assert payload["selection_rule"] == "partition_consistency_without_sampling_variance_ablation"
    assert payload["legacy_selection_rule"] == "crossfit_without_sampling_variance_ablation"


def test_reviewer_mode_count_rules_and_diagnostics_are_exposed():
    torch.manual_seed(7)
    model = TinyTransferNet()
    report = calibrate_mdl_tangent_core(
        model, loader(), nn.CrossEntropyLoss(), device="cpu",
        mode_count_rule="harmonic", r_scale=0.5,
        calibration_fraction=0.5, partition_mode="seeded_random", partition_seed=11,
        svd_power_iterations=1,
    )
    payload = getattr(model, "_mdl_tangent_report")
    assert report.mode_count_rule == "harmonic"
    assert report.r_scale == 0.5
    assert payload["mode_count_rule"] == "harmonic"
    assert payload["global_selected_modes"] == report.global_selected_modes
    assert payload["global_numerical_support"] >= payload["global_selected_modes"]
    assert payload["global_shannon_effective_modes"] >= 1.0
    assert payload["candidate_modes"] >= payload["global_numerical_support"]
    assert payload["partition_mode"] == "seeded_random"
    assert payload["partition_seed"] == 11
    assert isinstance(payload["layer_ranks"], dict)
    assert payload["rank_min"] <= payload["rank_max"]


def test_alternative_mode_count_rules_are_ordered_between_d1_and_d0():
    from models.tuning_modules.mdl_tangent_core import _mode_count_value
    d0, d1 = 100, 10.0
    values = {
        name: _mode_count_value(d0, d1, name)
        for name in ("shannon", "harmonic", "geometric", "arithmetic")
    }
    assert values["shannon"] == d1
    assert d1 <= values["harmonic"] <= d0
    assert d1 <= values["geometric"] <= d0
    assert d1 <= values["arithmetic"] <= d0
