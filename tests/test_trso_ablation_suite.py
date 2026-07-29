from argparse import Namespace

from models.tuning_modules.mdl_tangent_core import TRSO_ABLATIONS
from tools.run_trso_ablation import build_ablation_specs


def test_ablation_suite_covers_all_fixed_variants_and_seeds(tmp_path):
    args = Namespace(
        dataset="fake", data_path="./data", task="auto", backbone="resnet18",
        model_source="torchvision", weights="DEFAULT", pretrained=False,
        seeds="0,1", ablations=",".join(TRSO_ABLATIONS), epochs=1, batch_size=4,
        num_workers=0, input_size=32, optimizer="sgd", lr=1e-3,
        weight_decay=0.0, warmup_epochs=0, min_lr=0.0, device="cpu",
        output_root=str(tmp_path), profile_efficiency=False, final_test=False,
        allow_val_as_test=True,
    )
    specs = build_ablation_specs(args)
    assert len(specs) == len(TRSO_ABLATIONS) * 2
    assert {spec.name for spec in specs} == set(TRSO_ABLATIONS)
    assert all(spec.parameters["tuning_method"] == "trso" for spec in specs)
    assert all(spec.parameters["trso_fast_inference"] is True for spec in specs)
    assert all(spec.parameters["trso_ablation"] in TRSO_ABLATIONS for spec in specs)
