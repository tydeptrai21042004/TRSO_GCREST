#!/usr/bin/env python3
"""Offline structural/gradient preflight for strict paper baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, resnet50
from torchvision.models.vision_transformer import VisionTransformer

from main import set_trainability_policy
from models.tuning_modules.adaptformer import apply_adaptformer
from models.tuning_modules.piggyback import apply_piggyback
from models.tuning_modules.conv_adapter import apply_conv_adapter_resnet50, set_conv_adapter_trainability
from models.tuning_modules.lora_transformer import apply_lora_transformer
from models.tuning_modules.prompter import VisualPromptingClassifier
from models.tuning_modules.repadapter import apply_repadapter, merge_repadapter_, set_repadapter_trainability
from models.tuning_modules.arc import apply_arc, merge_arc_, set_arc_trainability
from models.tuning_modules.side_tuning import SideTuningClassifier
from models.tuning_modules.ssf import apply_ssf, set_ssf_trainability
from models.tuning_modules.vpt import VisualPromptTuning, set_vpt_trainability
from models.tuning_modules.convpass_transformer import apply_convpass, set_convpass_trainability
from models.tuning_modules.fact import apply_fact, merge_fact_, set_fact_trainability
from models.tuning_modules.vqt import VisualQueryTuning, set_vqt_trainability
from models.tuning_modules.spt import calibrate_spt, merge_spt_


def step_ok(model, x):
    model.eval()
    model.zero_grad(set_to_none=True)
    y = model(x)
    y.float().square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    return list(y.shape), bool(grads and any(g is not None for g in grads))


def run():
    torch.manual_seed(7)
    x = torch.randn(2, 3, 32, 32)
    report = {}

    prompt = VisualPromptingClassifier(
        resnet18(weights=None, num_classes=1000),
        5,
        prompt_size=2,
        image_size=32,
        prompt_type="padding",
    )
    set_trainability_policy(prompt, SimpleNamespace(tuning_method="prompt", weight_decay=1e-4))
    report["visual_prompt"] = step_ok(prompt, x)

    conv = resnet50(weights=None, num_classes=5)
    apply_conv_adapter_resnet50(conv, mode="conv_parallel")
    set_conv_adapter_trainability(conv)
    report["conv_adapter"] = step_ok(conv, x)


    ssf = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_ssf(ssf)
    set_ssf_trainability(ssf)
    report["ssf"] = step_ok(ssf, x)

    lora = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_lora_transformer(lora, rank=2, alpha=4)
    set_trainability_policy(lora, SimpleNamespace(tuning_method="lora"))
    report["lora"] = step_ok(lora, x)

    bitfit = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    set_trainability_policy(bitfit, SimpleNamespace(tuning_method="bitfit", bitfit_train_head=True, weight_decay=1e-4, fair_protocol=False))
    report["bitfit"] = step_ok(bitfit, x)

    adaptformer = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_adaptformer(adaptformer, bottleneck=4, scale=0.1)
    set_trainability_policy(adaptformer, SimpleNamespace(tuning_method="adaptformer"))
    report["adaptformer"] = step_ok(adaptformer, x)

    repadapter = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
    apply_repadapter(repadapter, hidden_dim=8, groups=2, scale=1.0, dropout=0.0)
    set_repadapter_trainability(repadapter)
    report["repadapter"] = step_ok(repadapter, x)
    repadapter.eval()
    with torch.no_grad():
        before_merge = repadapter(x)
    report["repadapter_merged_updates"] = merge_repadapter_(repadapter)
    with torch.no_grad():
        after_merge = repadapter(x)
    report["repadapter_merge_max_abs_error"] = float((before_merge - after_merge).abs().max())

    arc = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5, dropout=0.0, attention_dropout=0.0)
    apply_arc(arc, adapter_dim=4, dropout=0.0)
    set_arc_trainability(arc)
    report["arc"] = step_ok(arc, x)
    arc.eval()
    with torch.no_grad():
        arc_before_merge = arc(x)
    report["arc_merged_updates"] = merge_arc_(arc)
    with torch.no_grad():
        arc_after_merge = arc(x)
    report["arc_merge_max_abs_error"] = float((arc_before_merge - arc_after_merge).abs().max())

    piggyback = resnet50(weights=None, num_classes=5)
    apply_piggyback(piggyback, threshold=5e-3, mask_init="ones")
    set_trainability_policy(piggyback, SimpleNamespace(tuning_method="piggyback", piggyback_train_head=True))
    report["piggyback"] = step_ok(piggyback, x)

    side = SideTuningClassifier(
        resnet18(weights=None),
        num_classes=5,
        side_arch="lightweight",
        side_width=16,
        side_depth=4,
    )
    set_trainability_policy(side, SimpleNamespace(tuning_method="sidetune"))
    report["side_tuning"] = step_ok(side, x)

    for deep, name in ((False, "vpt_shallow"), (True, "vpt_deep")):
        vpt = VisualPromptTuning(
            VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5),
            num_tokens=3, deep=deep, dropout=0.0,
        )
        set_vpt_trainability(vpt)
        report[name] = step_ok(vpt, x)

    for attn_only, name in ((False, "convpass"), (True, "convpass_attn")):
        convpass = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
        apply_convpass(convpass, bottleneck=4, scale=1.0, dropout=0.0, attn_only=attn_only)
        set_convpass_trainability(convpass)
        report[name] = step_ok(convpass, x)

    for variant, rank in (("tt", 4), ("tk", 8)):
        fact = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=128, num_classes=5)
        apply_fact(fact, variant=variant, rank=rank, scale=1.0)
        set_fact_trainability(fact)
        report[f"fact_{variant}"] = step_ok(fact, x)
        report[f"fact_{variant}_merged_updates"] = merge_fact_(fact)

    vqt = VisualQueryTuning(
        VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5),
        num_classes=5, query_length=1,
    )
    set_vqt_trainability(vqt)
    report["vqt"] = step_ok(vqt, x)

    spt_data = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    for variant in ("lora", "adapter"):
        spt = VisionTransformer(image_size=32, patch_size=8, num_layers=1, num_heads=2, hidden_dim=32, mlp_dim=64, num_classes=5)
        calibrate_spt(
            spt, DataLoader(spt_data, batch_size=2, shuffle=False), torch.device("cpu"),
            variant=variant, budget=64, sensitivity_samples=4, rank=2, adapter_dim=2, alpha=2.0,
        )
        report[f"spt_{variant}"] = step_ok(spt, x)
        report[f"spt_{variant}_merged_updates"] = merge_spt_(spt)

    execution_rows = {name: value for name, value in report.items() if isinstance(value, tuple)}
    metadata = {name: value for name, value in report.items() if not isinstance(value, tuple)}
    strict_names = {
        "visual_prompt", "conv_adapter", "ssf", "adaptformer", "repadapter", "arc",
        "piggyback", "vpt_shallow", "vpt_deep", "convpass",
    }
    paper_ablation_names = {"convpass_attn"}
    candidate_names = {"fact_tt", "fact_tk", "vqt", "spt_lora", "spt_adapter"}
    nonpaper_names = {"lora", "bitfit", "side_tuning"}
    strict_rows = {name: value for name, value in execution_rows.items() if name in strict_names}
    paper_ablation_rows = {name: value for name, value in execution_rows.items() if name in paper_ablation_names}
    candidate_rows = {name: value for name, value in execution_rows.items() if name in candidate_names}
    control_rows = {name: value for name, value in execution_rows.items() if name in nonpaper_names}
    failures = [name for name, (_shape, grad_ok) in execution_rows.items() if not grad_ok]
    encode = lambda rows: {
        name: {"output_shape": shape, "gradient_ok": grad}
        for name, (shape, grad) in rows.items()
    }
    return {
        "strict_paper_baselines": encode(strict_rows),
        "paper_internal_ablations": encode(paper_ablation_rows),
        "qualified_reimplementation_candidates": encode(candidate_rows),
        "opt_in_nonpaper_controls": encode(control_rows),
        "metadata": metadata,
        "all_ok": not failures,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    report = run()
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
