"""Paper-reproduction tuning modules."""
from .prompter import (
    FixedPatchPrompter,
    PadPrompter,
    RandomPatchPrompter,
    VisualPromptingClassifier,
    build_prompter,
)
from .conv_adapter import ConvAdapter, ConvAdapterBottleneck, apply_conv_adapter_resnet50
from .program_module import ProgramModule
from .ssf import SSF, SSFPost, SSFMultiheadAttention, apply_ssf, merge_ssf_
from .lora_transformer import (
    LoRALinear,
    LoRAQKVLinear,
    LoRAMultiheadAttention,
    apply_lora_transformer,
)
from .side_tuning import ConvSideNetwork, SideTuningClassifier
from .bitfit import set_bitfit_trainability
from .adaptformer import AdaptFormerAdapter, apply_adaptformer, set_adaptformer_trainability
from .repadapter import RepAdapter, apply_repadapter, set_repadapter_trainability, merge_repadapter_
from .arc import ARCProjectionBank, ARCAdapter, apply_arc, set_arc_trainability, merge_arc_
from .vpt import VisualPromptTuning, set_vpt_trainability
from .convpass_transformer import ConvPassAdapter, apply_convpass, set_convpass_trainability
from .fact import apply_fact, set_fact_trainability, merge_fact_
from .vqt import VisualQueryTuning, set_vqt_trainability
from .spt import calibrate_spt, set_spt_trainability, merge_spt_
from .ml_decoder import MLDecoder, MLDecoderClassifier, set_ml_decoder_trainability
from .segadapter import (
    SegAttention, SegAdapterBlock, SegAdapterStage, apply_segadapter_lraspp,
    collect_segadapter_aux, set_segadapter_trainability,
)
from .piggyback import (
    BinaryMaskSTE, PiggybackConv2d, PiggybackLinear, apply_piggyback,
    set_piggyback_trainability, piggyback_storage, export_binary_masks,
)


def set_tuning_config(tuning_method, args):
    method = str(tuning_method).strip().lower().replace("-", "_")
    aliases = {
        "conv_adapter": "conv",
        "adapter": "conv",
        "task_response": "trso",
        "trso_adapter": "trso",
        "side_tuning": "sidetune",
        "adapt_former": "adaptformer",
        "piggy_back": "piggyback",
    }
    method = aliases.get(method, method)
    if method == "prompt":
        return {
            "method": method,
            "prompt_size": getattr(args, "prompt_size", 30),
            "prompt_type": getattr(args, "prompt_type", "padding"),
        }
    if method == "conv":
        return {
            "method": method,
            "kernel_size": getattr(args, "kernel_size", 3),
            "mode": getattr(args, "conv_adapter_mode", "conv_parallel"),
            "width": getattr(args, "adapt_size", 8),
            "scale": getattr(args, "adapt_scale", 1.0),
        }
    if method == "trso":
        return {"method": method, "proposal": "mdl_evidence_adaptive_tangent_core_v6"}
    if method == "ssf":
        return {"method": method, "init_std": getattr(args, "ssf_init_std", 0.02)}
    if method == "lora":
        return {"method": method, "rank": getattr(args, "lora_r", 8), "alpha": getattr(args, "lora_alpha", 16.0)}
    if method == "sidetune":
        return {
            "method": method,
            "alpha": getattr(args, "sidetune_alpha", 0.5),
            "side_arch": getattr(args, "sidetune_arch", "lightweight"),
            "side_width": getattr(args, "sidetune_width", 64),
            "side_depth": getattr(args, "sidetune_depth", 4),
        }
    if method == "bitfit":
        return {
            "method": method,
            "bias_scope": getattr(args, "bitfit_bias_scope", "all"),
            "train_head": getattr(args, "bitfit_train_head", True),
        }
    if method in {"vpt_shallow", "vpt_deep"}:
        return {"method": method, "num_tokens": getattr(args, "vpt_num_tokens", 10), "deep": method == "vpt_deep"}
    if method in {"convpass", "convpass_attn"}:
        return {"method": method, "dim": getattr(args, "convpass_dim", 8), "scale": getattr(args, "convpass_scale", 1.0)}
    if method in {"fact_tt", "fact_tk"}:
        variant = method.split("_")[-1]
        configured = int(getattr(args, "fact_rank", 0))
        rank = configured if configured > 0 else (4 if variant == "tt" else 8)
        return {"method": method, "variant": variant, "rank": rank}
    if method == "vqt":
        return {"method": method, "query_length": getattr(args, "vqt_query_length", 1)}
    if method == "ml_decoder":
        return {
            "method": method,
            "decoder_embedding": getattr(args, "ml_decoder_embedding", 768),
            "num_groups": getattr(args, "ml_decoder_num_groups", -1),
        }
    if method == "segadapter":
        return {
            "method": method,
            "kernel_size": getattr(args, "segadapter_kernel_size", 5),
            "ffn_ratio": getattr(args, "segadapter_ffn_ratio", 3.0),
            "aux_weight": getattr(args, "segadapter_aux_weight", 0.4),
        }
    if method in {"spt_lora", "spt_adapter"}:
        return {"method": method, "budget": getattr(args, "spt_budget", 400000), "samples": getattr(args, "spt_sensitivity_samples", 800)}
    if method == "adaptformer":
        return {
            "method": method,
            "bottleneck": getattr(args, "adaptformer_dim", 16),
            "scale": getattr(args, "adaptformer_scale", 0.1),
            "dropout": getattr(args, "adaptformer_dropout", 0.0),
        }
    if method == "repadapter":
        return {
            "method": method,
            "hidden_dim": getattr(args, "repadapter_dim", 8),
            "groups": getattr(args, "repadapter_groups", 2),
            "scale": getattr(args, "repadapter_scale", 1.0),
            "dropout": getattr(args, "repadapter_dropout", 0.1),
        }
    if method == "arc":
        return {
            "method": method,
            "adapter_dim": getattr(args, "arc_dim", 50),
            "dropout": getattr(args, "arc_dropout", 0.1),
            "merge": getattr(args, "arc_merge", True),
        }
    if method == "piggyback":
        return {
            "method": method,
            "threshold": getattr(args, "piggyback_threshold", 5e-3),
            "mask_init": getattr(args, "piggyback_mask_init", "ones"),
            "mask_linear": getattr(args, "piggyback_mask_linear", False),
        }
    if method in {"full", "linear"}:
        return {"method": method}
    if method in {"lora_conv", "lora_conv2d"}:
        raise NotImplementedError(
            "LoRA-Conv is an experimental repository control, not an original-paper baseline, "
            "and is excluded from strict reproduction runs."
        )
    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")


from .mdl_tangent_core import (
    MDLTangentCoreParametrization,
    MDLTangentRecord,
    MDLTangentReport,
    calibrate_mdl_tangent_core,
    iter_mdl_tangent_parametrizations,
    mdl_tangent_basis_value_count,
    mdl_tangent_parameter_count,
    merge_mdl_tangent_cores_,
    set_mdl_tangent_trainability,
)

__all__ = [name for name in globals() if not name.startswith("_")]
