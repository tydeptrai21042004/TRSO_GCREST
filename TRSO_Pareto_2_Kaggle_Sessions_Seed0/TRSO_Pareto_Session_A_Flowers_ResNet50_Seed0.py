# Kaggle one-cell/session script
# Experiment A: Flowers-102 / ResNet-50 — CNN Accuracy–Parameter Trade-off
# Exactly ONE optimization seed: seed=0

from __future__ import annotations
import csv, json, os, re, shutil, subprocess, sys, zipfile
from pathlib import Path

SESSION = "A"
SEED = 0
SPLIT_SEED = 2026
EPOCHS = 30
WARMUP = 3
BATCH = 16
INPUT_SIZE = 224
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
DATA = WORK / "trso_pareto_data"
OUT = WORK / "trso_pareto_session_A_flowers_resnet50_seed0"
REPO = WORK / "TRSO_GCREST_PARETO_A"
GITHUB = os.environ.get("TRSO_GITHUB_REPO", "https://github.com/tydeptrai21042004/TRSO_GCREST.git")
REF = os.environ.get("TRSO_GITHUB_REF", "main")


def run(cmd, cwd=None):
    cmd = [str(x) for x in cmd]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def find_uploaded_repo_zip():
    roots = [Path("/kaggle/input"), Path("/mnt/data")]
    pats = ["TRSO_GCREST-main*.zip", "TRSO_GCREST*.zip"]
    candidates = []
    for root in roots:
        if not root.exists():
            continue
        for pat in pats:
            candidates.extend(root.rglob(pat))
    return sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)[0] if candidates else None


def setup_repo():
    if REPO.exists():
        shutil.rmtree(REPO)
    source_zip = find_uploaded_repo_zip()
    if source_zip is not None:
        tmp = WORK / "_trso_extract_A"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        print(f"Using uploaded repo ZIP: {source_zip}")
        with zipfile.ZipFile(source_zip) as zf:
            zf.extractall(tmp)
        roots = [p for p in tmp.rglob("main.py") if (p.parent / "models").is_dir() and (p.parent / "datasets").is_dir()]
        if not roots:
            raise RuntimeError("Uploaded ZIP does not contain the expected TRSO_GCREST repository.")
        shutil.copytree(roots[0].parent, REPO)
    else:
        print("No uploaded TRSO ZIP found; cloning GitHub fallback.")
        run(["git", "clone", "--depth", "1", "--branch", REF, GITHUB, REPO])
    if not (REPO / "main.py").is_file():
        raise RuntimeError("main.py missing after repository setup")


setup_repo()
run([sys.executable, "-m", "pip", "install", "-q", "timm>=0.9,<2", "pandas>=2", "scipy>=1.10", "scikit-learn>=1.3", "tqdm>=4.65"])

import torch
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this session.")
print("GPU:", torch.cuda.get_device_name(0))

DATA.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

# New experiment: Full/Linear/Proposal are one-point anchors.
# Only baselines with a genuine CNN capacity knob are swept.
CONFIGS = [
    {"label":"Full FT", "method":"full", "knob":"canonical", "value":None, "extra":[]},
    {"label":"Linear", "method":"linear", "knob":"canonical", "value":None, "extra":[]},
    {"label":"Visual Prompt p=5",  "method":"prompt", "knob":"prompt_size", "value":5,  "extra":["--prompt_size","5"]},
    {"label":"Visual Prompt p=10", "method":"prompt", "knob":"prompt_size", "value":10, "extra":["--prompt_size","10"]},
    {"label":"Visual Prompt p=20", "method":"prompt", "knob":"prompt_size", "value":20, "extra":["--prompt_size","20"]},
    {"label":"Visual Prompt p=30", "method":"prompt", "knob":"prompt_size", "value":30, "extra":["--prompt_size","30"]},
    {"label":"Conv-Adapter d=2",  "method":"conv", "knob":"adapt_size", "value":2,  "extra":["--adapt_size","2"]},
    {"label":"Conv-Adapter d=4",  "method":"conv", "knob":"adapt_size", "value":4,  "extra":["--adapt_size","4"]},
    {"label":"Conv-Adapter d=8",  "method":"conv", "knob":"adapt_size", "value":8,  "extra":["--adapt_size","8"]},
    {"label":"Conv-Adapter d=16", "method":"conv", "knob":"adapt_size", "value":16, "extra":["--adapt_size","16"]},
    {"label":"Conv-Adapter d=32", "method":"conv", "knob":"adapt_size", "value":32, "extra":["--adapt_size","32"]},
    {"label":"Piggyback", "method":"piggyback", "knob":"canonical", "value":None, "extra":[]},
    {"label":"Proposal-Auto", "method":"trso", "knob":"automatic", "value":None, "extra":[]},
]

(OUT / "session_plan.json").write_text(json.dumps({
    "session": SESSION,
    "experiment": "Flowers-102 / ResNet-50 CNN accuracy-parameter trade-off",
    "seed": SEED,
    "split_seed": SPLIT_SEED,
    "epochs": EPOCHS,
    "warmup_epochs": WARMUP,
    "batch_size": BATCH,
    "configs": CONFIGS,
}, indent=2), encoding="utf-8")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def common_cmd(cfg, run_dir):
    keep_head = "True" if cfg["method"] == "prompt" else "False"
    return [
        sys.executable, "main.py",
        "--dataset", "flowers102",
        "--task", "single_label",
        "--download", "True",
        "--data_path", str(DATA),
        "--backbone", "resnet50",
        "--model_source", "torchvision",
        "--weights", "DEFAULT",
        "--pretrained", "True",
        "--tuning_method", cfg["method"],
        "--keep_pretrained_head", keep_head,
        "--seed", str(SEED),
        "--split_seed", str(SPLIT_SEED),
        "--epochs", str(EPOCHS),
        "--batch_size", str(BATCH),
        "--num_workers", "4",
        "--input_size", str(INPUT_SIZE),
        "--fair_protocol", "True",
        "--fair_optimizer", "adamw",
        "--fair_peft_lr", "1e-3",
        "--fair_full_lr", "1e-4",
        "--fair_linear_lr", "1e-3",
        "--fair_weight_decay", "1e-4",
        "--fair_warmup_epochs", str(WARMUP),
        "--fair_min_lr", "1e-6",
        "--scheduler", "cosine",
        "--train_aug", "standard",
        "--aa", "rand-m9-mstd0.5-inc1",
        "--color_jitter", "0.2",
        "--mixup", "0.2",
        "--cutmix", "0.0",
        "--smoothing", "0.1",
        "--reprob", "0.1",
        "--peft_head_lr_scale", "1.0",
        "--peft_freeze_head", "False",
        "--head_init_policy", "random",
        "--legacy_auto_hparams", "False",
        "--paper_hparams", "False",
        "--profile_efficiency", "True",
        "--measure_eval_latency", "True",
        "--evaluate_before_training", "True",
        "--save_history", "True",
        "--save_ckpt", "True",
        "--save_ckpt_num", "1",
        "--final_test", "True",
        "--auto_resume", "False",
        "--device", "cuda",
        "--output_dir", str(run_dir),
        "--experiment_suite", "pareto_flowers_resnet50",
        "--experiment_name", cfg["label"],
        "--experiment_run_id", f"A_seed{SEED}_{slug(cfg['label'])}",
        *cfg["extra"],
    ]

for i, cfg in enumerate(CONFIGS, 1):
    run_dir = OUT / "runs" / f"{i:02d}_{slug(cfg['label'])}"
    done = run_dir / "test_summary.json"
    if done.is_file() and (run_dir / "parameter_summary.json").is_file():
        print(f"[SKIP completed] {cfg['label']}")
        continue
    run_dir.mkdir(parents=True, exist_ok=True)
    run(common_cmd(cfg, run_dir), cwd=REPO)

# Aggregate exactly the metrics needed for the Pareto plot.
rows = []
for i, cfg in enumerate(CONFIGS, 1):
    run_dir = OUT / "runs" / f"{i:02d}_{slug(cfg['label'])}"
    p = json.loads((run_dir / "parameter_summary.json").read_text())
    t = json.loads((run_dir / "test_summary.json").read_text())
    timing = json.loads((run_dir / "timing_summary.json").read_text()) if (run_dir / "timing_summary.json").is_file() else {}
    row = {
        "session":"A", "seed":SEED, "dataset":"flowers102", "backbone":"resnet50",
        "label":cfg["label"], "method":cfg["method"], "capacity_knob":cfg["knob"], "capacity_value":cfg["value"],
        "acc1":t.get("acc1"), "acc5":t.get("acc5"), "macro_f1":t.get("macro_f1"), "mAP":t.get("mAP", t.get("map")),
        "trainable_params":p.get("trainable_params"), "adapter_trainable_params":p.get("adapter_trainable_params"),
        "head_trainable_params":p.get("head_trainable_params"), "total_params":p.get("total_params"),
        "latency_ms_per_image":t.get("latency_ms_per_image"),
        "calibration_time_sec":timing.get("proposal_calibration_time_sec", 0.0),
        "frozen_basis_values":p.get("mdl_frozen_basis_values"),
    }
    rows.append(row)

csv_path = OUT / "pareto_results.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# Plot one seed only; no error bars by design.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 6))
groups = {}
for row in rows:
    groups.setdefault(row["method"], []).append(row)
for method, gr in groups.items():
    gr = sorted(gr, key=lambda r: int(r["trainable_params"] or 0))
    xs = [r["trainable_params"] for r in gr]
    ys = [r["acc1"] for r in gr]
    if len(gr) > 1:
        ax.plot(xs, ys, marker="o", label=method)
    else:
        marker = "*" if method == "trso" else "o"
        size = 180 if method == "trso" else 70
        ax.scatter(xs, ys, marker=marker, s=size, label=method)
ax.set_xscale("log")
ax.set_xlabel("Total trainable parameters (log scale)")
ax.set_ylabel("Test Acc@1")
ax.set_title("Flowers-102 / ResNet-50 — seed 0")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "accuracy_vs_parameters.png", dpi=220)
plt.close(fig)

# Zip all session outputs.
zip_path = WORK / "trso_pareto_session_A_flowers_resnet50_seed0.zip"
if zip_path.exists(): zip_path.unlink()
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for p in OUT.rglob("*"):
        if p.is_file():
            zf.write(p, p.relative_to(WORK))
print("\nDONE")
print("CSV:", csv_path)
print("Plot:", OUT / "accuracy_vs_parameters.png")
print("ZIP:", zip_path)
