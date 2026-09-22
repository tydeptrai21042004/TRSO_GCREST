# Paper reproduction commands

These commands demonstrate the strict model construction. Replace dataset paths
and checkpoints with the exact assets used by the corresponding paper when
reproducing reported accuracy.

## Visual prompting

```bash
python main.py \
  --tuning_method prompt \
  --backbone resnet50 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data \
  --prompt_size 30 \
  --prompt_type padding \
  --prompt_output_indices "$(seq -s, 0 101)" \
  --paper_hparams True
```

Only the prompt tensors are optimized. No downstream classifier is created.

## Conv-Adapter

```bash
python main.py \
  --tuning_method conv \
  --backbone resnet50 \
  --weights DEFAULT \
  --conv_adapter_mode conv_parallel \
  --adapt_size 8 \
  --adapt_scale 1.0 \
  --dataset flowers102 \
  --data_path /path/to/data
```

## RepAdapter (RepBlock)

```bash
python main.py \
  --tuning_method repadapter \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --task single_label \
  --dataset dtd \
  --data_path /path/to/dtd \
  --repadapter_dim 8 \
  --repadapter_groups 2 \
  --repadapter_scale 1.0 \
  --repadapter_dropout 0.1 \
  --repadapter_merge True
```

The strict route reproduces the paper's ViT-B/16 RepBlock classification setting. Only RepAdapter branches and the task head train. With `--repadapter_merge True`, the trained branches are folded into the attention and first MLP projections before final inference.


```bash
python main.py \
  --tuning_method residual \
  --ra_mode parallel \
  --ra_pretrained_checkpoint /path/to/official/shared_resnet26.pth \
  --dataset imagefolder \
  --data_path /path/to/task
```

## SSF

```bash
python main.py \
  --tuning_method ssf \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data
```

## LoRA

```bash
python main.py \
  --tuning_method lora \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_merge_weights True \
  --dataset flowers102 \
  --data_path /path/to/data
```

## BitFit

```bash
python main.py \
  --tuning_method bitfit \
  --backbone vit_b_16 \
  --weights DEFAULT \
  --dataset flowers102 \
  --data_path /path/to/data
```

## Side-Tuning

```bash
python main.py \
  --tuning_method sidetune \
  --backbone resnet18 \
  --weights DEFAULT \
  --sidetune_alpha 0.5 \
  --sidetune_arch lightweight \
  --sidetune_width 64 \
  --sidetune_depth 4 \
  --dataset flowers102 \
  --data_path /path/to/data
```

## Automated controlled comparison

Use the active generic runner:

```bash
python -m tools.run_fair_suite --help
```

G-CREST-TRSO has no manually supplied global rank budget, evidence threshold, or layer list: the default retained-mode count and tensor/mode allocation are derived from one model-wide partition-consistency-weighted calibration-gradient evidence distribution. Realized capacity can still depend on calibration construction through the candidate spectral cap.
