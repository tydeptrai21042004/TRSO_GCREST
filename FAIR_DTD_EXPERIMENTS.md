# Fair DTD experiments

Use the generic controlled runner for DTD:

```bash
python -m tools.run_fair_suite \
  --dataset dtd \
  --task auto \
  --data_path ./data \
  --download True \
  --dataset_args_json '{"dtd_partition":1}' \
  --backbones resnet50@torchvision,vit_tiny_patch16_224@timm \
  --methods linear,trso,full,lora,bitfit,ssf,adaptformer,conv \
  --seeds 0,1,2,3,4 \
  --epochs 30 \
  --batch_size 64 \
  --input_size 0 \
  --execute
```

The runner creates a task-aware linear head once per backbone and seed, then
loads that same head before every compatible PEFT method and before
parameter-free tangent-projection calibration. Unsupported combinations are written to the
compatibility report with an explicit reason.

Verify the protocol with:

```bash
python -m tools.verify_fairness \
  --manifest experiments/fair_manifest.json \
  --compatibility experiments/fair_manifest_compatibility.json
```
