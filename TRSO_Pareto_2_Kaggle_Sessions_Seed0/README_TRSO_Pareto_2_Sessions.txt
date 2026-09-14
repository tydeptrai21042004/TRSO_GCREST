TRSO Pareto — two independent Kaggle sessions, one optimization seed each
==========================================================================

Session A
---------
File: TRSO_Pareto_Session_A_Flowers_ResNet50_Seed0.py
Dataset/backbone: Flowers-102 / ResNet-50
Seed: 0 only
Batch size: 16
30 epochs, 3 warm-up epochs
Methods:
  Full Fine-Tuning
  Linear Probing
  Visual Prompting: prompt_size = 5,10,20,30
  Conv-Adapter: adapt_size = 2,4,8,16,32
  Piggyback: canonical
  Proposal/TRSO: automatic canonical point
Total runs: 13

Session B
---------
File: TRSO_Pareto_Session_B_DTD_ViTB16_Seed0.py
Dataset/backbone: DTD partition 1 / ViT-B/16 torchvision
Seed: 0 only
Batch size: 8
30 epochs, 3 warm-up epochs
Methods:
  Full Fine-Tuning
  Linear Probing
  SSF: canonical
  AdaptFormer: dim = 8,16,32,64,128
  RepAdapter: dim = 2,4,8,16,32 (groups=2)
  VPT-Deep: tokens = 1,5,10,20,50
  ConvPass: dim = 2,4,8,16,32
  FacT-TT: rank = 1,2,4,8,16
  Proposal/TRSO: automatic canonical point
Total runs: 29

Protocol shared by both
-----------------------
  seed=0 only
  split_seed=2026
  AdamW controlled fair protocol
  PEFT LR=1e-3
  Full FT LR=1e-4
  Linear LR=1e-3
  weight_decay=1e-4
  min_lr=1e-6
  cosine schedule
  strong single-label augmentation
  fresh/random downstream head policy
  validation-selected checkpoint, then one final held-out test

Each script:
  1) uses an uploaded TRSO_GCREST ZIP if found in /kaggle/input;
  2) otherwise clones the public GitHub repo;
  3) downloads the benchmark dataset;
  4) runs every configuration sequentially for seed 0;
  5) writes pareto_results.csv;
  6) writes accuracy_vs_parameters.png;
  7) exports a ZIP containing all session outputs.

Kaggle usage
------------
Create one Kaggle notebook/session per file, enable GPU + Internet, paste the complete
.py content into one cell (or upload/run the file), and execute it. Do NOT run both
scripts in the same Kaggle session if you want the requested two-session separation.
