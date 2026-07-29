# Experiment manifests

Generate the canonical 46-run, six-session protocol with:

```bash
python -m tools.minimal_paper_protocol \
  --output experiments/minimal_paper_46_protocol.json
```

Run strict literature baselines, corrected reference controls, or the full proposal through `tools.run_fair_suite`. Full Fine-Tuning and Linear Probing must be requested as `--methods reference_controls`; `--methods auto` remains strict-literature-baseline only.

Generate the fixed G-CREST-TRSO structural proposal ablations with:

```bash
python -m tools.run_trso_ablation --dataset <name> --backbone <name>
```

The six ready-to-run Kaggle files are documented in `kaggle/README.md`. All tools write JSON and CSV manifests with deterministic run IDs. Use `python -m tools.verify_fairness` before interpreting benchmark results.
