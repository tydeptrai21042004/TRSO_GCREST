# Final G-CREST Release Test Report

## Software validation

- Full test suite: 147 passed.
- Python compilation: passed.
- G-CREST controlled seed-0 rerun from the final package: reproduced 100.0% TinyViT accuracy and 100.0% TinyCNN accuracy.
- Exact-merge tests: passed through the repository suite.
- Release contract: no V6/V7 method document, no Linear fallback phrase, and the G-CREST method document is required.

## Controlled evidence

See `validation/GCREST_CONTROLLED_SUMMARY.json` and `GCREST_NOVELTY_AND_EVIDENCE_AUDIT.md`.

## Kaggle protocol

All six session runners clone `tydeptrai21042004/trso_adapter@main`, require the G-CREST release, use 30 epochs and shared strong augmentation, and record the resolved commit in `run_summary.json`.

## Limitation

The real DTD Session 1 benchmark has not been rerun for G-CREST in this environment. The release is runnable and tested, but no real-dataset improvement is asserted until the GPU result is produced.
