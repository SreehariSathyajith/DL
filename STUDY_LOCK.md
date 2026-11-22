# Study lock for Track C2

This lock is committed before the first final test evaluation. The lock commit is supplied to `c2_final_eval.py`; the evaluator refuses to run from a different commit, if its locked model and pipeline source files have local changes, or if any selected checkpoint/input hash differs.

## Choices fixed from training and validation only

- Issue months, partition boundaries, and 24-month inputs: Annex D3 implementation in `enso_data.py`; split file `T03_C2_forecast.csv` (505 train, 109 validation, 229 test, 22 excluded). Grid monthly climatology and all standardisation statistics are fit on the training period only.
- Random seed: 22. LSTM, attention and map-CNN architectures: `models.py` and `c2_checkpoint.py`; trained with the saved Week 1 / checkpoint configurations. No retraining or checkpoint selection follows test evaluation.
- Baselines: persistence; ridge autoregression with alpha 0.1. Alpha 0.1 was selected among 0.1, 1 and 10 by validation mean of the five lead RMSEs.
- LSTM checkpoint: `checkpoint_submission/results/lstm_best.pt` (run R005).
- Attention checkpoint: `checkpoint_submission/results/attention_best.pt` (run R006), reported as the ablation.
- Map-CNN checkpoint: `out/cnn_best.pt` (run R008).
- Fusion: 0.25 LSTM weight and 0.75 CNN weight, selected from `{0, 0.25, 0.5, 0.75, 1}` by validation mean of the five lead RMSEs; exact ties go to the weight closest to 0.5.
- Final metrics: per-lead ACC and RMSE at 1, 3, 6, 9 and 12 months for persistence, selected ridge, LSTM, attention, CNN and the fixed fusion. No test result can change these choices.

## Freeze and evaluation order

1. Commit this file, `study_lock.json`, the guarded evaluator and all source changes. Record that full commit hash as the lock commit.
2. Run the evaluator's `--preflight` mode. It checks the commit, inputs, partition counts and checkpoint hashes without making predictions or computing test metrics.
3. Run the final evaluator exactly once to a new output directory. It generates every fixed prediction before extracting test target values, then computes the required metrics.
4. Record the test evaluation date and lock commit in the preliminary report and set `test_scored=yes` for the evaluated fit rows in the run log. Do not tune or select anything from test results.

The two pre-existing checkpoint disclosures remain in `RUN_PROVENANCE.md`; earlier temporary target materialization was not used for fitting, selection, or scoring.
