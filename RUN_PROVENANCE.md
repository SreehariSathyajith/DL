# Run-history reconciliation for the C2 freeze

This note preserves what the surviving records establish and calls out missing provenance. It does not replace the checkpoint artifacts or their run log.

## CNN runs

- **R001** is the Week 1 reference CNN row. The preserved `verification/r001_week1_measurements.json` matches its first-epoch loss, best validation MSE, epoch count, seed, parameter count and six intra-op/one inter-op threads. It records peak working-set bytes `494596096`, which is `471.7 MiB` (the old `494.6` value was decimal MB). The measurement JSON records `3.851369 s` elapsed and `22.515625 s` CPU, while the checkpoint run log records `3.215 s` and `18.844 s`. The timing discrepancy is unresolved; the run log keeps the original reported timings. The exact original command and script version were not preserved, so this record cannot establish whether R001 used the first timed-script version. No saved R001 validation prediction array was found; its mean five-lead RMSE is therefore unavailable and was not regenerated.
- **R008** is an additional Week 1 repeat using the same seed and CNN architecture, with PyTorch's default 16/16 thread settings. The saved `out/week1_timed_cnn.json` records 9 epochs, 144 updates, `3.491204 s` elapsed, `49.984375 s` CPU, and `574.707 MiB` lifetime peak working set. Its mean five-lead validation RMSE from the saved validation predictions is `0.53864700`. No separate development objective was recorded.
- The saved `out/cnn_best.pt` SHA-256 is `C70A0B1DEA6AAC20BB3ED570BCFABCECB35DA484AEBF059309704F60A00F3B4F`. The adjacent `out/week1_timed_cnn.json` matches R008, and `c2_checkpoint.py` loads `out/cnn_best.pt` by default. The saved CNN validation metrics match R008. Together, these support that R008 supplied the CNN predictions used by the core comparison. The exact invocation, any path override, and a hash captured at training time were not recorded, so this is a post-hoc association.

## Ridge history

- The three ridge validation arrays in `checkpoint_submission/results` were written on 4 November by `c2_checkpoint.py`, whose `full()` routine recomputes each ridge solution. Thus R002-R004 are the 4 November checkpoint calculations, separate from the earlier alpha results reported in the 12 October proposal.
- The proposal establishes that earlier ridge calculations existed for alpha `0.1`, `1` and `10`, used for validation alpha selection. Their original execution dates, settings beyond the alpha/split protocol, timings, commit IDs and prediction arrays were not retained. They are recorded as R009-R011 with date/results marked unrecorded rather than assigning the proposal date as their run date.

## Other validation records and memory

- R005's saved validation predictions give mean five-lead RMSE `0.61425626`; its optimization MSE is `0.41481620` at epoch 14. R006's saved predictions give `0.66582388`; its optimization MSE is `0.47725064` at epoch 29. R007's selected LSTM weight is `0.25`, with mean five-lead RMSE `0.53293741`; its recorded elapsed/CPU values are `0.465115 s` and `1.125 s` and are counted once.
- R001 and R008 use process-lifetime peak working-set measurements. R005-R006 use sampled process RSS during training. R002-R004 record an RSS snapshot from the ridge routine, not a process-lifetime peak. The R007 memory method and raw bytes were not recorded. The `peak_mem_mb` column's numeric values from `/2^20` are MiB, not decimal MB; raw bytes are available only for R001.
- Week 1 saved measurement records report NumPy `2.4.2`. The later locked environment uses NumPy `2.5.3`.

## Test-set handling

Checkpoint logs and saved validation summaries have `test_scored=no` / `test_scored=false`. Earlier review accepted the disclosed temporary materialization of targets because they were not used for fitting, selection or scoring. This note contains no test metrics.
