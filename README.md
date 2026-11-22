Smoke test, run from the repository root after setting up the locked environment and placing the issued files there:

```powershell
uv run python c2_checkpoint.py --smoke --index .\ersst5.nino.mth.91-20.ascii --grid .\TrackC_ERSSTv5_subset.npz --split .\T03_C2_forecast.csv --out .\checkpoint_submission\smoke_freeze
```

# C2 ENSO forecasting

## Environment and input files

Use Python 3.12 and the committed `uv.lock` file:

```powershell
uv python install 3.12
uv sync --frozen
```

Place these files in the repository root:

- `ersst5.nino.mth.91-20.ascii` — issued CPC index.
- `TrackC_ERSSTv5_subset.npz` — issued ERSSTv5 subset.
- `T03_C2_forecast.csv` — issued forecast split.

Check the issued data-file hashes in PowerShell:

```powershell
(Get-FileHash .\ersst5.nino.mth.91-20.ascii -Algorithm SHA256).Hash
(Get-FileHash .\TrackC_ERSSTv5_subset.npz -Algorithm SHA256).Hash
```

Expected SHA-256 values:

- Index: `bedcd0683c07f7ccb2ae358a90cb39d57413dcc8881027e0c30a1811f6d58012`
- Issued grid subset: `aa35edf96391272199fdd62e4428ec89d5a2b58ba9267acfd875e77026a888cf`

The split validator checks 505/109/229 issue months, 22 excluded rows, sample IDs against the reconstructed issue months, and partition membership. The smoke command runs this check. The downloaded full-grid file is not required by this pipeline; the issued subset is.

The locked environment is described by `pyproject.toml` and `uv.lock`. It uses NumPy 2.5.3, pandas 3.0.6, PyTorch 2.14.1+cpu, netCDF4 1.7.4, psutil 7.2.2 and matplotlib 3.11.2. The saved Week 1 measurements record NumPy 2.4.2 for those runs; that is separate from the lock-file environment used for later work.

## Smoke test

The smoke test checks the split, runs persistence and all three ridge alphas, and makes two-update examples of the index LSTM, attention LSTM and map CNN. It checks the attention ablation, selects the fusion weight from the five prescribed values using mean RMSE across leads (ties go to the weight closest to 0.5), and runs unscored inference on test-shaped inputs. It does not load test targets or compute test metrics. Smoke-test runs are not fit rows in the run log.

## Fit and reuse commands

`full` is the default mode. It fits all three ridge alphas, then fits the LSTM and attention LSTM. It loads the saved map-CNN checkpoint from `out\cnn_best.pt`; it does not train that CNN. A full run is a new set of fits and writes new output files.

```powershell
$fitDate = Get-Date -Format 'yyyy-MM-dd'
$fitCommit = (git rev-parse --short=12 HEAD).Trim()
uv run python c2_checkpoint.py --index .\ersst5.nino.mth.91-20.ascii --grid .\TrackC_ERSSTv5_subset.npz --split .\T03_C2_forecast.csv --out .\checkpoint_submission\new_fit --cnn-checkpoint .\out\cnn_best.pt --date $fitDate --commit $fitCommit
```

`--recompute` uses saved validation predictions and model checkpoints to recalculate validation summaries and fusion selection. It does not fit models or load test targets. It writes `validation_results.json` and `T03_C2_runlog.csv` in its output folder, so copy the saved result folder first if the originals must be preserved.

```powershell
New-Item -ItemType Directory -Path .\checkpoint_submission\recompute_copy
Copy-Item .\checkpoint_submission\results\* .\checkpoint_submission\recompute_copy\
uv run python c2_checkpoint.py --recompute --index .\ersst5.nino.mth.91-20.ascii --grid .\TrackC_ERSSTv5_subset.npz --split .\T03_C2_forecast.csv --out .\checkpoint_submission\recompute_copy --cnn-checkpoint .\out\cnn_best.pt
```

The Week 1 CNN script is a separate training run. It reads the index and grid from the two input paths and writes `cnn_best.pt`, its loss history and measurement JSON to the output directory (default `week1_out`). The command below uses a separate output directory to preserve the checkpoint in `out`:

```powershell
uv run python week1_timed_cnn.py .\ersst5.nino.mth.91-20.ascii .\TrackC_ERSSTv5_subset.npz .\week1_out
```

The full pipeline uses `out\cnn_best.pt` by default. A fresh run of the Week 1 script must use the intended output path explicitly if its CNN is to be used by the full pipeline.

## Training protocol

Reference fits use seed 22, Adam at 0.001, batch size 32, CPU, six intra-op threads, one inter-op thread, at most 100 epochs or 15 minutes, and early stopping after five validation checks without improvement. The CNN is the Week 1 fit. Validation chooses the ridge alpha and one global fusion weight from `{0, 0.25, 0.5, 0.75, 1}`. The fusion weight multiplies the LSTM prediction; the complement multiplies the CNN prediction.

Final test evaluation is a separate, one-time step. First commit all model, preprocessing, checkpoint and fusion choices using training and validation evidence. Then run each required final evaluation once, record the lock commit, evaluation date and run IDs, and set `test_scored=yes` on the corresponding existing run-log rows. Do not tune from test results.

## Final evaluation after lock

Once `STUDY_LOCK.md` and `study_lock.json` are committed, use that full commit hash. Run the preflight first; it verifies the lock, input hashes, split and selected checkpoint hashes without producing test predictions or metrics. Then run the final evaluation exactly once, to a new output directory:

```powershell
$lockCommit = (git rev-parse HEAD).Trim()
uv run python c2_final_eval.py --index .\ersst5.nino.mth.91-20.ascii --grid .\TrackC_ERSSTv5_subset.npz --split .\T03_C2_forecast.csv --lock-commit $lockCommit --out .\final_test_2026-11-22 --preflight
uv run python c2_final_eval.py --index .\ersst5.nino.mth.91-20.ascii --grid .\TrackC_ERSSTv5_subset.npz --split .\T03_C2_forecast.csv --lock-commit $lockCommit --out .\final_test_2026-11-22
```

The evaluator refuses to run if the committed source has changed, the lock commit differs from `HEAD`, an input or checkpoint hash differs, or the output directory already exists. It generates the locked predictions before extracting test targets and writes per-lead ACC/RMSE plus predictions under the chosen output directory. Do not rerun it or make choices from its test results.

