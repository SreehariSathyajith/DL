Smoke test, from the repository root:

```powershell
.\.venv\Scripts\python.exe c2_checkpoint.py --smoke --index ersst5.nino.mth.91-20.ascii --grid TrackC_ERSSTv5_subset.npz --split T03_C2_forecast.csv --out checkpoint_submission/smoke
```

# C2 ENSO forecasting

The smoke test checks the issued split file and runs persistence and ridge baselines, the index LSTM, the SST-map CNN, temporal attention with its no-attention ablation, the named LSTM–CNN fusion, and validation calculations for both core experiments. Each neural component trains for two updates, validates on a small validation subset, and runs unscored inference on test-shaped inputs. It constructs no test-target matrix and computes no test metric.

Full reference fits use seed 22, Adam at 0.001, batch size 32, CPU, six intra-op threads and one inter-op thread, up to 100 epochs or 15 minutes per fit, with early stopping after five validation checks without improvement. The Week 1 map-CNN checkpoint is reused. Input and split paths are command-line arguments; the workflow downloads nothing.
