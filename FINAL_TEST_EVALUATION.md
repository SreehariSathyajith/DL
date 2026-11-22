# Final test evaluation record

- Evaluation date: 2026-11-22
- Study lock commit: `3fb7ced65ab52af0bbe2e7261e5e0b3f6bcfbd5e`
- Issue months scored: 229 (all five target leads per issue month)
- Fixed choices: ridge alpha `0.1`; fusion LSTM weight `0.25` (CNN weight `0.75`).
- Evaluated rows: R002 (selected ridge), R005 (LSTM), R006 (attention ablation), R007 (fusion), and R008 (map CNN). Persistence was evaluated as a non-fitted baseline and has no fit row.
- Test scoring was run once after the lock commit. No model, preprocessing, checkpoint, threshold, or fusion choice was changed after seeing test results.
- Per-lead ACC and RMSE and saved predictions are in `final_test_2026-11-22/final_test_results.json` and `final_test_2026-11-22/final_test_predictions.npz`.
- Result JSON SHA-256: `9857f44d1d18993f6ab37ac4a94294c17ab5054b2f88c1dd92688986d18538b2`
- Prediction archive SHA-256: `698d106515dba75e11e97b20aea1df6574e1b4b18c22938a34b5393f5702345a`
