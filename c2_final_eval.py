from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch

from c2_checkpoint import ALPHAS, AttentionLSTM, IndexLSTM, MapCNN, metrics, ridge_fit, settings, validate_manifest
from enso_data import LEADS, build, load_raw


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_lock(lock_path: Path, lock_commit: str, root: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if head != lock_commit:
        raise SystemExit(f"Refusing evaluation: HEAD {head} differs from --lock-commit {lock_commit}.")
    code_files = ["c2_final_eval.py", "c2_checkpoint.py", "enso_data.py", "models.py", "STUDY_LOCK.md", "study_lock.json"]
    changed = subprocess.check_output(["git", "status", "--porcelain", "--", *code_files], cwd=root, text=True)
    if changed.strip():
        raise SystemExit("Refusing evaluation: locked evaluation or model source files have uncommitted changes.")
    for rel, expected in lock["artifacts"].items():
        actual = sha256(root / rel)
        if actual != expected:
            raise SystemExit(f"Refusing evaluation: checkpoint hash mismatch for {rel}.")
    return lock


def ridge_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float) -> np.ndarray:
    xm, ym = xtr.mean(0), ytr.mean(0)
    xc, yc = xtr - xm, ytr - ym
    b = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ yc)
    return (xte - xm) @ b + ym


def collect_predictions(root: Path, data: dict, lock: dict) -> dict[str, np.ndarray]:
    settings()
    preds: dict[str, np.ndarray] = {}
    preds["persistence"] = np.repeat(data["Xraw_test"][:, -1, None], len(LEADS), axis=1)
    alpha = float(lock["ridge_alpha"])
    preds["ridge"] = ridge_predict(data["Xraw_train"], data["Y_train"], data["Xraw_test"], alpha)

    lstm = IndexLSTM()
    lstm.load_state_dict(torch.load(root / "checkpoint_submission/results/lstm_best.pt", map_location="cpu", weights_only=True))
    lstm.eval()
    attn = AttentionLSTM()
    attn.load_state_dict(torch.load(root / "checkpoint_submission/results/attention_best.pt", map_location="cpu", weights_only=True))
    attn.eval()
    cnn = MapCNN()
    cnn.load_state_dict(torch.load(root / "out/cnn_best.pt", map_location="cpu", weights_only=True))
    cnn.eval()
    with torch.no_grad():
        preds["lstm"] = lstm(torch.from_numpy(data["Xidx_test"])).cpu().numpy()
        preds["attention"] = attn(torch.from_numpy(data["Xidx_test"])).cpu().numpy()
        preds["cnn"] = cnn(torch.from_numpy(data["Xmap_test"])).cpu().numpy()
    w = float(lock["fusion_weight_lstm"])
    preds["fusion"] = w * preds["lstm"] + (1 - w) * preds["cnn"]
    for key, pred in preds.items():
        if pred.shape != (229, 5) or not np.isfinite(pred).all():
            raise RuntimeError(f"Invalid final prediction array for {key}: {pred.shape}")
    return preds


def main() -> None:
    p = argparse.ArgumentParser(description="One-time C2 final test evaluation using committed locked choices.")
    p.add_argument("--index", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--out", required=True, help="New output directory; existing paths are refused.")
    p.add_argument("--lock-commit", required=True, help="Full commit hash that locked all choices before test scoring.")
    p.add_argument("--preflight", action="store_true", help="Check locked inputs, split, and checkpoints without inference or test scoring.")
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    lock = verify_lock(root / "study_lock.json", args.lock_commit, root)
    out = Path(args.out).resolve()
    if args.preflight:
        if sha256(Path(args.index)) != lock["inputs"]["index_sha256"] or sha256(Path(args.grid)) != lock["inputs"]["grid_sha256"]:
            raise SystemExit("Preflight failed: issued input hash mismatch.")
        if out.exists():
            raise SystemExit("Preflight output path already exists; choose a fresh path.")
        d = build(args.index, args.grid, include_test_labels=False)
        _, counts = validate_manifest(args.split, d)
        if counts != (505, 109, 229, 22):
            raise SystemExit(f"Unexpected split counts: {counts}")
        print("Preflight passed: lock commit, source files, inputs, split and checkpoint hashes verified; no test predictions or metrics computed.")
        return
    if out.exists():
        raise SystemExit("Refusing evaluation: output path already exists. The final evaluation is one-time.")
    started = time.perf_counter()
    if sha256(Path(args.index)) != lock["inputs"]["index_sha256"] or sha256(Path(args.grid)) != lock["inputs"]["grid_sha256"]:
        raise SystemExit("Refusing evaluation: issued input hash mismatch.")
    data = build(args.index, args.grid, include_test_labels=False)
    split_file, counts = validate_manifest(args.split, data)
    if counts != (505, 109, 229, 22):
        raise SystemExit(f"Unexpected split counts: {counts}")

    # Generate every fixed prediction before extracting any test target values.
    preds = collect_predictions(root, data, lock)
    y_all, _, _ = load_raw(args.index, args.grid)
    test_ids = data["idx"]["test"]
    y_test = np.stack([[y_all[i + h] for h in LEADS] for i in test_ids]).astype(np.float32)
    if y_test.shape != (229, 5):
        raise RuntimeError(f"Unexpected test target shape: {y_test.shape}")
    scored = {name: metrics(y_test, pred) for name, pred in preds.items()}
    result = {
        "protocol": "single final evaluation after committed lock; no selection or tuning on test results",
        "lock_commit": args.lock_commit,
        "evaluation_date": date.today().isoformat(),
        "evaluation_elapsed_s": time.perf_counter() - started,
        "test_issue_months": len(test_ids),
        "lead_months": LEADS,
        "runs": {"persistence": "baseline", "ridge": "R002", "lstm": "R005", "attention": "R006", "cnn": "R008", "fusion": "R007"},
        "selected_choices": {"ridge_alpha": lock["ridge_alpha"], "fusion_weight_lstm": lock["fusion_weight_lstm"]},
        "metrics": scored,
        "test_scored": True,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "final_test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(out / "final_test_predictions.npz", **preds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

