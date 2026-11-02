"""Week-1 timed run (handout §8 item 7): ONE fit of the map CNN on the 24-month grid stack (Annex D0, cards C1-C3).
Limits: seed 22, <=100 epochs, <=15 min elapsed, early stop after 5 validation checks without improvement.
Test LABELS are never loaded: only test-shaped INPUTS are used for one timed inference pass.
Usage:  python week1_timed_cnn.py ersst5.nino.mth.91-20.ascii TrackC_ERSSTv5_subset.npz [outdir]
"""
import sys, os, time, json, platform, random
import numpy as np, pandas as pd, psutil, torch, torch.nn as nn
from enso_data import build
from models import MapCNN, n_params

ascii_path, npz_path = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "week1_out"
os.makedirs(out, exist_ok=True)

SEED, MAX_EPOCHS, MAX_SECONDS, PATIENCE = 22, 100, 15 * 60, 5
BATCH, LR = 32, 1e-3                      # my choices (card leaves optimiser/lr/batch to the group)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
proc = psutil.Process()

t_prep = time.perf_counter()
d = build(ascii_path, npz_path, include_test_labels=False)
Xtr, Ytr = torch.from_numpy(d["Xmap_train"]), torch.from_numpy(d["Y_train"])
Xva, Yva = torch.from_numpy(d["Xmap_val"]), torch.from_numpy(d["Y_val"])
Xte = torch.from_numpy(d["Xmap_test"])            # inputs only
prep_s = time.perf_counter() - t_prep
rss_before = proc.memory_info().rss / 2**20

model = MapCNN()
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
g = torch.Generator().manual_seed(SEED)
ckpt = os.path.join(out, "cnn_best.pt")

hist, best, bad, stop = [], float("inf"), 0, None
cpu0, wall0 = time.process_time(), time.perf_counter()   # clock: first training batch -> end of last val pass + checkpoint
n_updates = 0
for ep in range(1, MAX_EPOCHS + 1):
    te = time.perf_counter()
    model.train()
    perm = torch.randperm(len(Xtr), generator=g)
    tl = 0.0
    for b in range(0, len(Xtr), BATCH):
        ii = perm[b:b + BATCH]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[ii]), Ytr[ii])
        loss.backward(); opt.step(); n_updates += 1
        tl += loss.item() * len(ii)
    tl /= len(Xtr)
    model.eval()
    with torch.no_grad():
        vl = loss_fn(model(Xva), Yva).item()
    if vl < best:
        best, bad = vl, 0
        torch.save(model.state_dict(), ckpt)
    else:
        bad += 1
    hist.append(dict(epoch=ep, train_mse=tl, val_mse=vl, epoch_s=time.perf_counter() - te))
    if bad >= PATIENCE:
        stop = "early stopping (5 validation checks without improvement)"; break
    if time.perf_counter() - wall0 >= MAX_SECONDS:
        stop = "15-minute clock"; break
else:
    stop = "100-epoch limit"
elapsed = time.perf_counter() - wall0
cpu = time.process_time() - cpu0          # user+sys over all threads of this process; no worker processes exist
def peak_memory_mb():   # process-lifetime peak resident memory incl. data prep; works on Windows, macOS, Linux
    mi = proc.memory_info()
    if hasattr(mi, "peak_wset"):                       # Windows
        return mi.peak_wset / 2**20
    import resource
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 2**20 if sys.platform == "darwin" else kb / 1024   # macOS reports bytes, Linux KB
peak_mb = peak_memory_mb()

# one timed inference pass over test-shaped inputs (229 x 24 x 21 x 81), best checkpoint, no labels, no metrics
model.load_state_dict(torch.load(ckpt)); model.eval()
ti = time.perf_counter()
with torch.no_grad():
    pred = torch.cat([model(Xte[b:b + 64]) for b in range(0, len(Xte), 64)])
inf_s = time.perf_counter() - ti
finite = bool(torch.isfinite(pred).all()) and tuple(pred.shape) == (len(Xte), 5)

# context from validation only (allowed): persistence RMSE on the same val examples
pers = np.sqrt(((d["Y_val"] - d["Xraw_val"][:, -1:]) ** 2).mean())
hw = dict(os=platform.platform(), cpu=next((l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l), platform.processor()) if os.path.exists("/proc/cpuinfo") else platform.processor(),
          logical_cores=os.cpu_count(), physical_cores=psutil.cpu_count(logical=False), ram_gb=round(psutil.virtual_memory().total / 2**30, 1))
res = dict(hardware=hw, torch=torch.__version__, python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
           torch_threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(), dataloader_workers=0,
           params=n_params(model), n_train=len(Xtr), n_val=len(Xva), n_test_inputs=len(Xte),
           epochs=len(hist), updates=n_updates, batch=BATCH, lr=LR, elapsed_s=elapsed, cpu_s=cpu, s_per_epoch=elapsed / len(hist),
           stopped_by=stop, peak_rss_mb=peak_mb, rss_before_training_mb=rss_before, data_prep_s=prep_s,
           best_val_mse=best, best_epoch=int(np.argmin([h["val_mse"] for h in hist])) + 1,
           first_epoch=hist[0], last_epoch=hist[-1], persistence_val_mse=float(pers ** 2),
           inference_s=inf_s, inference_finite=finite, val_finished=True)
json.dump(dict(result=res, history=hist), open(os.path.join(out, "week1_timed_cnn.json"), "w"), indent=1)
import csv
with open(os.path.join(out, "cnn_loss_curve.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(hist[0])); w.writeheader(); w.writerows(hist)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(5.5, 3.2))
    plt.plot([h["epoch"] for h in hist], [h["train_mse"] for h in hist], label="train MSE")
    plt.plot([h["epoch"] for h in hist], [h["val_mse"] for h in hist], label="validation MSE")
    plt.axhline(res["persistence_val_mse"], ls=":", c="grey", label="persistence (val)")
    plt.xlabel("epoch"); plt.ylabel("MSE (°C²), 5 outputs"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out, "cnn_loss_curve.png"), dpi=150)
except Exception as e:
    print("plot skipped:", e)

print(json.dumps(res, indent=1))
print(f"\n--- paste into readiness note, item 7 ---\nHardware: {hw['cpu']}, {hw['physical_cores']} physical / {hw['logical_cores']} logical cores, {hw['ram_gb']} GB RAM, {hw['os']}. "
      f"torch {torch.__version__}, {res['torch_threads']} intra-op threads, {res['interop_threads']} inter-op, 0 data-loader workers.\n"
      f"Map CNN ({res['params']:,} params), seed {SEED}, Adam lr {LR}, batch {BATCH}: elapsed {elapsed:.1f} s, CPU {cpu:.1f} s, {res['epochs']} epochs, {n_updates} updates, stopped by {stop}; "
      f"{res['s_per_epoch']:.2f} s/epoch; peak RSS {peak_mb:.0f} MB. Val MSE {hist[0]['val_mse']:.3f} -> best {best:.3f} (epoch {res['best_epoch']}); train MSE {hist[0]['train_mse']:.3f} -> {hist[-1]['train_mse']:.3f}. "
      f"Test-shaped inference (229 inputs, no labels): {inf_s*1000:.0f} ms, finite={finite}.")
