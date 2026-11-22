"""C2 checkpoint fits and smoke test. Test labels are never requested."""
from __future__ import annotations
import argparse, csv, json, os, random, threading, time
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import torch
from torch import nn
from enso_data import build
from models import IndexLSTM, MapCNN, n_params

LEADS = [1, 3, 6, 9, 12]
ALPHAS = (0.1, 1.0, 10.0)
SEED = 22

class AttentionLSTM(nn.Module):
    """The fixed 84-unit index LSTM with temporal softmax attention."""
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 84, batch_first=True)
        self.score = nn.Linear(84, 1)
        self.head = nn.Linear(84, 5)
    def forward(self, x):
        seq, _ = self.lstm(x)
        weights = torch.softmax(self.score(seq).squeeze(-1), dim=1)
        context = torch.sum(seq * weights.unsqueeze(-1), dim=1)
        return self.head(context)

def settings():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(6)
    try: torch.set_num_interop_threads(1)
    except RuntimeError: pass

def validate_manifest(path, data):
    f = pd.read_csv(path)
    if list(f.columns)[:4] != ["sample_id", "unit_id", "partition", "label"]:
        raise ValueError("Unexpected forecast split-file header")
    counts = f.partition.value_counts().to_dict()
    got = (counts.get("train", 0), counts.get("val", 0), counts.get("test", 0), counts.get("excluded", 0))
    if got != (505, 109, 229, 22): raise ValueError(f"split file counts {got}")
    expected_ids = data["table"].issue.astype(str).tolist()
    if f.sample_id.astype(str).tolist() != expected_ids:
        raise ValueError("split file sample_id values differ from reconstructed issue IDs")
    expected = data["table"].partition.tolist()
    if f.partition.tolist() != expected: raise ValueError("split file rows differ from Annex D3 reconstruction")
    return f, got

def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    per_lead=np.sqrt(np.mean((y-p)**2, axis=0))
    return {"rmse": per_lead.tolist(),
            "acc": [float(np.corrcoef(y[:,j],p[:,j])[0,1]) for j in range(y.shape[1])],
            "mean_rmse": float(per_lead.mean())}

def ridge_fit(xtr, ytr, xva, yva, alpha):
    xm, ym = xtr.mean(0), ytr.mean(0)
    xc, yc = xtr-xm, ytr-ym
    b = np.linalg.solve(xc.T@xc + alpha*np.eye(xc.shape[1]), xc.T@yc)
    pred = (xva-xm)@b + ym
    return pred

def fit_torch(name, model, xtr, ytr, xva, yva, out):
    settings(); tx=torch.from_numpy(xtr); ty=torch.from_numpy(ytr)
    vx=torch.from_numpy(xva); vy=torch.from_numpy(yva)
    opt=torch.optim.Adam(model.parameters(), lr=1e-3); lossfn=nn.MSELoss()
    gen=torch.Generator().manual_seed(SEED); best=float("inf"); best_epoch=0; stale=0; hist=[]
    proc=psutil.Process(); peak=[proc.memory_info().rss/2**20]; stop_monitor=threading.Event()
    def sample_memory():
        while not stop_monitor.wait(.01): peak[0]=max(peak[0],proc.memory_info().rss/2**20)
    monitor=threading.Thread(target=sample_memory,daemon=True); monitor.start()
    start=time.perf_counter(); cpu0=time.process_time()
    best_state=None; updates=0; stop_reason="100-epoch limit"
    for ep in range(1,101):
        model.train(); order=torch.randperm(len(tx),generator=gen); total=0.0
        for ids in order.split(32):
            opt.zero_grad(set_to_none=True); loss=lossfn(model(tx[ids]),ty[ids]); loss.backward(); opt.step()
            total += loss.item()*len(ids); updates+=1
            if time.perf_counter()-start >= 900: break
        tr=total/len(tx); model.eval()
        with torch.no_grad(): val=float(lossfn(model(vx),vy))
        hist.append({"epoch":ep,"train_mse":tr,"validation_mse":val})
        if val < best-1e-10:
            best=val; best_epoch=ep; stale=0
            best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=5:
            stop_reason="early_stop"; break
        if time.perf_counter()-start>=900:
            stop_reason="clock"; break
    elapsed=time.perf_counter()-start; cpu=time.process_time()-cpu0
    peak[0]=max(peak[0],proc.memory_info().rss/2**20); stop_monitor.set(); monitor.join()
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad(): pred=model(vx).cpu().numpy()
    out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(),out/f"{name}_best.pt")
    pd.DataFrame(hist).to_csv(out/f"{name}_loss.csv",index=False)
    return pred, {"component":name,"epochs":len(hist),"updates":updates,"best_epoch":best_epoch,
        "val_mse":best,"elapsed_s":elapsed,"cpu_s":cpu,"peak_mem_mb":peak[0],
        "params":n_params(model),"stop_reason":stop_reason,"history":hist}

def smoke(data, split, out, total_start):
    settings(); split_file, counts=validate_manifest(split,data)
    tr=np.arange(min(12,len(data["Y_train"]))); va=np.arange(min(6,len(data["Y_val"])))
    logs=[]; started=time.perf_counter()
    ytr=data["Y_train"][tr]; yv=data["Y_val"][va]
    xtr=data["Xraw_train"][tr]; xv=data["Xraw_val"][va]
    preds={}
    best_alpha=None; best_loss=float("inf")
    for a in ALPHAS:
        t=time.perf_counter(); p=ridge_fit(xtr,ytr,xv,yv,a)
        score=metrics(yv,p)["mean_rmse"]
        if score < best_loss: best_alpha,best_loss=a,score
        logs.append((f"ridge alpha={a}",len(tr),time.perf_counter()-t,"validation fit and scoring"))
    last=np.repeat(data["Xraw_val"][:6,-1,None],5,axis=1)
    logs.append(("persistence",6,0.0,"validation baseline; no fitted parameters"))
    for name, model, key in [("1-D LSTM",IndexLSTM(),"Xidx"),("attention LSTM",AttentionLSTM(),"Xidx"),
                             ("2-D map CNN",MapCNN(),"Xmap")]:
        t=time.perf_counter(); x=torch.from_numpy(data[f"{key}_train"][:12]); y=torch.from_numpy(ytr)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3); model.train()
        for _ in range(2):
            opt.zero_grad(set_to_none=True); loss=nn.MSELoss()(model(x),y); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp=model(torch.from_numpy(data[f"{key}_val"][:6])).numpy()
            tp=model(torch.from_numpy(data[f"{key}_test"][:8])).numpy()
        if not np.isfinite(vp).all() or not np.isfinite(tp).all(): raise ValueError(f"non-finite {name} output")
        preds[name]=vp; logs.append((name,12,time.perf_counter()-t,"2 updates, validation, unscored 8-example test-shaped inference"))
    # Attention ablation is the same no-attention LSTM fit, not another architecture.
    logs.append(("attention ablation (same no-attention LSTM)",12,0.0,"reuse of 1-D LSTM component"))
    fuse_t=time.perf_counter(); candidates=[]
    for w in (0,.25,.5,.75,1):
        q=w*preds["1-D LSTM"]+(1-w)*preds["2-D map CNN"]
        candidates.append((float(np.sqrt(np.mean((yv-q)**2))),abs(w-.5),w))
    best_w=min(candidates,key=lambda z:(z[0],z[1]))[2]
    logs.append((f"fusion (index LSTM + map CNN), selected weight={best_w:g}",6,time.perf_counter()-fuse_t,"validation RMSE selection; no new fit"))
    # The core experiments are validation comparisons and lead-wise metrics from these same predictions.
    _=metrics(yv,preds["1-D LSTM"]); _=metrics(yv,preds["2-D map CNN"])
    logs += [("core experiment 1: sequential vs spatial",6,0.0,"validation comparison"),
             ("core experiment 2: lead-wise skill and cost",6,0.0,"reuse same validation predictions")]
    out.mkdir(parents=True,exist_ok=True)
    total=time.perf_counter()-total_start
    lines=["C2 smoke test; seed=22; unscored test-shaped inference; no test metrics computed.",
           f"Split file validated: train={counts[0]}, val={counts[1]}, test={counts[2]}, excluded={counts[3]}"]
    for name,n,secs,detail in logs: lines.append(f"{name}: samples={n}; elapsed_s={secs:.4f}; {detail}")
    lines.append(f"TOTAL elapsed_s={total:.4f}")
    (out/"smoke_log.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

def full(args):
    settings(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    data=build(args.index,args.grid,include_test_labels=False)
    _,counts=validate_manifest(args.split,data)
    rows=[]; proc=psutil.Process(); id0=args.run_offset
    # Three declared ridge alphas; validation chooses among these only.
    for j,a in enumerate(ALPHAS):
        cpu0=time.process_time(); t=time.perf_counter()
        pred=ridge_fit(data["Xraw_train"],data["Y_train"],data["Xraw_val"],data["Y_val"],a)
        elapsed=time.perf_counter()-t; cpu=time.process_time()-cpu0
        np.save(out/f"ridge_alpha_{a:g}_val.npy",pred)
        rows.append([f"R{j+2+id0:03d}",args.date,args.commit,"baseline",f"ridge_alpha={a:g}",22,505,"","","classical",f"{elapsed:.6f}",f"{cpu:.6f}",f"{proc.memory_info().rss/2**20:.2f}","intra-op=6;inter-op=1",0,"","mean_rmse",f"{metrics(data['Y_val'],pred)['mean_rmse']:.8f}","","no"])
    vals=[(metrics(data["Y_val"],np.load(out/f"ridge_alpha_{a:g}_val.npy"))["mean_rmse"],a) for a in ALPHAS]
    best_alpha=min(vals,key=lambda x:(x[0],abs(x[1]-1)))[1]
    lp,lres=fit_torch("lstm",IndexLSTM(),data["Xidx_train"],data["Y_train"],data["Xidx_val"],data["Y_val"],out)
    ap,ares=fit_torch("attention",AttentionLSTM(),data["Xidx_train"],data["Y_train"],data["Xidx_val"],data["Y_val"],out)
    # Reuse the Week-1 map-CNN fit; evaluate it only on validation and test-shaped inputs.
    cnn=MapCNN(); cnn.load_state_dict(torch.load(args.cnn_checkpoint,map_location="cpu",weights_only=True)); cnn.eval()
    with torch.no_grad():
        cp=cnn(torch.from_numpy(data["Xmap_val"])).numpy()
        _=cnn(torch.from_numpy(data["Xmap_test"][:64]))
    for name,res,pred,component in [("lstm",lres,lp,"1d"),("attention",ares,ap,"attention")]:
        rows.append([f"R{(5 if name=='lstm' else 6)+id0:03d}",args.date,args.commit,component,"reference",22,505,res["epochs"],res["updates"],res["stop_reason"],f"{res['elapsed_s']:.6f}",f"{res['cpu_s']:.6f}",f"{res['peak_mem_mb']:.2f}","intra-op=6;inter-op=1",0,res["params"],"mean_mse",f"{res['val_mse']:.8f}",res["best_epoch"],"no"])
        np.save(out/f"{name}_val.npy",pred)
        np.save(out/f"{name}_val.npy",pred)
    cm=metrics(data["Y_val"],cp); lm=metrics(data["Y_val"],lp)
    fuse_start=time.perf_counter(); fuse_cpu=time.process_time(); fusion_candidates=[]
    for w in (0,.25,.5,.75,1):
        p=w*lp+(1-w)*cp; fusion_candidates.append((metrics(data["Y_val"],p)["mean_rmse"],abs(w-.5),w,p))
    f_rmse,_,weight,fp=min(fusion_candidates,key=lambda x:(x[0],x[1]))
    np.save(out/"fusion_val.npy",fp)
    _=weight*lp+(1-weight)*cp
    fuse_elapsed=time.perf_counter()-fuse_start
    rows.append([f"R{7+id0:03d}",args.date,args.commit,"fusion",f"weight_lstm={weight:g}",22,109,"","","none",f"{fuse_elapsed:.6f}",f"{time.process_time()-fuse_cpu:.6f}","","intra-op=6;inter-op=1",0,"","mean_rmse",f"{f_rmse:.8f}","","no"])
    np.save(out/"cnn_val.npy",cp)
    # CNN Week-1 reference fit: taken from its saved readiness record; its historical commit was not captured.
    rows.insert(0,["R001","2026-10-02","unavailable","reference","matched",22,505,9,144,"early_stop","3.215000","18.844000","494.60","6/1",0,30229,"mean_mse","0.32016855",4,"no"])
    hdr="run_id,date,commit,component,condition,seed,n_train,epochs,updates,stop_reason,elapsed_s,cpu_s,peak_mem_mb,threads,workers,trainable_params,val_metric,val_value,best_epoch,test_scored".split(",")
    with (out/"T03_C2_runlog.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f,lineterminator="\n"); w.writerow(hdr); w.writerows(rows)
    allmetrics={"counts":{"train":counts[0],"validation":counts[1],"test_inputs_only":counts[2],"excluded":counts[3]},
       "baseline":{"selected_alpha":best_alpha,"validation_by_alpha":{str(a):metrics(data["Y_val"],np.load(out/f"ridge_alpha_{a:g}_val.npy")) for a in ALPHAS}},
       "lstm":metrics(data["Y_val"],lp),"cnn":cm,"attention":metrics(data["Y_val"],ap),
       "fusion":{"weight_lstm":weight,"validation":metrics(data["Y_val"],fp)},
       "support":{"validation_issue_months":109,"independent_events":4},
       "test_scored":False,"lead_months":LEADS}
    (out/"validation_results.json").write_text(json.dumps(allmetrics,indent=2),encoding="utf-8")
    print(json.dumps(allmetrics,indent=2))

def recompute(args):
    """Re-score saved validation predictions using mean of the five lead RMSEs."""
    out=Path(args.out); d=build(args.index,args.grid,include_test_labels=False)
    validate_manifest(args.split,d); y=d["Y_val"]
    result={"counts":{"train":505,"validation":109,"test_inputs_only":229,"excluded":22},
        "support":{"validation_issue_months":109,"independent_events":4},"lead_months":LEADS,"test_scored":False}
    for name in ("lstm","attention","cnn"):
        result[name]=metrics(y,np.load(out/f"{name}_val.npy"))
    baseline={}; options=[]
    for a in ALPHAS:
        baseline[str(a)]=metrics(y,np.load(out/f"ridge_alpha_{a:g}_val.npy"))
        options.append((baseline[str(a)]["mean_rmse"],a))
    result["baseline"]={"selected_alpha":min(options)[1],"validation_by_alpha":baseline}
    fuse_start=time.perf_counter(); fuse_cpu=time.process_time(); candidates=[]
    for w in (0,.25,.5,.75,1):
        m=metrics(y,w*np.load(out/"lstm_val.npy")+(1-w)*np.load(out/"cnn_val.npy"))
        candidates.append((m["mean_rmse"],abs(w-.5),w,m))
    _,_,w,fm=min(candidates,key=lambda z:(z[0],z[1]))
    lm=IndexLSTM(); lm.load_state_dict(torch.load(out/"lstm_best.pt",map_location="cpu",weights_only=True)); lm.eval()
    cm=MapCNN(); cm.load_state_dict(torch.load(args.cnn_checkpoint,map_location="cpu",weights_only=True)); cm.eval()
    with torch.no_grad():
        lt=lm(torch.from_numpy(d["Xidx_test"])).numpy()
        ct=cm(torch.from_numpy(d["Xmap_test"])).numpy()
        _=w*lt+(1-w)*ct
    fuse_elapsed=time.perf_counter()-fuse_start; fuse_cpu_s=time.process_time()-fuse_cpu
    result["fusion"]={"weight_lstm":w,"validation":fm,
        "candidate_mean_rmse":{str(q):next(z[0] for z in candidates if z[2]==q) for q in (0,.25,.5,.75,1)},
        "selection_and_test_shaped_inference_s":fuse_elapsed,"inference_test_targets_loaded":False}
    (out/"validation_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    log=out/"T03_C2_runlog.csv"
    with log.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f)); fields=list(rows[0])
    for row in rows:
        if row["component"]=="baseline":
            a=f"{float(row['condition'].split('=')[1]):.1f}"; row["val_value"]=f"{baseline[a]['mean_rmse']:.8f}"
        if row["component"]=="fusion":
            row["condition"]=f"weight_lstm={w:g}"; row["val_value"]=f"{fm['mean_rmse']:.8f}"
            row["elapsed_s"]=f"{fuse_elapsed:.6f}"; row["cpu_s"]=f"{fuse_cpu_s:.6f}"
    with log.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--index",required=True); p.add_argument("--grid",required=True); p.add_argument("--split",required=True)
    p.add_argument("--out",default="checkpoint_submission/results"); p.add_argument("--cnn-checkpoint",default="out/cnn_best.pt")
    p.add_argument("--date",default="2026-11-04"); p.add_argument("--smoke",action="store_true"); p.add_argument("--recompute",action="store_true")
    p.add_argument("--commit",default="unavailable"); p.add_argument("--run-offset",type=int,default=0)
    a=p.parse_args(); total_start=time.perf_counter(); settings()
    if a.recompute: recompute(a); return
    if a.smoke:
        d=build(a.index,a.grid,include_test_labels=False); smoke(d,a.split,Path(a.out),total_start)
    else: full(a)
if __name__=="__main__": main()
