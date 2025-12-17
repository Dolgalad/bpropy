# train_var_logits.py
import os, time, json, argparse, numpy as np, torch
from dataclasses import dataclass
import sys
sys.path.append("../..")
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar
import torch_scatter

from bpropy.datasets.mrf_dataset_2 import MRFDataset, MRFDatasetConfig
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.datasets.collator import mrf_collate
from bpropy.datasets.prefetcher import CUDAPrefetcher

# Import the new models:
from bpropy.models.logit_models import GCNVarLogits, HeteroGCNVarLogits, HeteroGCNFactorLogits
from bpropy.models.mrf_models import Hetero_GCN_MRF_v4

MODEL_REGISTRY = {
    "GCNVarLogits": GCNVarLogits,
    #"GCNFactorLogits": GCNFactorLogits,
    "HeteroGCNVarLogits": HeteroGCNVarLogits,
    "HeteroGCNFactorLogits": HeteroGCNFactorLogits,
    "Hetero_GCN_MRF_v4": Hetero_GCN_MRF_v4,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset_root")
    p.add_argument("--split", default="train")
    p.add_argument("--val_split", default="val")
    p.add_argument("--log_dir", default="runs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--feature_mode", default="EB")

    # graph build toggles
    p.add_argument("--hetero", action="store_true")
    p.add_argument("--with_lbp", action="store_true")
    p.add_argument("--with_nmf", action="store_true")

    # labels (choose which assignment to learn)
    p.add_argument("--solver", default="greedy")

    # model selection via name + JSON args
    p.add_argument("--model_class", default="GCNVarLogits",
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--model_args", default="{}",
                   help="JSON dict of kwargs passed to the model ctor")
    # processing-dir
    p.add_argument("--processed_dir", default="processed",
                   help="subdirectory under <root>/processed/ used for cached features")
    #p.add_argument("--greedy-fn", required=True,
    #               help="Dotted path 'pkg.mod:func' for greedy solver, signature (pb, scores)->(y_hat, obj) or ->y_hat")
    p.add_argument("--patience", type=int, default=20,
                   help="epochs without gap improvement before early stop")
    p.add_argument("--resume", default="", help="Path to checkpoint (.pt/.pth) to resume training")
    p.add_argument("--maximize", action="store_true", help="Optimization is a maximization problem")
    p.add_argument("--steps_per_epoch", type=int, default=10000000)
    p.add_argument("--steps_per_val", type=int, default=10000000)
    p.add_argument("--loss", default="factor_ce", choices=["factor_ce", "focal_ce"])


    return p.parse_args()

@dataclass
class EpochState:
    epoch = None
    start = None
    best_gap = None
    best_epoch = None
    epochs_bad = None

def save_checkpoint(path, epoch_state, model, optimizer, extra=None):
    ckpt = {
        "epoch": int(epoch_state.epoch),
        "best_gap": float(epoch_state.best_gap),
        "best_epoch": int(epoch_state.best_epoch),
        "epochs_bad": int(epoch_state.epochs_bad),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    epoch_state = EpochState()
    epoch_state.epoch = int(ckpt.get("epoch",0))+1
    epoch_state.best_gap = float(ckpt.get("best_gap", "inf"))
    epoch_state.best_epoch = int(ckpt.get("best_epoch", 1))
    epoch_state.epochs_bad = int(ckpt.get("epochs_bad", 0))

    return epoch_state, ckpt

def bce_on_variables(logits, y):
    return F.binary_cross_entropy_with_logits(logits, y.float())

def make_ds(root, split, cfg, hetero, feature_mode, with_lbp, with_nmf, processed_dir):
    return MRFDataset(
        root, split, cfg=cfg,
        preprocess=lambda pb: problem_to_data(
            pb, hetero=hetero, feature_mode=feature_mode,
            with_lbp=with_lbp, with_nmf=with_nmf
        ),
        processed_dir=processed_dir
    )

def _sync(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize()

def factor_ce(pred, data, reduction="mean", eps=1e-8):
    m_stride = data.lbp_edges.m_stride.to(dtype=torch.long, device=data.y_ss.device)   # [nF], each = 2**|F|
    m_idx    = torch.arange(m_stride.numel(), device=data.y_ss.device).repeat_interleave(m_stride)
    nF = m_stride.numel()
    _, global_labels = torch_scatter.scatter_max(data.y_ss, m_idx, dim=0, dim_size=nF)   # labels shape [nF]

    logp = pred
    ce_per_factor = -logp[global_labels]
    if reduction=="mean":
        return ce_per_factor.mean()
    if reduction=="sum":
        return ce_per_factor.sum()
    return ce_per_factor

def factor_focal_ce(pred, data, reduction="mean", eps=1e-8, alpha=1.0, gamma=2.0):
    m_stride = data.lbp_edges.m_stride.to(dtype=torch.long, device=data.y_ss.device)   # [nF], each = 2**|F|
    m_idx    = torch.arange(m_stride.numel(), device=data.y_ss.device).repeat_interleave(m_stride)
    nF = m_stride.numel()
    _, global_labels = torch_scatter.scatter_max(data.y_ss, m_idx, dim=0, dim_size=nF)   # labels shape [nF]

    p_t = pred[global_labels]
    l = -alpha * ((1 - p_t.exp()) ** gamma) * p_t
    if reduction=="mean":
        return l.mean()
    if reduction=="sum":
        return l.sum()
    return l

def evaluate_on_train_split(
    args, device, log_dir, model,
    make_ds_fn, mrf_collate, MRFDatasetConfig, loss_fn
):
    """
    Evaluates on the TRAIN split with (pb, data, y) per item.
    Computes BCE loss, greedy gap, and times (prediction, greedy, total).
    Saves mean/std to <log_dir>/train_eval.json.
    """
    # config that yields (pb, data, y)
    eval_cfg = MRFDatasetConfig(
        feature_mode=args.feature_mode,
        yield_mode="both",
        refresh=False,
        return_assignment=True,
        assignment_solver=args.solver,
    )
    ds_train_eval = make_ds_fn(
        args.dataset_root, "test", eval_cfg,
        args.hetero, args.feature_mode, args.with_lbp, args.with_nmf,
        args.processed_dir
    )
    loader_eval = DataLoader(
        ds_train_eval, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )

    model.eval()
    losses, gaps = [], []
    t_pred, t_grd, t_tot = [], [], []

    with torch.no_grad():
        bar = Bar(f"[Test]          ", max=len(loader_eval))
        for pb, data, y in CUDAPrefetcher(loader_eval, device):
            # timing with CUDA-sync for accuracy
            _sync(device)
            t0 = time.perf_counter()

            # prediction
            t1 = time.perf_counter()
            logits = model(data)
            _sync(device)
            t2 = time.perf_counter()

            # greedy on CPU scores (your pb API)
            scores = logits.detach().cpu().numpy()
            t3 = time.perf_counter()
            grd_sol = pb[0].greedy(scores=scores)
            t4 = time.perf_counter()

            # metrics
            loss = loss_fn(logits, data).item()

            # same gap definition you used during validation
            if args.maximize:
                gap =  (y.sum().item() - grd_sol.sum()) / (y.sum().item() + 1e-8)
            else:
                gap =  (grd_sol.sum() - y.sum().item()) / (y.sum().item() + 1e-8)


            losses.append(loss)
            gaps.append(gap)
            t_pred.append(t2 - t1)
            t_grd.append(t4 - t3)
            t_tot.append(t4 - t0)
            bar.next()
        bar.finish()

    def _m_s(x):
        x = np.array(x, dtype=np.float64)
        return float(x.mean()) if x.size else 0.0, float(x.std(ddof=1)) if x.size > 1 else 0.0

    out = {
        "loss": {"mean": _m_s(losses)[0], "std": _m_s(losses)[1]},
        "gap":  {"mean": _m_s(gaps)[0],   "std": _m_s(gaps)[1]},
        "time_pred_s": {"mean": _m_s(t_pred)[0], "std": _m_s(t_pred)[1]},
        "time_greedy_s": {"mean": _m_s(t_grd)[0], "std": _m_s(t_grd)[1]},
        "time_total_s": {"mean": _m_s(t_tot)[0], "std": _m_s(t_tot)[1]},
        "count": len(losses),
    }

    out_path = os.path.join(log_dir, "test_eval.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"[test-eval] saved stats to {out_path}")

def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.model_class}_{'hetero' if args.hetero else 'homo'}_{int(time.time())}_{int(args.seed)}"
    log_dir = os.path.join(args.log_dir, run_name)
    print("Log dir : ", log_dir)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Dataset config: return assignment tensor y
    cfg = MRFDatasetConfig(
        feature_mode=args.feature_mode,
        yield_mode="data",
        refresh=False,
        return_assignment=True,
        assignment_solver=args.solver,
    )

    val_cfg = MRFDatasetConfig(
        feature_mode=args.feature_mode,
        yield_mode="both",
        refresh=False,
        return_assignment=True,
        assignment_solver=args.solver,
    )

    ds_train = make_ds(args.dataset_root, args.split, cfg,
                       args.hetero, args.feature_mode, args.with_lbp, args.with_nmf, args.processed_dir)
    ds_val   = make_ds(args.dataset_root, args.val_split, val_cfg,
                       args.hetero, args.feature_mode, args.with_lbp, args.with_nmf, args.processed_dir)

    loader_train = DataLoader(
        ds_train, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )

    # Build model from registry + JSON args
    model_kwargs = json.loads(args.model_args)
    run_hparams = {
        "timestamp": int(time.time()),
        "device": str(device),
        "dataset_root": args.dataset_root,
        "split": args.split,
        "val_split": args.val_split,
        "feature_mode": args.feature_mode,
        "hetero": args.hetero,
        "with_lbp": args.with_lbp,
        "with_nmf": args.with_nmf,
        "solver": args.solver,
        "processed_dir": args.processed_dir,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "model_class": args.model_class,
        "model_args": model_kwargs,
        "run_name": run_name,
        "log_dir": log_dir,
    }
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(run_hparams, f, indent=2, sort_keys=True)
    # you must set in_dim according to your features (e.g., EB features size)
    Model = MODEL_REGISTRY[args.model_class]
    model = Model(**model_kwargs).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr * args.batch_size)
    loss_fn = factor_ce
    if args.loss == "focal_ce":
        loss_fn = factor_focal_ce

    start_epoch = 1
    best_gap = float("inf")
    best_epoch = 1
    epochs_bad = 0

    if args.resume:
        epoch_state, _ = load_checkpoint(args.resume, model, opt, map_location=device)
        start_epoch = epoch_state.epoch
        best_gap = epoch_state.best_gap
        best_epoch = epoch_state.best_epoch
        epochs_bad = epoch_state.epochs_bad
        print(f"[resume] Loaded checkpoint from {args.resume}, resuming at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_losses = []
        bar = Bar(f"[Train] epoch {epoch}", max=len(loader_train))
        c = 0
        for data, y in CUDAPrefetcher(loader_train, device):
            opt.zero_grad()
            logits = model(data)
            loss = loss_fn(logits, data)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
            bar.next()
            c += 1
            if c >= args.steps_per_epoch:
                break

        bar.finish()
        tr = float(np.mean(tr_losses)) if tr_losses else 0.0

        # ---- Val ----
        model.eval()
        va_losses = []
        va_opts = []
        with torch.no_grad():
            bar = Bar(f"[Val]   epoch {epoch}", max=len(loader_val))
            c = 0
            for pb, data, y in CUDAPrefetcher(loader_val, device):
                logits = model(data)
                loss = loss_fn(logits, data)
                va_losses.append(loss.item()); bar.next()
                # solve greedy; scores used are the unary factor values
                scores = logits[:2*pb[0].nv()].reshape(-1,2)[:,1].exp().cpu().numpy()
                grd_sol = pb[0].greedy(scores=scores)
                # gap
                if args.maximize:
                    gap = (y.sum().item() - grd_sol.sum()) / y.sum().item()
                else:
                    gap = (grd_sol.sum() - y.sum().item()) / y.sum().item()

                va_opts.append(gap)
                #print("logits 0: ", logits[y==0].mean(), " logits 1: ", logits[y==1].mean())
                c += 1
                if c >= args.steps_per_val:
                    break
            bar.finish()
        va = float(np.mean(va_losses)) if va_losses else 0.0
        gap = float(np.mean(va_opts)) if va_opts else 0.0

        writer.add_scalars("loss", {"train": tr, "val": va, "gap": gap}, epoch)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | gap {gap:.4f}")


        # epoch state
        save_best = False
        end_train = False
        epoch_state = EpochState()
        epoch_state.epoch = epoch

        if gap < best_gap:
            best_gap = gap
            best_epoch = epoch
            epochs_bad = 0
            epoch_state.best_gap = gap
            epoch_state.best_epoch = best_epoch
            epoch_state.epochs_bad = 0
            save_best = True
        else:
            epochs_bad += 1
            epoch_state.best_gap = gap
            epoch_state.best_epoch = best_epoch
            epoch_state.epochs_bad = epochs_bad

            if epochs_bad >= args.patience:
                end_train = True


        last_ckpt = os.path.join(log_dir, "checkpoints", "last.pt")
        save_checkpoint(
            last_ckpt, epoch_state, model, opt,
            extra={"best_gap": best_gap, "best_epoch": best_epoch}
        )


        if save_best:
            #best_gap = gap
            #best_epoch = epoch
            best_ckpt_path = os.path.join(log_dir, "checkpoints", "best.pt")
            save_checkpoint(
                best_ckpt_path, epoch_state, model, opt,
                extra={"best_gap": best_gap, "best_epoch": best_epoch}
            )
        else:
            if end_train:
                print(f"[early-stop] No gap improvement for {epochs_bad} epochs "
                      f"(patience={args.patience}). Stopping at epoch {epoch}.")
                break

    writer.close()
    print("Done. Best opt:", best_gap)

    # Update hparams file with summary
    summary = {
        **run_hparams,
        "best_val_opt_gap": best_gap,
        "best_epoch": best_epoch,
    }
    with open(os.path.join(log_dir, "hparams.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    # --- Load best model before final evaluation ---
    best_ckpt_path = os.path.join(log_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        print(f"[eval] Loading best checkpoint from {best_ckpt_path}")
        _, ckpt = load_checkpoint(best_ckpt_path, model, optimizer=None, map_location=device)
    else:
        print("[eval] No best checkpoint found, using final model.")

    # test model
    evaluate_on_train_split(
        args, device, log_dir, model,
        make_ds_fn=make_ds,
        mrf_collate=mrf_collate,
        MRFDatasetConfig=MRFDatasetConfig,
    )

if __name__ == "__main__":
    main()

