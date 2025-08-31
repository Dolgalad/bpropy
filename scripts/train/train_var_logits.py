# train_var_logits.py
import os, time, json, argparse, numpy as np, torch
import sys
sys.path.append("../..")
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from progress.bar import Bar

from bpropy.datasets.mrf_dataset_2 import MRFDataset, MRFDatasetConfig
from bpropy.adapters.vc_to_data import problem_to_data
from bpropy.datasets.collator import mrf_collate
from bpropy.datasets.prefetcher import CUDAPrefetcher

# Import the new models:
from bpropy.models.logit_models import GCNVarLogits, HeteroGCNVarLogits

MODEL_REGISTRY = {
    "GCNVarLogits": GCNVarLogits,
    "HeteroGCNVarLogits": HeteroGCNVarLogits,
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


    return p.parse_args()

def save_checkpoint(path, epoch, model, optimizer, extra=None):
    ckpt = {
        "epoch": int(epoch),
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
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    return start_epoch, ckpt

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


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{args.model_class}_{'hetero' if args.hetero else 'homo'}_{int(time.time())}"
    log_dir = os.path.join(args.log_dir, run_name)
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
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
        collate_fn=mrf_collate,
    )
    loader_val = DataLoader(
        ds_val, batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2,
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

    start_epoch = 1
    if args.resume:
        start_epoch, _ = load_checkpoint(args.resume, model, opt, map_location=device)
        print(f"[resume] Loaded checkpoint from {args.resume}, resuming at epoch {start_epoch}")

    best_gap = float("inf")
    best_epoch = 1
    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        tr_losses = []
        bar = Bar(f"[Train] epoch {epoch}", max=len(loader_train))
        for data, y in CUDAPrefetcher(loader_train, device):
            opt.zero_grad()
            logits = model(data)
            loss = bce_on_variables(logits, y.to(device, non_blocking=True))
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
            bar.next()
        bar.finish()
        tr = float(np.mean(tr_losses)) if tr_losses else 0.0

        # ---- Val ----
        model.eval()
        va_losses = []
        va_opts = []
        with torch.no_grad():
            bar = Bar(f"[Val]   epoch {epoch}", max=len(loader_val))
            for pb, data, y in CUDAPrefetcher(loader_val, device):
                logits = model(data)
                loss = bce_on_variables(logits, y.to(device, non_blocking=True))
                va_losses.append(loss.item()); bar.next()
                # solve greedy
                grd_sol = pb[0].greedy(scores=logits.cpu().numpy())
                # gap
                gap = ( grd_sol.sum() - y.sum().item()) / y.sum().item()
                va_opts.append(gap)
            bar.finish()
        va = float(np.mean(va_losses)) if va_losses else 0.0
        gap = float(np.mean(va_opts)) if va_opts else 0.0

        writer.add_scalars("loss", {"train": tr, "val": va, "gap": gap}, epoch)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | gap {gap:.4f}")

        last_ckpt = os.path.join(log_dir, "checkpoints", "last.pt")
        save_checkpoint(
            last_ckpt, epoch, model, opt,
            extra={"best_gap": best_gap, "best_epoch": best_epoch}
        )


        if gap < best_gap:
            best_gap = gap
            best_epoch = epoch
            best_ckpt_path = os.path.join(log_dir, "checkpoints", "best.pt")
            save_checkpoint(
                best_ckpt_path, epoch, model, opt,
                extra={"best_opt": best_opt, "best_epoch": best_epoch}
            )

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

if __name__ == "__main__":
    main()

