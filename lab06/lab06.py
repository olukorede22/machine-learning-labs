"""
Lab 6: Training pipelines (Fashion-MNIST, model from Lab 3), driven by config.yaml.

    pip install -r requirements.txt
    python lab06.py                    # everything (tasks 1-7), about 30-90 min on a CPU
    python lab06.py --task train       # task 1 only: one training run logged to MLflow
    python lab06.py --task profile     # task 2        --task workers   # task 3
    python lab06.py --task batch       # task 4        --task sched     # task 5
    python lab06.py --task accum       # task 6        --task resume    # task 7
    python lab06.py --quick            # tiny subset + few epochs, for a fast test
    python lab06.py --synthetic        # random data (no download), code test only

View MLflow runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (then open http://127.0.0.1:5000)
Outputs: ./results/*.csv|*.md|*.png, ./checkpoints/<run>/{last,best}.pt, ./mlflow.db, ./mlruns
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

try:
    import mlflow
except ImportError:  # the pipeline still runs, but task 1 needs MLflow for the marks
    mlflow = None

HERE = Path(__file__).resolve().parent
SYNTHETIC = False  # set by --synthetic


# --------------------------------------------------------------------------
# Models (the three Lab 3 architectures; choose one with model.name in the YAML)
# --------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                 nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.net(x)


class PatchAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(49, 64)
        self.pos = nn.Parameter(torch.zeros(1, 16, 64))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(64, 4, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        b = x.shape[0]
        p = x.unfold(2, 7, 7).unfold(3, 7, 7).reshape(b, 16, 49)
        h = self.encoder(self.embed(p) + self.pos)
        return self.head(h.mean(dim=1))


MODELS = {"mlp": MLP, "cnn": CNN, "attention": PatchAttention}


# --------------------------------------------------------------------------
# Config, seeds, small helpers
# --------------------------------------------------------------------------
def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def override(cfg, changes):
    """Copy of the config with dotted keys changed, e.g. {'train.batch_size': 64}."""
    c = copy.deepcopy(cfg)
    for key, val in changes.items():
        node, parts = c, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = val
    return c


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        out.update(flatten(v, key) if isinstance(v, dict) else {key: v})
    return out


def apply_quick(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["data"]["subset"] = [6000, 1000, 1000]
    s = cfg["studies"]
    s.update({"workers": [0, 2], "batch_sizes": [32, 128, 512], "batch_epochs": 2,
              "scheduler_epochs": 3, "accum": {**s["accum"], "epochs": 2},
              "resume": {"total_epochs": 4, "interrupt_at": 2}})
    cfg["train"]["epochs"] = 2
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg):
    want = cfg.get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def save_study(res_dir, name, df, notes):
    df.to_csv(res_dir / f"{name}.csv", index=False)
    (res_dir / f"{name}.md").write_text(
        f"## {name}\n\n{md_table(df)}\n\n" + "\n".join(f"- {n}" for n in notes) + "\n",
        encoding="utf-8")
    print(f"\n== {name} ==\n{df.to_string(index=False)}")
    for n in notes:
        print("  *", n)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def make_synthetic(n, seed):
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 10, (n,), generator=g)
    x = torch.randn(n, 1, 28, 28, generator=g) * 0.5 + (y.float().view(-1, 1, 1, 1) - 4.5) * 0.15
    return TensorDataset(x, y)


def get_datasets(cfg, synthetic=False):
    d, seed = cfg["data"], cfg["seed"]
    if synthetic:
        train, val, test = (make_synthetic(5000, seed), make_synthetic(1000, seed + 1),
                            make_synthetic(1000, seed + 2))
    else:
        from torchvision import datasets, transforms
        tf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((d["mean"],), (d["std"],))])
        full = datasets.FashionMNIST(d["root"], train=True, download=True, transform=tf)
        test = datasets.FashionMNIST(d["root"], train=False, download=True, transform=tf)
        train, val = random_split(full, [len(full) - d["val_size"], d["val_size"]],
                                  generator=torch.Generator().manual_seed(seed))
    sub = d.get("subset")
    if sub:
        train, val, test = (Subset(train, range(min(sub[0], len(train)))),
                            Subset(val, range(min(sub[1], len(val)))),
                            Subset(test, range(min(sub[2], len(test)))))
    return train, val, test


def make_loaders(cfg, datasets, device):
    train, val, test = datasets
    nw, bs = cfg["data"]["num_workers"], cfg["train"]["batch_size"]
    gen = torch.Generator().manual_seed(cfg["seed"])  # makes shuffling reproducible/resumable
    common = dict(num_workers=nw, pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    return (DataLoader(train, bs, shuffle=True, generator=gen, **common),
            DataLoader(val, 512, shuffle=False, **common),
            DataLoader(test, 512, shuffle=False, **common), gen)


# --------------------------------------------------------------------------
# Memory tracking (CUDA allocator peak, or sampled process RSS on CPU)
# --------------------------------------------------------------------------
class MemTracker:
    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.peak = 0.0

    @staticmethod
    def _rss():
        p = psutil.Process()
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            try:
                total += c.memory_info().rss
            except psutil.Error:
                pass
        return total

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._rss())
            time.sleep(0.05)

    def __enter__(self):
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            self.start = self._rss()
            self.peak = self.start
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._loop, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *a):
        if not self.cuda:
            self._stop.set()
            self._t.join()
            self.peak = max(self.peak, self._rss())

    @property
    def peak_mb(self):
        return (torch.cuda.max_memory_allocated() if self.cuda else self.peak) / 1e6

    @property
    def delta_mb(self):
        return self.peak_mb if self.cuda else (self.peak - self.start) / 1e6


# --------------------------------------------------------------------------
# Optimizer, scheduler, checkpoints
# --------------------------------------------------------------------------
def build_optimizer(model, t):
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                               weight_decay=t["weight_decay"])
    raise ValueError(t["optimizer"])


def build_scheduler(opt, t):
    name = t["scheduler"]
    if name == "none":
        return None
    p = dict(t["scheduler_params"][name])
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(opt, **p)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["T_max"] or t["epochs"])
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", **p)
    raise ValueError(name)


def save_checkpoint(path, model, opt, sched, epoch, history, cfg, gen):
    torch.save({
        "epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None, "history": history, "config": cfg,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "loader": gen.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, path)


def load_checkpoint(path, model, opt, sched, gen):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if sched and ck["scheduler"]:
        sched.load_state_dict(ck["scheduler"])
    r = ck["rng"]
    random.setstate(r["python"])
    np.random.set_state(r["numpy"])
    torch.set_rng_state(r["torch"].cpu())
    gen.set_state(r["loader"].cpu())
    if r["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(r["cuda"])
    return ck["epoch"], ck["history"]


# --------------------------------------------------------------------------
# One epoch, evaluation, full training run (task 1 + building block for all studies)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    return loss_sum / n, correct / n


def train_epoch(model, loader, opt, device, accum):
    """Gradient accumulation: optimizer step every `accum` micro-batches."""
    model.train()
    loss_sum, correct, n, wait = 0.0, 0, 0, 0.0
    opt.zero_grad(set_to_none=True)
    it, n_batches, t0 = iter(loader), len(loader), time.perf_counter()
    n_total, bs = len(loader.dataset), loader.batch_size
    for i in range(n_batches):
        tw = time.perf_counter()
        x, y = next(it)
        wait += time.perf_counter() - tw  # time blocked waiting for the data loader
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = F.cross_entropy(out, y)
        # Weight each micro-batch by its share of the accumulation group, so the summed gradient
        # equals the gradient of one big batch exactly (also for the last, smaller group).
        group_start = (i // accum) * accum * bs
        group_size = min(accum * bs, n_total - group_start)
        (loss * (y.numel() / group_size)).backward()
        if (i + 1) % accum == 0 or i == n_batches - 1:
            opt.step()
            opt.zero_grad(set_to_none=True)
        loss_sum += loss.item() * y.numel()
        correct += (out.argmax(1) == y).sum().item()
        n += y.numel()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return {"train_loss": loss_sum / n, "train_acc": correct / n, "epoch_time_s": elapsed,
            "data_wait_s": wait, "samples_per_s": n / elapsed}


def run_training(cfg, datasets, run_name, experiment=None, stop_after=None, resume=None,
                 final_test=True, quiet=False):
    t, device = cfg["train"], get_device(cfg)
    set_seed(cfg["seed"])
    model = MODELS[cfg["model"]["name"]]().to(device)
    opt = build_optimizer(model, t)
    sched = build_scheduler(opt, t)
    train_loader, val_loader, test_loader, gen = make_loaders(cfg, datasets, device)
    out_dir = HERE / cfg["checkpoint_dir"] / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history, start = [], 0
    if resume:
        start, history = load_checkpoint(resume, model, opt, sched, gen)
        print(f"[{run_name}] resumed from {resume} at epoch {start}")

    run = None
    if mlflow is not None:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(experiment or cfg["mlflow"]["experiment"])
        run = mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v)[:250] for k, v in flatten(cfg).items()})
        mlflow.log_param("resumed_from_epoch", start)
        mlflow.log_param("resolved_device", str(device))
        mlflow.log_param("trainable_params", sum(p.numel() for p in model.parameters()))

    best_acc = max([h["val_acc"] for h in history], default=-1.0)
    interrupted = False
    with MemTracker(device) as mem:
        for epoch in range(start + 1, t["epochs"] + 1):
            lr = opt.param_groups[0]["lr"]
            row = train_epoch(model, train_loader, opt, device, t["accumulation_steps"])
            val_loss, val_acc = evaluate(model, val_loader, device)
            if sched:
                sched.step(val_loss) if t["scheduler"] == "plateau" else sched.step()
            row.update(epoch=epoch, lr=lr, val_loss=val_loss, val_acc=val_acc)
            history.append(row)
            if not quiet:
                print(f"[{run_name}] epoch {epoch}/{t['epochs']} lr={lr:.5f} "
                      f"train_loss={row['train_loss']:.4f} val_acc={val_acc:.4f} "
                      f"{row['samples_per_s']:.0f} samples/s ({row['epoch_time_s']:.1f}s)", flush=True)
            save_checkpoint(out_dir / "last.pt", model, opt, sched, epoch, history, cfg, gen)
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint(out_dir / "best.pt", model, opt, sched, epoch, history, cfg, gen)
            if run is not None:
                mlflow.log_metrics({k: row[k] for k in ("train_loss", "train_acc", "val_loss",
                                    "val_acc", "lr", "epoch_time_s", "samples_per_s",
                                    "data_wait_s")}, step=epoch)
            if stop_after and epoch >= stop_after and epoch < t["epochs"]:
                interrupted = True
                print(f"[{run_name}] INTERRUPTED after epoch {epoch} (checkpoint saved)")
                break

    result = {"history": history, "peak_mem_mb": mem.peak_mb, "mem_delta_mb": mem.delta_mb,
              "run_name": run_name, "interrupted": interrupted, "checkpoint": out_dir / "last.pt"}
    if final_test and not interrupted:
        result["test_loss"], result["test_acc"] = evaluate(model, test_loader, device)
        print(f"[{run_name}] final test accuracy = {result['test_acc']:.4f}")
    if run is not None:
        mlflow.log_metric("peak_memory_mb", mem.peak_mb)
        mlflow.log_metric("memory_delta_mb", mem.delta_mb)
        if "test_acc" in result:
            mlflow.log_metrics({"test_acc": result["test_acc"], "test_loss": result["test_loss"]})
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        for f in ("history.csv", "config.yaml", "last.pt", "best.pt"):
            if (out_dir / f).exists():
                mlflow.log_artifact(str(out_dir / f))
        mlflow.end_run()
    return result



# --------------------------------------------------------------------------
# Isolated runs: a fresh process per run gives a clean peak-memory measurement
# --------------------------------------------------------------------------
def _child(cfg, synthetic, run_name, experiment, queue):
    os.chdir(HERE)
    try:
        r = run_training(cfg, get_datasets(cfg, synthetic), run_name, experiment)
        r["checkpoint"] = str(r["checkpoint"])
        queue.put(("ok", r))
    except Exception as e:  # report the failure to the parent instead of hanging
        queue.put(("error", repr(e)))


def run_isolated(cfg, run_name, experiment):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(cfg, SYNTHETIC, run_name, experiment, q))
    p.start()
    while p.is_alive() and q.empty():
        time.sleep(0.5)
    status, payload = q.get(timeout=30)
    p.join()
    if status != "ok":
        raise RuntimeError(f"run {run_name} failed: {payload}")
    return payload

# --------------------------------------------------------------------------
# Task 2: torch.profiler on one epoch
# --------------------------------------------------------------------------
def task_profile(cfg, datasets, res):
    from torch.profiler import ProfilerActivity, profile, record_function
    device = get_device(cfg)
    set_seed(cfg["seed"])
    model = MODELS[cfg["model"]["name"]]().to(device)
    opt = build_optimizer(model, cfg["train"])
    loader, _, _, _ = make_loaders(cfg, datasets, device)
    cuda = device.type == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    phases = ["data_loading", "forward", "backward", "optimizer_step"]
    wall = dict.fromkeys(phases, 0.0)

    model.train()
    for i, (x, y) in enumerate(loader):  # warm-up (not profiled)
        F.cross_entropy(model(x.to(device)), y.to(device)).backward()
        opt.zero_grad(set_to_none=True)
        if i == 4:
            break
    sync()

    it = iter(loader)
    with profile(activities=acts) as prof:
        while True:
            t0 = time.perf_counter()
            with record_function("data_loading"):
                try:
                    x, y = next(it)
                except StopIteration:
                    break
                x, y = x.to(device), y.to(device)
            sync(); t1 = time.perf_counter(); wall["data_loading"] += t1 - t0
            with record_function("forward"):
                loss = F.cross_entropy(model(x), y)
            sync(); t2 = time.perf_counter(); wall["forward"] += t2 - t1
            with record_function("backward"):
                loss.backward()
            sync(); t3 = time.perf_counter(); wall["backward"] += t3 - t2
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            sync(); wall["optimizer_step"] += time.perf_counter() - t3

    ev = {e.key: e for e in prof.key_averages()}
    prof_s = {p: ev[p].cpu_time_total / 1e6 if p in ev else float("nan") for p in phases}
    tot_p, tot_w = sum(prof_s.values()), sum(wall.values())
    df = pd.DataFrame({"phase": phases,
                       "profiler_seconds": [prof_s[p] for p in phases],
                       "profiler_share_pct": [100 * prof_s[p] / tot_p for p in phases],
                       "wallclock_seconds": [wall[p] for p in phases],
                       "wallclock_share_pct": [100 * wall[p] / tot_w for p in phases]})
    df = df.round(4)
    (res / "profile_top_ops.txt").write_text(
        prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20), encoding="utf-8")
    top = df.loc[df.profiler_share_pct.idxmax()]
    save_study(res, "profile_breakdown", df,
               [f"One epoch of {cfg['model']['name']}, batch {cfg['train']['batch_size']}, "
                f"{cfg['data']['num_workers']} workers on {device}.",
                f"Largest share: {top.phase} ({top.profiler_share_pct:.1f}% by torch.profiler).",
                "profiler_* = torch.profiler record_function ranges; wallclock_* = independent "
                "perf_counter cross-check. Top operators: results/profile_top_ops.txt."])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(phases, df["profiler_share_pct"], color=["#3b6fd8", "#2a9d6a", "#d8763b", "#8e5bd6"])
    ax.set_ylabel("Share of epoch time (%)")
    ax.set_title("torch.profiler: one training epoch")
    fig.tight_layout()
    fig.savefig(res / "profile_breakdown.png", dpi=150)
    plt.close(fig)
    if mlflow is not None:
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment("lab06-profile")
        with mlflow.start_run(run_name="profile_one_epoch"):
            mlflow.log_params({k: str(v)[:250] for k, v in flatten(cfg).items()})
            mlflow.log_metrics({f"share_{r.phase}_pct": r.profiler_share_pct for r in df.itertuples()})
            for f in ("profile_breakdown.csv", "profile_breakdown.png", "profile_top_ops.txt"):
                mlflow.log_artifact(str(res / f))


# --------------------------------------------------------------------------
# Task 3: num_workers
# --------------------------------------------------------------------------
def task_workers(cfg, datasets, res):
    rows = []
    for w in cfg["studies"]["workers"]:
        c = override(cfg, {"data.num_workers": w, "train.epochs": 2, "train.scheduler": "none"})
        r = run_training(c, datasets, f"workers_{w}", "lab06-workers", final_test=False)
        h = r["history"][1]  # epoch 2 = steady state; epoch 1 includes worker start-up
        rows.append({"num_workers": w, "epoch1_s": r["history"][0]["epoch_time_s"],
                     "epoch_s": h["epoch_time_s"], "samples_per_s": h["samples_per_s"],
                     "data_wait_s": h["data_wait_s"],
                     "compute_s": h["epoch_time_s"] - h["data_wait_s"],
                     "data_wait_pct": 100 * h["data_wait_s"] / h["epoch_time_s"]})
    df = pd.DataFrame(rows).round(3)
    base, best = df.iloc[0], df.loc[df.samples_per_s.idxmax()]
    if base.data_wait_pct > 20:
        verdict = (f"With 0 workers the loop waits for data {base.data_wait_pct:.0f}% of the time, "
                   "so data loading is the bottleneck there.")
    else:
        verdict = (f"With 0 workers only {base.data_wait_pct:.0f}% of the epoch is spent waiting "
                   "for data, so the model compute is the bottleneck, not data loading.")
    save_study(res, "workers_study", df,
               [verdict, f"Best throughput: {int(best.num_workers)} workers, "
                f"{best.samples_per_s:.0f} samples/s ({best.samples_per_s / base.samples_per_s:.2f}x "
                "the 0-worker speed).",
                "More workers than CPU cores, or more than the model can consume, only add "
                "process start-up and inter-process copy overhead (see epoch1_s)."])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(df))
    ax[0].plot(df.num_workers, df.samples_per_s, "o-")
    ax[0].set_xlabel("num_workers"); ax[0].set_ylabel("Training throughput (samples/s)")
    ax[1].bar(x, df.compute_s, label="compute")
    ax[1].bar(x, df.data_wait_s, bottom=df.compute_s, label="waiting for data")
    ax[1].set_xticks(x); ax[1].set_xticklabels(df.num_workers)
    ax[1].set_xlabel("num_workers"); ax[1].set_ylabel("Epoch time (s)"); ax[1].legend()
    fig.suptitle("DataLoader workers study")
    fig.tight_layout()
    fig.savefig(res / "workers_study.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Task 4: batch size
# --------------------------------------------------------------------------
def task_batch(cfg, datasets, res):
    rows = []
    for bs in cfg["studies"]["batch_sizes"]:
        c = override(cfg, {"train.batch_size": bs, "train.epochs": cfg["studies"]["batch_epochs"]})
        r = run_isolated(c, f"batch_{bs}", "lab06-batchsize")
        h = r["history"]
        rows.append({"batch_size": bs,
                     "samples_per_s": float(np.median([x["samples_per_s"] for x in h])),
                     "peak_mem_mb": r["peak_mem_mb"], "mem_delta_mb": r["mem_delta_mb"],
                     "final_train_loss": h[-1]["train_loss"], "final_val_acc": h[-1]["val_acc"],
                     "test_acc": r["test_acc"]})
    df = pd.DataFrame(rows).round(4)
    save_study(res, "batch_size_study", df,
               [f"Throughput {df.samples_per_s.iloc[0]:.0f} -> {df.samples_per_s.iloc[-1]:.0f} "
                f"samples/s from batch {df.batch_size.iloc[0]} to {df.batch_size.iloc[-1]}.",
                "The learning rate is the same for every batch size, so accuracy differences mix "
                "the effect of batch size with fewer optimizer steps per epoch.",
                "peak_mem_mb = CUDA allocator peak on GPU, or peak process RSS on CPU; "
                "mem_delta_mb = peak minus the value at the start of the run."])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, col, lab in zip(ax, ["samples_per_s", "mem_delta_mb", "test_acc"],
                           ["Throughput (samples/s)", "Peak memory above start (MB)",
                            "Test accuracy after training"]):
        a.plot(df.batch_size, df[col], "o-"); a.set_xscale("log", base=2)
        a.set_xlabel("Batch size"); a.set_ylabel(lab); a.grid(alpha=.3)
    fig.suptitle("Batch-size study")
    fig.tight_layout()
    fig.savefig(res / "batch_size_study.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Task 5: schedulers and optimizers (10-epoch budgets)
# --------------------------------------------------------------------------
def task_sched(cfg, datasets, res):
    E, s = cfg["studies"]["scheduler_epochs"], cfg["studies"]
    base = {"train.epochs": E, "train.scheduler_params.step.step_size": max(1, E // 3),
            "train.scheduler_params.cosine.T_max": E}
    runs = {"adam + constant lr": {"train.optimizer": "adam", "train.scheduler": "none"},
            "adam + StepLR": {"train.optimizer": "adam", "train.scheduler": "step"},
            "adam + CosineAnnealingLR": {"train.optimizer": "adam", "train.scheduler": "cosine"},
            "adam + ReduceLROnPlateau": {"train.optimizer": "adam", "train.scheduler": "plateau"},
            "sgd+momentum + CosineAnnealingLR": {"train.optimizer": "sgd", "train.lr": s["sgd_lr"],
                                                 "train.scheduler": "cosine"}}
    rows, hist = [], {}
    for name, ch in runs.items():
        r = run_training(override(cfg, {**base, **ch}), datasets,
                         "sched_" + name.replace(" ", "").replace("+", "_"), "lab06-schedulers")
        h = r["history"]
        hist[name] = h
        rows.append({"run": name, "final_lr": h[-1]["lr"], "best_val_acc": max(x["val_acc"] for x in h),
                     "final_val_acc": h[-1]["val_acc"], "test_acc": r["test_acc"],
                     "final_train_loss": h[-1]["train_loss"],
                     "total_time_s": sum(x["epoch_time_s"] for x in h)})
    df = pd.DataFrame(rows).round(4)
    best = df.loc[df.test_acc.idxmax()]
    adam, sgd = df[df.run == "adam + CosineAnnealingLR"].iloc[0], df[df.run.str.startswith("sgd")].iloc[0]
    save_study(res, "scheduler_optimizer_study", df,
               [f"Best test accuracy in a {E}-epoch budget: {best.run} ({best.test_acc:.4f}).",
                f"Adam vs SGD+momentum, both with cosine decay: {adam.test_acc:.4f} vs {sgd.test_acc:.4f}.",
                f"Adam lr {cfg['train']['lr']}, SGD lr {s['sgd_lr']} (momentum {cfg['train']['momentum']}); "
                "differences of a few tenths of a percent from one seed are within noise."])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in hist.items():
        ax[0].plot([x["epoch"] for x in h], [x["val_acc"] for x in h], marker="o", label=name)
        ax[1].plot([x["epoch"] for x in h], [x["lr"] for x in h], marker="o", label=name)
    ax[0].set_ylabel("Validation accuracy"); ax[1].set_ylabel("Learning rate")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("Epoch"); a.grid(alpha=.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f"Schedulers and optimizers ({E} epochs)")
    fig.tight_layout()
    fig.savefig(res / "scheduler_optimizer_study.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Task 6: gradient accumulation
# --------------------------------------------------------------------------
def task_accum(cfg, datasets, res):
    a = cfg["studies"]["accum"]
    common = {"train.epochs": a["epochs"], "train.scheduler": "none"}
    runs = {f"batch {a['direct_batch']} (direct)": {"train.batch_size": a["direct_batch"],
                                                    "train.accumulation_steps": 1},
            f"batch {a['micro_batch']} x accumulate {a['steps']}": {"train.batch_size": a["micro_batch"],
                                                                    "train.accumulation_steps": a["steps"]},
            f"batch {a['micro_batch']} (no accumulation)": {"train.batch_size": a["micro_batch"],
                                                            "train.accumulation_steps": 1}}
    hist, rows = {}, []
    for name, ch in runs.items():
        r = run_isolated(override(cfg, {**common, **ch}),
                         "accum_" + name.replace(" ", "_").replace("(", "").replace(")", ""),
                         "lab06-accumulation")
        hist[name] = r["history"]
        rows.append({"run": name, "final_train_loss": r["history"][-1]["train_loss"],
                     "final_val_acc": r["history"][-1]["val_acc"], "test_acc": r["test_acc"],
                     "peak_mem_mb": r["peak_mem_mb"]})
    names = list(runs)
    d_loss = max(abs(x["train_loss"] - y["train_loss"]) for x, y in zip(hist[names[0]], hist[names[1]]))
    d_val = max(abs(x["val_acc"] - y["val_acc"]) for x, y in zip(hist[names[0]], hist[names[1]]))
    df = pd.DataFrame(rows).round(4)
    save_study(res, "grad_accumulation_check", df,
               [f"Per-epoch curves, direct batch {a['direct_batch']} vs {a['micro_batch']}x{a['steps']}: "
                f"max |train-loss difference| = {d_loss:.2e}, max |val-accuracy difference| = {d_val:.4f}.",
                "Same seed, same data order, and each micro-batch weighted by its share of the group, "
                "so the two runs match up to floating-point rounding.",
                "Accumulation trades speed for memory: the micro-batch needs less activation memory."])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in hist.items():
        ax[0].plot([x["epoch"] for x in h], [x["train_loss"] for x in h], marker="o", label=name)
        ax[1].plot([x["epoch"] for x in h], [x["val_acc"] for x in h], marker="o", label=name)
    ax[0].set_ylabel("Train loss"); ax[1].set_ylabel("Validation accuracy")
    for x in ax:
        x.set_xlabel("Epoch"); x.grid(alpha=.3)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "grad_accumulation_check.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Task 7: interrupt and resume
# --------------------------------------------------------------------------
def task_resume(cfg, datasets, res):
    r_cfg = cfg["studies"]["resume"]
    c = override(cfg, {"train.epochs": r_cfg["total_epochs"], "train.optimizer": "adam",
                       "train.scheduler": "cosine",
                       "train.scheduler_params.cosine.T_max": r_cfg["total_epochs"]})
    ref = run_training(c, datasets, "resume_reference", "lab06-resume")
    part = run_training(c, datasets, "resume_interrupted", "lab06-resume",
                        stop_after=r_cfg["interrupt_at"], final_test=False)
    res2 = run_training(c, datasets, "resume_continued", "lab06-resume", resume=part["checkpoint"])
    joined = res2["history"]  # checkpoint history (epochs 1..k) + the continued epochs
    rows = []
    for a, b in zip(ref["history"], joined):
        rows.append({"epoch": a["epoch"], "ref_train_loss": a["train_loss"],
                     "resumed_train_loss": b["train_loss"], "ref_val_acc": a["val_acc"],
                     "resumed_val_acc": b["val_acc"],
                     "abs_loss_diff": abs(a["train_loss"] - b["train_loss"]),
                     "abs_acc_diff": abs(a["val_acc"] - b["val_acc"])})
    df = pd.DataFrame(rows)
    k = r_cfg["interrupt_at"]
    after = df[df.epoch > k]
    df = df.round(6)
    save_study(res, "resume_check", df,
               [f"Training was interrupted after epoch {k} and resumed from checkpoint {part['checkpoint'].name}.",
                f"After resuming (epochs {k + 1}-{r_cfg['total_epochs']}): max |train-loss difference| vs the "
                f"uninterrupted run = {after.abs_loss_diff.max():.2e}, max |val-accuracy difference| = "
                f"{after.abs_acc_diff.max():.4f}.",
                "The checkpoint stores model, optimizer, scheduler and all RNG states (including the data "
                "shuffling generator), so the curve continues as if nothing happened. On a GPU, tiny "
                "differences from non-deterministic kernels are normal."])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ep = [h["epoch"] for h in ref["history"]]
    ax[0].plot(ep, [h["train_loss"] for h in ref["history"]], "o-", label="uninterrupted")
    ax[0].plot(ep, [h["train_loss"] for h in joined], "x--", label="interrupted + resumed")
    ax[1].plot(ep, [h["val_acc"] for h in ref["history"]], "o-", label="uninterrupted")
    ax[1].plot(ep, [h["val_acc"] for h in joined], "x--", label="interrupted + resumed")
    for a, lab in zip(ax, ["Train loss", "Validation accuracy"]):
        a.axvline(k, color="grey", ls=":"); a.set_xlabel("Epoch"); a.set_ylabel(lab); a.grid(alpha=.3)
    ax[0].legend()
    fig.suptitle(f"Interrupt after epoch {k} and resume")
    fig.tight_layout()
    fig.savefig(res / "resume_check.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
def write_summary(res):
    parts = ["# Lab 6 results summary\n"]
    for name in ("profile_breakdown", "workers_study", "batch_size_study",
                 "scheduler_optimizer_study", "grad_accumulation_check", "resume_check"):
        f = res / f"{name}.md"
        if f.exists():
            parts.append(f.read_text(encoding="utf-8"))
    (res / "summary.md").write_text("\n".join(parts), encoding="utf-8")
    print(f"\nSummary written to {res / 'summary.md'}")


TASKS = ["train", "profile", "workers", "batch", "sched", "accum", "resume"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--task", default="all", choices=["all"] + TASKS + ["summary"])
    ap.add_argument("--quick", action="store_true", help="small subset and few epochs")
    ap.add_argument("--synthetic", action="store_true", help="random data (code test only)")
    ap.add_argument("--resume", default=None, help="checkpoint path for --task train")
    args = ap.parse_args()
    os.chdir(HERE)  # relative paths (data, results, mlflow.db) resolve next to this script
    cfg = load_config(args.config)
    if args.quick:
        cfg = apply_quick(cfg)
    res = HERE / cfg["output_dir"]
    res.mkdir(exist_ok=True)
    if mlflow is None:
        print("WARNING: mlflow is not installed (pip install mlflow); runs will NOT be logged.")
    global SYNTHETIC
    SYNTHETIC = args.synthetic
    if args.synthetic:
        print("WARNING: SYNTHETIC random data, results are meaningless (code test only).")
    print(f"device={get_device(cfg)} model={cfg['model']['name']} torch={torch.__version__}")

    if args.task == "summary":
        write_summary(res)
        return
    datasets = get_datasets(cfg, args.synthetic)
    todo = TASKS if args.task == "all" else [args.task]
    fns = {"profile": task_profile, "workers": task_workers, "batch": task_batch,
           "sched": task_sched, "accum": task_accum, "resume": task_resume}
    for task in todo:
        print(f"\n######## task: {task} ########")
        if task == "train":
            r = run_training(cfg, datasets, "train_main", resume=args.resume)
            pd.DataFrame(r["history"]).round(5).to_csv(res / "train_main_history.csv", index=False)
        else:
            fns[task](cfg, datasets, res)
    write_summary(res)


if __name__ == "__main__":
    main()
