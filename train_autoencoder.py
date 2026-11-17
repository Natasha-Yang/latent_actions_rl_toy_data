"""Train an action autoencoder on expert demos, logging to wandb.

    python train_autoencoder.py                          # cfgs/train_autoencoder.yaml
    python train_autoencoder.py --config cfgs/train_autoencoder.yaml

All keys in REQUIRED must be in the config; see cfgs/train_autoencoder.yaml for what each one means.
One task -> single-task autoencoder; more -> one multitask autoencoder trained on all of them.
Only rows of the dataset whose task is in `tasks` are used. Actions are in [-1, 1],
matching the decoders' tanh output.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader, TensorDataset

import model.autoencoder as autoencoders

ROOT = os.path.dirname(os.path.abspath(__file__))

REQUIRED = ("model", "model_kwargs", "num_epochs", "lr", "batch_size", "val_frac", "kl_weight", "l1_weight",
            "seed",
            "checkpoint_dir", "dataset", "tasks", "wandb")
REQUIRED_WANDB = ("project", "entity", "mode")


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED if k not in cfg]
    missing += [f"wandb.{k}" for k in REQUIRED_WANDB if k not in (cfg.get("wandb") or {})]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    cfg["model_kwargs"] = cfg["model_kwargs"] or {}
    cfg["tasks"] = [str(t).upper() for t in cfg["tasks"]]  # YAML reads `8` as an int; dataset stores upper case
    if not hasattr(autoencoders, cfg["model"]):
        raise ValueError(f"model {cfg['model']!r} not found in model/autoencoder.py")
    return cfg


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_data(cfg):
    """Return actions (in [-1, 1]), task index per row and episode id per row, for the configured tasks only."""
    d = np.load(resolve(cfg["dataset"]))
    task = d["task"]
    available = set(np.unique(task).tolist())
    unknown = [t for t in cfg["tasks"] if t not in available]
    if unknown:
        raise ValueError(f"tasks {unknown} not in {cfg['dataset']} (has {sorted(available)})")

    keep = np.isin(task, cfg["tasks"])
    actions = d["action"][keep].astype(np.float32)
    task_idx = np.array([cfg["tasks"].index(t) for t in task[keep]], dtype=np.int64)
    return actions, task_idx, d["episode"][keep]


def split_by_episode(episode, val_frac, rng):
    """Hold out whole episodes so consecutive, near-identical steps don't leak into validation."""
    eps = np.unique(episode)
    n_val = int(round(val_frac * len(eps)))
    if val_frac > 0:
        n_val = min(max(n_val, 1), len(eps) - 1)
    val_eps = rng.choice(eps, size=n_val, replace=False)
    val = np.isin(episode, val_eps)
    return ~val, val


def compute_loss(net, x, cfg):
    """Total loss to minimise, per-sample reconstruction MSE, and a dict of per-sample regulariser terms
    ({"kl": ...} for VariationalAutoencoder, {"l1": ..., "l0": ...} for SparseAutoencoder, {} otherwise)."""
    out = net(x)
    terms = {}
    if isinstance(net, autoencoders.VariationalAutoencoder):  # (recon, mu, logvar)
        recon, mu, logvar = out
        terms["kl"] = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    elif isinstance(net, autoencoders.SparseAutoencoder):  # (recon, z)
        recon, z = out
        terms["l1"] = z.abs().sum(1)
        terms["l0"] = (z > 0).float().sum(1)  # active latents per sample; logged only, not part of the loss
    else:
        recon = out
    mse = F.mse_loss(recon, x, reduction="none").mean(1)
    # Both penalties are summed over latent dims, then divided by action_dim to match the per-dim MSE.
    # KL: kl_weight = 1 is the ELBO for a decoder noise variance of 0.5, far above the data's per-dim variance
    # (~0.01), so the optimum is posterior collapse; use kl_weight ~0.01 or below.
    weights = {"kl": cfg["kl_weight"], "l1": cfg["l1_weight"]}
    loss = mse.mean() + sum(w * terms[k].mean() for k, w in weights.items() if k in terms) / x.shape[1]
    return loss, mse, terms


@torch.no_grad()
def evaluate(net, loader, cfg, device):
    net.eval()
    n_tasks = len(cfg["tasks"])
    sums = {"loss": 0.0, "mse": 0.0}
    task_mse, task_n, n = torch.zeros(n_tasks), torch.zeros(n_tasks), 0
    for x, t in loader:
        x, t = x.to(device), t.to(device)
        loss, mse, terms = compute_loss(net, x, cfg)
        sums["loss"] += loss.item() * len(x)
        sums["mse"] += mse.sum().item()
        for k, v in terms.items():
            sums[k] = sums.get(k, 0.0) + v.sum().item()
        task_mse += torch.zeros(n_tasks, device=device).index_add_(0, t, mse).cpu()
        task_n += torch.bincount(t, minlength=n_tasks).cpu().float()
        n += len(x)
    stats = {k: v / n for k, v in sums.items()}
    if n_tasks > 1:
        for i, name in enumerate(cfg["tasks"]):
            if task_n[i] > 0:
                stats[f"mse/{name}"] = (task_mse[i] / task_n[i]).item()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "cfgs", "train_autoencoder.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actions, task_idx, episode = load_data(cfg)
    train, val = split_by_episode(episode, cfg["val_frac"], rng)
    multitask = len(cfg["tasks"]) > 1
    print(f"{'multitask' if multitask else 'single-task'} {cfg['model']} on {cfg['tasks']}: "
          f"{train.sum()} train / {val.sum()} val steps from {len(np.unique(episode))} episodes, "
          f"action_dim {actions.shape[1]}, device {device}")

    def loader(mask, shuffle):
        ds = TensorDataset(torch.from_numpy(actions[mask]), torch.from_numpy(task_idx[mask]))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle)

    train_loader = loader(train, shuffle=True)
    val_loader = loader(val, shuffle=False) if val.any() else None

    model_kwargs = {"action_dim": actions.shape[1], **cfg["model_kwargs"]}
    net = getattr(autoencoders, cfg["model"])(**model_kwargs).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    run_name = f"{cfg['model']}_{'-'.join(cfg['tasks'])}"
    run = wandb.init(
        project=cfg["wandb"]["project"], entity=cfg["wandb"]["entity"], mode=cfg["wandb"]["mode"],
        name=run_name, config={**cfg, "multitask": multitask, "action_dim": actions.shape[1],
                               "n_train": int(train.sum()), "n_val": int(val.sum())},
        tags=["multitask" if multitask else "single-task", *cfg["tasks"]],
    )
    ckpt_dir = os.path.join(resolve(cfg["checkpoint_dir"]), f"{run_name}_{run.id}")
    os.makedirs(ckpt_dir, exist_ok=True)

    def save(path, epoch, metrics):
        torch.save({"model": cfg["model"], "model_kwargs": model_kwargs, "state_dict": net.state_dict(),
                    "config": cfg, "epoch": epoch, "metrics": metrics}, path)

    best_val, step = float("inf"), 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        net.train()
        for x, _ in train_loader:
            x = x.to(device)
            loss, mse, terms = compute_loss(net, x, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if hasattr(net, "normalize_decoder"):  # SparseAutoencoder: keep decoder columns at unit norm
                net.normalize_decoder()
            step += 1
            log = {"train/batch_loss": loss.item(), "train/batch_mse": mse.mean().item()}
            log.update({f"train/batch_{k}": v.mean().item() for k, v in terms.items()})
            wandb.log(log, step=step)

        log = {"epoch": epoch, **{f"train/{k}": v for k, v in evaluate(net, train_loader, cfg, device).items()}}
        if val_loader is not None:
            log.update({f"val/{k}": v for k, v in evaluate(net, val_loader, cfg, device).items()})
        wandb.log(log, step=step)
        print(f"epoch {epoch:3d}  " + "  ".join(f"{k} {v:.5f}" for k, v in log.items()
                                                 if k.endswith(("/loss", "/mse", "/kl", "/l1", "/l0"))))

        score = log.get("val/loss", log["train/loss"])
        if score < best_val:
            best_val = score
            save(os.path.join(ckpt_dir, "best.pt"), epoch, log)
            run.summary["best_val_loss"], run.summary["best_epoch"] = best_val, epoch

    save(os.path.join(ckpt_dir, "last.pt"), cfg["num_epochs"], log)
    print(f"checkpoints -> {ckpt_dir}")
    run.finish()


if __name__ == "__main__":
    main()
