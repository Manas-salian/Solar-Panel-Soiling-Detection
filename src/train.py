"""
Two-stage transfer-learning routine.

Stage 1 (warm-up)  : backbone frozen, train only the new classification head.
Stage 2 (fine-tune): everything trainable, discriminative learning rates
                     (backbone lower than head), 1-epoch linear warm-up then
                     cosine decay, label smoothing, class-weighted loss,
                     bf16 autocast, gradient clipping, early stopping on
                     validation macro-F1.
"""

import copy
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.config import (
    BACKBONE_LR, HEAD_LR, LABEL_SMOOTHING, NUM_CLASSES, WEIGHT_DECAY,
)
from src.data import class_weights_from_counts, seed_everything
from src.evaluate import compute_metrics, plot_history, predict_loader
from src.model import (
    backbone_parameters, build_model, head_parameters, set_backbone_frozen, set_train_mode,
)


def _run_epoch(model, loader, criterion, optimizer, device, use_amp, scheduler=None,
               backbone_frozen=False, max_grad_norm=1.0):
    set_train_mode(model, backbone_frozen)
    total_loss, correct, seen = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.float(), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        seen += x.size(0)
    return total_loss / seen, correct / seen


def _warmup_cosine(total_steps: int, warmup_steps: int, floor: float = 0.02):
    def fn(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def _evaluate(model, loader, device, use_amp, class_names):
    probs, labels = predict_loader(model, loader, device, use_amp)
    metrics = compute_metrics(probs, labels, class_names)
    ce = nn.CrossEntropyLoss()
    loss = ce(torch.log(torch.tensor(probs).clamp_min(1e-8)), torch.tensor(labels)).item()
    return loss, metrics


def train_backbone(backbone: str, loaders: dict, train_counts, class_names, *,
                   epochs: int, warmup_epochs: int, patience: int, seed: int,
                   device: torch.device, run_dir: Path, verbose: bool = True) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    use_amp = device.type == "cuda"

    model = build_model(backbone, NUM_CLASSES, pretrained=True).to(device)
    class_w = class_weights_from_counts(train_counts).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w, label_smoothing=LABEL_SMOOTHING)

    history = defaultdict(list)
    best = {"f1": -1.0, "epoch": 0, "state": None, "val_metrics": None, "bad_epochs": 0}
    t_start = time.time()

    def log_and_track(stage, ep, n_ep, tr_loss, tr_acc, va_loss, va):
        """Record the epoch and keep the best-so-far weights (by val macro-F1) across both stages."""
        history["stage"].append(stage)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va["accuracy"])
        history["val_macro_f1"].append(va["macro_f1"])
        if verbose:
            print(f"  [{backbone}] {stage:9s} {ep:>2d}/{n_ep}  "
                  f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va['accuracy']:.3f} macroF1 {va['macro_f1']:.3f}",
                  flush=True)
        if va["macro_f1"] > best["f1"] + 1e-4:
            best.update(f1=va["macro_f1"], epoch=len(history["train_loss"]), val_metrics=va, bad_epochs=0,
                        state=copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}))
        else:
            best["bad_epochs"] += 1

    # ---- Stage 1: head only ------------------------------------------------ #
    set_backbone_frozen(model, True)
    opt = AdamW(head_parameters(model), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
    for ep in range(1, warmup_epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, opt, device, use_amp,
                                     backbone_frozen=True)
        va_loss, va = _evaluate(model, loaders["val"], device, use_amp, class_names)
        log_and_track("warm-up", ep, warmup_epochs, tr_loss, tr_acc, va_loss, va)

    # ---- Stage 2: full fine-tune ------------------------------------------- #
    set_backbone_frozen(model, False)
    opt = AdamW([
        {"params": backbone_parameters(model), "lr": BACKBONE_LR},
        {"params": head_parameters(model), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(loaders["train"])
    sched = LambdaLR(opt, _warmup_cosine(epochs * steps_per_epoch, steps_per_epoch))

    best["bad_epochs"] = 0   # patience counts fine-tune epochs only
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, opt, device, use_amp, sched)
        va_loss, va = _evaluate(model, loaders["val"], device, use_amp, class_names)
        log_and_track("fine-tune", ep, epochs, tr_loss, tr_acc, va_loss, va)
        if best["bad_epochs"] >= patience:
            if verbose:
                print(f"  [{backbone}] early stop: no val macro-F1 gain for {patience} epochs")
            break

    best_f1, best_epoch, best_state, best_val_metrics = best["f1"], best["epoch"], best["state"], best["val_metrics"]

    train_time = time.time() - t_start
    model.load_state_dict(best_state)

    ckpt_path = run_dir / "best_model.pt"
    torch.save({"backbone": backbone, "class_names": list(class_names), "state_dict": best_state}, ckpt_path)
    plot_history(history, run_dir / "training_curves.png",
                 title=f"{backbone}: training history", warmup_epochs=warmup_epochs, best_epoch=best_epoch)

    return {
        "backbone": backbone,
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "best_val_macro_f1": best_f1,
        "best_val_metrics": best_val_metrics,
        "history": dict(history),
        "train_time_sec": train_time,
        "checkpoint": str(ckpt_path),
    }
