import argparse
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import ltc_dataset
from core import MA_RAFT


MAX_FLOW = 1000
SUM_FREQ = 100
VAL_FREQ = 1000


class Logger:
    def __init__(self, optimizer, log_dir: Path):
        self.optimizer = optimizer
        self.total_steps = 0
        self.running_loss = {}
        log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(log_dir / time.strftime("%Y%m%d-%H%M%S")))

    def _print_training_status(self):
        metric_names = sorted(self.running_loss.keys())
        metrics_data = [self.running_loss[k] / SUM_FREQ for k in metric_names]
        current_lr = self.optimizer.param_groups[0]["lr"]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps, current_lr)
        metrics_str = ", ".join(f"{name}: {value:.4f}" for name, value in zip(metric_names, metrics_data))
        print(training_str + metrics_str)

        for key, value in self.running_loss.items():
            self.writer.add_scalar(f"train/{key}", value / SUM_FREQ, self.total_steps)
        self.running_loss = {}

    def push(self, metrics):
        for key, value in metrics.items():
            self.running_loss[key] = self.running_loss.get(key, 0.0) + value
        if self.total_steps > 0 and self.total_steps % SUM_FREQ == 0:
            self._print_training_status()

    def step(self):
        self.total_steps += 1

    def write_dict(self, results):
        for key, value in results.items():
            self.writer.add_scalar(f"validation/{key}", value, self.total_steps)

    def close(self):
        self.writer.close()


def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        weight = gamma ** (n_predictions - i - 1)
        loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += weight * (valid[:, None] * loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    metrics = {
        "loss": flow_loss.item(),
        "epe": epe.mean().item() if epe.numel() > 0 else 0.0,
        "1px": (epe < 1).float().mean().item() if epe.numel() > 0 else 0.0,
        "3px": (epe < 3).float().mean().item() if epe.numel() > 0 else 0.0,
        "5px": (epe < 5).float().mean().item() if epe.numel() > 0 else 0.0,
    }
    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    warmup_iters = min(args.warmup_steps, max(args.num_steps - 1, 1))
    cosine_iters = max(args.num_steps - warmup_iters, 1)
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_iters,
    )
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_iters,
        eta_min=args.min_lr,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_iters],
    )
    return optimizer, scheduler


def load_weights_intelligently(model, checkpoint_path, device):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = OrderedDict((k.removeprefix("module."), v) for k, v in state_dict.items())

    model_state = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and value.shape == model_state[key].shape:
            matched[key] = value
        else:
            skipped.append(key)

    model_state.update(matched)
    model.load_state_dict(model_state, strict=True)

    print(f"Loaded {len(matched)} / {len(model_state)} tensors.")
    if skipped:
        print(f"Skipped {len(skipped)} unmatched tensors.")
        for key in skipped[:20]:
            print(f"  - {key}")
    return model


@torch.no_grad()
def validate(model, args, logger, device):
    model.eval()
    results = {}
    val_loader = ltc_dataset.fetch_custom_dataloader(args, split=args.val_split)

    for data_blob in val_loader:
        image1, image2, flow_gt, valid_gt = [x.to(device, non_blocking=True) for x in data_blob]
        flow_predictions = model(image1, image2, iters=args.val_iters)
        _, val_metrics = sequence_loss(flow_predictions, flow_gt, valid_gt, args.gamma)
        for key, value in val_metrics.items():
            results[key] = results.get(key, 0.0) + value

    if len(val_loader) > 0:
        avg_results = {key: value / len(val_loader) for key, value in results.items()}
        print("\n--- Validation Results ---")
        for key, value in avg_results.items():
            print(f"{key.upper():>6}: {value:.4f}")
        logger.write_dict(avg_results)

    model.train()
    model.freeze_bn()


def build_model(args, device):
    model = MA_RAFT(args).to(device)
    dummy = torch.zeros(1, 3, args.image_size[0], args.image_size[1], device=device)
    with torch.no_grad():
        model(dummy, dummy, iters=1)

    if args.restore_ckpt:
        model = load_weights_intelligently(model, Path(args.restore_ckpt), device)

    return model


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.mixed_precision = bool(args.mixed_precision and device.type == "cuda")

    model = build_model(args, device)
    print(f"Created MA_RAFT with {count_parameters(model):,} trainable parameters.")
    model.train()
    model.freeze_bn()

    train_loader = ltc_dataset.fetch_custom_dataloader(args, split=args.train_split)
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=args.mixed_precision)

    run_root = Path(args.output_dir)
    checkpoint_dir = run_root / "checkpoints"
    log_dir = run_root / "runs" / args.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(optimizer, log_dir=log_dir)

    should_keep_training = True
    while should_keep_training:
        for data_blob in train_loader:
            image1, image2, flow, valid = [x.to(device, non_blocking=True) for x in data_blob]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.mixed_precision):
                flow_predictions = model(image1, image2, iters=args.iters)
                loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            logger.push(metrics)
            logger.step()

            if logger.total_steps % VAL_FREQ == 0 and logger.total_steps > 0:
                save_path = checkpoint_dir / f"{logger.total_steps}_{args.name}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"\nSaved checkpoint: {save_path}")
                if args.validate:
                    validate(model, args, logger, device)

            if logger.total_steps >= args.num_steps:
                should_keep_training = False
                break

    logger.close()
    final_path = checkpoint_dir / f"{args.name}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final checkpoint: {final_path}")
    return final_path


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="MA_RAFT training script")
    parser.add_argument("--name", default="ma_raft", help="Experiment name.")
    parser.add_argument("--path", type=Path, required=True, help="Dataset root containing train/test folders.")
    parser.add_argument("--output-dir", type=Path, default=project_dir, help="Directory for runs and checkpoints.")
    parser.add_argument("--restore-ckpt", default=None, help="Optional checkpoint path.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--validate", action="store_true", help="Run validation whenever a checkpoint is saved.")
    parser.add_argument("--num-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--val-iters", type=int, default=12)
    parser.add_argument("--wdecay", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(1234)
    np.random.seed(1234)
    train(parse_args())
