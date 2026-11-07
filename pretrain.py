"""Pretrain the MENDR encoder on the TUH EEG Corpus (preprocessed cache, see preprocess_tuh.py).

  python pretrain.py --cache_dir cache/tuh_200hz --checkpoint_dir checkpoints/mendr_pretrain

Defaults are the settings of the released encoder (weights/pretrained.pth): 30 epochs, batch 128, AdamW lr 5e-4 and
weight decay 0.05, 3 warm-up epochs then cosine decay, 10 random 6 s epochs per recording per epoch, seed 42, loss =
spatial + 0.5 * Frechet + 0.1 * MSR + 1.0 * filter diversity. The encoder of epoch k is saved as encoder_ep{k}.pth
(the released encoder, pretrained.pth, is the last epoch, encoder_ep29.pth); latest.pth allows resuming with --resume.
"""
import argparse
import glob
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mendr.pretraining import MENDRPretrainer
from mendr.train import configure_precision


class CachedTUHDataset(IterableDataset):
    """Streams random epochs from the cache: every recording file (.npz with 'epochs' [N, 19, 1200]) yields up to
    `epochs_per_file` epochs drawn without replacement; files are shuffled and split across DataLoader workers."""

    def __init__(self, cache_dir, epochs_per_file=10, epoch_samples=1200, max_files=0):
        self.epoch_samples = epoch_samples
        self.epochs_per_file = epochs_per_file
        self.npz_files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.npz"), recursive=True))
        if max_files and max_files > 0:
            self.npz_files = self.npz_files[:max_files]
        print(f"  {len(self.npz_files):,} cached recordings in {cache_dir}", flush=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        files = self.npz_files
        if worker_info is not None:
            per_worker = len(files) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(files)
            files = files[start:end]
        random.shuffle(files)
        for fpath in files:
            try:
                epochs = np.load(fpath)['epochs']
                if len(epochs) == 0:
                    continue
                for idx in np.random.choice(len(epochs), min(self.epochs_per_file, len(epochs)), replace=False):
                    ep = epochs[idx]
                    if ep.shape[-1] != self.epoch_samples:
                        continue
                    yield torch.from_numpy(np.ascontiguousarray(ep, dtype=np.float32))
            except Exception:
                continue


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--frechet_weight", type=float, default=0.5)
    p.add_argument("--temporal_weight", type=float, default=0.1)
    p.add_argument("--diversity_weight", type=float, default=1.0)
    p.add_argument("--epochs_per_file", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_files", type=int, default=0, help="use only the first N cached recordings (quick tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tpx_spans", type=int, default=4)
    p.add_argument("--tpx_span_len", type=int, default=32)
    p.add_argument("--tpx_kernel", type=int, default=15)
    args = p.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision()
    print(f"Device: {device}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = MENDRPretrainer(d_spatial=32, n_filters=4, sfreq=200, window_samples=64, stride_samples=32, d_model=128,
                            n_heads=4, n_transformer_layers=4, d_temporal=64, temporal_kernel=15, n_temporal_layers=2,
                            mask_ratio=0.5, dropout=0.1, ca_d_key=64, ca_n_heads=4, tpx_spans=args.tpx_spans,
                            tpx_span_len=args.tpx_span_len, tpx_kernel=args.tpx_kernel).to(device)
    print(f"  {sum(p_.numel() for p_ in model.parameters()):,} parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < 3:
            return (epoch + 1) / 3
        return 0.5 * (1 + np.cos(np.pi * (epoch - 3) / max(1, args.epochs - 3)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    start_epoch, best_loss = 0, float('inf')
    ckpt_path = os.path.join(args.checkpoint_dir, "latest.pth")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch, best_loss = ckpt['epoch'] + 1, ckpt['best_loss']
        print(f"  resumed from epoch {start_epoch}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        # each epoch draws the same random stream whether or not the run was resumed
        torch.manual_seed(args.seed + 1000 * (epoch + 1)); np.random.seed(args.seed + 1000 * (epoch + 1))
        model.tpx_reseed(args.seed + 1000 * (epoch + 1) + 17)
        loader = DataLoader(CachedTUHDataset(args.cache_dir, args.epochs_per_file, max_files=args.max_files),
                            batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                            collate_fn=lambda batch: torch.stack(batch))
        model.train()
        sums, n_batches, n_nan = np.zeros(3), 0, 0
        for batch_idx, x in enumerate(loader):
            x = x.to(device)
            optimizer.zero_grad()
            spatial, frechet, temporal, diversity = model(x)
            loss = spatial + args.frechet_weight * frechet + args.temporal_weight * temporal + args.diversity_weight * diversity
            if torch.isnan(loss):
                n_nan += 1
                continue
            loss.backward()
            if any(p_.grad is not None and torch.isnan(p_.grad).any() for p_ in model.parameters()):
                optimizer.zero_grad()
                n_nan += 1
                continue
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums += [spatial.item(), frechet.item(), temporal.item()]
            n_batches += 1
            if batch_idx > 0 and batch_idx % 500 == 0:
                s, f, t = sums / n_batches
                print(f"    ep{epoch} batch {batch_idx} | spatial={s:.4f} frechet={f:.4f} msr={t:.4f} | "
                      f"nan={n_nan} | {time.time() - t0:.0f}s", flush=True)
        scheduler.step()
        if n_batches == 0:
            print("  no batches: check --cache_dir", flush=True)
            break
        s, f, t = sums / n_batches
        total = s + args.frechet_weight * f + args.temporal_weight * t
        if total < best_loss:
            best_loss = total
            torch.save(model.get_encoder_state_dict(), os.path.join(args.checkpoint_dir, "best_encoder.pth"))
        print(f"  ep {epoch:3d} | spatial={s:.4f} frechet={f:.4f} msr={t:.4f} total={total:.4f} | batches={n_batches} "
              f"nan={n_nan} | {time.time() - t0:.0f}s", flush=True)
        torch.save(model.get_encoder_state_dict(), os.path.join(args.checkpoint_dir, f"encoder_ep{epoch}.pth"))
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(), 'best_loss': best_loss}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        print(f"    filters: {['%.1f-%.1f Hz' % (lo, hi) for lo, hi in model.get_filter_frequencies()]}", flush=True)


if __name__ == "__main__":
    main()
