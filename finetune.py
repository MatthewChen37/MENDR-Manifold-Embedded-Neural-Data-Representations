"""Fine-tune MENDR on one downstream task with one seed (the Table 1 protocol).

  python finetune.py --task tuab --data_root /data/TUAB/v3.0.1/edf --encoder weights/pretrained.pth --seed 789 \
      --out results/tuab_s789.json [--save_weights weights_out/tuab_s789.pth] [--cache_dir cache/]

Selection uses the validation split only; the test split is evaluated once, with the selected weights. Table 1 reports
the mean and standard deviation over seeds 789, 1024, 31, 64 and 99. PhysioMI runs its 5 folds in one call and reports
the mean over folds; with --save_weights each fold's weights go to <path>_fold<k>.pth.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mendr.tasks import TASKS, build_model, build_splits, config, load_encoder, loaders, physio_fold_sets
from mendr.train import configure_precision, finetune, set_seed


def run_split(task, splits, cfg, args, device, save_to):
    train, val, test = splits
    tr, va, te = loaders(task, train, val, test)
    model = build_model(task).to(device)
    if args.encoder:
        load_encoder(model, task, args.encoder, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    {len(train)} train, {len(val)} val, {len(test)} test, {n_params:,} params", flush=True)
    train_labels = [int(train[i][1]) for i in range(len(train))]
    run_cfg = {**cfg, "pretrained": bool(args.encoder), "save_weights_to": save_to}
    best_val, test_metrics, history = finetune(model, tr, va, te, train_labels, run_cfg, device, cfg["num_classes"],
                                               is_binary=cfg["num_classes"] == 2)
    return {"n_params": n_params, "best_val": best_val, "test": test_metrics, "history": history}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--data_root", required=True, help="dataset directory (see README, 'Data')")
    p.add_argument("--encoder", default="weights/pretrained.pth", help="pretrained encoder; '' trains from scratch")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True, help="result JSON")
    p.add_argument("--save_weights", default="", help="save the selected weights here (never overwrites)")
    p.add_argument("--cache_dir", default="", help="cache the preprocessed splits here (recommended)")
    p.add_argument("--epochs", type=int, default=None, help="override the number of epochs (default 50)")
    args = p.parse_args()

    cfg = config(args.task)
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision()
    print(f"Device: {device} | task={args.task} seed={args.seed} encoder={args.encoder or '(none)'}", flush=True)
    print(f"Config: {json.dumps(cfg, sort_keys=True)}", flush=True)
    t0 = time.time()
    set_seed(args.seed)
    splits = build_splits(args.task, args.data_root, args.cache_dir or None)
    if args.task == "physio":
        data, folds = splits
        records = []
        for fi, fold in enumerate(folds):
            print(f"  [fold {fi}] {len(fold[0])} train, {len(fold[1])} val, {len(fold[2])} test", flush=True)
            save_to = args.save_weights[:-4] + f"_fold{fi}.pth" if args.save_weights else ""
            r = run_split(args.task, physio_fold_sets(data, fold), cfg, args, device, save_to)
            records.append({"fold": fi, **r})
        fold_scores = [r["test"]["f1"] for r in records]
        record = {"folds": records, "fold_scores": fold_scores, "f1_mean_over_folds": float(np.mean(fold_scores)),
                  "f1_std_over_folds": float(np.std(fold_scores, ddof=1))}
        score = record["f1_mean_over_folds"]
    else:
        record = run_split(args.task, splits, cfg, args, device, args.save_weights)
        score = record["test"][cfg["metric"]]
    out = {"task": args.task, "seed": args.seed, "metric": cfg["metric"], "score": score, "encoder": args.encoder,
           "config": cfg, "record": record, "elapsed_sec": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\n{args.task} seed {args.seed}: test {cfg['metric']} = {score:.4f}  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
