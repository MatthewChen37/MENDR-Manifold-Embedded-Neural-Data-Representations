"""Evaluate released fine-tuned MENDR weights on a task's test split.

  python evaluate.py --task tuab --data_root /data/TUAB/v3.0.1/edf --weights weights/Finetuned/Tuab/seed789.pth
  python evaluate.py --task physio --data_root /data/eegmmidb/1.0.0 --weights weights/Finetuned/Physio/seed789.pth
      (PhysioMI: the five files seed789_fold{0..4}.pth are evaluated on their folds and averaged)

MENDR evaluates with per-band BatchNorm statistics (mendr/perband_bn.py). They are not stored in the weights file:
like the original run, this script recomputes them on 50 batches of the task's training split (fixed seed) before
scoring the test split, which reproduces the run's recorded test score (up to floating-point differences between
devices). `--plain_bn` skips this step and uses the ordinary running BatchNorm statistics (not the reported protocol).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mendr.perband_bn import PerBandBN
from mendr.tasks import TASKS, build_model, build_splits, config, physio_fold_sets
from mendr.train import collate_fn, configure_precision, evaluate_test


def evaluate_weights(task, weights, train, test, device, plain_bn=False):
    model = build_model(task).to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True), strict=True)
    bs = config(task)["batch_size"]
    tr = DataLoader(train, batch_size=bs, collate_fn=collate_fn)     # PerBandBN draws its own seeded batches from tr.dataset
    te = DataLoader(test, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    pbn = None if plain_bn else PerBandBN(model, config(task)["bn_perband_eval"], seed=0)
    return evaluate_test(model, tr, te, device, is_binary=config(task)["num_classes"] == 2, pbn=pbn)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=sorted(TASKS))
    p.add_argument("--data_root", required=True)
    p.add_argument("--weights", required=True, help="weights file (PhysioMI: the path without _fold<k>)")
    p.add_argument("--cache_dir", default="")
    p.add_argument("--plain_bn", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_precision()
    metric = config(args.task)["metric"]
    splits = build_splits(args.task, args.data_root, args.cache_dir or None)
    if args.task == "physio":
        data, folds = splits
        per_fold = []
        for fi, fold in enumerate(folds):
            train, _, test = physio_fold_sets(data, fold)
            m = evaluate_weights(args.task, args.weights[:-4] + f"_fold{fi}.pth", train, test, device, args.plain_bn)
            print(f"  fold {fi}: test {metric} = {m[metric]:.4f}", flush=True)
            per_fold.append(m)
        score, result = float(np.mean([m[metric] for m in per_fold])), {"folds": per_fold}
    else:
        train, _, test = splits
        result = evaluate_weights(args.task, args.weights, train, test, device, args.plain_bn)
        score = result[metric]
    print(f"{args.task}: test {metric} = {score:.4f}  ({'plain BatchNorm' if args.plain_bn else 'per-band BatchNorm'})", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"task": args.task, "weights": args.weights, "metric": metric, "score": score, "result": result},
                      fh, indent=2, default=float)


if __name__ == "__main__":
    main()
