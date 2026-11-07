#!/bin/bash
# All Table 1 MENDR runs: 6 tasks x seeds 789 1024 31 64 99. Usage: bash scripts/run_table1.sh data_roots.env
# data_roots.env defines TUAB TUEV PHYSIO SEEDV CHBMIT ISRUC (dataset directories, see README 'Data').
set -euo pipefail
source "$1"
cd "$(dirname "$0")/.."
declare -A ROOT=([tuab]=$TUAB [tuev]=$TUEV [physio]=$PHYSIO [seedv]=$SEEDV [chbmit]=$CHBMIT [isruc]=$ISRUC)
for task in tuab tuev physio seedv chbmit isruc; do
  for seed in 789 1024 31 64 99; do
    python finetune.py --task $task --data_root "${ROOT[$task]}" --encoder weights/pretrained.pth --seed $seed \
      --out results/${task}_s${seed}.json --cache_dir cache/
  done
done
python - <<'PY'
import json, glob, numpy as np
for t in ["tuab", "tuev", "physio", "seedv", "chbmit", "isruc"]:
    s = [json.load(open(f))["score"] for f in sorted(glob.glob(f"results/{t}_s*.json"))]
    if s: print(f"{t:7s} {np.mean(s):.4f} +/- {np.std(s, ddof=1) if len(s) > 1 else 0:.4f}  (n={len(s)})")
PY
