# MENDR: Manifold-Embedded Neural Data Representations

Code, pretrained encoder and fine-tuned weights for
**"MENDR: Manifold-Embedded Neural Data Representations for Channel-Agnostic EEG Foundation Modeling"** (NeurIPS 2026).

MENDR maps EEG with any number of channels to 32 virtual electrodes by cross-attention, splits them into four learnable
frequency bands, and represents each band as a sequence of windowed covariance matrices embedded in the Log-Euclidean
tangent space (spatial path, transformer encoder). A convolutional temporal path runs on the same band signals, and an MLP
head classifies the fused features.

```
raw EEG [C, T] --cross-attention--> 32 virtual electrodes --4 sinc bands--> band signals
  band signals --windowed covariance --> matrix log --> tokens --> transformer --> spatial feature
  band signals --temporal convolutions --> attention pooling --> mean over bands --> temporal feature
  [spatial, temporal] --> MLP --> logits
```

## Contents

```
mendr/
  layers.py        sinc filter bank, windowed covariance, temporal paths
  model.py         MENDRClassifier, matrix logarithm, encoder loading
  pretraining.py   pretraining objectives (masked SPD prediction, Frechet mean, filter diversity, masked span reconstruction)
  perband_bn.py    per-band BatchNorm statistics for evaluation
  train.py         fine-tuning loop, metrics
  tasks.py         per-task settings of Table 1, data and model builders
  data/            loaders for TUAB, TUEV, PhysioMI, SEED-V, CHB-MIT, ISRUC (+ TUAB/TUEV subject splits)
finetune.py        fine-tune on one task with one seed
evaluate.py        evaluate released fine-tuned weights
pretrain.py        pretrain the encoder on the TUH EEG Corpus
preprocess_tuh.py  build the pretraining cache from the TUH EDF files
scripts/run_table1.sh   all Table 1 MENDR runs (6 tasks x 5 seeds)
```

## Installation

```
pip install -r requirements.txt        # Python 3.12
```

`requirements.txt` pins the versions of the paper's runs: the NVIDIA PyTorch container 25.10 (PyTorch 2.9.0a0, NumPy 2.2.6,
SciPy 1.16.2) with scikit-learn 1.8.0 and MNE 1.11.0, on NVIDIA H100 / H200 GPUs. On GPU the scripts enable TF32 for
float32 matrix products and convolutions (`mendr.train.configure_precision`), as in that container.

## Weights

Download from **https://huggingface.co/MatthewChen37/MENDR-Manifold-Embedded-Neural-Data-Representations** and place the files under `weights/`:

```
pip install huggingface_hub
hf download MatthewChen37/MENDR-Manifold-Embedded-Neural-Data-Representations --local-dir weights
cd weights && sha256sum -c SHA256SUMS
```

| File | Content |
|---|---|
| `weights/encoder_ep29.pth` | pretrained encoder (TUH EEG Corpus, 30 epochs); sha256 `17be5e4492a67185a6675e8539a0f7001ba18960c541b6f648315c355de3eb9f` |
| `weights/finetuned/<task>_s<seed>.pth` | fine-tuned classifier for each task and seed (seeds 789, 1024, 31, 64, 99) |
| `weights/finetuned/physio_s<seed>_fold<k>.pth` | PhysioMI: one classifier per cross-validation fold (k = 0..4) |
| `weights/MANIFEST.json` | per file: sha256, task, seed, fold, the run's validation and test scores (PhysioMI: per fold, plus the mean over folds) |

Each fine-tuned file is the validation-selected `state_dict` of `MENDRClassifier` (build it with
`mendr.tasks.build_model(task)`). The fine-tuned weights were produced by re-running the Table 1 configuration with
weight saving switched on. Their test scores (`MANIFEST.json`) differ from the printed Table 1 per seed by up to 0.018,
and their means agree with Table 1 to within 0.003 (table below).

## Data

Every dataset is public; obtain it from its provider under its own terms and pass its directory as `--data_root`.

| Task | `--data_root` | Source |
|---|---|---|
| `tuab` | TUAB v3.0.1 `edf/` (contains `train/`, `eval/`) | TUH EEG Abnormal Corpus, Temple University Hospital |
| `tuev` | TUEV v2.0.1 `edf/` (contains `train/`, `eval/`; `.edf` + `.rec`) | TUH EEG Events Corpus |
| `physio` | `eegmmidb/1.0.0/` (contains `S001/` ... `S109/`) | PhysioNet EEG Motor Movement/Imagery Dataset |
| `seedv` | directory of the raw `.cnt` files (`{subject}_{session}_*.cnt`); `7_1_20180411.cnt` is not used (it cannot be read as a Neuroscan .cnt file) | SEED-V, BCMI laboratory, SJTU |
| `chbmit` | directory of `chbXX/chbXX_NN.edf.pkl` | CHB-MIT Scalp EEG (PhysioNet), converted to per-recording pickles by the CHB-MIT `process1.py` of the BIOT repository (github.com/ycq091044/BIOT) |
| `isruc` | directory with `{1..100}.rec` and `{1..100}_1.txt` (extracted from the per-subject `.rar` archives) | ISRUC-Sleep, subgroup 1 |

Preprocessing, montages and splits are documented at the top of each file in `mendr/data/`. Splits: TUAB and TUEV use
the official eval set as test set and a fixed 10 % of the training subjects as validation set
(`mendr/data/splits/`); PhysioMI uses 5-fold cross-validation over subjects with 15 % of each fold's training subjects
for validation; SEED-V uses trials 0-4 / 5-9 / 10-14 of every session; CHB-MIT uses subjects chb01-19 / chb20-21 /
chb22-23; ISRUC uses subjects 1-80 / 81-90 / 91-100. Use `--cache_dir` to store the preprocessed splits after the first
run. The first build of a cache holds a split twice while writing it; peak RAM is about 100 GB for TUAB, 65 GB for
CHB-MIT, 20 GB for ISRUC and under 15 GB for the others. Afterwards the TUAB cache is memory-mapped and TUAB fine-tuning
needs about 20 GB; CHB-MIT fine-tuning needs about 32 GB.

Built from the raw files with this code, the TUAB, TUEV, SEED-V and ISRUC inputs are bit-identical to those of the paper's
runs. The PhysioMI and CHB-MIT inputs of the paper's runs were preprocessed in an earlier software environment and with a
different order of the scaling operations, so a rebuild differs from them at the float32 rounding level (at most 4.8e-7
and 3.8e-6 in absolute value; labels and sample order are identical).

## Reproducing Table 1

```
python finetune.py --task tuab --data_root /data/TUAB/v3.0.1/edf --encoder weights/encoder_ep29.pth \
    --seed 789 --out results/tuab_s789.json --cache_dir cache/
cp scripts/data_roots.env.example data_roots.env          # set the six dataset directories
bash scripts/run_table1.sh data_roots.env                  # all tasks and seeds
```

Protocol: 5 epochs of linear probing (head only), then full fine-tuning; 50 epochs in total at most, with early stopping
when the validation metric has not improved for 15 epochs; the test split is evaluated once with the validation-selected weights. Metrics: AUROC for
TUAB and CHB-MIT, weighted F1 for TUEV, PhysioMI, SEED-V and ISRUC. Per-task settings are in `mendr/tasks.py` (window
length and stride of the covariance tokens, temporal path version, augmentation, whether the pretrained channel
projection is loaded, and, for SEED-V, the 4-8 / 8-13 / 13-30 / 30-45 Hz filter plan with no weight decay on the cutoffs).
Parameter counts of the full classifier, including the classification head and an unused 5,760-parameter coordinate
encoder in the channel projection: 1,296,459 (TUAB, CHB-MIT), 1,297,231 (TUEV), 1,297,038 (ISRUC), 1,296,845 (PhysioMI),
1,408,973 (SEED-V).

| Task | Metric | Table 1 (mean ± sd, 5 seeds) | Released weights (mean ± sd) |
|---|---|---|---|
| TUAB | AUROC | 0.9028 ± 0.0022 | 0.9033 ± 0.0015 |
| TUEV | weighted F1 | 0.7690 ± 0.0157 | 0.7697 ± 0.0081 |
| PhysioMI | weighted F1 | 0.5789 ± 0.0063 | 0.5767 ± 0.0076 |
| SEED-V | weighted F1 | 0.3886 ± 0.0068 | 0.3889 ± 0.0069 |
| CHB-MIT | AUROC | 0.9866 ± 0.0105 | 0.9885 ± 0.0033 |
| ISRUC | weighted F1 | 0.7533 ± 0.0164 | 0.7553 ± 0.0167 |

Training on GPU is not bit-reproducible: repeating the 30 runs on a mix of H100 and H200 GPUs changed individual seeds by up
to about 0.02, with the task means within 0.01 of Table 1.

## Evaluating the released weights

```
python evaluate.py --task tuab --data_root /data/TUAB/v3.0.1/edf --weights weights/finetuned/tuab_s789.pth --cache_dir cache/
python evaluate.py --task physio --data_root /data/eegmmidb/1.0.0 --weights weights/finetuned/physio_s789.pth
```
For PhysioMI, `--weights` names the seed (`physio_s789.pth`); the script evaluates `physio_s789_fold0.pth` ...
`physio_s789_fold4.pth` on their folds and averages them.

MENDR evaluates the temporal path's BatchNorm layers with per-band statistics (`mendr/perband_bn.py`): the temporal path
is applied to each frequency band separately, so each band is normalised with its own statistics, as during training.
These statistics are not stored in the weights file. `evaluate.py` recomputes them from 50 batches of the task's
training split (fixed seed), exactly as the original run did, and therefore needs the training split as well as the test
split. This reproduces the recorded test score up to floating-point differences between devices. Measured on CPU for all 50 released files: AUROC (TUAB, CHB-MIT) within 1e-4 of the recorded GPU value; weighted F1 within 1e-3 on SEED-V and ISRUC, and up to 6e-3 (TUEV) and 8e-3 (single PhysioMI folds, about 2,000 test epochs each) where a few dozen test windows change class; every task's mean over seeds within 2e-3.
`--plain_bn` evaluates with ordinary running statistics instead, which is not the reported protocol.

## Pretraining

```
python preprocess_tuh.py --tuh_root /data/tuh_eeg/v2.0.1/edf --cache_dir cache/tuh_200hz --workers 16
python pretrain.py --cache_dir cache/tuh_200hz --checkpoint_dir checkpoints/mendr_pretrain
```

`pretrain.py` defaults to the settings of the released encoder (30 epochs, batch 128, AdamW lr 5e-4, weight decay 0.05,
seed 42, loss = spatial + 0.5 Frechet + 0.1 masked span reconstruction + 1.0 filter diversity); the released encoder is
the last epoch, `encoder_ep29.pth`. The released encoder was trained on a cache of 50,210 TUH recordings from 14,559 subjects
preprocessed with the parameters that `preprocess_tuh.py` implements (recordings of 130 s or less skipped, 60 s trimmed at
both ends of the others, 60 Hz notch, 0.1-75 Hz band-pass, 200 Hz, non-overlapping 6 s epochs, epochs above 100 uV
rejected, x1e4 scaling). Checked on the TUAB recordings, which are part of the corpus: `preprocess_tuh.py` reproduces the
cache files of 200 randomly drawn recordings bit-identically, and it skips all 69 TUAB recordings that the cache does not
contain. TUAB and TUEV are drawn from the TUH EEG
Corpus and overlap the (unlabeled) pretraining data, so they are not out-of-domain tests.

## Citation

```
@inproceedings{mendr2026,
  title     = {MENDR: Manifold-Embedded Neural Data Representations for Channel-Agnostic EEG Foundation Modeling},
  author    = {Chen, Matthew and Nnamdi, Micky C. and Marteau, Benoit Louis and Tan, Shaun Qien Yeau and Tamo, J. Ben and
               Wang, May Dongmei},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). The datasets are not redistributed here and remain under their providers' terms.
