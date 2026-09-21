# RLMF-m6APred

[English](README.md) | [简体中文](README_zh-CN.md)

RLMF-m6APred predicts RNA m6A sites from 41-nt sequences. This repository
provides the inference code and the datasets required to reproduce the
independent-test results. The released model weights are hosted on
[Hugging Face](https://huggingface.co/Yu-star/RLMF-m6APred).

**Web server:** [https://xiaof-xy--rlmf.modal.run](https://xiaof-xy--rlmf.modal.run)

## 1. Environment

The experiments reported in the paper were conducted on a workstation with
three NVIDIA A10 GPUs (24 GB each). Reproducing the released test results only
requires inference and can run on one CUDA GPU or on CPU.

```bash
conda env create -f environment.yml
conda activate rlmf_m6apred
```

## 2. Download the released weights

Download the domain-adapted RNA backbone and all 55 fold checkpoints from
Hugging Face directly into the required directories:

```bash
python scripts/download_model.py
```

After downloading, the relevant layout is:

```text
RLMF-m6APred/
|-- pretrained/rna_language_model/
|-- weights/
|   |-- H_b/fold_01.pt ... fold_05.pt
|   |-- ...
|   `-- R_l/fold_01.pt ... fold_05.pt
|-- data/independent_test/
`-- evaluate_independent.py
```

## 3. Reproduce the independent-test results

Evaluate one dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_independent.py --dataset H_b
```

Evaluate all 11 datasets:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_independent.py --all
```

CPU inference is also supported by omitting `CUDA_VISIBLE_DEVICES=0`. The five
fold probabilities are averaged, and ACC, MCC, AUC, AUPRC, F1, precision,
recall, and specificity are written to `reproduced_results/`.

## 4. Evaluate another labeled dataset

The input CSV must contain `sequence` and `label` columns:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --dataset H_b \
  --input path/to/data.csv
```

The model accepts 41-nt sequences. RNA `U` is normalized to `T` during input
processing.
