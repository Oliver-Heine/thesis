# Notebooks — Jupyter Experiments

This directory contains Jupyter notebooks used for exploratory analysis and
prototyping on an Apple M1 Mac (MPS acceleration).

## Contents

| Notebook | Description |
|----------|-------------|
| *(to be added)* | Initial EDA on the malicious-urls dataset |
| *(to be added)* | Per-model training curves and confusion matrices |
| *(to be added)* | ONNX inference latency benchmarks |

## Setup

```bash
cd model_training
pip install -r requirements.txt
pip install jupyter
jupyter lab
```

> **Note:** Large model weights and `.csv` dataset files are excluded from
> version control via `.gitignore`. Notebooks that depend on them must be
> run locally after placing the required files in `model_training/`.
