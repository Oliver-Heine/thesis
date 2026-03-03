# Docs

This directory holds thesis documentation, experimental results, and figures.

## Contents (planned)

| Path | Description |
|------|-------------|
| `results/` | Per-model metrics tables (accuracy, F1, precision, recall, specificity) |
| `figures/` | Confusion matrices, training curves, architecture diagrams |
| `thesis/`  | LaTeX source for the written thesis |

## Generating Results

Run the model training and evaluation pipeline first:

```bash
cd model_training
python src/train.py       # trains all four models
python src/evaluate.py    # writes results/metrics.csv and prints comparison table
```

The `results/metrics.csv` file can then be imported into the thesis write-up.
