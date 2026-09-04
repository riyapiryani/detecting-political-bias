# Detecting Political Bias in News Articles

A classical ML pipeline that predicts the political lean of a news article — left, center, or right — along with a continuous −1 to +1 bias score reflecting the strength of that lean.
## Overview

Given the text of a news article, the model outputs:
- A categorical label: **left**, **center**, or **right**
- A continuous bias score from **−1 (far left) to +1 (far right)**

This lets two articles with the same label be distinguished by intensity of lean, rather than collapsing everything into three hard buckets.

## Dataset

We use the **Article-Bias-Prediction** dataset ([GitHub](https://github.com/ramybaly/Article-Bias-Prediction)):

- 37,554 news articles crawled from AllSides.com
- Labeled at the **article level** (not just inherited from outlet-level lean), avoiding the label-leakage risk of assuming every article from a given outlet shares that outlet's overall bias
- Two official evaluation splits provided by the dataset authors:
  - **`media` split** — train and test outlets do not overlap; tests true generalization to unseen sources
  - **`random` split** — articles randomly split; same outlets can appear in both train and test

Clone the dataset into this repo (or point `--data_root` at wherever you clone it):

```bash
git clone https://github.com/ramybaly/Article-Bias-Prediction.git
```

## Setup

```bash
pip install pandas scikit-learn nltk joblib --break-system-packages
```

(`nltk`/VADER is optional — the script degrades gracefully and skips sentiment features if unavailable, e.g. due to SSL certificate issues. On macOS with the python.org installer, fix this by running `Install Certificates.command` in your Python.app Applications folder.)

## Usage

### Train the model

```bash
python3 train_bias_model.py --data_root ./Article-Bias-Prediction/data --split media
```

Options:
- `--split {random, media}` — which official split to use (default: `media`)
- `--out_dir ./model_output` — where trained models and predictions are saved (default: `./model_output`)
- `--cv_folds N` — number of folds for cross-validation on the train set; set to `0` to skip (default: `5`)

This will:
1. Load articles and labels for the chosen split
2. Print class distribution across train/valid/test
3. Run k-fold cross-validation on the train set (a tuning signal independent of the official validation split)
4. Train Naive Bayes (baseline) and Linear SVM (main model) on TF-IDF (+ VADER sentiment, if available) features
5. Evaluate both models on validation and test sets
6. Derive a continuous bias score from the SVM's decision function, scaled to −1 to +1
7. Save trained models, vectorizer, scaler, and test-set predictions to `--out_dir`

### Run inference on new text

```bash
python3 demo_predict.py --model_dir ./model_output --text "Some headline or article text"
```

Or omit `--text` to be prompted interactively.

### Pick clean demo examples from the test set

```bash
python3 pick_demo_examples.py --predictions ./model_output/test_predictions_with_scores.csv --data_root ./Article-Bias-Prediction/data
```

Surfaces the most confidently-scored, correctly-classified left/center/right examples from the test set — useful for live demos or sanity-checking the model.

## Results

Evaluated on the `media` split (held-out outlets):

| Model | 5-fold CV macro-F1 (train) | Held-out test macro-F1 | Held-out test accuracy |
|---|---|---|---|
| Naive Bayes | 0.677 | 0.524 | 55.8% |
| Linear SVM | **0.788** | 0.511 | 53.5% |

**Key finding:** the gap between cross-validation performance (~79% macro-F1, same outlets seen in both train and validation folds) and held-out-outlet test performance (~51%) suggests the model partly relies on outlet-specific stylistic signal rather than purely transferable bias language — directly quantifying a labeling/generalization risk raised during project scoping.

Note: the official `media`-split validation set has a skewed class distribution (69.6% left, 4.2% right) relative to train and test, which explains an initially confusing validation accuracy (~26–30%) that looked much worse than test accuracy (~53–56%). This is a property of the split, not a modeling bug — see cross-validation results for a distribution-independent performance signal.

## Limitations & Future Work

- Model still partly reliant on outlet-specific style rather than pure bias language
- 3-class label collapses a real ideological spectrum
- SVM's `decision_function` gives a relative score, not a calibrated probability — `CalibratedClassifierCV` would give a more principled continuous score
- Planned: evaluate our own scraped articles, aggregated to outlet level, against AllSides/MediaBiasFactCheck outlet ratings as an additional real-world validation step
