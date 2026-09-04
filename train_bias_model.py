"""
Political Bias Classification — Article-Bias-Prediction dataset
==================================================================

Trains classical ML models (Naive Bayes + linear SVM) on the AllSides-labeled
article dataset from Baly et al. (2020), "We Can Detect Your Bias."

Dataset source: https://github.com/ramybaly/Article-Bias-Prediction

Usage:
    python3 train_bias_model.py --data_root ./data --split random
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib

VADER_AVAILABLE = False
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        VADER_AVAILABLE = True
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=False)
            nltk.data.find("sentiment/vader_lexicon.zip")
            VADER_AVAILABLE = True
        except Exception as e:
            print(f"WARNING: could not download VADER lexicon ({e}). "
                  "Proceeding WITHOUT VADER sentiment features. "
                  "Fix your SSL certs (macOS: run 'Install Certificates.command' "
                  "in your Python.app Applications folder) and rerun to enable them.")
except ImportError:
    print("WARNING: nltk not installed — proceeding WITHOUT VADER sentiment features. "
          "Run `pip install nltk --break-system-packages` to enable them.")


# The dataset's numeric labels: 0 = left, 1 = center, 2 = right (per Baly et al.)
LABEL_MAP = {0: "left", 1: "center", 2: "right"}


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_split(data_root: str, split: str, subset: str) -> pd.DataFrame:
    """Load one subset (train/valid/test) of one split type (random/media) and
    join the article IDs to their full JSON content."""
    split_path = os.path.join(data_root, "splits", split, f"{subset}.tsv")
    jsons_dir = os.path.join(data_root, "jsons")

    if not os.path.exists(split_path):
        sys.exit(f"Could not find split file: {split_path}\n"
                  f"Check --data_root and --split arguments.")

    ids_df = pd.read_csv(split_path, sep="\t", header=None,
                          names=["ID", "bias"])

    records = []
    for _, row in ids_df.iterrows():
        json_path = os.path.join(jsons_dir, f"{row['ID']}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            article = json.load(f)
        records.append({
            "ID": row["ID"],
            "bias": int(row["bias"]),
            "source": article.get("source", ""),
            "title": article.get("title", ""),
            "content": article.get("content", "") or article.get("content_original", ""),
        })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} articles for {split}/{subset}")
    return df


def print_class_distribution(train_df, valid_df, test_df):
    """Print label counts and proportions for each split, side by side, so
    it's easy to spot mismatches between train/valid/test class balance."""
    print("\n=== Class distribution across splits ===")
    rows = []
    for name, df_ in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        counts = df_["bias"].value_counts().sort_index()
        total = len(df_)
        for label_id in sorted(LABEL_MAP):
            n = int(counts.get(label_id, 0))
            pct = 100 * n / total if total else 0.0
            rows.append({
                "split": name,
                "label": LABEL_MAP[label_id],
                "count": n,
                "pct": round(pct, 1),
            })
    dist_df = pd.DataFrame(rows)
    pivot_counts = dist_df.pivot(index="label", columns="split", values="count")
    pivot_pcts = dist_df.pivot(index="label", columns="split", values="pct")
    pivot_counts = pivot_counts.reindex(["left", "center", "right"])
    pivot_pcts = pivot_pcts.reindex(["left", "center", "right"])
    print("\nCounts:")
    print(pivot_counts)
    print("\nPercent of split:")
    print(pivot_pcts)
    print()


def add_vader_features(df: pd.DataFrame) -> pd.DataFrame:
    if not VADER_AVAILABLE:
        return df
    sia = SentimentIntensityAnalyzer()

    def score(text):
        # VADER works on shorter text; truncate very long articles for speed
        s = sia.polarity_scores(text[:5000])
        return s["compound"], s["pos"], s["neg"], s["neu"]

    scores = df["content"].fillna("").apply(score)
    df["vader_compound"] = scores.apply(lambda x: x[0])
    df["vader_pos"] = scores.apply(lambda x: x[1])
    df["vader_neg"] = scores.apply(lambda x: x[2])
    df["vader_neu"] = scores.apply(lambda x: x[3])
    return df


def cross_validate_on_train(train_df, k=5):
    """
    Run stratified k-fold CV on the TRAIN set only. This gives a tuning/
    sanity-check signal that doesn't depend on the official validation set,
    which (for the 'media' split) can have a very different class balance
    than train/test and is therefore unreliable for model selection.
    """
    print(f"\n=== {k}-fold cross-validation on TRAIN set ===")

    text = train_df["title"].fillna("") + " " + train_df["content"].fillna("")
    y = train_df["bias"].values

    # Fit a fresh vectorizer here (separate from the main one used later) so
    # CV folds don't leak information between each other.
    cv_vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=3,
        sublinear_tf=True,
    )
    X = cv_vectorizer.fit_transform(text)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    nb = MultinomialNB(alpha=0.1)
    nb_scores = cross_val_score(nb, X, y, cv=skf, scoring="f1_macro")
    print(f"Naive Bayes  — macro-F1 per fold: {np.round(nb_scores, 3)}  "
          f"mean={nb_scores.mean():.4f}  std={nb_scores.std():.4f}")

    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    svm_scores = cross_val_score(svm, X, y, cv=skf, scoring="f1_macro")
    print(f"Linear SVM   — macro-F1 per fold: {np.round(svm_scores, 3)}  "
          f"mean={svm_scores.mean():.4f}  std={svm_scores.std():.4f}")

    # Also try SVM without class_weight="balanced" for comparison, since that
    # setting seemed to hurt on the skewed validation set.
    svm_unbalanced = LinearSVC(C=1.0, max_iter=5000)
    svm_unbalanced_scores = cross_val_score(svm_unbalanced, X, y, cv=skf, scoring="f1_macro")
    print(f"Linear SVM (no class_weight) — macro-F1 per fold: "
          f"{np.round(svm_unbalanced_scores, 3)}  "
          f"mean={svm_unbalanced_scores.mean():.4f}  std={svm_unbalanced_scores.std():.4f}")
    print()


# ---------------------------------------------------------------------------
# 2. Training
# ---------------------------------------------------------------------------

def train_and_evaluate(train_df, valid_df, test_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    y_train = train_df["bias"].values
    y_valid = valid_df["bias"].values
    y_test = test_df["bias"].values

    # --- TF-IDF features (title + body combined) ---
    train_df["text"] = (train_df["title"].fillna("") + " " + train_df["content"].fillna(""))
    valid_df["text"] = (valid_df["title"].fillna("") + " " + valid_df["content"].fillna(""))
    test_df["text"] = (test_df["title"].fillna("") + " " + test_df["content"].fillna(""))

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),      # unigrams + bigrams, per proposal
        stop_words="english",
        min_df=3,
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(train_df["text"])
    X_valid_tfidf = vectorizer.transform(valid_df["text"])
    X_test_tfidf = vectorizer.transform(test_df["text"])

    print(f"TF-IDF vocab size: {len(vectorizer.vocabulary_)}")

    # --- VADER features (optional, appended as dense columns) ---
    if VADER_AVAILABLE:
        for df_ in (train_df, valid_df, test_df):
            add_vader_features(df_)
        vader_cols = ["vader_compound", "vader_pos", "vader_neg", "vader_neu"]

    # --- Model 1: Multinomial Naive Bayes (baseline, TF-IDF only — NB needs non-negative features) ---
    print("\n=== Training Naive Bayes (baseline) ===")
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    evaluate_model(nb, X_valid_tfidf, y_valid, "NB — validation")
    evaluate_model(nb, X_test_tfidf, y_test, "NB — test")

    # --- Model 2: Linear SVM (main model) ---
    print("\n=== Training Linear SVM ===")
    svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    svm.fit(X_train_tfidf, y_train)
    evaluate_model(svm, X_valid_tfidf, y_valid, "SVM — validation")
    evaluate_model(svm, X_test_tfidf, y_test, "SVM — test")

    # --- Continuous bias score from SVM decision function ---
    # decision_function gives one score per class in a 1-vs-rest scheme.
    # For a left/center/right ordinal setup, project onto a single left(-1) to right(+1)
    # axis: score = P(right)-like signal minus P(left)-like signal.
    print("\n=== Deriving continuous bias scores from SVM decision_function ===")
    test_scores = svm_scores_to_continuous(svm, X_test_tfidf)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    continuous_scores = scaler.fit_transform(test_scores.reshape(-1, 1)).flatten()

    test_df = test_df.copy()
    test_df["bias_label"] = test_df["bias"].map(LABEL_MAP)
    test_df["continuous_bias_score"] = continuous_scores
    print(test_df[["ID", "source", "bias_label", "continuous_bias_score"]].head(10))

    # --- Save everything ---
    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(nb, os.path.join(out_dir, "naive_bayes_model.joblib"))
    joblib.dump(svm, os.path.join(out_dir, "svm_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "score_scaler.joblib"))
    test_df.to_csv(os.path.join(out_dir, "test_predictions_with_scores.csv"), index=False)
    print(f"\nSaved models and predictions to: {out_dir}")

    return nb, svm, vectorizer


def svm_scores_to_continuous(svm_model, X):
    """
    Convert LinearSVC's one-vs-rest decision_function output (shape:
    n_samples x n_classes, in label order left=0, center=1, right=2) into a
    single continuous left-to-right axis: right_score - left_score.
    """
    decision = svm_model.decision_function(X)  # shape (n_samples, 3)
    left_score = decision[:, 0]
    right_score = decision[:, 2]
    return right_score - left_score


def evaluate_model(model, X, y_true, label):
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n[{label}] accuracy={acc:.4f}  macro-F1={f1_macro:.4f}")
    print(classification_report(y_true, y_pred,
                                 target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))
    print("Confusion matrix (rows=true, cols=pred), order left/center/right:")
    print(confusion_matrix(y_true, y_pred))


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train political bias classifier.")
    parser.add_argument("--data_root", required=True,
                         help="Path to the Article-Bias-Prediction/data folder")
    parser.add_argument("--split", default="media", choices=["random", "media"],
                         help="Which official split to use. 'media' is the harder, "
                              "more realistic split (held-out outlets); 'random' is easier.")
    parser.add_argument("--out_dir", default="./model_output",
                         help="Where to save trained models and predictions")
    parser.add_argument("--cv_folds", type=int, default=5,
                         help="Number of folds for cross-validation on the train "
                              "set. Set to 0 to skip cross-validation.")
    args = parser.parse_args()

    print(f"Loading '{args.split}' split from {args.data_root} ...")
    train_df = load_split(args.data_root, args.split, "train")
    valid_df = load_split(args.data_root, args.split, "valid")
    test_df = load_split(args.data_root, args.split, "test")

    print_class_distribution(train_df, valid_df, test_df)

    if args.cv_folds > 0:
        cross_validate_on_train(train_df, k=args.cv_folds)

    train_and_evaluate(train_df, valid_df, test_df, args.out_dir)


if __name__ == "__main__":
    main()