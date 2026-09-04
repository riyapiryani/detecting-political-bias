"""
Demo inference script — Political Bias Detector
==================================================

Loads the trained TF-IDF vectorizer, SVM model, and score scaler saved by
train_bias_model.py, and predicts a label + continuous bias score for any
text you paste in. made specifically for the live demo recording.

Usage:
    python demo_predict.py --model_dir ./model_output
    (then paste/type article text when prompted, or pass --text "...")

    python demo_predict.py --model_dir ./model_output --text "Some headline or article text here"
"""

import argparse
import joblib
import numpy as np


LABEL_MAP = {0: "left", 1: "center", 2: "right"}


def predict(text, vectorizer, svm, scaler):
    X = vectorizer.transform([text])
    pred_label = svm.predict(X)[0]
    decision = svm.decision_function(X)[0]  # shape (3,) — left, center, right
    raw_score = decision[2] - decision[0]    # right_score - left_score
    continuous_score = scaler.transform([[raw_score]])[0][0]
    return LABEL_MAP[pred_label], continuous_score, decision


def main():
    parser = argparse.ArgumentParser(description="Predict political bias for a piece of text.")
    parser.add_argument("--model_dir", default="./model_output",
                         help="Folder containing tfidf_vectorizer.joblib, svm_model.joblib, score_scaler.joblib")
    parser.add_argument("--text", default=None,
                         help="Text to classify. If omitted, you'll be prompted to paste text.")
    args = parser.parse_args()

    vectorizer = joblib.load(f"{args.model_dir}/tfidf_vectorizer.joblib")
    svm = joblib.load(f"{args.model_dir}/svm_model.joblib")
    scaler = joblib.load(f"{args.model_dir}/score_scaler.joblib")

    if args.text:
        texts = [args.text]
    else:
        print("Paste article text (or a headline), then press Enter:")
        texts = [input("> ")]

    for text in texts:
        label, score, decision = predict(text, vectorizer, svm, scaler)
        print("\n--- Prediction ---")
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Predicted label:      {label}")
        print(f"Continuous bias score: {score:+.3f}   (-1 = far left, 0 = center, +1 = far right)")
        print(f"Raw decision scores:   left={decision[0]:.3f}  center={decision[1]:.3f}  right={decision[2]:.3f}")


if __name__ == "__main__":
    main()
