from __future__ import annotations

import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_npz_features(path: str) -> DatasetSplit:
    """Loads a feature-vector dataset from .npz.

    Supported formats:
      - X, y (then auto-splits 80/20)
      - X_train, y_train, X_test, y_test

    Labels expected: 0=benign, 1=malicious
    """
    data = np.load(path, allow_pickle=False)

    if {"X_train", "y_train", "X_test", "y_test"}.issubset(set(data.files)):
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        return DatasetSplit(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    if {"X", "y"}.issubset(set(data.files)):
        X = data["X"]
        y = data["y"]
        n = X.shape[0]
        split = int(0.8 * n)
        return DatasetSplit(X_train=X[:split], y_train=y[:split], X_test=X[split:], y_test=y[split:])

    raise ValueError(f"Unsupported .npz keys: {data.files}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train a lightweight feature-vector malware classifier (.npz)")
    p.add_argument("--data", required=True, help="Path to .npz containing features")
    p.add_argument(
        "--out",
        default=str(Path("outputs") / "fusion" / "feature_model.joblib"),
        help="Output model path",
    )
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)

    ds = load_npz_features(args.data)

    # Model choice: linear classifier trained with SGD.
    # This scales well, is small, and is common for EMBER-style features.
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    import joblib

    clf = SGDClassifier(
        loss="log_loss",
        alpha=1e-5,
        max_iter=20,
        tol=1e-3,
        random_state=args.seed,
    )

    clf.fit(ds.X_train, ds.y_train)

    preds = clf.predict(ds.X_test)
    cm = confusion_matrix(ds.y_test, preds)
    report = classification_report(ds.y_test, preds, target_names=["Benign", "Malicious"])

    print("CONFUSION MATRIX:\n", cm)
    print("\nREPORT:\n", report)

    joblib.dump(clf, args.out)
    print(f"Saved: {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
