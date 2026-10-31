from __future__ import annotations
import os
import pandas as pd
from engine.profiler import profile_dataset, ColType

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "eval_datasets")

GROUND_TRUTH = {
    "titanic.csv": {
        "PassengerId": ColType.ID, "Survived": ColType.BOOLEAN, "Pclass": ColType.CATEGORICAL,
        "Name": ColType.TEXT, "Sex": ColType.CATEGORICAL, "Age": ColType.NUMERIC,
        "Fare": ColType.NUMERIC, "Embarked": ColType.CATEGORICAL,
    },
    "iris.csv": {
        "sepal_length": ColType.NUMERIC, "sepal_width": ColType.NUMERIC,
        "petal_length": ColType.NUMERIC, "petal_width": ColType.NUMERIC,
        "species": ColType.CATEGORICAL,
    },
    "tips.csv": {
        "total_bill": ColType.NUMERIC, "tip": ColType.NUMERIC, "sex": ColType.CATEGORICAL,
        "smoker": ColType.CATEGORICAL, "day": ColType.CATEGORICAL, "time": ColType.CATEGORICAL,
        "size": ColType.NUMERIC,
    },
    "diamonds.csv": {
        "carat": ColType.NUMERIC, "cut": ColType.CATEGORICAL, "color": ColType.CATEGORICAL,
        "clarity": ColType.CATEGORICAL, "depth": ColType.NUMERIC, "table": ColType.NUMERIC,
        "price": ColType.NUMERIC,
    },
    "flights.csv": {
        "year": ColType.NUMERIC, "month": ColType.CATEGORICAL, "passengers": ColType.NUMERIC,
    },
    "penguins.csv": {
        "species": ColType.CATEGORICAL, "island": ColType.CATEGORICAL,
        "bill_length_mm": ColType.NUMERIC, "bill_depth_mm": ColType.NUMERIC,
        "flipper_length_mm": ColType.NUMERIC, "body_mass_g": ColType.NUMERIC,
        "sex": ColType.CATEGORICAL,
    },
}


def run_eval() -> float:
    total, correct, mismatches = 0, 0, []
    for fname, expected_cols in GROUND_TRUTH.items():
        path = os.path.join(EVAL_DIR, fname)
        if not os.path.exists(path):
            print(f"SKIP {fname}: run scripts/fetch_eval_datasets.py first")
            continue
        df = pd.read_csv(path)
        profile = profile_dataset(df)
        for col, expected in expected_cols.items():
            if col not in profile.columns:
                continue
            total += 1
            actual = profile.columns[col].dtype
            if actual == expected:
                correct += 1
            else:
                mismatches.append((fname, col, expected.value, actual.value))

    accuracy = correct / total if total else 0.0
    print(f"\nType detection accuracy: {correct}/{total} ({accuracy:.1%})")
    for fname, col, exp, act in mismatches:
        print(f"  MISMATCH {fname}:{col} expected={exp} actual={act}")
    return accuracy


if __name__ == "__main__":
    acc = run_eval()
    assert acc >= 0.8, "Type detection accuracy below 80% threshold"
