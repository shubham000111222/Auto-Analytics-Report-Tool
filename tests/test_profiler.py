import os
from engine.loader import load_file
from engine.profiler import profile_dataset, ColType
from engine.analyzers.categorical import analyze_categorical

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def _load(name):
    with open(os.path.join(CORPUS_DIR, name), "rb") as f:
        return load_file(name, f.read())


def test_constant_column_detected():
    p = profile_dataset(_load("constant_column.csv").df)
    assert p.columns["a"].dtype == ColType.CONSTANT


def test_all_nulls_detected():
    p = profile_dataset(_load("all_nulls.csv").df)
    assert all(c.dtype == ColType.CONSTANT and c.missing_pct == 100.0 for c in p.columns.values())


def test_inconsistent_categories_flagged():
    df = _load("inconsistent_categories.csv").df
    assert analyze_categorical(df["city"])["possible_inconsistent_categories"] > 0


def test_duplicates_counted():
    assert profile_dataset(_load("duplicates.csv").df).duplicate_rows > 0


def test_wide_file_profiles_without_crash():
    assert profile_dataset(_load("wide_500_cols.csv").df).n_cols == 500


def test_mixed_types_flagged():
    assert profile_dataset(_load("mixed_types.csv").df).columns["col"].dtype == ColType.MIXED
