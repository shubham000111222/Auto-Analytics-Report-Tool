import pandas as pd
from engine.profiler import profile_dataset
from engine.cleaner import generate_all_suggestions, apply_cleaning


def test_missing_and_duplicate_suggestions_applied():
    df = pd.DataFrame({
        "amount": [10, 20, None, 10000, 15],
        "city": ["Delhi", "delhi ", "Mumbai", "mumbai", None],
    })
    df = pd.concat([df, df.iloc[[0]]])
    profile = profile_dataset(df)
    suggestions = generate_all_suggestions(df, profile)

    ids = {s.id for s in suggestions}
    assert any(i.startswith("missing::amount") for i in ids)
    assert any(i.startswith("duplicates::") for i in ids)

    cleaned, log = apply_cleaning(df, suggestions, ids)
    assert cleaned.duplicated().sum() == 0
    assert cleaned["amount"].isna().sum() == 0
    assert len(log) == len(suggestions)
