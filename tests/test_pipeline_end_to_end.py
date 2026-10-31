import pandas as pd
from engine.pipeline import run_pipeline


def test_pipeline_runs_without_api_key_or_narrative():
    df = pd.DataFrame({
        "amount": [1, 2, None, 100, 5],
        "city": ["Delhi", "delhi ", "Mumbai", "mumbai", None],
        "date": pd.date_range("2023-01-01", periods=5),
    })
    result = run_pipeline("t.csv", df.to_csv(index=False).encode(),
                           skip_narrative=True, skip_preparse_check=True)
    assert result.profile.n_rows == 5
    assert "data_quality" in result.charts
    assert result.narrative is None


def test_cleaning_suggestions_applied_through_pipeline():
    df = pd.DataFrame({"amount": [1, 2, None, 100, 5]})
    raw = df.to_csv(index=False).encode()
    first = run_pipeline("t.csv", raw, skip_narrative=True, skip_preparse_check=True)
    ids = {s.id for s in first.cleaning_suggestions}
    second = run_pipeline("t.csv", raw, accepted_cleaning_ids=ids,
                           skip_narrative=True, skip_preparse_check=True)
    assert second.cleaned_df["amount"].isna().sum() == 0
