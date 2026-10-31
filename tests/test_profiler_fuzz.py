import pandas as pd
from hypothesis import given, settings, strategies as st
from engine.profiler import profile_dataset, ColType

column_values = st.one_of(
    st.integers(), st.floats(allow_nan=True, allow_infinity=False),
    st.text(max_size=20), st.booleans(), st.none(),
)


@given(st.lists(st.dictionaries(st.sampled_from(["a", "b", "c"]), column_values), min_size=0, max_size=50))
@settings(max_examples=50, deadline=None)
def test_profiler_never_crashes_on_random_data(rows):
    df = pd.DataFrame(rows)
    profile = profile_dataset(df)
    assert profile.n_cols == df.shape[1]
    for col_profile in profile.columns.values():
        assert isinstance(col_profile.dtype, ColType)
