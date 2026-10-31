import glob
import os
import pytest
from engine.loader import load_file, validate_file

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")


def _corpus_files():
    return [f for f in glob.glob(os.path.join(CORPUS_DIR, "*"))
            if os.path.isfile(f) and not f.endswith(".py") and "large_rows" not in f]


@pytest.mark.parametrize("path", _corpus_files())
def test_load_does_not_crash(path):
    with open(path, "rb") as f:
        raw = f.read()
    filename = os.path.basename(path)
    if validate_file(filename, len(raw)):
        pytest.skip("expected rejection")
    try:
        result = load_file(filename, raw)
    except Exception as e:
        pytest.fail(f"load_file raised on {filename}: {e}")
    assert result.df is not None
