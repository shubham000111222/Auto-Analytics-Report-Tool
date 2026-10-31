import os
import urllib.request

EVAL_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "eval_datasets")

DATASETS = {
    "titanic.csv": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "iris.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
    "tips.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
    "diamonds.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
    "flights.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/flights.csv",
    "penguins.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv",
}


def main():
    os.makedirs(EVAL_DIR, exist_ok=True)
    for fname, url in DATASETS.items():
        dest = os.path.join(EVAL_DIR, fname)
        if os.path.exists(dest):
            print(f"SKIP {fname} (cached)")
            continue
        print(f"Downloading {fname} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
