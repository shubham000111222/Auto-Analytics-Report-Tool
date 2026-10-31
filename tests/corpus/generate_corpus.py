import os
import numpy as np
import pandas as pd
from openpyxl import Workbook

OUT = os.path.dirname(__file__)


def write(name, df):
    df.to_csv(os.path.join(OUT, name), index=False)


def main():
    os.makedirs(OUT, exist_ok=True)

    open(os.path.join(OUT, "empty.csv"), "w").close()
    pd.DataFrame(columns=["a", "b", "c"]).to_csv(os.path.join(OUT, "header_only.csv"), index=False)
    write("one_column.csv", pd.DataFrame({"value": range(100)}))
    write("all_nulls.csv", pd.DataFrame({"a": [None] * 50, "b": [np.nan] * 50}))

    n = 1_000_000
    write("large_rows.csv", pd.DataFrame({
        "id": range(n), "value": np.random.randn(n),
        "category": np.random.choice(["A", "B", "C"], n),
        "date": pd.date_range("2020-01-01", periods=n, freq="min"),
    }))

    pd.DataFrame({"name": ["Jose", "Muller", "Asa", "naive"], "amount": [1, 2, 3, 4]}) \
        .to_csv(os.path.join(OUT, "latin1_encoded.csv"), index=False, encoding="latin-1")

    write("wide_500_cols.csv", pd.DataFrame(np.random.randn(100, 500),
                                             columns=[f"col_{i}" for i in range(500)]))

    with open(os.path.join(OUT, "messy_headers.csv"), "w") as f:
        f.write(",,Revenue,,\nRegion,Rep,Q1,Q2,Q3\nEast,Jane,100,110,120\nWest,John,90,95,100\n")

    write("inconsistent_categories.csv", pd.DataFrame({
        "city": ["Delhi", "delhi ", "DELHI", "Mumbai", "mumbai", " Mumbai"],
        "sales": [10, 20, 15, 30, 25, 28],
    }))
    write("mixed_types.csv", pd.DataFrame({"col": [1, "two", 3.0, "4", None]}))

    wb = Workbook()
    ws = wb.active
    ws.merge_cells("A1:C1")
    ws["A1"] = "Quarterly Report"
    ws.append(["Region", "Rep", "Sales"])
    ws.append(["East", "Jane", 100])
    ws.append(["West", "John", 90])
    wb.save(os.path.join(OUT, "merged_cells.xlsx"))

    write("single_row.csv", pd.DataFrame({"a": [1], "b": ["x"]}))
    write("constant_column.csv", pd.DataFrame({"a": [5] * 20, "b": range(20)}))
    base = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    write("duplicates.csv", pd.concat([base, base, base]))

    print(f"Corpus written to {OUT}")


if __name__ == "__main__":
    main()
