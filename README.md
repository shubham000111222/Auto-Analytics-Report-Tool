# Auto Analytics Report Tool

Upload any CSV/Excel file and get a full analytics report -- profiling, charts,
plain-language findings, and exports to HTML/PDF/Excel.

Most "upload a CSV, get insights" tools are thin wrappers around
`ydata-profiling`. This one builds the profiling, chart-selection, and
statistics engine from scratch, and uses an LLM *only* to phrase computed
numbers in plain language -- never to do the math. Every number in a report is
reproducible by re-running the underlying code.

## Pipeline

```
Upload -> Parse (loader.py) -> Profile columns (profiler.py)
       -> Suggest cleaning (cleaner.py, opt-in, never silent)
       -> Analyze by type (analyzers/) -> Auto-chart (charts.py)
       -> Narrate computed stats via Claude (narrative.py, validated)
       -> Export (exporters.py): HTML / PDF / Excel
```

Two front ends share this engine:
- **Streamlit app** (`app/streamlit_app.py`) -- fastest path, single-process demo.
- **FastAPI service** (`api/`) -- async upload, background processing, Postgres
  job metadata.

## Quickstart (local)

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# WeasyPrint (PDF export) needs system libraries:
#   Ubuntu/Debian: sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
#   macOS:         brew install pango
#   Windows:       see https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#windows

cp .env.example .env               # then edit ANTHROPIC_API_KEY (optional)

streamlit run app/streamlit_app.py
# or
uvicorn api.main:app --reload
```

## Running tests

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m tests.corpus.generate_corpus
pytest tests/ -v -k "not large_rows"     # the 1M-row corpus file is slow; CI skips it by default

# Optional: type-detection accuracy against real public datasets
python scripts/fetch_eval_datasets.py
python scripts/eval_type_detection.py
```

Or simply: `make install-dev && make test`

## Docker

```bash
cp .env.example .env
docker compose up --build
# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000/docs
```

## Design decisions worth noting

- **Compute first, narrate second.** `narrative.py` sends only aggregated
  JSON statistics to Claude, and validates every number the model outputs
  against that JSON before showing it. If validation fails twice, the report
  falls back to a templated summary.
- **Cleaning is suggested, never silent.** `cleaner.py` produces a list of
  `CleaningSuggestion`s with before/after previews; nothing changes until the
  user opts in via the Streamlit "Cleaning" tab, and the applied log is shown
  in the final report's Caveats section.
- **Exact stats for sampled data.** When a file is too large to hold in full
  for charting, DuckDB computes missing/duplicate/cardinality counts and
  numeric aggregates (mean/median/std/outliers) against the *entire* dataset
  via SQL, while charts are drawn from a bounded sample. This is stated
  explicitly in the report rather than silently assumed. Datetime trend and
  forecast analyses still run on the sample -- also stated explicitly.
- **Charts are capped and ranked**, not exhaustive -- at most 12 per section
  and 40 total, picked by variance/correlation/effect size, so a 500-column
  file doesn't produce a 500-chart report.
- **PDF vs HTML rendering.** HTML exports embed Plotly.js fully (works
  offline). PDFs render each chart to a static PNG via `kaleido` and assemble
  with WeasyPrint -- no headless browser, no JavaScript execution, no network
  calls, which is far more robust in locked-down deployments.

## Privacy & security

- Only CSV/TSV/XLSX/XLS/JSON are accepted; macro-enabled workbooks (.xlsm)
  are rejected outright.
- Uploaded `.xlsx` files are checked for zip-bomb-style compression ratios
  before being parsed.
- Every upload runs through a resource-capped subprocess pre-check
  (`engine/preparse_check.py`) with a CPU time limit, memory limit, and hard
  timeout before the real pipeline touches it. **This is not full OS-level
  sandboxing** (no seccomp/namespaces/filesystem jail) -- for full isolation,
  run the service in a container with a non-root user and read-only
  filesystem, which the provided Dockerfiles are set up for.
- No formulas or macros from uploaded files are ever executed.
- Uploaded files and generated reports are deleted after `RETENTION_HOURS`
  (default 24h) via `api/cleanup.py`.
- **Raw data is never sent to the LLM** -- only column-level aggregated
  statistics. No cell-level values leave the server.

## Repo structure

```
engine/                   # the actual product
  analyzers/               # numeric, categorical, datetime, relationships
  loader.py profiler.py cleaner.py charts.py narrative.py
  exact_stats.py optional_analysis.py exporters.py report_sections.py
  pipeline.py config.py security.py sandbox.py preparse_check.py
app/streamlit_app.py
api/                       # FastAPI + Postgres job tracking
templates/                 # Jinja2 HTML + PDF templates
tests/
  corpus/                   # generator for ugly test files
  eval_datasets/            # auto-fetched public datasets (gitignored)
scripts/                   # eval_type_detection.py, fetch_eval_datasets.py
docker-compose.yml Dockerfile.api Dockerfile.streamlit
```

## Known limitations (stated honestly)

- JSON large files don't get the DuckDB exact-stats treatment.
- Datetime trend/seasonality and forecasting run on the sample even when
  exact stats are used elsewhere (noted in the report's Caveats section).
- The sandbox covers the pre-parse check only, not the full pipeline, since
  the main pipeline holds a live DuckDB connection that can't cross a
  process boundary.
- Background processing uses FastAPI's `BackgroundTasks`, not Celery --
  fine for single-worker demo traffic, swap in Celery for concurrent
  large-file processing at scale.

## Roadmap

- [ ] Celery worker swap-in for concurrent processing.
- [ ] S3-backed storage for multi-instance deployment.
- [ ] DuckDB semantic layer + natural-language query copilot on top of the
      uploaded dataset (see original plan, Section 7).
