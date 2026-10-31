.PHONY: install install-dev test run-streamlit run-api docker-up corpus eval

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

corpus:
	python -m tests.corpus.generate_corpus

test: corpus
	pytest tests/ -v -k "not large_rows"

eval:
	python scripts/fetch_eval_datasets.py
	python scripts/eval_type_detection.py

run-streamlit:
	streamlit run app/streamlit_app.py

run-api:
	uvicorn api.main:app --reload

docker-up:
	docker compose up --build
