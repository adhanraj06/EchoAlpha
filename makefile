PYTHON := python3
VENV_DIR := venv
REQS := src/requirements.txt
MAIN := main.py
BACKTEST_DIR := backtest/
ANALYSIS_DIR := ticker_analysis/
STUDY_DIR := study/
DATA_DIR := data/
DOCS_DIR := docs/
DOC_PORT := 8000

# Default target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  install    - Create virtual environment and install dependencies"
	@echo "  run        - Run the main analysis pipeline"
	@echo "  test       - Run tests (if any)"
	@echo "  lint       - Check code style with flake8"
	@echo "  format     - Format code with black"
	@echo "  clean      - Remove generated files and virtual environment"
	@echo "  clean-data - Remove only the data files"

# Installation and setup
.PHONY: install
install:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && \
		pip install --upgrade pip && \
		pip install -r $(REQS) && \
		pip install black flake8

# Main execution
.PHONY: run
run:
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(MAIN)

# Documentation
.PHONY: docs
docs:
	mkdir -p $(DOCS_DIR)
	. $(VENV_DIR)/bin/activate && \
		PYTHONPATH=.:./src pdoc -o $(DOCS_DIR) main.py src

# Code quality
.PHONY: lint
lint:
	. $(VENV_DIR)/bin/activate && \
		flake8 src/ --max-line-length=88 --exclude=__pycache__

.PHONY: format
format:
	. $(VENV_DIR)/bin/activate && \
		black src/ $(MAIN)

# Cleanup
.PHONY: clean
clean: clean-data
	rm -rf $(VENV_DIR)
	rm -rf **/__pycache__
	rm -rf .idea

.PHONY: clean-data
clean-data:
	rm -rf $(DOCS_DIR) $(DATA_DIR) $(BACKTEST_DIR) $(ANALYSIS_DIR) $(STUDY_DIR)

# Default target
.DEFAULT_GOAL := help