PYTHON=3.8
BASENAME=$(shell basename $(CURDIR))
CONDA_CH=conda-forge defaults

env:
	conda create -n $(BASENAME)  python=$(PYTHON)

setup:
	conda install --file requirements.txt $(addprefix -c ,$(CONDA_CH))
	pre-commit install

format:
	black .
	isort .

lint:
	pytest src --flake8 --pylint --mypy

cov:
	open htmlcov/index.html
