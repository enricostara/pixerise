# https://github.com/casey/just

# comment to not use powershell (for Linux, MacOS, and the BSDs)
# set shell := ["powershell.exe", "-c"]

@default: 
	just --list --unsorted

dev-sync:
    pdm sync --clean --dev

prod-sync:
	pdm sync --clean --prod

install-pre-commit:
	pdm run pre-commit install

format:
	pdm run ruff format

lint:
	pdm run ruff check --fix

test:
	pdm run pytest --verbose --color=yes tests

doc:
	pydoc-markdown --render-toc -I src > src/README.md

example:
	pdm run python examples/rendering_obj_file.py

validate: format lint test

bench:
  for file in `ls ./benchmarks`; do pdm run python benchmarks/$file; done

cache:
	find . -type d -name "__pycache__" -exec rm -r {} + && find . -type f -name "*.pyc" -delete

build: cache
	pdm build

publish:
	pdm publish --no-build