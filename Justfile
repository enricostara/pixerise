# https://github.com/casey/just

# comment to not use powershell (for Linux, MacOS, and the BSDs)
# set shell := ["powershell.exe", "-c"]

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
