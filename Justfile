# https://github.com/casey/just

# uncomment to use powershell
set shell := ["powershell.exe", "-c"]

dev-sync:
    pdm sync --clean --dev

prod-sync:
	pdm sync --clean --prod

install-hooks:
	uv run pre-commit install

format:
	pdm run ruff format

lint:
	pdm run ruff check --fix

test:
	pdm run pytest --verbose --color=yes tests

validate: format lint test
