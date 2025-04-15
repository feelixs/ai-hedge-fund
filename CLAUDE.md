# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands
- Install dependencies: `poetry install`
- Run the hedge fund: `poetry run python src/main.py --ticker TICKER1,TICKER2`
- Run the backtester: `poetry run python src/backtester.py --ticker TICKER1,TICKER2`
- Use local LLMs: Add `--ollama` flag to the run commands

## Environment Setup
- Python version: 3.9+ (as specified in pyproject.toml)
- Architecture note: If running on Apple Silicon (M1/M2/M3), ensure Poetry is using an arm64 Python interpreter:
  ```
  # Check current Python architecture
  python -c "import platform; print(platform.machine())"
  
  # Configure Poetry to use arm64 Python if needed
  poetry env use /path/to/arm64/python
  poetry install
  ```
- Required API keys: Set in .env file (see README for details)

## Code Style Guidelines
- Line length: 420 characters (as per tool.black config)
- Formatting: Use black for code formatting
- Imports: Use isort for import sorting
- Linting: flake8 is used for linting
- Error handling: Use try/except with specific exception types and helpful error messages
- Type hints: Use Python type hints with imports from typing and pydantic
- Naming: snake_case for variables/functions, PascalCase for classes, and UPPER_CASE for constants
- Class definitions: Use Pydantic models for structured data with Field descriptions
- Docstrings: Include for functions and classes (use triple quotes)
- Agent pattern: Follow established agent structure with input/output handling

## Testing
- Tests should be placed in a `tests/` directory (create if not present)
- Use pytest for testing