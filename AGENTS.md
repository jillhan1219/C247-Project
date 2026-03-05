# AGENTS.md - Agent Coding Guidelines

This document provides guidance for AI agents working on the emg2qwerty codebase.

## Project Overview

emg2qwerty is a Python project for modeling QWERTY typing from surface electromyography (EMG). It uses PyTorch, PyTorch Lightning, and implements CTC-based decoders for gesture/typing recognition.

## Build, Lint, and Test Commands

### Environment Setup
```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate emg2qwerty

# Install package in editable mode
pip install -e .
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest emg2qwerty/tests/decoder_test.py

# Run a single test function
pytest emg2qwerty/tests/decoder_test.py::test_logsumexp

# Run tests with coverage
pytest --cov=emg2qwerty --cov-report=html

# Run tests in parallel (faster)
pytest -n auto

# Run tests with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_greedy"

# Re-run failed tests on next run
pytest --reruns 3
```

### Linting and Formatting
```bash
# Run pre-commit checks on all files
pre-commit run --all-files

# Run individual linters
black --check .           # Check formatting
black .                   # Apply formatting
ufmt check .              # Check import sorting + formatting
ufmt format .             # Apply import sorting + formatting
flake8 .                  # Lint
ruff check .               # Lint and auto-fix
ruff check --fix .
mypy .                    # Type checking
```

### Hydra Configuration
This project uses Hydra for configuration management. Configs are in `config/` and `hydra_configs/`. Override config values via CLI:
```bash
python -m emg2qwerty.train key1=value1 key2=value2
```

## Code Style Guidelines

### General Principles
- Follow existing code patterns in the repository
- Keep functions focused and small
- Write docstrings for public APIs
- Use type hints consistently
- Run linting tools before committing

### Imports
- Use `from __future__ import annotations` at the top of all files
- Group imports in this order (use ufmt/black formatting):
  1. Standard library
  2. Third-party libraries
  3. Local application imports
- Use absolute imports: `from emg2qwerty.charset import CharacterSet`
- Use type annotations from `typing` and `collections.abc` for generics

### Formatting
- Maximum line length: 88 characters (Black default)
- Use Black for all formatting
- Use ufmt (Black + usort) for import sorting
- One blank line between top-level definitions
- No trailing whitespace

### Type Hints
- Use Python 3.10+ type hints (e.g., `list[str]` instead of `List[str]`)
- Use `typing.ClassVar` for class variables in dataclasses
- Use `typing.Any` sparingly
- Run mypy to verify types: `mypy .`
- Configure mypy in `setup.cfg`

### Naming Conventions
- Classes: `PascalCase` (e.g., `CharacterSet`, `EMGSessionData`)
- Functions/methods: `snake_case` (e.g., `logsumexp`, `str_to_labels`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `HDF5_GROUP`)
- Private methods: prefix with underscore (e.g., `_post_init`)
- Type aliases: `PascalCase` (e.g., `UniChar = str`)

### Dataclasses
- Use `@dataclass` for data containers
- Use `field(default_factory=...)` for mutable defaults
- Use `ClassVar` for class-level constants
- Define `__post_init__` for validation

### Error Handling
- Use assertions for internal invariants
- Raise specific exceptions with clear messages
- Handle exceptions at appropriate boundaries
- Use `pytest.raises()` for testing exceptions

### Testing
- Use pytest as the test framework
- Use Hypothesis for property-based testing
- Place tests in `emg2qwerty/tests/` directory
- Name test files: `*_test.py`
- Name test functions: `test_*`
- Use `@pytest.mark.parametrize` for parameterized tests
- Use descriptive docstrings for test functions

### Code Headers
All source files must include the Meta license header:
```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
```

### Pre-commit Hooks
This project uses pre-commit. Install hooks with:
```bash
pre-commit install
```

Hooks include:
- ufmt (Black + usort)
- flake8
- mypy
- ruff
- Standard checks (YAML, TOML, trailing whitespace, etc.)

## Key Dependencies
- torch, pytorch-lightning (ML framework)
- hydra-core, omegaconf (configuration)
- numpy, scipy (numerical computing)
- pytest, hypothesis (testing)
- kenlm (language modeling)
- flake8, black, mypy, ruff (linting/type checking)

## File Structure
```
emg2qwerty/
  __init__.py
  charset.py       # Character set handling
  data.py          # Data loading (HDF5)
  decoder.py       # CTC decoders
  transforms.py    # Data transforms
  modules.py       # Neural network modules
  lightning.py     # PyTorch Lightning wrappers
  metrics.py       # Evaluation metrics
  utils.py         # Utilities
  train.py         # Training script
  tests/           # Test suite
config/            # Hydra configs
hydra_configs/     # Additional Hydra configs
```
