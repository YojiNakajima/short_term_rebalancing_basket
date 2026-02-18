"""Compatibility shim for legacy import path.

The implementation was moved to `interfaces.cli.build_initial_portfolio` (CLI entry).
This module keeps `import infrastructure.mt5.build_initial_portfolio` working for existing code/tests.

Important: This shim must resolve to the *same module object* as the implementation so that
`unittest.mock.patch("infrastructure.mt5.build_initial_portfolio.*")` affects runtime behavior.
"""

from __future__ import annotations

import sys
from importlib import import_module

_impl = import_module("interfaces.cli.build_initial_portfolio")

# Make this import path an alias to the implementation module.
sys.modules[__name__] = _impl
