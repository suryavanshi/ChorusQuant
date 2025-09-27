"""Run mypy against the schema module to guarantee type safety annotations."""
from __future__ import annotations

from pathlib import Path

import pytest

mypy = pytest.importorskip("mypy.api")


def test_schema_module_type_checks_cleanly() -> None:
    result = mypy.run([str(Path("agents/normalize/schema.py"))])
    stdout, stderr, exit_status = result
    assert stderr == ""
    assert exit_status == 0, f"mypy reported issues:\n{stdout}"
