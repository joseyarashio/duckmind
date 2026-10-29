"""Parts of the repository that not every checkout has (not a test file).

The public repository leaves out the research records (``docs/design``, the measurement
evidence under ``docs/validation/evidence``) and the owner's own measurement tools under
``tools/``; tests that read them skip there instead of failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "docs" / "design"
EVIDENCE = ROOT / "docs" / "validation" / "evidence"
TOOLS = ROOT / "tools"

needs_design = pytest.mark.skipif(
    not DESIGN.is_dir(), reason="the design records (docs/design) are not in this checkout"
)
needs_evidence = pytest.mark.skipif(
    not EVIDENCE.is_dir(),
    reason="the measurement records (docs/validation/evidence) are not in this checkout",
)


def has_tool(name: str) -> bool:
    """Whether ``tools/<name>`` (a file or a folder) is in this checkout."""
    return (TOOLS / name).exists()


def needs_tool(name: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not has_tool(name),
        reason=f"tools/{name} (an internal measurement tool) is not in this checkout",
    )
