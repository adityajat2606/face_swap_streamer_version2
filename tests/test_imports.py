"""Module-boundary enforcement via AST (CLAUDE.md §4.2).

* No module imports from cli.py.
* flicker.py does not import the swap engine.
* quality_validator.py contains no detection/swap logic (no detector/swapper import).
* The package root imports without torch/cv2/insightface (import-light, §11.3).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src" / "face_swap"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)
    return names


@pytest.mark.parametrize("pyfile", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_no_module_imports_cli(pyfile: Path):
    if pyfile.name in ("cli.py", "__main__.py"):
        return
    imports = _imports(pyfile)
    assert not any("cli" in i.split(".")[-1] and i.endswith("cli") for i in imports), (
        f"{pyfile.name} imports cli"
    )
    assert "face_swap.cli" not in imports, f"{pyfile.name} imports face_swap.cli"


def test_flicker_does_not_import_swap_engine():
    imports = _imports(SRC / "flicker.py")
    assert not any("swap_engine" in i for i in imports)
    assert not any("face_detector" in i for i in imports)


def test_quality_validator_has_no_swap_or_detect_logic():
    imports = _imports(SRC / "quality_validator.py")
    assert not any("swap_engine" in i for i in imports)
    assert not any("face_detector" in i for i in imports)


def test_package_root_is_import_light():
    """Importing face_swap must not require torch/cv2/insightface."""
    root_imports = _imports(SRC / "__init__.py")
    for heavy in ("torch", "cv2", "insightface", "onnxruntime"):
        assert not any(heavy in i for i in root_imports)
