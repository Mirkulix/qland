"""Load the generalized qal-ast lowering."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[2] / "qal-ast" / "lower.py"
_spec = importlib.util.spec_from_file_location("qal_ast_lower", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
lower = _mod.lower
