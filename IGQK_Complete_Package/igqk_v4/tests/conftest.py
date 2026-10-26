"""
Pytest configuration for IGQK v4 tests.
Ensures repository and package paths are resolvable regardless of working directory.
"""

import sys
import os
from pathlib import Path

test_dir = Path(__file__).resolve().parent
v4_dir = test_dir.parent
complete_pkg_dir = v4_dir.parent
workspace_root = complete_pkg_dir.parent
v1_dir = complete_pkg_dir / "igqk"

for p in [str(workspace_root), str(complete_pkg_dir), str(v4_dir), str(v1_dir)]:
    if p not in sys.path and os.path.exists(p):
        sys.path.insert(0, p)
