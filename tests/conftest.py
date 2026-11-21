"""Shared pytest bootstrap: put the project root on sys.path.

Allows tests to `import src.data_prep`, `import src.model`, `import src.explainer`
without installing the package. Kept intentionally tiny and side-effect free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
