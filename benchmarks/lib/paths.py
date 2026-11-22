"""Path resolution and runtime directory helpers for Star Hunters benchmarks."""

from __future__ import annotations

import os
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "results"
DATA_DIR = BENCH_DIR / "data"
DEFAULT_BACKEND_DIR = BENCH_DIR.parent / "backend"


def ensure_runtime_dirs() -> None:
    """Create the git-ignored results/ and data/ directories if absent."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_backend_dir(backend_dir: Path | str | None = None) -> Path:
    """Resolve the Star Hunters backend directory under test.

    Resolution order:
    1. Explicit ``--backend-dir`` CLI argument.
    2. ``STAR_HUNTERS_BACKEND_DIR`` environment variable.
    3. Repository-local ``BENCH_DIR.parent / "backend"``.
    """
    raw = backend_dir or os.environ.get("STAR_HUNTERS_BACKEND_DIR")
    candidate = Path(raw).expanduser().resolve() if raw else DEFAULT_BACKEND_DIR
    if not (candidate / "app").is_dir():
        raise FileNotFoundError(
            f"Backend directory '{candidate}' does not contain an 'app/' package."
        )
    return candidate
