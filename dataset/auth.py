"""
auth.py — HuggingFace credential management for local model inference.

Loads HF_TOKEN from .env files in the project root on import, then exposes
:func:`resolve_hf_token` for use by any script that needs gated model access.

.env file layout (project root)
────────────────────────────────
  .env.huggingface  → HF_TOKEN
  .env              → generic fallback (does not override the above)

Load order: provider-specific file first, generic .env last.
Keys already set in the environment (or by an earlier file) are never
overridden.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Project root is one level above this file (dataset/../)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Load .env files at import time ──────────────────────────────────────────
try:
    from dotenv import load_dotenv as _load_dotenv

    for _env_file in (
        _PROJECT_ROOT / ".env.huggingface",
        _PROJECT_ROOT / ".env",          # generic fallback
    ):
        if _env_file.exists():
            _load_dotenv(_env_file, override=False)
            logger.debug(f"auth: loaded env from {_env_file.name}")
except ImportError:
    logger.debug("auth: python-dotenv not installed; skipping .env loading")


def resolve_hf_token(explicit_token: str | None = None) -> str | None:
    """
    Return the HuggingFace token for accessing gated models.

    Resolution order:
      1. *explicit_token* — e.g. a value passed via ``--hf-token`` on the CLI.
      2. The ``HF_TOKEN`` env var (set via ``.env.huggingface`` or system env).

    Returns ``None`` if no token is found (non-gated models work without one).

    Parameters
    ----------
    explicit_token:
        An override value supplied directly by the caller (optional).

    Returns
    -------
    str | None
        The resolved HuggingFace token, or ``None`` if not available.
    """
    if explicit_token:
        return explicit_token

    return os.environ.get("HF_TOKEN")


# Backward-compatible alias used by construct_dataset.py and generate.py
def resolve_api_key(backend: str = "huggingface", explicit_key: str | None = None) -> str | None:
    """Backward-compatible wrapper — delegates to :func:`resolve_hf_token`."""
    return resolve_hf_token(explicit_key)
