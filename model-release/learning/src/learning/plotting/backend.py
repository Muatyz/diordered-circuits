"""Matplotlib backend setup for batch figure generation."""

from __future__ import annotations


def use_headless_backend() -> None:
    """Use a non-GUI backend before importing pyplot."""
    import matplotlib

    matplotlib.use("Agg", force=True)
