"""Sorbed — explainable pressure-injury (bedsore) image analysis and grading.

Sorbed is clinical **decision-support** software, not a medical device and not a
diagnostic tool. Every value it reports is computed from real image pixels; it
never invents a measurement. See ``DISCLAIMER.md`` before any clinical use.
"""

from __future__ import annotations

from sorbed.version import __version__

__all__ = ["__version__"]
