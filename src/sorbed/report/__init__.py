"""Report assembly: canonical JSON, annotated images, HTML, and PDF.

The PDF path (``build_grading_report_html`` / ``build_followup_report_html`` →
:func:`sorbed.report.pdf.html_to_pdf`) grounds each report in a clinical
directive pack and renders a modern, self-contained document via headless
Chromium. ``pdf`` is an optional extra; importing the builders is cheap and the
Chromium dependency is only touched when a PDF is actually rendered.
"""

from __future__ import annotations

from sorbed.report.builder import write_report
from sorbed.report.html_report import build_html
from sorbed.report.pdf import PdfRenderError, html_to_pdf
from sorbed.report.templates import (
    build_followup_report_html,
    build_grading_report_html,
    build_summary_report_html,
)

__all__ = [
    "PdfRenderError",
    "build_followup_report_html",
    "build_grading_report_html",
    "build_html",
    "build_summary_report_html",
    "html_to_pdf",
    "write_report",
]
