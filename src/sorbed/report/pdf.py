"""Render an HTML report to PDF with headless Chromium.

Chromium's print pipeline gives full modern-CSS control (grid, variable fonts,
background colours) that classic PDF toolkits lack. Playwright is imported
lazily so it is only required when a PDF is actually requested — install it via
the ``pdf`` extra.
"""

from __future__ import annotations

import os
from pathlib import Path


class PdfRenderError(RuntimeError):
    """Raised when the PDF backend is unavailable or fails to render."""


def _chromium_executable() -> str | None:
    """Locate a Chromium build, honouring a pinned env override.

    Managed environments sometimes ship a Chromium whose build differs from the
    one Playwright expects; ``SORBED_CHROMIUM`` pins an explicit binary.
    """
    override = os.environ.get("SORBED_CHROMIUM")
    if override and Path(override).exists():
        return override
    root = Path("/opt/pw-browsers")
    if root.is_dir():
        for pat in ("chromium-*/chrome-linux/chrome", "chromium/chrome-linux/chrome"):
            hits = sorted(root.glob(pat))
            if hits:
                return str(hits[0])
    return None


def html_to_pdf(
    html: str,
    out_path: str | Path,
    *,
    prefer_css_page_size: bool = True,
) -> Path:
    """Render ``html`` to a PDF file at ``out_path`` and return its path."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise PdfRenderError(
            "PDF rendering needs Playwright. Install the extra: pip install 'sorbed[pdf]' "
            "and run 'playwright install chromium'."
        ) from exc

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    exe = _chromium_executable()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                args=["--no-sandbox"], executable_path=exe
            )
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.emulate_media(media="print")
            page.pdf(
                path=str(out),
                print_background=True,
                prefer_css_page_size=prefer_css_page_size,
            )
            browser.close()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise PdfRenderError(f"Chromium failed to render the PDF: {exc}") from exc
    return out
