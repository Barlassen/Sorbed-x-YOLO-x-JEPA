"""FastAPI service tests using a real in-memory image."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sorbed.api.app import create_app
from tests.synth import GRANULATION, make_wound


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_health(client: TestClient):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_formats_lists_png(client: TestClient):
    resp = client.get("/v1/formats")
    assert resp.status_code == 200
    assert resp.json().get("png") is True


def test_analyze_returns_grade(client: TestClient):
    wound = make_wound(fill=GRANULATION, seed=31)
    files = {"file": ("wound.png", wound.to_png_bytes(), "image/png")}
    resp = client.post("/v1/analyze", files=files, data={"mm_per_px": "0.2"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The analysis (possibly nested) must carry a decision stage.
    text = str(body)
    assert "stage" in text
    assert "disclaimer" in text.lower()


def test_analyze_rejects_non_image(client: TestClient):
    files = {"file": ("x.txt", b"this is not an image", "text/plain")}
    resp = client.post("/v1/analyze", files=files)
    assert resp.status_code in (415, 422)
