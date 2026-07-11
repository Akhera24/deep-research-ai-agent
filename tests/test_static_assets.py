"""
Phase D2 — /static mount contract.

Assets in src/api/static/ are content-hashed (PLAN.md D2 / review R5): Starlette
StaticFiles sends no long-lived Cache-Control by itself, so far-future caching
is only safe because a changed asset gets a new name. These tests pin that
contract: immutable cache headers on hits, no cache poisoning on misses, no
traversal, and the naming rule that makes it all sound.
"""

import re
from pathlib import Path

import pytest

from tests.test_api_endpoints import client  # reuse the app fixture

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "api" / "static"
HASHED_NAME = re.compile(r"^[a-z0-9_-]+\.[0-9a-f]{8}\.[a-z0-9]+$")
IMMUTABLE = "public, max-age=31536000, immutable"
# The mount serves WITHOUT the report CSP — scriptable types (html, svg, js…)
# must never ship here; they'd dodge the /sample-report header contract.
SAFE_EXTENSIONS = {".webp", ".avif", ".png", ".jpg", ".jpeg", ".ico",
                   ".mp4", ".webm", ".vtt"}


def _any_asset() -> str:
    files = sorted(p.name for p in STATIC_DIR.iterdir() if p.is_file())
    assert files, "src/api/static/ must ship at least one asset"
    return files[0]


class TestStaticMount:
    def test_asset_served_with_immutable_cache_headers(self, client):
        r = client.get(f"/static/{_any_asset()}")
        assert r.status_code == 200
        assert r.headers["cache-control"] == IMMUTABLE
        assert r.headers["x-content-type-options"] == "nosniff"

    def test_webp_content_type(self, client):
        webps = [p.name for p in STATIC_DIR.iterdir() if p.suffix == ".webp"]
        assert webps, "expected the D2 webp assets to be present"
        r = client.get(f"/static/{webps[0]}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/webp"
        assert r.content[:4] == b"RIFF"  # actual bytes, not an error page

    def test_missing_asset_404_without_immutable_header(self, client):
        """A cached 404 under an immutable header would be forever-broken."""
        r = client.get("/static/nope.00000000.webp")
        assert r.status_code == 404
        assert r.headers.get("cache-control") != IMMUTABLE

    @pytest.mark.parametrize("path", [
        "/static/../main.py",
        "/static/%2e%2e/main.py",
        "/static/..%2fmain.py",
    ])
    def test_traversal_blocked(self, client, path):
        r = client.get(path)
        assert r.status_code in (400, 404)
        assert b"create_app" not in r.content

    def test_all_shipped_assets_are_content_hashed(self):
        """R5: far-future caching is only safe if every filename embeds a hash."""
        for p in STATIC_DIR.iterdir():
            if p.is_file() and not p.name.startswith("."):
                assert HASHED_NAME.match(p.name), (
                    f"{p.name} is not content-hashed (name.<sha8>.ext) — "
                    "unhashed files must not ship under the immutable mount"
                )

    def test_no_scriptable_types_under_the_cspless_mount(self):
        """OWASP A05: /static sends no CSP — html/svg/js here would bypass the
        sample-report header contract. Media/image types only."""
        for p in STATIC_DIR.iterdir():
            if p.is_file() and not p.name.startswith("."):
                assert p.suffix in SAFE_EXTENSIONS, (
                    f"{p.name}: {p.suffix} is scriptable or unvetted — serve it "
                    "through a route with explicit headers instead"
                )
