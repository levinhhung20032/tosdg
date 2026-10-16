"""tools/fetch_backgrounds.py: the photos the library lists but this copy lacks are found through ATTRIBUTION.md and the
Wikimedia Commons API (faked here: no network), downloaded at the indexed width and checked against the indexed sha256.

Contents:
  FetchTests (5 tests)
"""
import json
import shutil
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path

from helpers import ROOT, fixture_library

sys.path.insert(0, str(ROOT / "tools"))
import fetch_backgrounds as fb  # noqa: E402
from tosdg.backgrounds import BackgroundLibrary  # noqa: E402

ROWS = """| File | Kích thước | Giấy phép | Tác giả | Nguồn |
|---|---|---|---|---|
| fx_dusk.png | 480×270 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | A | [Dusk (1).png](https://commons.wikimedia.org/wiki/File:Dusk_(1).png) |
| fx_day_clear.png | 480×270 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | B | [Clear.png](https://commons.wikimedia.org/wiki/File:Clear.png) |
"""


class FakeCommons:
    """API answers and file bytes: Dusk (1).png is 960 px wide on Commons (a 480 px thumbnail is served), Clear.png 480 px."""

    def __init__(self, files: dict[str, bytes]):
        self.files, self.calls = files, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url.startswith(fb.API):
            q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
            w = int(q["iiurlwidth"][0])
            pages = []
            for t in q["titles"][0].split("|"):
                if t == "File:Dusk (1).png":
                    pages.append({"title": t, "imageinfo": [{"url": "https://up/dusk_orig.png", "width": 960, "thumburl": f"https://up/dusk_{w}.png"}]})
                elif t == "File:Clear.png":
                    pages.append({"title": t, "imageinfo": [{"url": "https://up/clear.png", "width": 480}]})
                else:
                    pages.append({"title": t, "missing": True})
            return json.dumps({"query": {"pages": pages}}).encode()
        return self.files[url]


class FetchTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.lib_dir = Path(self._td.name) / "lib"
        shutil.copytree(fixture_library(), self.lib_dir)
        (self.lib_dir / "ATTRIBUTION.md").write_text(ROWS, encoding="utf-8")
        self.lib = BackgroundLibrary(self.lib_dir)
        self.dusk = self.lib.path(self.lib.by_id["fx_dusk"])
        self.clear = self.lib.path(self.lib.by_id["fx_day_clear"])
        self.dusk_bytes, self.clear_bytes = self.dusk.read_bytes(), self.clear.read_bytes()

    def tearDown(self):
        self._td.cleanup()

    def test_attribution_rows_with_brackets_in_the_url(self):
        self.assertEqual(fb.parse_attribution(ROWS), {"fx_dusk.png": "File:Dusk (1).png", "fx_day_clear.png": "File:Clear.png"})

    def test_missing_photos_are_downloaded_at_the_indexed_width(self):
        self.dusk.unlink()
        self.clear.unlink()
        get = FakeCommons({"https://up/dusk_480.png": self.dusk_bytes, "https://up/clear.png": self.clear_bytes})
        rc = fb.fetch(self.lib_dir, get=get, log=lambda *_: None)
        self.assertEqual(rc, 0)                                      # the two other fixtures are on disk: not needed
        self.assertEqual((self.dusk.read_bytes(), self.clear.read_bytes()), (self.dusk_bytes, self.clear_bytes))
        self.assertIn("https://up/dusk_480.png", get.calls)          # the thumbnail of the indexed width, not the original

    def test_nothing_to_do_when_everything_is_there(self):
        get = FakeCommons({})
        lines = []
        rc = fb.fetch(self.lib_dir, get=get, log=lines.append)
        self.assertEqual((rc, get.calls), (0, []))
        self.assertIn("all photos are here", lines[-1])

    def test_a_photo_without_a_source_is_reported(self):
        self.lib.path(self.lib.by_id["fx_elevated"]).unlink()
        lines = []
        self.assertEqual(fb.fetch(self.lib_dir, get=FakeCommons({}), log=lines.append), 1)
        self.assertTrue(any("fx_elevated: no Commons source" in x for x in lines))

    def test_different_bytes_are_reported(self):
        self.clear.unlink()
        lines = []
        rc = fb.fetch(self.lib_dir, get=FakeCommons({"https://up/clear.png": b"not the same photo"}), log=lines.append)
        self.assertEqual(rc, 1)
        self.assertTrue(any("differs from the indexed one" in x for x in lines))


if __name__ == "__main__":
    unittest.main()
