"""Download the background photos the library lists but this copy does not have (they are not in git: size and licences).

The library index (data/backgrounds/index.json, in git) says which photos were chosen, with their size and sha256;
ATTRIBUTION.md (in git) gives the Wikimedia Commons page of each. For every photo that is missing (or, with --force, whose
bytes differ) the Commons API gives the file at the indexed width (the 1920 px rendition, or the original when it is
that size), which is downloaded and checked against the indexed sha256. Ground masks that are missing are detected again
(SegFormer, needs torch + transformers: pip install -e .[detect]) when possible. The index is never changed.

    python tools/fetch_backgrounds.py [LIBRARY] [--force] [--no-masks] [--dry-run]

Exit code 0 when every photo (and mask) is there and matches the index, 1 otherwise.

Contents:
  constants: API, USER_AGENT
  parse_attribution — Commons file title of every photo in ATTRIBUTION.md
  missing — Library entries whose photo is missing or differs
  commons_urls — Download URL of each title at a given width
  fetch — Download the missing photos, then the missing masks
  main
  (+ 2 private helpers)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tosdg.backgrounds import BackgroundLibrary, sha256_file  # noqa: E402

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "tosdg-fetch-backgrounds/1.0 (https://github.com/levinhhung20032/tosdg)"
_ROW = re.compile(r"^\|\s*([^|]+?)\s*\|.*\]\((https://commons\.wikimedia\.org/wiki/\S+)\)\s*\|\s*$")   # the URL may hold "(...)"


def _get(url: str, tries: int = 4) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(2 + 3 * k)
    raise RuntimeError("unreachable")


def parse_attribution(text: str) -> dict[str, str]:
    """{file name in images/: Commons title "File:..."} from the table rows of ATTRIBUTION.md."""
    out = {}
    for line in text.splitlines():
        m = _ROW.match(line.strip())
        if m:
            page = urllib.parse.unquote(m.group(2).split("/wiki/", 1)[1])
            out[m.group(1)] = page.replace("_", " ")
    return out


def missing(lib: BackgroundLibrary, force: bool = False) -> list[dict]:
    """Entries whose photo file is missing (or, with force, differs from the indexed sha256)."""
    out = []
    for e in lib.index["backgrounds"]:
        p = lib.path(e)
        if not p.exists() or (force and sha256_file(p) != e["sha256"]):
            out.append(e)
    return out


def commons_urls(titles: list[str], width: int, get=_get) -> dict[str, str]:
    """{title: URL of the file at `width` px} (the original when it is not wider), 50 titles per request."""
    out = {}
    for i in range(0, len(titles), 50):
        q = {"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo", "iiprop": "url|size",
             "iiurlwidth": width, "format": "json", "formatversion": 2}
        data = json.loads(get(API + "?" + urllib.parse.urlencode(q)))
        norm = {n["to"]: n["from"] for n in data.get("query", {}).get("normalized", [])}
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [None])[0]
            if info:
                url = info["url"] if info["width"] <= width else info.get("thumburl", info["url"])
                out[norm.get(page["title"], page["title"])] = url
    return out


def _detect_masks(lib: BackgroundLibrary, log) -> list[str]:
    """Detect the ground masks whose file is missing (the index is not changed). Problems, as messages."""
    todo = [e for e in lib.index["backgrounds"] if e.get("ground") and not (lib.root / e["ground"]["file"]).exists()]
    if not todo:
        return []
    from PIL import Image
    from tosdg import ground as groundmod
    probs = []
    for e in todo:
        if not lib.path(e).exists():
            probs.append(f"{e['id']}: no photo, so no ground mask")
            continue
        try:
            with Image.open(lib.path(e)) as im:
                mask, _ = groundmod.detect_ground(im, e.get("horizon_y_frac"))
        except groundmod.GroundDetectorUnavailable as err:
            return probs + [f"{len(todo)} ground mask(s) missing and {err}"]
        dst = lib.root / e["ground"]["file"]
        groundmod.save_mask(mask, dst)
        same = sha256_file(dst) == e["ground"]["sha256"]
        log(f"{e['id']}: ground mask detected again" + ("" if same else " (differs from the indexed one: check it in the Studio)"))
        if not same:
            probs.append(f"{e['id']}: ground mask differs from the index")
    return probs


def fetch(library: Path, force: bool = False, masks: bool = True, dry_run: bool = False, get=_get, log=print) -> int:
    lib = BackgroundLibrary(library)
    attr = library / "ATTRIBUTION.md"
    sources = parse_attribution(attr.read_text(encoding="utf-8")) if attr.exists() else {}
    todo = missing(lib, force)
    log(f"{len(lib.index['backgrounds'])} photos in {library}; {len(todo)} to download")
    probs = []
    by_width: dict[int, list[dict]] = {}
    for e in todo:
        name = Path(e["file"]).name
        if name not in sources:
            probs.append(f"{e['id']}: no Commons source in ATTRIBUTION.md (added from elsewhere): copy {e['file']} by hand")
            continue
        by_width.setdefault(int(e["width"]), []).append(e)
    for width, entries in sorted(by_width.items()):
        urls = commons_urls([sources[Path(e["file"]).name] for e in entries], width, get)
        for e in entries:
            title = sources[Path(e["file"]).name]
            if title not in urls:
                probs.append(f"{e['id']}: {title} not found on Commons (deleted or renamed?)")
                continue
            if dry_run:
                log(f"{e['id']}: would download {urls[title]}")
                continue
            data = get(urls[title])
            dst = lib.path(e)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            ok = hashlib.sha256(data).hexdigest() == e["sha256"]
            log(f"{e['id']}: downloaded {len(data) // 1024} KB" + ("" if ok else " - DIFFERENT BYTES than the indexed photo"))
            if not ok:
                probs.append(f"{e['id']}: the downloaded photo differs from the indexed one (Commons re-made the file?); "
                             "its annotations may not fit it")
            time.sleep(0.3)
    if masks and not dry_run:
        probs += _detect_masks(lib, log)
    for p in probs:
        log("PROBLEM " + p)
    if dry_run:
        log(f"dry run: {len(todo)} photo(s) would be downloaded, {len(probs)} problem(s)")
        return 0 if not probs else 1
    log("all photos are here and match the index" if not probs else f"{len(probs)} problem(s)")
    return 0 if not probs else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("library", nargs="?", type=Path, default=ROOT / "data" / "backgrounds")
    ap.add_argument("--force", action="store_true", help="also download again photos whose bytes differ from the index")
    ap.add_argument("--no-masks", action="store_true", help="do not detect missing ground masks")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    a = ap.parse_args(argv)
    return fetch(a.library, a.force, not a.no_masks, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
