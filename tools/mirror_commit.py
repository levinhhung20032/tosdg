"""Mirror the last commit of one tosdg copy into another and commit it there with the same message and author.

tosdg lives in two git repositories: the team repository (P44, as the sub-folder source-tool/tosdg) and a personal one
(tosdg at the root). Their histories differ (the team history must not be copied: it once held local-only files), so a
change is committed in one copy and then mirrored, file for file, into the other:

    python tools/mirror_commit.py DEST [--push] [--dry-run]

run from (or with --src pointing at) the copy that has the commit. The committed tree of the source tosdg folder (git
archive, exact bytes) replaces the tracked files of DEST: changed and new files are written, files tracked in DEST but no
longer in the source are removed, untracked / ignored files (photos, models, local notes) are left alone. The commit
message keeps its body; the "source-tool/tosdg: " subject prefix of the team repository is added or removed to match DEST.

Both working copies must be clean (no uncommitted change to tracked files, no untracked file that is not ignored).

Contents:
  run — Mirror the source's HEAD into DEST (returns the new commit id, or None when nothing changed)
  main
  (+ 4 private helpers)
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

TEAM_PREFIX = "source-tool/tosdg"


def _git(repo: Path, *args: str, binary: bool = False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"error: git {' '.join(args)} in {repo}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode("utf-8", errors="replace")


def _where(path: Path) -> tuple[Path, str]:
    """(top level of the repository, prefix of the tosdg folder inside it, "" or "source-tool/tosdg/")."""
    top = Path(_git(path, "rev-parse", "--show-toplevel").strip())
    return top, _git(path, "rev-parse", "--show-prefix").strip()


def _clean(path: Path) -> None:
    st = _git(path, "status", "--porcelain", "--", ".")
    if st.strip():
        raise SystemExit(f"error: {path} has uncommitted or untracked files; commit or remove them first:\n{st}")


def _message(msg: str, dest_prefix: str) -> str:
    subject, _, body = msg.partition("\n")
    label = TEAM_PREFIX + ": "
    if dest_prefix and not subject.startswith(label):
        subject = label + subject
    elif not dest_prefix and subject.startswith(label):
        subject = subject[len(label):]
    return subject + ("\n" + body if body else "")


def run(src: Path, dest: Path, push: bool = False, dry_run: bool = False) -> str | None:
    src_top, src_prefix = _where(src)
    dest_top, dest_prefix = _where(dest)
    _clean(src)
    _clean(dest)
    tree = f"HEAD:{src_prefix}" if src_prefix else "HEAD"
    archive = tarfile.open(fileobj=io.BytesIO(_git(src_top, "archive", "--format=tar", tree, binary=True)))
    files = {m.name: m for m in archive.getmembers() if m.isfile()}
    tracked = set(_git(dest, "ls-files", "-z").split("\0")) - {""}
    gone = sorted(tracked - set(files))
    if dry_run:
        print(f"{len(files)} files from {src} -> {dest}; would remove {len(gone)}: {gone}")
        return None
    for name, m in files.items():
        p = dest / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(archive.extractfile(m).read())
    for name in gone:
        _git(dest, "rm", "-q", "--", name)
    _git(dest, "add", "-A", "--", ".")
    if not _git(dest, "status", "--porcelain", "--", ".").strip():
        print(f"{dest}: already the same as {src} HEAD; nothing to commit")
        return None
    msg = _git(src, "log", "-1", "--format=%B").rstrip() + "\n"
    author = _git(src, "log", "-1", "--format=%an <%ae>").strip()
    date = _git(src, "log", "-1", "--format=%aI").strip()
    r = subprocess.run(["git", "-C", str(dest), "commit", "-q", "-F", "-", f"--author={author}", f"--date={date}"],
                       input=_message(msg, dest_prefix).encode("utf-8"), capture_output=True, check=False)
    if r.returncode != 0:
        raise SystemExit(f"error: commit in {dest}: {r.stderr.decode(errors='replace').strip()}")
    sha = _git(dest, "rev-parse", "--short", "HEAD").strip()
    print(f"{dest}: committed {sha} ({len(files)} files, {len(gone)} removed)")
    if push:
        _git(dest, "push", "-q")
        print(f"{dest}: pushed")
    return sha


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dest", type=Path, help="the other tosdg copy (its tosdg folder)")
    ap.add_argument("--src", type=Path, default=Path(__file__).resolve().parents[1], help="the copy with the commit (default: this one)")
    ap.add_argument("--push", action="store_true", help="push DEST's branch after committing")
    ap.add_argument("--dry-run", action="store_true", help="only list what would change")
    a = ap.parse_args(argv)
    run(a.src.resolve(), a.dest.resolve(), a.push, a.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
