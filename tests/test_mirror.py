"""tools/mirror_commit.py: a commit of one tosdg copy (team repository sub-folder / personal repository root) is mirrored into
the other with the same files, message and author.

Contents:
  MirrorTests (3 tests)
"""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helpers import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import mirror_commit  # noqa: E402


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True, text=True).stdout


def new_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    git(path, "init", "-q")
    git(path, "config", "user.name", "Tester")
    git(path, "config", "user.email", "tester@example.com")
    git(path, "config", "core.autocrlf", "false")
    return path


@unittest.skipUnless(shutil.which("git"), "git not installed")
class MirrorTests(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        td = Path(self._td.name)
        self.team = new_repo(td / "team")
        self.p44 = self.team / "source-tool" / "tosdg"
        self.p44.mkdir(parents=True)
        (self.team / "outside.txt").write_text("not tosdg\n")
        for d, text in ((self.p44, "a\n"),):
            (d / "a.txt").write_text(text)
            (d / "old.txt").write_text("old\n")
            (d / ".gitignore").write_text("/local.md\n")
        git(self.team, "add", "-A")
        git(self.team, "commit", "-q", "-m", "start")
        self.personal = new_repo(td / "personal")
        for name in ("a.txt", "old.txt", ".gitignore"):
            shutil.copy(self.p44 / name, self.personal / name)
        (self.personal / "local.md").write_text("local only\n")          # ignored: never touched
        git(self.personal, "add", "-A")
        git(self.personal, "commit", "-q", "-m", "start")

    def tearDown(self):
        self._td.cleanup()

    def test_team_commit_goes_to_personal(self):
        (self.p44 / "a.txt").write_text("a2\n")
        (self.p44 / "sub").mkdir()
        (self.p44 / "sub" / "b.txt").write_text("b\n")
        git(self.team, "rm", "-q", "source-tool/tosdg/old.txt")
        git(self.team, "add", "-A")
        git(self.team, "commit", "-q", "-m", "source-tool/tosdg: change a, add b\n\nbody line")
        self.assertIsNotNone(mirror_commit.run(self.p44, self.personal))
        self.assertEqual((self.personal / "a.txt").read_text(), "a2\n")
        self.assertEqual((self.personal / "sub" / "b.txt").read_text(), "b\n")
        self.assertFalse((self.personal / "old.txt").exists())
        self.assertEqual((self.personal / "local.md").read_text(), "local only\n")
        self.assertFalse((self.personal / "outside.txt").exists())                   # only the tosdg folder
        self.assertEqual(git(self.personal, "log", "-1", "--format=%s|%an").strip(), "change a, add b|Tester")
        self.assertEqual(git(self.personal, "log", "-1", "--format=%b").strip(), "body line")
        self.assertIsNone(mirror_commit.run(self.p44, self.personal))                 # already the same

    def test_personal_commit_goes_to_the_team_folder(self):
        (self.personal / "a.txt").write_text("from personal\n")
        git(self.personal, "commit", "-q", "-am", "edit a")
        mirror_commit.run(self.personal, self.p44)
        self.assertEqual((self.p44 / "a.txt").read_text(), "from personal\n")
        self.assertEqual(git(self.team, "log", "-1", "--format=%s").strip(), "source-tool/tosdg: edit a")
        self.assertEqual((self.team / "outside.txt").read_text(), "not tosdg\n")

    def test_uncommitted_changes_stop_it(self):
        (self.personal / "a.txt").write_text("dirty\n")
        with self.assertRaises(SystemExit):
            mirror_commit.run(self.p44, self.personal)


if __name__ == "__main__":
    unittest.main()
