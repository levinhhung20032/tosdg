"""Command line and dataset hygiene: argument checks, one-line errors, the output directory, portable paths, reproduce.

Contents:
  run_cli()
  OutputDirTests (4 tests)
  ArgumentTests (6 tests)
  PortablePathTests (4 tests)
  ReproduceTests (1 tests)
  BackgroundImportTests (5 tests)
  ProfileTests (1 tests)
"""
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from helpers import ROOT, fixture_library, make_config, poses
from tosdg.backgrounds import BackgroundLibrary
from tosdg.cli import main
from tosdg.generate import portable_path, prepare_output_dir, resolve_portable_path, run
from tosdg.scene import GenerationContext
from tosdg.schema import SchemaError, schema_errors
from tosdg.validate import validate_dataset

FAST = {"backend": "proxy", "uniform_brightness_range": [1.0, 1.0]}


def run_cli(*argv) -> tuple[int, str, str]:
    """(exit code, stdout, stderr) of `tosdg <argv>`; argparse errors (SystemExit) count as their exit code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = main([str(a) for a in argv])
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


class OutputDirTests(unittest.TestCase):
    def test_a_new_or_empty_directory_is_used(self):
        with tempfile.TemporaryDirectory() as td:
            prepare_output_dir(Path(td) / "new" / "deep")
            self.assertTrue((Path(td) / "new" / "deep").is_dir())
            prepare_output_dir(Path(td) / "new" / "deep")             # still empty: fine

    def test_an_existing_dataset_is_never_mixed_with_a_new_one(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ctx = GenerationContext(make_config(td, num_samples=2, **FAST))
            run(ctx, td / "ds", log=lambda *_: None)
            before = sorted(p.name for p in (td / "ds").iterdir())
            with self.assertRaisesRegex(ValueError, "not empty.*--overwrite"):
                run(GenerationContext(make_config(td, num_samples=1, **FAST)), td / "ds", log=lambda *_: None)
            self.assertEqual(sorted(p.name for p in (td / "ds").iterdir()), before)     # nothing touched

    def test_overwrite_replaces_the_dataset_and_leaves_a_valid_one(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            run(GenerationContext(make_config(td, num_samples=3, **FAST)), td / "ds", log=lambda *_: None)
            run(GenerationContext(make_config(td, num_samples=1, **FAST)), td / "ds", log=lambda *_: None, overwrite=True)
            self.assertEqual(validate_dataset(td / "ds"), [])
            self.assertEqual(len(json.loads((td / "ds" / "manifest.json").read_text())["samples"]), 1)

    def test_overwrite_refuses_a_directory_with_other_files(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td) / "mine"
            d.mkdir()
            (d / "notes.txt").write_text("keep me")
            (d / "x.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "not part of a tosdg dataset"):
                prepare_output_dir(d, overwrite=True)
            self.assertTrue((d / "notes.txt").exists() and (d / "x.json").exists())


class ArgumentTests(unittest.TestCase):
    def test_num_and_seed_are_checked_by_the_parser(self):
        for args in (["--num", "0"], ["--num", "-1"], ["--seed", "-5"], ["--num", "two"]):
            rc, _, err = run_cli("generate", ROOT / "configs" / "smoke.json", "--out", "unused", *args)
            self.assertEqual(rc, 2, args)
            self.assertIn("must be" if "two" not in args else "not an integer", err)

    def test_command_line_overrides_obey_the_config_schema(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(SchemaError, "overrides.*num_samples"):
                GenerationContext(make_config(td, **FAST), {"num_samples": 0})
            with self.assertRaisesRegex(SchemaError, "master_seed"):
                GenerationContext(make_config(td, **FAST), {"master_seed": -1})

    def test_user_mistakes_are_one_line_errors_not_tracebacks(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "bad.json").write_text("{not json")
            cases = [("generate", td / "missing.json", "--out", td / "o"), ("generate", td / "bad.json", "--out", td / "o"),
                     ("overlay", td / "no_dataset"), ("pose-preview", "no_such_pose", "--out", td / "p.png"),
                     ("validate-profile", td / "missing.json")]
            for argv in cases:
                rc, out, err = run_cli(*argv)
                self.assertNotEqual(rc, 0, argv)
                self.assertNotIn("Traceback", out + err, argv)
            self.assertIn("known:", run_cli("pose-preview", "no_such_pose", "--out", td / "p.png")[2])

    def test_config_ranges_backend_and_library_are_explained(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            with self.assertRaisesRegex(ValueError, r"officer_yaw_deg must be \[min, max\]"):
                GenerationContext(make_config(td, officer_yaw_deg=[90, -90], **FAST))
            with self.assertRaisesRegex(ValueError, r"lighting.sun_intensity must be \[min, max\]"):
                GenerationContext(make_config(td, lighting={"sun_intensity": [1.2, 0.8]}, **FAST))
            with self.assertRaisesRegex(ValueError, "backend 'unreal' is not implemented"):
                GenerationContext(make_config(td, backend="unreal", uniform_brightness_range=[1.0, 1.0]))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                GenerationContext(make_config(td, background_library=str(td / "nowhere"), **FAST))

    def test_import_model_refuses_a_name_without_letters(self):
        before = (ROOT / "profiles" / "VN" / "uniforms.json").read_bytes()
        rc, out, _ = run_cli("import-model", ROOT / "profiles" / "VN" / "models" / "beta.glb", "--name", "", "--no-activate")
        self.assertEqual(rc, 1)
        self.assertIn("no letter or digit", out)
        self.assertEqual((ROOT / "profiles" / "VN" / "uniforms.json").read_bytes(), before)

    def test_unknown_poses_name_the_known_ones(self):
        with self.assertRaisesRegex(ValueError, "unknown body pose 'nope'.*hello_r"):
            poses().body_pose("nope")
        with self.assertRaisesRegex(ValueError, "unknown hand pose 'nope'.*open_flat"):
            poses().hand_pose("nope", "r")


class PortablePathTests(unittest.TestCase):
    def test_paths_inside_the_repo_are_relative_and_outside_absolute(self):
        self.assertEqual(portable_path(ROOT / "data" / "backgrounds"), "data/backgrounds")
        with tempfile.TemporaryDirectory() as td:
            p = portable_path(td)
            self.assertNotIn("\\", p)
            self.assertTrue(Path(p).is_absolute())
            self.assertEqual(resolve_portable_path(p).resolve(), Path(td).resolve())
        self.assertEqual(resolve_portable_path("data/backgrounds"), ROOT / "data" / "backgrounds")

    def test_manifest_has_no_backslash_paths_and_validates(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            man = run(GenerationContext(make_config(td, num_samples=1, **FAST)), td / "ds", log=lambda *_: None)
            self.assertEqual(man["schema_version"], "1.1")
            self.assertEqual(schema_errors(man, "manifest"), [])
            self.assertFalse([k for k in man["asset_hashes"] if "\\" in k])
            self.assertNotIn("\\", man["background_library_resolved"])
            self.assertEqual(man["config"]["background_library"], man["background_library_resolved"])

    def test_old_manifests_still_validate(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            man = run(GenerationContext(make_config(td, num_samples=1, **FAST)), td / "ds", log=lambda *_: None)
            man["schema_version"] = "1.0"
            man["background_library_resolved"] = "C:\\Users\\someone\\lib"
            self.assertEqual(schema_errors(man, "manifest"), [])

    def test_library_index_uses_slashes_and_reads_old_backslash_entries(self):
        with tempfile.TemporaryDirectory() as td:
            lib = BackgroundLibrary(Path(td) / "lib")
            e = lib.add(next((fixture_library() / "images").iterdir()), ["t"], horizon_y_frac=0.4)
            self.assertEqual(e["file"], f"images/{e['id']}.png")
            lib.save()
            doc = json.loads(lib.index_path.read_text())
            doc["backgrounds"][0]["file"] = doc["backgrounds"][0]["file"].replace("/", "\\")     # as Windows wrote it before
            lib.index_path.write_text(json.dumps(doc))
            again = BackgroundLibrary(Path(td) / "lib")
            self.assertTrue(again.path(again.index["backgrounds"][0]).exists())
            self.assertEqual(again.integrity_errors(), [])


class ReproduceTests(unittest.TestCase):
    def test_reproduce_checks_the_files_on_disk_too(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            run(GenerationContext(make_config(td, num_samples=2, **FAST)), td / "ds", log=lambda *_: None)
            rc, out, _ = run_cli("reproduce", td / "ds")
            self.assertEqual(rc, 0, out)
            self.assertIn("2 identical", out)
            rgb = next((td / "ds").glob("*.rgb.png"))
            a = np.asarray(Image.open(rgb)).copy()
            a[0, 0] = (a[0, 0].astype(int) + 1) % 256
            Image.fromarray(a).save(rgb)
            rc, out, _ = run_cli("reproduce", td / "ds")
            self.assertEqual(rc, 1)
            self.assertIn("ON DISK", out)
            self.assertIn("rgb sha256 mismatch", out)


class BackgroundImportTests(unittest.TestCase):
    def image(self):
        return next((fixture_library() / "images").iterdir())

    def test_horizon_must_be_a_fraction(self):
        with tempfile.TemporaryDirectory() as td:
            lib = BackgroundLibrary(Path(td))
            for bad in (1.7, -0.2):
                with self.assertRaisesRegex(ValueError, r"in \[0, 1\]"):
                    lib.add(self.image(), [], horizon_y_frac=bad)
            self.assertEqual(lib.index["backgrounds"], [])
            rc, _, err = run_cli("backgrounds", "import", td, self.image(), "--horizon", "1.7")
            self.assertEqual(rc, 2)
            self.assertIn("[0, 1]", err)

    def test_a_file_that_is_not_an_image_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "x.png"
            bad.write_text("not an image")
            rc, out, err = run_cli("backgrounds", "import", Path(td) / "lib", bad)
            self.assertEqual(rc, 2)
            self.assertIn("not an image", err)
            self.assertNotIn("Traceback", out + err)

    def test_remove_takes_the_photo_and_its_mask_out_of_the_library(self):
        with tempfile.TemporaryDirectory() as td:
            lib_dir = Path(td) / "lib"
            shutil.copytree(fixture_library(), lib_dir)
            lib = BackgroundLibrary(lib_dir)
            gone = next(b for b in lib.index["backgrounds"] if b.get("ground"))
            files = [lib_dir / gone["file"], lib_dir / gone["ground"]["file"]]
            n = len(lib.index["backgrounds"])
            rc, out, _ = run_cli("backgrounds", "remove", lib_dir, gone["id"])
            self.assertEqual(rc, 0, out)
            lib = BackgroundLibrary(lib_dir)
            self.assertNotIn(gone["id"], lib.by_id)
            self.assertEqual(len(lib.index["backgrounds"]), n - 1)
            self.assertFalse(any(f.exists() for f in files))
            self.assertEqual(lib.integrity_errors(), [])
            rc, _, err = run_cli("backgrounds", "remove", lib_dir, "no_such_id")
            self.assertEqual(rc, 2)
            self.assertIn("no_such_id", err)

    def test_id_names_the_background(self):
        with tempfile.TemporaryDirectory() as td:
            rc, out, _ = run_cli("backgrounds", "import", td, self.image(), "--id", "my_street", "--horizon", "0.5")
            self.assertEqual(rc, 0, out)
            self.assertIn("my_street", BackgroundLibrary(td).by_id)
            rc, _, err = run_cli("backgrounds", "import", td, self.image(), self.image(), "--id", "two")
            self.assertEqual(rc, 2)
            self.assertIn("one at a time", err)

    def test_a_duplicate_id_is_refused_without_a_traceback(self):
        with tempfile.TemporaryDirectory() as td:
            run_cli("backgrounds", "import", td, self.image(), "--horizon", "0.5")
            rc, out, err = run_cli("backgrounds", "import", td, self.image(), "--horizon", "0.5")
            self.assertEqual(rc, 2)
            self.assertIn("already in library", err)
            self.assertNotIn("Traceback", out + err)


class ProfileTests(unittest.TestCase):
    def test_the_shipped_config_validates_with_its_own_character(self):
        """Every gesture's base pose must be valid on the skeleton the config uses (tripo_v1), not only on proxy_v1: the
        SLOW_DOWN poses collided with the torso once tripo_v1 got the model's narrower shoulders, and no test noticed."""
        with tempfile.TemporaryDirectory() as td:          # configs/smoke.json with the fixture library (data/ is not in the repo)
            rc, out, _ = run_cli("validate-profile", make_config(td))
        self.assertEqual(rc, 0, out)
        self.assertIn("-> OK", out)


if __name__ == "__main__":
    unittest.main()
