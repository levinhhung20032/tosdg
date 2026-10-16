"""Brightness controls cannot blow up a render or make the officer invisible (schema limits, exposure clamp, render check).

Contents:
  BoundsTests (3 tests)
"""
import tempfile
import unittest
from pathlib import Path

from helpers import make_config
from test_cli import FAST
from test_people import generate
from tosdg.scene import EXPOSURE_RANGE, GenerationContext
from tosdg.schema import SchemaError


class BoundsTests(unittest.TestCase):
    def test_absurd_lighting_ranges_are_refused(self):
        for bad in ({"exposure_ev": [-2000, 2000]}, {"sun_intensity": [-1, 1]}, {"ambient_intensity": [0, 50]}):
            with tempfile.TemporaryDirectory() as td:
                with self.assertRaises(SchemaError):
                    GenerationContext(make_config(Path(td), lighting=bad, **FAST))

    def test_uniform_brightness_out_of_range_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(SchemaError):
                GenerationContext(make_config(Path(td), uniform_brightness_range=[0.01, 5.0]))

    def test_the_final_exposure_is_clamped(self):
        with tempfile.TemporaryDirectory() as td:
            for m in generate(Path(td), num_samples=2, lighting={"exposure_ev": [3, 3]}):
                self.assertLessEqual(m["lighting"]["exposure_ev"], EXPOSURE_RANGE[1])


if __name__ == "__main__":
    unittest.main()
