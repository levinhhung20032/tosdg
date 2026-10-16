import unittest

from PIL import Image

from tosdg.horizon import detect_horizon


class HorizonTest(unittest.TestCase):
    def test_sky_ground_split(self):
        im = Image.new("RGB", (320, 200), (90, 90, 90))
        im.paste((150, 190, 240), (0, 0, 320, 80))
        self.assertAlmostEqual(detect_horizon(im), 0.4, delta=0.03)

    def test_flat_image_has_no_horizon(self):
        self.assertIsNone(detect_horizon(Image.new("RGB", (320, 200), (120, 120, 120))))


if __name__ == "__main__":
    unittest.main()
