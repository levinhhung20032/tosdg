"""The v2 gesture list: the ten signals of the VN infographic (QCVN 41:2019/BGTVT) plus the no-signal hard negative.

Contents:
  GestureListTests (9 tests)
"""
import unittest

import numpy as np

from helpers import poses, skeleton, vn_gestures
from tosdg import rng
from tosdg.gestures import pose_validity
from tosdg.poses import merge
from tosdg.rng import weighted_choice
from tosdg.schema import load_json

NONE = "NONE"
# gesture id -> instruction for (front, back, left, right) of the officer, as read from the infographic / the user's description
EXPECTED = {
    "VN_SPEED_UP_LEFT": (NONE, NONE, "SPEED_UP", NONE),
    "VN_SPEED_UP_RIGHT": (NONE, NONE, NONE, "SPEED_UP"),
    "VN_SLOW_DOWN_LEFT": (NONE, NONE, "SLOW_DOWN", NONE),
    "VN_SLOW_DOWN_RIGHT": (NONE, NONE, NONE, "SLOW_DOWN"),
    "VN_STOP_LEFT": (NONE, NONE, "STOP", NONE),
    "VN_STOP_RIGHT": (NONE, NONE, NONE, "STOP"),
    "VN_STOP_FRONT": ("STOP", NONE, NONE, NONE),
    "VN_TURN_LEFT_IN_FRONT": (NONE, NONE, "TURN_LEFT", NONE),
    "VN_ARM_VERTICAL": ("STOP", "STOP", "STOP", "STOP"),                      # restored v1 gestures
    "VN_ARMS_HORIZONTAL": ("STOP", "STOP", "PROCEED", "PROCEED"),
    "VN_RIGHT_ARM_FORWARD": ("TURN_RIGHT", "STOP", "PROCEED_ANY_DIRECTION", "STOP"),
    "VN_NO_SIGNAL": (NONE, NONE, NONE, NONE),
    "VN_HELLO": (NONE, NONE, NONE, NONE),                                      # opt-in (weight 0), not traffic signals
    "VN_THUMB_UP": (NONE, NONE, NONE, NONE),
    "VN_A_POSE": (NONE, NONE, NONE, NONE),                                     # opt-in (weight 0): the model's own A-pose
}


class GestureListTests(unittest.TestCase):
    def test_hello_and_thumb_up_are_told_apart_by_the_hand(self):
        """Same arm, different hand: an open hand is HELLO only, a thumb-up hand THUMB_UP only (Studio's Gesture match showed both)."""
        lib, sk = vn_gestures(), skeleton("tripo_v1")
        for side, body in (("r", "hello_r"), ("l", "hello_l")):
            other = "l" if side == "r" else "r"
            for hand, expect in (("open_spread", "VN_HELLO"), ("thumb_up", "VN_THUMB_UP")):
                pose = merge(poses().body_pose(body), poses().hand_pose(hand, side), poses().hand_pose("relaxed", other))
                ok = sorted(gid for gid in ("VN_HELLO", "VN_THUMB_UP")
                            for v in lib.gestures[gid]["pose_variants"] if not pose_validity(sk, lib.gestures[gid], v, pose)[0])
                self.assertEqual(ok, [expect], (side, hand))
    @classmethod
    def setUpClass(cls):
        cls.lib, cls.sk, cls.pl = vn_gestures(), skeleton(), poses()

    def test_the_gesture_list_and_its_instructions(self):
        self.assertEqual(set(self.lib.gestures), set(EXPECTED))
        for gid, (f, b, l, r) in EXPECTED.items():
            self.assertEqual(self.lib.gestures[gid]["instructions"], {"front": f, "back": b, "left": l, "right": r}, gid)

    def test_instructions_are_in_the_profile_vocabulary(self):
        vocab = set(load_json(self.lib.poses.root.parent / "profiles" / "VN" / "profile.json", "profile")["regulation"]["instruction_vocabulary"])
        for g in self.lib.gestures.values():
            self.assertLessEqual(set(g["instructions"].values()), vocab, g["id"])

    def test_every_variant_samples_valid_poses_and_the_signalling_arm_matches_its_side(self):
        for gid, g in self.lib.gestures.items():
            for v in g["pose_variants"]:
                for i in range(25):
                    sp = self.sample_variant(g, v, i)
                    errs, fk = pose_validity(self.sk, g, v, sp)
                    self.assertEqual(errs, [], f"{gid}/{v['id']} seed {i}")
                    for arm in v["signal_arms"]:      # the signalling arm is not hanging down
                        hang = fk.positions[f"hand_{arm}"][2] < fk.positions["pelvis"][2] - 10 and abs(fk.positions[f"hand_{arm}"][1]) < 35
                        self.assertFalse(hang, f"{gid}/{v['id']}: {arm} arm hangs")

    def sample_variant(self, g, v, i):
        """A pose drawn from exactly this variant (the library picks variants by weight)."""
        solo = dict(g, pose_variants=[dict(v, weight=1.0)])
        lib = type(self.lib)({"schema_version": "1.0", "country": "VN", "common_variation": self.lib.common_variation, "gestures": [solo]}, self.pl)
        return lib.sample(g["id"], self.sk, rng.stream(500 + i, "pose")).pose

    def test_left_and_right_poses_are_mirror_images(self):
        for name in ("speed_up_ear", "speed_up_chest", "slow_down", "stop_bent", "stop_handup"):
            fl, fr = self.sk.fk(self.pl.body_pose(f"{name}_l")), self.sk.fk(self.pl.body_pose(f"{name}_r"))
            for joint in ("lowerarm", "hand"):
                la, rb = fl.positions[f"{joint}_l"], fr.positions[f"{joint}_r"]
                self.assertAlmostEqual(la[0], rb[0], delta=0.5, msg=name)
                self.assertAlmostEqual(la[1], -rb[1], delta=0.5, msg=name)       # y mirrored
                self.assertAlmostEqual(la[2], rb[2], delta=0.5, msg=name)

    def test_every_pose_is_used_by_a_gesture_and_the_restored_v1_poses_exist(self):
        self.assertLessEqual({"arm_forward_r", "arm_horizontal_l", "arm_horizontal_r", "arm_vertical_l", "arm_vertical_r", "arms_horizontal",
                              "neutral_stand", "stop_forward_r"}, set(self.pl.body))
        used = {v["pose"] for g in self.lib.gestures.values() for v in g["pose_variants"]}
        self.assertEqual(used, set(self.pl.body))                        # no pose is left without a gesture

    def test_hello_and_thumb_up_are_opt_in_gestures(self):
        """VN_HELLO / VN_THUMB_UP have weight 0: the default gesture draw (and so every existing seed) is unchanged."""
        for gid in ("VN_HELLO", "VN_THUMB_UP", "VN_A_POSE"):
            g = self.lib.gestures[gid]
            self.assertEqual((g["weight"], g["signalling"], set(g["instructions"].values())), (0, False, {"NONE"}))
            self.assertEqual({v["id"] for v in g["pose_variants"]}, {"default"} if gid == "VN_A_POSE" else {"right", "left"})
        weights = {k: float(v.get("weight", 1.0)) for k, v in self.lib.gestures.items()}
        self.assertEqual(sum(1 for w in weights.values() if w > 0), 12)
        self.assertNotIn(weighted_choice(rng.stream(1, "gesture"), weights), ("VN_HELLO", "VN_THUMB_UP", "VN_A_POSE"))

    def test_a_pose_is_the_models_own_bind_pose_and_is_valid(self):
        """a_pose.json copies the arm and leg DOFs of the beta model's bind pose (the A-pose it came in), so the posed model equals the original."""
        from tosdg.uniform_mesh import read_glb
        bind = read_glb(self.lib.poses.root.parent / "profiles" / "VN" / "assets" / "beta.glb").info["bind_pose"]
        pose = self.pl.body_pose("a_pose")
        for joint, dofs in pose.items():
            for dof, v in dofs.items():
                self.assertAlmostEqual(v, bind[joint][dof], delta=0.06, msg=f"{joint}.{dof}")
        full = merge(pose, self.pl.hand_pose("open_spread", "l"), self.pl.hand_pose("open_spread", "r"))
        self.assertEqual(self.sk.pose_errors(full), [])
        gest = self.lib.gestures["VN_A_POSE"]
        self.assertEqual(pose_validity(self.sk, gest, gest["pose_variants"][0], full)[0], [])

    def test_hello_and_thumb_up_poses_are_valid_and_the_thumb_is_up(self):
        for name in ("hello", "thumb_up"):
            for s in "rl":
                pose = self.pl.body_pose(f"{name}_{s}")
                hands = self.pl.hand_pose("thumb_up" if name == "thumb_up" else "open_spread", s)
                full = merge(pose, hands)
                self.assertEqual(self.sk.pose_errors(full), [], f"{name}_{s}")
                self.assertEqual(self.sk.collisions(self.sk.fk(full)), [], f"{name}_{s}")
        fk = self.sk.fk(merge(self.pl.body_pose("thumb_up_r"), self.pl.hand_pose("thumb_up", "r")))
        tip, base = fk.positions["thumb_03_r"], fk.positions["thumb_01_r"]
        self.assertGreater(tip[2] - base[2], 3.0)                         # the thumb points up from the fist

    def test_stop_signals_have_the_palm_turned_outwards(self):
        for gid, out in (("VN_STOP_LEFT", (0, -1, 0)), ("VN_STOP_RIGHT", (0, 1, 0))):
            g = self.lib.gestures[gid]
            for v in g["pose_variants"]:
                for i in range(10):
                    fk = self.sk.fk(self.sample_variant(g, v, i))
                    side = v["signal_arms"][0]
                    n = self.sk.palm_normal(fk, side)
                    self.assertGreater(float(n @ out), 0.6, f"{gid}/{v['id']}")

    def test_stop_front_arm_is_straight_forward_with_an_upright_untilted_hand(self):
        g = self.lib.gestures["VN_STOP_FRONT"]
        v = g["pose_variants"][0]
        for i in range(25):
            fk = self.sk.fk(self.sample_variant(g, v, i))
            arm = fk.positions["hand_r"] - fk.positions["upperarm_r"]
            fingers = fk.positions["middle_01_r"] - fk.positions["hand_r"]
            for vec, want, tol in ((arm, (1, 0, 0), 12), (fingers, (0, 0, 1), 10)):
                cos = float(vec @ np.array(want) / np.linalg.norm(vec))
                self.assertGreater(cos, np.cos(np.radians(tol)), f"seed {i}")
            self.assertGreater(float(self.sk.palm_normal(fk, "r") @ np.array([1.0, 0, 0])), 0.96, f"seed {i}")


if __name__ == "__main__":
    unittest.main()
