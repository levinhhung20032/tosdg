"""Pinhole camera in Unreal world conventions.

World: X forward, Y right, Z up, centimetres. Camera looks along its local +X,
local +Y is image-right, local +Z is image-up (UE CameraComponent convention).
Orientation is an Unreal FRotator (pitch, yaw, roll in degrees); the rotation
matrix matches FRotationMatrix: R = Rz(yaw) @ Ry(-pitch) @ Rx(-roll), whose
columns are the camera's forward/right/up axes in world space.

Projection (pinhole, then the optional radial lens distortion of `lens.Lens`):
    p_cam = R^T (p_world - position)          # (depth, right, up)
    u = cx + fx * right / depth
    v = cy - fy * up    / depth               # image v grows downward
    (u, v) <- Lens.distort(u, v)              # only when the camera has a lens with k1 / k2 != 0
Pixel coordinates use the convention that pixel (i, j) covers [i, i+1) x [j, j+1),
i.e. the image spans u in [0, W], v in [0, H].

Contents:
  rotator_matrix
  class Camera
  look_at_rotator — (pitch, yaw) so that the camera's +X points from eye to target
  sample_camera — Place a camera around the officer according to a camera…
  constants: NEAR_SPREAD
  (+ 1 private helper)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from . import rng as rngmod
from .lens import Lens
from .skeleton import axis_rotation


def rotator_matrix(pitch: float, yaw: float, roll: float) -> np.ndarray:
    return axis_rotation("z", yaw) @ axis_rotation("y", -pitch) @ axis_rotation("x", -roll)


@dataclass
class Camera:
    profile: str
    position: tuple[float, float, float]
    pitch: float
    yaw: float
    roll: float
    focal_length_mm: float
    sensor_width_mm: float
    width: int
    height: int
    lens: Lens | None = None        # radial distortion + black image-circle border; None = ideal pinhole

    @property
    def sensor_height_mm(self) -> float:
        return self.sensor_width_mm * self.height / self.width

    @property
    def fx(self) -> float:
        return self.focal_length_mm * self.width / self.sensor_width_mm

    @property
    def fy(self) -> float:
        return self.fx  # square pixels

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    @property
    def hfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.sensor_width_mm / (2 * self.focal_length_mm))))

    @property
    def vfov_deg(self) -> float:
        return float(np.degrees(2 * np.arctan(self.sensor_height_mm / (2 * self.focal_length_mm))))

    @property
    def R(self) -> np.ndarray:
        return rotator_matrix(self.pitch, self.yaw, self.roll)

    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def to_camera(self, pts_world: np.ndarray) -> np.ndarray:
        """World points (N,3) -> camera frame (N,3) as (depth, right, up)."""
        return (np.atleast_2d(pts_world) - np.asarray(self.position)) @ self.R

    def project_pinhole(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Ideal pinhole projection (no lens). Returns (uv (N,2), depth (N,)); uv is NaN for points at/behind the camera."""
        pc = self.to_camera(pts_world)
        depth = pc[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.fx * pc[:, 1] / depth
            v = self.cy - self.fy * pc[:, 2] / depth
        uv = np.stack([u, v], axis=1)
        uv[depth <= 1e-3] = np.nan
        return uv, depth

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (uv (N,2), depth (N,)) as the lens camera sees them. uv is NaN for points at/behind the camera
        (and for points the lens cannot map, far outside the field of view)."""
        uv, depth = self.project_pinhole(pts_world)
        if self.lens is not None and self.lens.distorts:
            uv = self.lens.distort(uv, self.cx, self.cy, self.fx)
        return uv, depth

    def undistort_uv(self, uv: np.ndarray) -> np.ndarray:
        """Pixel positions in the final (distorted) image -> the pinhole pixel positions that see the same ray."""
        uv = np.asarray(uv, float)
        if self.lens is None or not self.lens.distorts:
            return uv
        return self.lens.undistort(uv, self.cx, self.cy, self.fx)[0]

    def in_image(self, uv: np.ndarray) -> np.ndarray:
        """True where a pixel position is inside the frame and inside the lens' lit image circle (False for NaN)."""
        uv = np.asarray(uv, float)
        with np.errstate(invalid="ignore"):
            ok = np.isfinite(uv).all(axis=-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= self.width) & (uv[..., 1] >= 0) & (uv[..., 1] <= self.height)
        if self.lens is not None:
            ok &= self.lens.inside_circle(uv, self.width, self.height)
        return ok

    @property
    def horizon_v(self) -> float:
        """Image row of the horizon at the middle column, with the lens' distortion (roll ignored)."""
        v0 = self.cy + self.fy * float(np.tan(np.radians(self.pitch)))
        if self.lens is None or not self.lens.distorts:
            return v0
        v = self.lens.distort(np.array([self.cx, v0]), self.cx, self.cy, self.fx)[1]
        return float(v) if np.isfinite(v) else v0

    def to_metadata(self) -> dict:
        d = asdict(self)
        d["lens"] = (self.lens or Lens()).to_dict()
        d["position"] = [round(float(x), 4) for x in self.position]
        d.update({"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy,
                  "hfov_deg": self.hfov_deg, "vfov_deg": self.vfov_deg,
                  "sensor_height_mm": self.sensor_height_mm,
                  "K": self.K().tolist(),
                  "R_world_from_camera": self.R.tolist(),
                  "convention": "UE world (X fwd, Y right, Z up, cm); FRotator degrees; pinhole + radial lens distortion (see lens)"})
        return d


def look_at_rotator(eye, target) -> tuple[float, float]:
    """(pitch, yaw) so that the camera's +X points from eye to target."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = float(np.degrees(np.arctan2(d[1], d[0])))
    pitch = float(np.degrees(np.arctan2(d[2], np.hypot(d[0], d[1]))))
    return pitch, yaw


def sample_camera(profile: dict, rng: np.random.Generator, officer_pos, officer_yaw_deg: float,
                  width: int, height: int, min_subject: tuple[float, float] | None = None) -> tuple[Camera, dict]:
    """Place a camera around the officer according to a camera profile.

    Azimuth is in the officer's frame: 0 = in front of the officer, +90 = the
    officer's right. Returns the camera and the sampled placement parameters.
    min_subject = (subject height cm, minimum height px): the distance is mapped into a range near enough for a subject of
    that height to be at least that tall in the picture (closer than the profile's range when it has to). Same draws, so
    the other placement values do not change."""
    dist = rngmod.uniform(rng, profile["distance_cm"])
    h = rngmod.uniform(rng, profile["height_cm"])
    az = rngmod.uniform(rng, profile["azimuth_deg"])
    focal = rngmod.uniform(rng, profile["focal_length_mm"])
    if min_subject is not None:
        dist = _near_enough(dist, profile["distance_cm"], focal * width / float(profile["sensor_width_mm"]), *min_subject)
    aim_h = rngmod.uniform(rng, profile.get("aim_height_cm", [100, 100]))
    jy = rngmod.uniform(rng, profile.get("aim_jitter_deg", [0, 0]))
    jp = rngmod.uniform(rng, profile.get("aim_jitter_deg", [0, 0]))
    roll = rngmod.uniform(rng, profile.get("roll_deg", [0, 0]))

    o = np.asarray(officer_pos, float)
    world_az = np.radians(officer_yaw_deg + az)
    eye = o + np.array([dist * np.cos(world_az), dist * np.sin(world_az), 0.0])
    eye[2] = o[2] + h
    pitch, yaw = look_at_rotator(eye, o + np.array([0, 0, aim_h]))
    cam = Camera(profile["id"], tuple(float(x) for x in eye), pitch + jp, yaw + jy, roll,
                 focal, float(profile["sensor_width_mm"]), width, height)
    placement = {"distance_cm": dist, "height_cm": h, "azimuth_deg": az, "aim_height_cm": aim_h,
                 "aim_jitter_yaw_deg": jy, "aim_jitter_pitch_deg": jp}
    return cam, placement


NEAR_SPREAD = 0.6       # forced closer than the profile allows: distances in [0.6, 1] x the cap, so sizes still vary


def _near_enough(dist: float, rng_cm, fx: float, subject_cm: float, min_px: float) -> float:
    """Map a distance drawn uniformly from rng_cm = [lo, hi] into [lo', hi'] with hi' = the farthest distance at which the
    subject is min_px tall (pinhole: fx * subject / distance), keeping its relative position in the range."""
    lo, hi = float(rng_cm[0]), float(rng_cm[1])
    cap = fx * subject_cm / min_px
    if cap >= hi:
        return dist
    t = (dist - lo) / (hi - lo) if hi > lo else 0.0
    new_lo = lo if cap >= lo else NEAR_SPREAD * cap
    return new_lo + t * (cap - new_lo)
