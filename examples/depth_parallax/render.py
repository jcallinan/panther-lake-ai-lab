"""2.5D re-projection of a single frame + relative inverse depth to an orbiting virtual camera (numpy/OpenCV).

Method ("forward-splat depth, backward-warp colour"):
  1. Lift every pixel to 3D: inverse depth d in [0, 1] -> Z = 1 / (a + (1 - a) d), X/Y from a pinhole
     camera. Rotate the cloud (yaw, pitch) about a pivot point at the scene's median depth and move the
     camera forward (dolly), then project.
  2. Forward-splat the new camera-space depth with a z-buffer (nearest wins) at reduced resolution.
     Fill holes from the *background* side (max-depth neighbour), so cracks close and disocclusions
     become stretched background rather than black; large disocclusions are also dimmed ("voids").
  3. For every target pixel, invert the exact 3D transform using its splatted depth to find the source
     pixel, then cv2.remap the full-resolution colour frame. Foreground edges stay crisp.
Monocular depth is relative (affine-invariant inverse depth), so the 3D is plausible, not metric.
"""
import math

import cv2
import numpy as np

NEAR_DISP = 0.25          # inverse depth of the farthest point (d = 0); sets the depth range (Z in 1..4)


class DepthNormalizer:
    """Map raw relative inverse depth to [0, 1] with EMA-smoothed percentiles (prevents flicker)."""

    def __init__(self, alpha: float = 0.12, lo_pct: float = 2.0, hi_pct: float = 98.0):
        self.alpha, self.lo_pct, self.hi_pct = alpha, lo_pct, hi_pct
        self.lo = self.hi = None

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(depth[::4, ::4], [self.lo_pct, self.hi_pct])
        if self.lo is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            self.lo += self.alpha * (float(lo) - self.lo)
            self.hi += self.alpha * (float(hi) - self.hi)
        return np.clip((depth - self.lo) / max(self.hi - self.lo, 1e-6), 0.0, 1.0).astype(np.float32)


def colorize(dn: np.ndarray, cmap=cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Normalised inverse depth -> BGR (bright = near)."""
    return cv2.applyColorMap((dn * 255).astype(np.uint8), cmap)


def _rotation(yaw: float, pitch: float) -> np.ndarray:
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    return rx @ ry


class ParallaxRenderer:
    """Re-projects frames of a fixed output size; one instance per size (caches pixel grids)."""

    def __init__(self, width: int, height: int, splat_scale: float = 0.5, fov_deg: float = 55.0,
                 overscan: float = 1.07):
        self.w, self.h = width, height
        self.sw, self.sh = int(width * splat_scale), int(height * splat_scale)
        self.f = 0.5 * width / math.tan(math.radians(fov_deg) / 2)
        self.ft = self.f * overscan                    # virtual camera zooms slightly: hides frame edges
        sx, sy = width / self.sw, height / self.sh
        # low-res grid expressed in full-res pixel coordinates (pixel centres)
        self.u, self.v = np.meshgrid((np.arange(self.sw, dtype=np.float32) + 0.5) * sx - 0.5,
                                     (np.arange(self.sh, dtype=np.float32) + 0.5) * sy - 0.5)
        self.kernel = np.ones((3, 3), np.uint8)
        self._kerns = {}

    def _kern(self, k: int):
        if k not in self._kerns:
            self._kerns[k] = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        return self._kerns[k]

    def _lift(self, d):
        z = 1.0 / (NEAR_DISP + (1.0 - NEAR_DISP) * d)
        return (self.u - self.w * 0.5) / self.f * z, (self.v - self.h * 0.5) / self.f * z, z

    def render(self, frame: np.ndarray, dn: np.ndarray, yaw: float, pitch: float, dolly: float = 0.0,
               pivot: float = 0.5, void_dim: float = 0.4):
        """frame: BGR (h, w); dn: normalised inverse depth (any size; bright/high = near).

        Returns (view BGR (h, w), target inverse depth at splat resolution in [0, 1]).
        """
        sw, sh, f, cx, cy = self.sw, self.sh, self.f, self.w * 0.5, self.h * 0.5
        d = cv2.resize(dn, (sw, sh), interpolation=cv2.INTER_AREA)
        rot = _rotation(yaw, pitch)
        zp = 1.0 / (NEAR_DISP + (1.0 - NEAR_DISP) * pivot)
        shift = np.float32(zp - dolly)                     # camera moves forward by `dolly` (Z units)
        x, y, z = self._lift(d)
        zc = z - zp
        x2 = rot[0, 0] * x + rot[0, 1] * y + rot[0, 2] * zc
        y2 = rot[1, 0] * x + rot[1, 1] * y + rot[1, 2] * zc
        z2 = np.maximum(rot[2, 0] * x + rot[2, 1] * y + rot[2, 2] * zc + shift, 0.05)
        # forward splat (target pixel grid at splat res), nearest point wins
        ft = self.ft
        tx = np.rint(((ft * x2 / z2 + cx) + 0.5) * (sw / self.w) - 0.5).astype(np.int32).ravel()
        ty = np.rint(((ft * y2 / z2 + cy) + 0.5) * (sh / self.h) - 0.5).astype(np.int32).ravel()
        ok = (tx >= 0) & (tx < sw) & (ty >= 0) & (ty < sh)
        zbuf = np.full(sh * sw, np.inf, np.float32)
        np.minimum.at(zbuf, ty[ok] * sw + tx[ok], z2.ravel()[ok])
        zt = zbuf.reshape(sh, sw)
        holes = ~np.isfinite(zt)
        # voids = disocclusions larger than splat cracks (opening removes 1-px cracks)
        voids = cv2.morphologyEx(holes.astype(np.uint8), cv2.MORPH_OPEN, self.kernel)
        if holes.any():
            # Background-side fill: multi-scale max filter (farthest neighbour) into holes only,
            # then smooth inside the holes so revealed regions get an even stretch of background.
            hole0 = holes.copy()
            filled = np.where(holes, -1.0, zt).astype(np.float32)
            for k in (3, 7, 17, 41, 101):
                dil = cv2.dilate(filled, self._kern(k))
                upd = holes & (dil > 0)
                filled[upd] = dil[upd]
                holes &= ~upd
                if not holes.any():
                    break
            filled[holes] = zp
            smooth = cv2.GaussianBlur(filled, (0, 0), 3.0)
            filled[hole0] = smooth[hole0]
            zt = filled
        # grey opening removes thin 'far peaks' (background seen through gaps in a stretched near
        # surface), then a median cleans isolated speckles
        zt = cv2.medianBlur(cv2.morphologyEx(zt, cv2.MORPH_OPEN, self._kern(5)), 3)
        # inverse transform per target pixel -> source pixel coordinates (full-res units)
        xt = (self.u - cx) / ft * zt
        yt = (self.v - cy) / ft * zt
        zt0 = zt - shift
        # P_c = R^T (P' - shift); R^T rows are R columns
        xs = rot[0, 0] * xt + rot[1, 0] * yt + rot[2, 0] * zt0
        ys = rot[0, 1] * xt + rot[1, 1] * yt + rot[2, 1] * zt0
        zs = np.maximum(rot[0, 2] * xt + rot[1, 2] * yt + rot[2, 2] * zt0 + zp, 0.05)
        map_x = cv2.resize(f * xs / zs + cx, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(f * ys / zs + cy, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        view = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # dim disocclusion voids (soft-edged), so the newly revealed space reads as depth
        v = cv2.GaussianBlur(voids.astype(np.float32), (0, 0), 1.5)
        v = 1.0 - void_dim * np.clip(v * 1.6, 0, 1)
        shade = cv2.resize(v, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        view = cv2.multiply(view, cv2.merge([shade] * 3), dtype=cv2.CV_8U)
        disp_t = np.clip(((1.0 / np.maximum(zs, 1e-3)) - NEAR_DISP) / (1.0 - NEAR_DISP), 0, 1)
        return view, disp_t.astype(np.float32)


def depth_shade(view: np.ndarray, d: np.ndarray, fog=(40, 22, 18), amount: float = 0.3) -> np.ndarray:
    """Subtle atmospheric depth cue: far pixels fade toward a cool dark fog colour (d may be low-res)."""
    f = cv2.resize(amount * (1.0 - d) ** 1.5, (view.shape[1], view.shape[0]), interpolation=cv2.INTER_LINEAR)
    fog_img = np.empty_like(view)
    fog_img[:] = fog
    return cv2.blendLinear(view, fog_img, (1.0 - f).astype(np.float32), f.astype(np.float32))


def center_crop(frame: np.ndarray, aspect: float = 16 / 9) -> np.ndarray:
    h, w = frame.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect))
        x = (w - nw) // 2
        return frame[:, x:x + nw]
    nh = int(round(w / aspect))
    y = (h - nh) // 2
    return frame[y:y + nh]
