"""Record demo frames to MP4 (local, gitignored) and size-capped GIF (for README embedding).

Uses imageio-ffmpeg's bundled ffmpeg binary, so no system ffmpeg is needed.
Frames are RGB uint8 numpy arrays (H, W, 3). OpenCV frames are BGR: pass bgr=True.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from . import MEDIA_DIR

GIF_MAX_BYTES = 5 * 1024 * 1024


class Recorder:
    """Collects frames during a demo, then writes `<name>.mp4` and `<name>.gif` under media/.

    mp4 goes to media/mp4/ (gitignored); gif goes to media/ (committed).
    GIF settings: downscaled to `gif_width`, sampled at `gif_fps`, re-encoded smaller until under 5 MB.
    """

    def __init__(self, name: str, fps: float = 20.0, gif_width: int = 720, gif_fps: float = 10.0,
                 max_seconds: float | None = None, out_dir: Path | None = None):
        self.name = name
        self.fps = fps
        self.gif_width = gif_width
        self.gif_fps = gif_fps
        self.max_frames = int(max_seconds * fps) if max_seconds else None
        self.out_dir = Path(out_dir) if out_dir else MEDIA_DIR
        self.frames: list[np.ndarray] = []

    @property
    def full(self) -> bool:
        return self.max_frames is not None and len(self.frames) >= self.max_frames

    def add(self, frame: np.ndarray, bgr: bool = False):
        if self.full:
            return
        if bgr:
            frame = frame[..., ::-1]
        self.frames.append(np.ascontiguousarray(frame, dtype=np.uint8))

    def save(self, mp4: bool = True, gif: bool = True) -> dict:
        if not self.frames:
            return {}
        out = {}
        if mp4:
            out["mp4"] = str(write_mp4(self.frames, self.out_dir / "mp4" / f"{self.name}.mp4", self.fps))
        if gif:
            step = max(1, round(self.fps / self.gif_fps))
            out["gif"] = str(write_gif(self.frames[::step], self.out_dir / f"{self.name}.gif",
                                       self.fps / step, self.gif_width))
        return out


def _even(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    return frame[: h - h % 2, : w - w % 2]


def write_mp4(frames, path: Path, fps: float) -> Path:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
        for frame in frames:
            writer.append_data(_even(frame))
    return path


def write_gif(frames, path: Path, fps: float, width: int, max_bytes: int = GIF_MAX_BYTES,
              quantize: str = "adaptive") -> Path:
    """Write an optimized looping GIF, shrinking width / dropping frames until it fits max_bytes.

    quantize="adaptive" (default, PIL ADAPTIVE palette) or "fastoctree" (no dithering: ~30 % smaller for
    noisy video footage, slight banding).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = list(frames)
    for _ in range(6):
        images = []
        for frame in frames:
            img = Image.fromarray(frame)
            if img.width > width:
                img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
            if quantize == "fastoctree":
                images.append(img.quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE))
            else:
                images.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))
        images[0].save(path, save_all=True, append_images=images[1:], loop=0,
                       duration=round(1000 / fps), optimize=True, disposal=2)
        if path.stat().st_size <= max_bytes or len(frames) < 4:
            break
        width = int(width * 0.85)
        if path.stat().st_size > 2 * max_bytes:
            frames, fps = frames[::2], fps / 2
    return path
