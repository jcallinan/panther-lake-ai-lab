"""Stock video clips for the demos: a small registry of freely licensed footage, downloaded on demand.

Every clip is hosted on Wikimedia Commons with a verifiable licence (CC0 or public domain) on its file
page. `get_clip(name)` downloads a Commons server-side transcode once, trims it to the segment listed here
(dropping title cards), scales it to <= 720p H.264 without audio, and caches the result as
assets/clips/<name>.mp4 (gitignored). Attribution for all clips: assets/CLIPS_ATTRIBUTION.md.

    from examples.common.clips import get_clip, list_clips
    path = get_clip("street")        # -> Path to assets/clips/street.mp4

CLI:  python -m examples.common.clips            (list)
      python -m examples.common.clips --fetch-all (download + trim every clip)
"""
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import CLIPS_DIR

USER_AGENT = "panther-lake-ai-lab/0.1 (demo clip downloader; python urllib)"
_UPLOAD = "https://upload.wikimedia.org/wikipedia/commons"


@dataclass(frozen=True)
class Clip:
    name: str
    title: str
    scene: str  # what the clip is good for
    url: str  # direct download (Commons transcode of the original file)
    source_page: str  # licence / description page
    license: str
    license_url: str
    author: str
    start_s: float  # trimmed segment
    duration_s: float
    original_url: str = ""

    @property
    def attribution(self) -> str:
        return f'"{self.title}" by {self.author}, {self.license}, via Wikimedia Commons ({self.source_page})'

    def as_dict(self) -> dict:
        return {"name": self.name, "title": self.title, "license": self.license, "license_url": self.license_url,
                "author": self.author, "source_page": self.source_page, "url": self.url,
                "segment_s": [self.start_s, self.start_s + self.duration_s], "attribution": self.attribution}


CLIPS: dict[str, Clip] = {c.name: c for c in [
    Clip(
        name="street",
        title="DiagonalCrosswalkYongeDundas.webm",
        scene="busy diagonal crosswalk (Yonge-Dundas, Toronto): many pedestrians, cars, a bus - detection",
        url=f"{_UPLOAD}/transcoded/d/d2/DiagonalCrosswalkYongeDundas.webm/DiagonalCrosswalkYongeDundas.webm.1080p.vp9.webm",
        original_url=f"{_UPLOAD}/d/d2/DiagonalCrosswalkYongeDundas.webm",
        source_page="https://commons.wikimedia.org/wiki/File:DiagonalCrosswalkYongeDundas.webm",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Raysonho @ Open Grid Scheduler / Grid Engine",
        start_s=4.0, duration_s=24.0,
    ),
    Clip(
        name="person",
        title="Video Blog- CBP Super Bowl LII Countdown to Kickoff - Day 4.webm",
        scene="one person facing the camera, upper body, talking and gesturing - segmentation / pose",
        url=(f"{_UPLOAD}/transcoded/e/e2/Video_Blog-_CBP_Super_Bowl_LII_Countdown_to_Kickoff_-_Day_4.webm/"
             "Video_Blog-_CBP_Super_Bowl_LII_Countdown_to_Kickoff_-_Day_4.webm.1080p.vp9.webm"),
        original_url=f"{_UPLOAD}/e/e2/Video_Blog-_CBP_Super_Bowl_LII_Countdown_to_Kickoff_-_Day_4.webm",
        source_page="https://commons.wikimedia.org/wiki/File:Video_Blog-_CBP_Super_Bowl_LII_Countdown_to_Kickoff_-_Day_4.webm",
        license="Public domain (work of the U.S. federal government, DHS / U.S. Customs and Border Protection; PD-USGov-DHS)",
        license_url="https://commons.wikimedia.org/wiki/Template:PD-USGov-DHS",
        author="Mani Albrecht, U.S. Customs and Border Protection Office of Public Affairs",
        start_s=1.0, duration_s=24.0,
    ),
    Clip(
        name="forest",
        title="Fern Canyon in Prairie Creek Redwoods State Park.webm",
        scene="narrow fern-walled canyon with a creek bed receding into the distance - depth",
        url=(f"{_UPLOAD}/transcoded/3/36/Fern_Canyon_in_Prairie_Creek_Redwoods_State_Park.webm/"
             "Fern_Canyon_in_Prairie_Creek_Redwoods_State_Park.webm.480p.vp9.webm"),
        original_url=f"{_UPLOAD}/3/36/Fern_Canyon_in_Prairie_Creek_Redwoods_State_Park.webm",
        source_page="https://commons.wikimedia.org/wiki/File:Fern_Canyon_in_Prairie_Creek_Redwoods_State_Park.webm",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Thomas Farley",
        start_s=4.0, duration_s=22.0,
    ),
    Clip(
        name="drive",
        title="Dashcam Recording (urban).ogv",
        scene="dashcam drive through an office district: road, cars, buildings receding - depth / detection",
        url=(f"{_UPLOAD}/transcoded/e/ea/Dashcam_Recording_%28urban%29.ogv/"
             "Dashcam_Recording_%28urban%29.ogv.480p.vp9.webm"),
        original_url=f"{_UPLOAD}/e/ea/Dashcam_Recording_%28urban%29.ogv",
        source_page="https://commons.wikimedia.org/wiki/File:Dashcam_Recording_(urban).ogv",
        license="CC0 1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        author="Fernost",
        start_s=40.0, duration_s=25.0,
    ),
]}


def list_clips() -> list[dict]:
    """Registry entries with licence info and whether each clip is already cached locally."""
    return [{**c.as_dict(), "scene": c.scene, "cached": (CLIPS_DIR / f"{c.name}.mp4").exists()}
            for c in CLIPS.values()]


def _download(url: str, dest: Path, retries: int = 6):
    """Download with a descriptive User-Agent; back off on HTTP 429 (Wikimedia rate limits)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    delay = 5.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            tmp.replace(dest)
            return
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                raise
            wait = float(exc.headers.get("Retry-After") or delay)
            print(f"[clips] HTTP {exc.code}, retrying in {wait:.0f} s", file=sys.stderr)
            time.sleep(wait)
            delay *= 2


def _trim(src: Path, dest: Path, start_s: float, duration_s: float, max_height: int = 720):
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-ss", str(start_s), "-i", str(src), "-t", str(duration_s), "-an",
           "-vf", f"scale=-2:'min({max_height},ih)'", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


def get_clip(name: str, force: bool = False) -> Path:
    """Path to the trimmed local copy of clip `name`, downloading + trimming it on first use."""
    if name not in CLIPS:
        raise KeyError(f"unknown clip {name!r}; available: {', '.join(CLIPS)}")
    clip = CLIPS[name]
    out = CLIPS_DIR / f"{name}.mp4"
    if out.exists() and not force:
        return out
    src = CLIPS_DIR / "_source" / f"{name}{Path(clip.url.split('?')[0]).suffix or '.webm'}"
    if not src.exists() or force:
        print(f"[clips] downloading {clip.title} ({clip.license}) ...", file=sys.stderr)
        _download(clip.url, src)
    print(f"[clips] trimming {name}: {clip.start_s:.0f}-{clip.start_s + clip.duration_s:.0f} s -> {out}",
          file=sys.stderr)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    _trim(src, tmp, clip.start_s, clip.duration_s)
    tmp.replace(out)
    src.unlink(missing_ok=True)
    try:
        src.parent.rmdir()
    except OSError:
        pass
    return out


def resolve_source(source: str):
    """Map a --source argument to (cv2.VideoCapture argument, clip-or-None, kind).

    kind is "clip", "file" or "webcam". A bare integer means a webcam index.
    """
    if source in CLIPS:
        return str(get_clip(source)), CLIPS[source], "clip"
    if source.isdigit():
        return int(source), None, "webcam"
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"--source {source!r} is not a clip name ({', '.join(CLIPS)}), file, or webcam index")
    return str(path), None, "file"


if __name__ == "__main__":
    import json

    if "--fetch-all" in sys.argv:
        for n in CLIPS:
            print(get_clip(n))
    print(json.dumps(list_clips(), indent=2))
