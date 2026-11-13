# Depth Parallax: flat video, orbiting 3D, on the NPU

![](../../media/depth_parallax.gif)

Every frame of an ordinary 2D video goes through **Depth Anything V2 Small** (monocular relative depth) on the
**Intel AI Boost NPU**. The CPU then lifts the frame into a 3D point cloud using that depth and re-renders it
from a slowly orbiting virtual camera. The flat clip gains a 3D "wiggle": near rocks and ferns slide against
the far canyon. The composite shows the input frame, the colourised NPU depth (bright = near), and the
re-projected hero view. A HUD shows NPU latency, and an inset compares the same model on NPU, GPU and CPU.

**Why the NPU:** per-frame depth is the expensive part. On this Core Ultra 5 322, the 364x364 model takes about
12.5 ms on the NPU. The CPU takes 134 ms, so the NPU is about 10.7x faster. The integrated GPU (2 Xe3
cores) takes 17.3 ms (NPU 1.4x faster). With the NPU doing inference, the CPU is free to do the 3D rendering.

## Run

```powershell
.venv\Scripts\python.exe -m examples.depth_parallax                          # default clip 'forest', live window (q/Esc quits)
.venv\Scripts\python.exe -m examples.depth_parallax --source drive           # any clip from examples.common.clips, a video file, or a webcam index
.venv\Scripts\python.exe -m examples.depth_parallax --image photo.jpg        # 3D-photo "Ken Burns" mode for a still image
.venv\Scripts\python.exe -m examples.depth_parallax --record --no-display    # writes media/depth_parallax.gif + media/mp4/depth_parallax.mp4
.venv\Scripts\python.exe -m examples.depth_parallax --device GPU --no-compare
```

| Flag | Meaning |
|---|---|
| `--device` | Device for the live model (default `NPU`). The demo never falls back to another device: if compilation fails, it exits with the error. |
| `--compare` / `--no-compare` | Run a short startup benchmark of the same model and input on NPU, GPU and CPU for the inset (on by default; CPU adds about 5 s). |
| `--record`, `--duration` | Record the composite. Duration defaults to 12 s when recording or headless. |
| `--orbit` | Multiplier for the camera orbit amplitude (default 1.0: about ±5° yaw). |
| `--output` | Results JSON, default `benchmarks/results/depth_parallax.json`. It holds per-device latency stats, `compile_s` with a cold/warm cache flag, `EXECUTION_DEVICES`, frame count, and the clip with its licence. |
| `--size` | Model input side, a multiple of 14 (default 364; 518 also works on the NPU at about 25 ms). |

The first NPU compile takes tens of seconds (17 s in the preliminary run; the clean run started from a warm cache, so
no cold figure was recorded). The blob is cached in `models/npucache` (gitignored), and a warm start loaded it in
1.7 s in the clean run. A splash screen says "compiling" while this happens.

## Measured (clean run, 2026-11-12, NPU driver 32.0.100.5540)

Idle laptop on AC power, graphics driver 32.0.101.9175, OpenVINO 2026.4.1.

| | NPU | GPU | CPU |
|---|---|---|---|
| Startup benchmark, median (p95) ms (364x364) | 12.5 (20.1) | 17.3 (17.4) | 133.9 (138.7) |
| Compile, warm cache / no cache | 1.7 s (warm) | 2.5 s (warm) | 4.6 s (no cache) |

(The preliminary 2026-11-11 run, with other demos sharing the machine and NPU driver 32.0.100.4724, measured the
NPU at 12.8 ms, the GPU at 19-29 ms and the CPU at 172-244 ms, with a 17.2 s cold NPU compile.)

Live run on the `forest` clip with recording on (359 frames):
- NPU latency per frame: p50 13.9 ms, p95 15.2 ms. This is `InferRequest.latency`. It is higher than the
  isolated benchmark because the host thread is busy at the same time.
- Host work per frame: 28.1 ms median. That covers preprocessing, depth normalisation, the 3D re-projection (14.7 ms)
  and the HUD.
- End to end: 32.7 FPS with recording on.

Inference of frame N+1 overlaps with rendering of frame N. The demo is therefore **CPU-render bound, not NPU
bound**, and the HUD reports the two separately.

## How the 3D view is made (`render.py`)

1. Normalise inverse depth to [0, 1] using percentiles. The percentiles are smoothed with an EMA so brightness doesn't flicker.
2. Lift pixels to 3D with Z = 1/(0.25 + 0.75·d) and a 55° pinhole camera. Rotate the cloud (yaw/pitch) about
   the median scene depth, and add a gentle dolly.
3. Forward-splat the new depth with a z-buffer at half resolution. Close cracks with a grey morphological opening.
   Fill disocclusions from the background side using a multi-scale max filter.
4. Invert the exact 3D transform per target pixel and `cv2.remap` the full-resolution colour frame. This keeps
   edges crisp. Revealed regions show stretched background and are slightly dimmed. A light fog darkens far regions.

## Caveats

- The depth is relative and monocular (affine-invariant inverse depth). The 3D is plausible, not metric, and
  objects touching the frame edge or very close to the camera can smear. The dashcam clip's mirror and A-pillar
  are an example, which is why `forest` is the default clip.
- `--image` repeats inference on the same still every frame so the HUD shows a live device rate.
- Timings vary with power mode and with whatever else is using the NPU or CPU.

## Licences and attribution

- Model: [onnx-community/depth-anything-v2-small](https://huggingface.co/onnx-community/depth-anything-v2-small), Apache-2.0
  (Depth Anything V2, Yang et al. 2024).
- Footage in the GIF: "Fern Canyon in Prairie Creek Redwoods State Park" by Thomas Farley, CC0 1.0, via Wikimedia Commons.
  All clips and their licences are listed in [assets/CLIPS_ATTRIBUTION.md](../../assets/CLIPS_ATTRIBUTION.md).
