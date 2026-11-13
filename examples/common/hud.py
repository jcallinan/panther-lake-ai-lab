"""Consistent heads-up-display drawing for the demos (OpenCV, BGR frames).

Device colours are shared across every demo and chart so NPU/GPU/CPU are recognisable at a glance.
"""
import cv2
import numpy as np

# BGR. Hex equivalents for matplotlib/charts in DEVICE_HEX.
DEVICE_BGR = {"NPU": (240, 170, 0), "GPU": (60, 200, 90), "CPU": (40, 120, 245)}
DEVICE_HEX = {"NPU": "#00AAF0", "GPU": "#5AC83C", "CPU": "#F57828"}
FONT = cv2.FONT_HERSHEY_DUPLEX


def device_color(device: str):
    return DEVICE_BGR.get(device.split(".")[0].upper(), (200, 200, 200))


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.6):
    """Darken a rectangle in place for legible text."""
    x2, y2 = min(x + w, frame.shape[1]), min(y + h, frame.shape[0])
    roi = frame[y:y2, x:x2]
    roi[:] = (roi * (1 - alpha)).astype(np.uint8)


def text(frame, s: str, x: int, y: int, scale: float = 0.6, color=(255, 255, 255), thickness: int = 1):
    # Outline via 1 px offset copies at the same thickness: in OpenCV 5 a thicker stroke also widens the glyph
    # advance, so a `thickness + 2` outline drifts right of the text and leaves dark ghost letters.
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(frame, s, (x + dx, y + dy), FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def device_badge(frame, device: str, lines: list[str], x: int = 10, y: int = 10):
    """Top-left badge: coloured device name plus metric lines (e.g. 'FPS 54.2', 'p50 6.1 ms')."""
    color = device_color(device)
    h = 34 + 24 * len(lines)
    panel(frame, x, y, 230, h)
    cv2.rectangle(frame, (x, y), (x + 6, y + h), color, -1)
    text(frame, device, x + 16, y + 26, 0.8, color, 2)
    for i, line in enumerate(lines):
        text(frame, line, x + 16, y + 52 + 24 * i, 0.55)


def tile(frames: list[np.ndarray], cols: int, cell_w: int, cell_h: int, gap: int = 4) -> np.ndarray:
    """Grid-compose BGR frames (resized to cell size) on a near-black background."""
    rows = -(-len(frames) // cols)
    canvas = np.full((rows * cell_h + (rows - 1) * gap, cols * cell_w + (cols - 1) * gap, 3), 18, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = r * (cell_h + gap), c * (cell_w + gap)
        canvas[y:y + cell_h, x:x + cell_w] = cv2.resize(f, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
    return canvas


def sparkline(frame, values, x: int, y: int, w: int, h: int, color, vmax: float | None = None):
    """Tiny line chart of recent values (e.g. latency history) drawn in place."""
    vals = np.asarray(list(values)[-w:], dtype=np.float32)
    if vals.size < 2:
        return
    top = vmax or float(vals.max()) or 1.0
    xs = np.linspace(x, x + w, vals.size).astype(np.int32)
    ys = (y + h - np.clip(vals / top, 0, 1) * h).astype(np.int32)
    cv2.polylines(frame, [np.stack([xs, ys], 1)], False, color, 2, cv2.LINE_AA)
