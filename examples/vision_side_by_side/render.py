"""Drawing for the vision side-by-side demo: device panels (blur effect + boxes + badge) and a stats panel."""
import cv2
import numpy as np

from examples.common import hud

BG = (18, 18, 18)
MUTED = (165, 165, 165)


def background_layer(frame: np.ndarray) -> np.ndarray:
    """Blurred, dimmed, slightly desaturated copy (uint8) shown behind the segmented foreground."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.6)
    blur = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    # subtle: 75 % brightness, ~35 % desaturated, soft blur, so the scene stays readable behind the boxes
    return cv2.addWeighted(blur, 0.49, gray, 0.26, 0)


def apply_segmentation(frame: np.ndarray, bg_layer: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Sharp foreground where mask ~1, blurred background elsewhere, thin device-coloured mask outline."""
    out = cv2.blendLinear(frame, bg_layer, mask, 1.0 - mask)
    edge = cv2.morphologyEx((mask > 0.5).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    sel = edge.astype(bool)
    out[sel] = (0.35 * out[sel] + 0.65 * np.asarray(color, np.float32)).astype(np.uint8)
    return out


def draw_detections(img: np.ndarray, dets, scale: float, color, max_labels: int = 14, min_label_h: int = 16):
    """Corner-bracket boxes plus a filled label chip in the device colour."""
    h, w = img.shape[:2]
    dets = sorted(dets, key=lambda d: -d.score)
    overlay = img.copy()
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in d.box]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        c = max(4, min(14, (x2 - x1) // 4, (y2 - y1) // 4))
        for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + dx * c, py), color, 2, cv2.LINE_AA)
            cv2.line(img, (px, py), (px, py + dy * c), color, 2, cv2.LINE_AA)
        if i < max_labels and y2 - y1 >= min_label_h:  # label only the confident, not-tiny boxes
            label = f"{d.label} {int(d.score * 100)}"
            (tw, th), _ = cv2.getTextSize(label, hud.FONT, 0.38, 1)
            ty = y1 - 3 if y1 - th - 6 > 0 else y1 + th + 5
            cv2.rectangle(img, (x1, ty - th - 3), (x1 + tw + 6, ty + 3), color, -1)
            cv2.putText(img, label, (x1 + 3, ty), hud.FONT, 0.38, (15, 15, 15), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)


def badge(img: np.ndarray, device: str, exec_devices: str, lines: list[str], history, vmax: float):
    """Device badge (shared hud style) with a latency sparkline underneath."""
    color = hud.device_color(device)
    hud.device_badge(img, device, lines, x=10, y=10)
    hud.text(img, exec_devices, 90, 35, 0.4, MUTED)  # EXECUTION_DEVICES next to the name
    y = 10 + 34 + 24 * len(lines)
    hud.panel(img, 10, y, 230, 30)
    cv2.rectangle(img, (10, y), (16, y + 30), color, -1)
    cv2.line(img, (26, y + 25), (230, y + 25), (70, 70, 70), 1)
    hud.sparkline(img, history, 26, y + 4, 204, 21, color, vmax=vmax)


def stats_panel(w: int, h: int, rows: list[dict], header: list[str], footer: list[str]) -> np.ndarray:
    """rows: dicts with device, p50, p95 (ms), fps (1000/p50), n_obj. Horizontal p50 bars with p95 whiskers."""
    img = np.full((h, w, 3), 24, np.uint8)
    y = 34
    hud.text(img, header[0], 18, y, 0.75, (255, 255, 255), 1)
    for line in header[1:]:
        y += 22
        hud.text(img, line, 18, y, 0.45, MUTED)
    y += 26
    hud.text(img, "inference latency per frame (ms)  bar = p50, tick = p95", 18, y, 0.42, MUTED)
    vmax = max([r["p95"] for r in rows if r.get("p95")] + [1.0]) * 1.15
    bar_x, bar_w = 92, w - 92 - 150
    for r in rows:
        y += 34
        color = hud.device_color(r["device"])
        hud.text(img, r["device"], 18, y + 4, 0.6, color, 1)
        cv2.rectangle(img, (bar_x, y - 12), (bar_x + bar_w, y + 8), (40, 40, 40), -1)
        if r.get("p50"):
            bw = int(bar_w * r["p50"] / vmax)
            cv2.rectangle(img, (bar_x, y - 12), (bar_x + bw, y + 8), color, -1)
            px = bar_x + int(bar_w * r["p95"] / vmax)
            cv2.line(img, (px, y - 15), (px, y + 11), (235, 235, 235), 2)
            hud.text(img, f"{r['p50']:.1f} ms  {r['fps']:.0f}/s", bar_x + bar_w + 10, y + 4, 0.45)
        else:
            hud.text(img, r.get("status", "n/a"), bar_x + 8, y + 4, 0.45, MUTED)
    y = h - 14 - 20 * (len(footer) - 1)
    for line in footer:
        hud.text(img, line, 18, y, 0.4, MUTED)
        y += 20
    return img
