"""Drawing for the offload dashboard: live 1280x720 OpenCV frame, end-of-run summary frame, static matplotlib PNG."""
from __future__ import annotations

import cv2
import numpy as np

from examples.common.hud import DEVICE_BGR, DEVICE_HEX, FONT

W, H = 1280, 720
BG = (22, 20, 18)
CARD = (38, 35, 32)
GRID = (62, 58, 54)
INK = (240, 240, 240)
INK2 = (178, 174, 168)
MUTED = (128, 124, 118)
BASE_BGR = (170, 170, 170)  # "AI off" baseline phase
BASE_HEX = "#8a8a8a"


def phase_color(device):
    return DEVICE_BGR.get(device, BASE_BGR) if device else BASE_BGR


def put(img, s, x, y, scale=0.5, color=INK, thick=1, align="left"):
    if align != "left":
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
        x = x - tw if align == "right" else x - tw // 2
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def card(img, x, y, w, h, color=CARD):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)


def fmt(v, spec="{:.0f}", none="n/a"):
    return none if v is None or (isinstance(v, float) and np.isnan(v)) else spec.format(v)


# ------------------------------------------------------------------------------------------------ live frame
def line_chart(img, x, y, w, h, title, t, series, t_total, phases, vmax, unit="", ref=None):
    """Panel with phase bands. series: list of (label, values, colors_per_point or single colour)."""
    card(img, x, y, w, h)
    put(img, title, x + 10, y + 18, 0.48, INK2)
    px, py, pw, ph = x + 40, y + 28, w - 52, h - 38
    for ph_ in phases:  # phase bands
        x0 = px + int(pw * ph_["t0"] / t_total)
        x1 = px + int(pw * ph_["t1"] / t_total)
        cv2.line(img, (x0, py), (x0, py + ph), GRID, 1)
    for frac in (0, 0.5, 1.0):
        yy = py + ph - int(ph * frac)
        cv2.line(img, (px, yy), (px + pw, yy), GRID, 1)
        put(img, f"{vmax * frac:.0f}{unit}", px - 6, yy + 4, 0.36, MUTED, align="right")
    if ref is not None:
        yy = py + ph - int(ph * min(ref / vmax, 1))
        for xx in range(px, px + pw, 8):
            cv2.line(img, (xx, yy), (xx + 3, yy), INK2, 1)
    if len(t) < 2:
        return
    tt = np.asarray(t, np.float32)
    xs = (px + pw * np.clip(tt / t_total, 0, 1)).astype(np.int32)
    for label, vals, col in series:
        v = np.asarray([np.nan if a is None else a for a in vals], np.float32)
        ys = (py + ph - ph * np.clip(v / vmax, 0, 1)).astype(np.float32)
        for i in range(1, len(xs)):
            if np.isnan(ys[i]) or np.isnan(ys[i - 1]):
                continue
            c = col[i] if isinstance(col, list) else col
            cv2.line(img, (int(xs[i - 1]), int(ys[i - 1])), (int(xs[i]), int(ys[i])), c, 2, cv2.LINE_AA)
        if not np.isnan(ys[-1]):
            c = col[-1] if isinstance(col, list) else col
            cv2.circle(img, (int(xs[-1]), int(ys[-1])), 4, c, -1, cv2.LINE_AA)


def dashboard_frame(video, st) -> np.ndarray:
    """st: dict with keys described in __main__.Dashboard.state()."""
    img = np.full((H, W, 3), BG, np.uint8)
    phases, cur = st["phases"], st["phase_idx"]
    # Header
    put(img, "Offload: always-on AI vs. your foreground work", 16, 32, 0.78, INK, 1)
    put(img, st["subtitle"], 16, 52, 0.42, MUTED)
    # Phase pills: AI placements as labelled pills, AI-off phases as small grey squares between them
    px = W - 16
    for i in reversed(range(len(phases))):
        p = phases[i]
        col = phase_color(p["device"])
        if p["device"] is None:
            x0 = px - 12
            if i == cur:
                cv2.rectangle(img, (x0, 25), (px, 37), col, -1)
            else:
                cv2.rectangle(img, (x0, 25), (px, 37), col if i < cur else GRID, 1 if i > cur else -1)
            px = x0 - 6
            continue
        label = p["short"]
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        x0 = px - (tw + 18)
        if i == cur:
            cv2.rectangle(img, (x0, 18), (px, 44), col, -1)
            put(img, label, x0 + 9, 36, 0.45, (20, 20, 20))
        else:
            cv2.rectangle(img, (x0, 18), (px, 44), col if i < cur else GRID, 1)
            put(img, label, x0 + 9, 36, 0.45, INK if i < cur else MUTED)
        px = x0 - 6
    # Video panel
    vx, vy, vw, vh = 16, 64, 640, 360
    img[vy:vy + vh, vx:vx + vw] = cv2.resize(video, (vw, vh)) if video.shape[:2] != (vh, vw) else video
    dev = st["device"]
    col = phase_color(dev)
    cv2.rectangle(img, (vx, vy), (vx + vw - 1, vy + vh - 1), col, 2)
    roi = img[vy + 10:vy + 72, vx + 10:vx + 262]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.rectangle(img, (vx + 10, vy + 10), (vx + 16, vy + 72), col, -1)
    put(img, f"AI on {dev}" if dev else "AI off (baseline)", vx + 26, vy + 36, 0.7, col, 2)
    put(img, st["video_caption"], vx + 26, vy + 60, 0.42, INK)
    # progress bar of the current phase
    cv2.rectangle(img, (vx, vy + vh + 4), (vx + vw, vy + vh + 8), GRID, -1)
    cv2.rectangle(img, (vx, vy + vh + 4), (vx + int(vw * st["phase_frac"]), vy + vh + 8), col, -1)

    # KPI tiles (current phase, rolling)
    k = st["kpi"]
    tiles = [("Your work", fmt(k["fg_pct"], "{:.0f}%"), "vs. last AI-off phase"),
             ("AI FPS", fmt(k["ai_fps"], "{:.1f}"), f"target {st['target_fps']:.0f}"),
             ("System CPU", fmt(k["cpu"], "{:.0f}%"), f"AI proc {fmt(k['ai_cpu'], '{:.0f}%')} of system"),
             ("UI lag p95", fmt(k["lag"], "{:.1f} ms"), "2 ms timer wake-up")]
    tx, ty, tw_, th = 16, 440, 154, 86
    for i, (name, val, sub) in enumerate(tiles):
        x = tx + i * (tw_ + 8)
        card(img, x, ty, tw_, th)
        put(img, name, x + 10, ty + 20, 0.42, INK2)
        put(img, val, x + 10, ty + 56, 0.95, INK, 2)
        put(img, sub, x + 10, ty + 76, 0.36, MUTED)

    # Scoreboard of finished phases
    sx, sy = 16, 536
    card(img, sx, sy, 640, 168)
    put(img, "Per placement (mean of rounds so far; work % vs. adjacent AI-off phases)", sx + 10, sy + 20, 0.42,
        INK2)
    cols = [("Placement", 10), ("AI FPS", 150), ("Your work", 240), ("AI CPU", 340), ("CPU ms/frame", 420),
            ("Accel", 550)]
    for name, cx in cols:
        put(img, name, sx + cx, sy + 44, 0.4, MUTED)
    by_dev = {p["device"]: p for p in st.get("placements") or []}
    devs = [None] + [d for d in dict.fromkeys(p["device"] for p in phases if p["device"])]
    for r, d in enumerate(devs):
        y = sy + 70 + r * 24
        p = by_dev.get(d)
        res = (p or {}).get("result") or {}
        c = phase_color(d)
        cv2.rectangle(img, (sx + 10, y - 10), (sx + 16, y + 2), c, -1)
        put(img, "AI off" if d is None else f"AI on {d}", sx + 24, y, 0.45, INK if res else MUTED)
        if not res.get("n_phases"):
            put(img, res.get("status") or ("running..." if d == st["device"] else "-"), sx + 150, y, 0.42, MUTED)
            continue
        acc = res.get("npu_util") if d == "NPU" else res.get("gpu_util") if d == "GPU" else None
        vals = [fmt(res.get("ai_fps"), "{:.1f}", "-"), fmt(res.get("fg_pct") if d else None, "{:.0f}%", "100%"),
                fmt(res.get("ai_proc_cpu_pct"), "{:.1f}%"), fmt(res.get("ai_proc_cpu_ms_per_frame"), "{:.1f} ms", "-"),
                fmt(acc, "{:.0f}%", "-")]
        for (name, cx), v in zip(cols[1:], vals):
            put(img, v, sx + cx, y, 0.45, INK)

    # Charts (whole-run timeline)
    t, hist, T = st["t"], st["hist"], st["t_total"]
    pc = [phase_color(d) for d in hist["device"]]
    cx_, cw = 672, 592
    ch, gap, cy = 152, 6, 64
    line_chart(img, cx_, cy, cw, ch, "Your work (zlib ops/s), % of the last AI-off phase", t,
               [("fg", hist["fg_pct"], pc)], T, phases, 120, "%", ref=100)
    line_chart(img, cx_, cy + (ch + gap), cw, ch, "CPU used by the AI pipeline process, % of all 6 threads", t,
               [("ai", hist["ai_cpu"], pc)], T, phases, 30, "%")
    line_chart(img, cx_, cy + 2 * (ch + gap), cw, ch, "AI frames per second", t, [("fps", hist["ai_fps"], pc)], T,
               phases, max(35, st["target_fps"] * 1.15), "", ref=st["target_fps"])
    acc_title = "Accelerator busy % (busiest engine)" if st["accel_ok"] else "Accelerator busy %: counters unavailable"
    line_chart(img, cx_, cy + 3 * (ch + gap), cw, ch, acc_title, t,
               [("GPU", hist["gpu"], DEVICE_BGR["GPU"]), ("NPU", hist["npu"], DEVICE_BGR["NPU"])], T, phases, 100,
               "%")
    ly = cy + 3 * (ch + gap) + 18
    for i, (name, c) in enumerate((("NPU", DEVICE_BGR["NPU"]), ("iGPU", DEVICE_BGR["GPU"])) ):
        lx = cx_ + cw - 120 + i * 60
        cv2.line(img, (lx, ly - 4), (lx + 14, ly - 4), c, 2)
        put(img, name, lx + 18, ly, 0.4, INK2)
    # phase labels under the bottom chart's x axis
    by = cy + 4 * ch + 3 * gap + 2
    for p in phases:
        x0 = cx_ + 40 + int((cw - 52) * p["t0"] / T)
        x1 = cx_ + 40 + int((cw - 52) * p["t1"] / T)
        cv2.rectangle(img, (x0 + 1, by + 2), (x1 - 1, by + 5), phase_color(p["device"]), -1)
        if p["device"]:  # AI-off segments are the short grey ones in between
            put(img, p["short"], (x0 + x1) // 2, by + 18, 0.4, INK2, align="center")
    return img


# ------------------------------------------------------------------------------------------------ summary frame
def summary_frame(placements, subtitle, target_fps) -> np.ndarray:
    """End card. `placements` = aggregated per-placement results (AI off + one entry per device)."""
    img = np.full((H, W, 3), BG, np.uint8)
    put(img, "Result: where should always-on AI run?", 16, 40, 0.9, INK, 1)
    put(img, subtitle, 16, 64, 0.45, MUTED)
    done = [p for p in placements if p.get("result")]
    ai = [p for p in done if p["device"] and p["result"].get("ai_fps") is not None]

    def bars(x, y, w, h, title, items, vmax, unit, ref=None, ref_label=""):
        card(img, x, y, w, h)
        put(img, title, x + 16, y + 30, 0.62, INK)
        n = max(1, len(items))
        top, bottom = y + 70, y + h - 60
        slot = (w - 40) / n
        bw = int(min(110, slot * 0.55))
        cv2.line(img, (x + 20, bottom), (x + w - 20, bottom), GRID, 1)
        if ref is not None:
            yy = bottom - int((bottom - top) * min(ref / vmax, 1))
            for xx in range(x + 20, x + w - 20, 10):
                cv2.line(img, (xx, yy), (xx + 4, yy), INK2, 1)
            for xx in range(x + 16, x + 40, 10):
                cv2.line(img, (xx, y + 52), (xx + 4, y + 52), INK2, 1)
            put(img, ref_label, x + 46, y + 56, 0.42, INK2)
        for i, (label, val, col) in enumerate(items):
            cx = int(x + 20 + slot * (i + 0.5))
            if val is not None:
                hh = int((bottom - top) * min(max(val, 0) / vmax, 1))
                cv2.rectangle(img, (cx - bw // 2, bottom - hh), (cx + bw // 2, bottom - 1), col, -1)
                txt = f"{val:.0f}{unit}" if unit == "%" else f"{val:.1f}{unit}"
                put(img, txt, cx, bottom - hh - 10, 0.75, INK, 2, align="center")
            else:
                put(img, "n/a", cx, bottom - 10, 0.6, MUTED, align="center")
            for j, part in enumerate(label.split("\n")):
                put(img, part, cx, bottom + 26 + 22 * j, 0.55, col, 2 if j == 0 else 1, align="center")

    items = lambda key: [(p.get("bar", p["label"]), p["result"].get(key), phase_color(p["device"]))  # noqa: E731
                         for p in ai]
    pw = 405
    bars(16, 84, pw, 470, "Foreground work kept", items("fg_pct"), 110, "%", ref=100,
         ref_label="= AI off (phases before + after)")
    bars(16 + pw + 16, 84, pw, 470, "AI frames per second", items("ai_fps"), max(35, target_fps * 1.15), "",
         ref=target_fps, ref_label=f"target {target_fps:.0f} FPS")
    ms = [p["result"].get("ai_proc_cpu_ms_per_frame") for p in ai]
    bars(16 + 2 * (pw + 16), 84, pw, 470, "CPU time per AI frame", items("ai_proc_cpu_ms_per_frame"),
         max([m for m in ms if m] + [10]) * 1.15, " ms")
    # takeaway lines
    y = 600
    card(img, 16, 570, 1248, 134)
    for i, p in enumerate(ai):
        r = p["result"]
        acc = r.get("npu_util") if p["device"] == "NPU" else r.get("gpu_util") if p["device"] == "GPU" else None
        s = (f"AI on {p['device']}: {fmt(r.get('ai_fps'), '{:.1f}')} FPS, your work keeps "
             f"{fmt(r.get('fg_pct'), '{:.0f}%')}, AI process uses {fmt(r.get('ai_proc_cpu_pct'), '{:.1f}%')} of the CPU "
             f"({fmt(r.get('ai_proc_cpu_ms_per_frame'), '{:.1f}')} ms/frame, of which pre/post "
             f"{fmt(r.get('prepost_ms'), '{:.1f}')} ms)" + (f", {p['device']} busy {acc:.0f}%" if acc is not None else ""))
        cv2.rectangle(img, (30, y + i * 32 - 12), (38, y + i * 32 + 2), phase_color(p["device"]), -1)
        put(img, s, 48, y + i * 32, 0.5, INK)
    return img


# ------------------------------------------------------------------------------------------------ static PNG
def static_chart(data: dict, path):
    import logging

    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    import matplotlib.pyplot as plt

    SURFACE, INKH, INK2H, MUTEDH, GRIDH = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 12.5,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2H, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "xtick.color": INK2H,
        "ytick.color": INK2H, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRIDH, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "text.color": INKH,
    })
    phases = [p for p in data.get("placements") or data["phases"] if p.get("result", {}).get("samples")
              or p.get("result", {}).get("n_phases")]
    ai = [p for p in phases if p["device"]]
    col = lambda d: DEVICE_HEX.get(d, BASE_HEX)  # noqa: E731
    lab = lambda p: p.get("bar", p["label"])  # noqa: E731

    fig, axes = plt.subplots(1, 4, figsize=(16, 6.2))

    def barplot(ax, ps, key, title, ylim, fmtv, ref=None, ref_label=None):
        xs = np.arange(len(ps))
        vals = [p["result"].get(key) for p in ps]
        ax.bar(xs, [v or 0 for v in vals], width=0.6, color=[col(p["device"]) for p in ps], edgecolor=SURFACE,
               linewidth=2)
        dots = False
        for x, p in zip(xs, ps):  # individual rounds as dots: the spread stays visible
            rv = [v for v in (p["result"].get("rounds") or {}).get(key, []) if v is not None]
            if len(rv) > 1:
                ax.scatter([x + 0.38] * len(rv), rv, s=26, color=SURFACE, edgecolor=INKH, linewidth=1.2, zorder=3,
                           clip_on=False)
                dots = True
        for x, p, v in zip(xs, ps, vals):
            if v is not None and v > ylim * 0.15:  # value inside the bar top, clear of the round dots
                ax.text(x, v - ylim * 0.03, fmtv(v), ha="center", va="top", fontsize=11, color="white",
                        fontweight="semibold")
            else:
                ax.text(x, (v or 0) + ylim * 0.015, "n/a" if v is None else fmtv(v), ha="center", va="bottom",
                        fontsize=11, color=INKH, fontweight="semibold")
        if dots:
            ax.scatter([], [], s=26, color=SURFACE, edgecolor=INKH, linewidth=1.2, label="individual rounds")
        if ref is not None:
            ax.axhline(ref, color=INK2H, linewidth=1, linestyle=(0, (4, 3)))
            ax.plot([], [], color=INK2H, linewidth=1, linestyle=(0, (4, 3)), label=ref_label)
        if ref is not None or dots:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=9, handlelength=2.2, ncol=1)
        ax.set_xticks(xs, [lab(p) for p in ps])
        ax.set_ylim(0, ylim)
        ax.set_title(title)

    barplot(axes[0], ai, "fg_pct", "Your work kept while AI runs\n% of zlib ops/s with AI off", 120,
            lambda v: f"{v:.0f}%", 100, "AI off (phases before + after)")
    tgt = data["config"]["target_fps"]
    barplot(axes[1], ai, "ai_fps", "AI pipeline FPS\n(blur + detection)", max(35, tgt * 1.2), lambda v: f"{v:.1f}",
            tgt, f"target {tgt:.0f}")
    # CPU breakdown: who used the CPU during each phase (stacked, % of all logical CPUs)
    ax = axes[2]
    xs = np.arange(len(phases))
    parts = [("fg_proc_cpu_pct", "foreground work", "#52514e", None),
             ("ai_proc_cpu_pct", "AI pipeline process", None, None),
             ("other_procs_cpu_pct", "everything else (OS, other apps, dashboard)", "#d6d4cc", None)]
    bottom = np.zeros(len(phases))
    for key, name, c, _ in parts:
        v = np.array([p["result"].get(key) or 0 for p in phases])
        if key == "dashboard":
            continue
        if key == "other_procs_cpu_pct":
            v = v + np.array([p["result"].get("dashboard_proc_cpu_pct") or 0 for p in phases])
        colors = [col(p["device"]) for p in phases] if c is None else c
        ax.bar(xs, v, bottom=bottom, width=0.6, color=colors, edgecolor=SURFACE, linewidth=1.5, label=name)
        if key == "ai_proc_cpu_pct":
            for x, b0, vv in zip(xs, bottom, v):
                if vv >= 1:
                    ax.text(x, b0 + vv + 1, f"{vv:.0f}%", ha="center", va="bottom", fontsize=10,
                            fontweight="semibold", bbox=dict(fc=SURFACE, ec="none", pad=1))
        bottom += v
    ax.set_xticks(xs, [lab(p) for p in phases])
    ax.set_ylim(0, 105)
    ax.set_title("Who used the CPU\n% of all logical CPUs (mean per placement)")
    handles, labels = ax.get_legend_handles_labels()
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#52514e"), Patch(color=col("CPU")), Patch(color="#d6d4cc")],
              ["foreground work", "AI process (device colour)", "everything else"], loc="upper center",
              bbox_to_anchor=(0.5, -0.17), fontsize=8.5, ncol=2, handlelength=1, columnspacing=0.8)
    # AI process CPU cost per frame: pre/post vs total pipeline-thread CPU vs process
    ax = axes[3]
    xs = np.arange(len(ai))
    pp = [p["result"].get("prepost_ms") or 0 for p in ai]
    proc = [p["result"].get("ai_proc_cpu_ms_per_frame") or 0 for p in ai]
    ax.bar(xs, proc, width=0.6, color=[col(p["device"]) for p in ai], edgecolor=SURFACE, linewidth=2)
    for x, a, b in zip(xs, pp, proc):
        ax.text(x, b + max(proc + [1]) * 0.02, f"{b:.1f} ms", ha="center", va="bottom", fontsize=11,
                fontweight="semibold")
        ax.text(x, b + max(proc + [1]) * 0.09, f"pre/post {a:.1f} ms", ha="center", va="bottom", fontsize=8.5,
                color=INK2H)
    ax.set_xticks(xs, [lab(p) for p in ai])
    ax.set_ylim(0, max(proc + [1]) * 1.3)
    ax.set_title("CPU time per AI frame (ms)\nwhole AI process, incl. inference threads")
    fig.suptitle(data["headline"], x=0.01, ha="left", fontsize=14.5, fontweight="semibold", y=0.99)
    import textwrap
    fig.text(0.01, 0.012, "\n".join(textwrap.wrap(data["footnote"], 215)), fontsize=8.8, color=MUTEDH, ha="left",
             va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path
