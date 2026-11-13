"""Regenerate the NPU showcase chart set (media/charts/*.png + README.md) from benchmarks/results/*.json.

Run:  .venv\\Scripts\\python.exe scripts/make_charts.py               # every chart + README
      .venv\\Scripts\\python.exe scripts/make_charts.py --only cpu_cost  # one chart (README still refreshed)
      .venv\\Scripts\\python.exe scripts/make_charts.py --list

The PNG file names are a contract with the slide decks and the GitHub Pages site: do not rename them.
A missing or malformed input skips the charts that need it with a warning; it never stops the run.
"""
from __future__ import annotations

import argparse
import json
import logging
import textwrap
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ARCHIVE = RESULTS / "archive" / "20261111_npu32.0.100.4724_preliminary"
OUT = ROOT / "media" / "charts"

# Device colours shared with the demo overlays (examples/common/hud.py DEVICE_HEX). Importing hud pulls in cv2,
# so fall back to the same literals if that import is unavailable.
try:
    sys.path.insert(0, str(ROOT))
    from examples.common.hud import DEVICE_HEX  # noqa: E402
except Exception:  # pragma: no cover - fallback only
    DEVICE_HEX = {"NPU": "#00AAF0", "GPU": "#5AC83C", "CPU": "#F57828"}
DEVICES = ["NPU", "GPU", "CPU"]

# Chrome tokens: the same light palette as media/perf_atlas.png (examples/perf_atlas/charts.py).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FAIL = "#fab219"          # status: ran, wrong output (same as the atlas)
PRELIM = "#b9b7ae"        # preliminary run (de-emphasised)
CLEAN = "#2b2a28"         # clean run
BATCH_RAMP = {1: "#9ad6f5", 8: "#2f8fd0", 32: "#0b4f80"}  # ordinal, NPU-blue family

W, H, DPI = 16, 9, 120    # 1920 x 1080 px

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 14, "axes.titlesize": 16,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 12,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.labelsize": 14, "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 13, "text.color": INK,
})

PLAIN = {"NPU": "NPU\n(AI chip)", "GPU": "GPU\n(graphics chip)", "CPU": "CPU\n(main processor)"}
PLAIN_1L = {"NPU": "NPU (AI chip)", "GPU": "GPU (graphics chip)", "CPU": "CPU (main processor)"}


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


class Skip(Exception):
    """Raised when a chart's inputs are missing or unusable."""


# ----------------------------------------------------------------------------------------------- inputs
_cache: dict = {}


def _rel(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load(name, base=RESULTS):
    """Parsed JSON for benchmarks/results/<name>, or raise Skip. Handles the UTF-8 BOM PowerShell writes."""
    path = base / name
    key = str(path)
    if key not in _cache:
        try:
            _cache[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            _cache[key] = Skip(f"{_rel(path)} not found")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            _cache[key] = Skip(f"{_rel(path)} is not valid JSON ({e})")
    v = _cache[key]
    if isinstance(v, Skip):
        raise v
    return v


def try_load(name, base=RESULTS):
    try:
        return load(name, base)
    except Skip:
        return None


def num(v):
    """float(v) when v is a finite number, else None."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def need(v, what):
    f = num(v)
    if f is None:
        raise Skip(f"missing/invalid value: {what}")
    return f


def first_num(d, keys):
    for k in keys:
        f = num(d.get(k)) if isinstance(d, dict) else None
        if f is not None:
            return f
    return None


def context():
    """Hardware / driver / date line shared by every chart."""
    summ = try_load("record_all_summary.json") or {}
    hw = try_load("hardware.json") or {}
    npu_drv = summ.get("npu_driver")
    gpu_drv = summ.get("graphics_driver")
    for a in hw.get("AcceleratorDrivers", []) or []:
        if "NPU" in str(a.get("DeviceName", "")) and not npu_drv:
            npu_drv = a.get("DriverVersion")
        if "Graphics" in str(a.get("DeviceName", "")) and not gpu_drv:
            gpu_drv = a.get("DriverVersion")
    date = str(summ.get("finished") or hw.get("Timestamp") or "")[:10]
    ov = None
    for f in ("perf_atlas.json", "vision_side_by_side.json"):
        d = try_load(f) or {}
        ov = ov or (d.get("meta", {}) or {}).get("openvino_version") or d.get("openvino_version")
    ov = (ov or "2026.4.1").split("-")[0]
    return {"npu_drv": npu_drv or "?", "gpu_drv": gpu_drv or "?", "date": date or "?", "ov": ov}


CTX: dict = {}


def subtitle():
    c = CTX
    return (f"Intel Core Ultra 5 322 (Panther Lake): NPU 5 “Intel AI Boost”, 46 TOPS INT8 · Intel Graphics, "
            f"2 Xe3 cores · CPU 2P + 4LP-E, 6 threads\n"
            f"NPU driver {c['npu_drv']} · graphics driver {c['gpu_drv']} · OpenVINO {c['ov']}  |  measured "
            f"{c['date']} on an idle laptop, on AC, one device at a time")


# ----------------------------------------------------------------------------------------------- layout helpers
def _fit(fig, text, x, y, size, min_size, max_w, line_gap, **kw):
    """Place text at (x, y); shrink to min_size, then wrap onto two lines, so it never runs off the figure.
    Returns the y of the last baseline."""
    r = fig.canvas.get_renderer()
    t = fig.text(x, y, text, fontsize=size, va="baseline", **kw)
    while t.get_window_extent(r).width > max_w * W * DPI and size > min_size:
        size -= 1
        t.set_fontsize(size)
    if t.get_window_extent(r).width > max_w * W * DPI:
        t.remove()
        words, best = text.split(" "), None
        for i in range(1, len(words)):  # most balanced split that fits
            l1, l2 = " ".join(words[:i]), " ".join(words[i:])
            w = max(len(l1), len(l2))
            if best is None or w < best[0]:
                best = (w, l1, l2)
        fig.text(x, y + line_gap, best[1], fontsize=size, va="baseline", **kw)
        fig.text(x, y, best[2], fontsize=size, va="baseline", **kw)
        return y
    return y


def figure(title, sub=None, simple=False):
    """Title (auto-fit), two-line subtitle. Plots should keep their (panel-title) tops below y = 0.76."""
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    if simple:
        _fit(fig, title, 0.05, 0.885, 46, 36, 0.9, 0.08, fontweight="bold", color=INK)
    else:
        _fit(fig, title, 0.04, 0.905, 27, 23, 0.93, 0.05, fontweight="bold", color=INK)
        fig.text(0.04, 0.845, sub or subtitle(), fontsize=13, color=INK2, va="top", linespacing=1.45)
    return fig


def footer(fig, source, simple=False):
    width = 175 if simple else 195
    fig.text(0.05 if simple else 0.04, 0.02, textwrap.fill(source, width), fontsize=13 if simple else 12, color=MUTED,
             va="bottom", linespacing=1.4)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def fmt_ms(v):
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.2f}"


def log_axis(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    a.set_major_formatter(fmt)
    a.set_minor_formatter(NullFormatter())


def device_bars(ax, values, fmt, unit="", ymax=None, label_size=15, width=0.6, labels=None, label_y=None):
    """One bar per device (NPU, GPU, CPU order), value printed on the cap; x tick = device name."""
    devs = [d for d in DEVICES if values.get(d) is not None]
    xs = np.arange(len(devs))
    vals = [values[d] for d in devs]
    ax.bar(xs, vals, width=width, color=[DEVICE_HEX[d] for d in devs], edgecolor=SURFACE, linewidth=2, zorder=3)
    top = ymax or max(vals) * 1.22
    for x, d, v in zip(xs, devs, vals):
        ax.text(x, max(v, (label_y or {}).get(d, v)) + top * (0.04 if d in (label_y or {}) else 0.015), fmt(v) + unit, ha="center", va="bottom", fontsize=label_size,
                fontweight="semibold", color=INK)
    ax.set_xticks(xs, labels or devs)
    ax.set_ylim(0, top)
    ax.grid(axis="x", visible=False)
    return devs


# ----------------------------------------------------------------------------------------------- data extractors
def atlas_rows(base=RESULTS):
    d = load("perf_atlas.json", base)
    rows = d.get("results")
    if not isinstance(rows, list) or not rows:
        raise Skip(f"{base.name}/perf_atlas.json has no results list")
    return d, rows


def atlas_get(rows, model, batch, precision, device):
    for r in rows:
        if (r.get("model"), r.get("batch"), r.get("precision"), r.get("device")) == (model, batch, precision, device):
            return r
    return None


def vision_lat(task):
    d = load("vision_side_by_side.json")
    out = {}
    for dev in DEVICES:
        lat = (((d.get("devices") or {}).get(dev) or {}).get("latency_ms") or {}).get(task) or {}
        p50, p95 = num(lat.get("median_ms")), num(lat.get("p95_ms"))
        if p50 is not None:
            out[dev] = (p50, p95)
    if not out:
        raise Skip(f"vision_side_by_side.json has no '{task}' latency")
    return out


def depth_lat():
    d = load("depth_parallax.json")
    out = {}
    for dev in DEVICES:
        b = (((d.get("devices") or {}).get(dev) or {}).get("benchmark")) or {}
        p50, p95 = num(b.get("median_ms")), num(b.get("p95_ms"))
        if p50 is not None:
            out[dev] = (p50, p95)
    if not out:
        raise Skip("depth_parallax.json has no per-device benchmark")
    return out


def llm_devices(base=RESULTS):
    d = load("llm_chat.json", base)
    devs = {}
    for rec in d.get("devices") or []:
        if isinstance(rec, dict) and rec.get("device") in DEVICES and rec.get("status") in ("executed", "ok", None):
            devs[rec["device"]] = rec
    if not devs:
        raise Skip("llm_chat.json has no executed devices")
    return d, devs


def llm_token_latency(rec):
    """Pooled inter-token intervals (ms) over all prompts, first token (TTFT) excluded."""
    gaps = []
    for p in rec.get("prompts") or []:
        t = [num(x) for x in (p.get("token_times_ms") or [])]
        t = [x for x in t if x is not None]
        if len(t) >= 2:
            gaps.extend(np.diff(t).tolist())
    if not gaps:
        return None
    return float(np.percentile(gaps, 50)), float(np.percentile(gaps, 95)), len(gaps)


def llm_summary(rec, key, prompt_key):
    v = num((rec.get("summary") or {}).get(key))
    if v is None:  # fall back to the median over prompts
        vals = [num(p.get(prompt_key)) for p in rec.get("prompts") or []]
        vals = [x for x in vals if x is not None]
        v = float(np.median(vals)) if vals else None
    return v


def llm_load(rec):
    load_ = rec.get("load") or {}
    first = num((load_.get("first") or {}).get("load_s"))
    cold = num(load_.get("cold_s"))
    cached = num(load_.get("cached_s"))
    label = (load_.get("first") or {}).get("cache")
    return {"first": cold if cold is not None else first, "cached": cached, "first_label": label}


OFFLOAD_JSON = [RESULTS / "offload_dashboard.json"]  # overridable with --offload-json (for testing new formats)


def _placement_device(p):
    dev = p.get("device")
    if isinstance(dev, str):
        dev = dev.split(".")[0].upper()
        return dev if dev in DEVICES else None
    text = " ".join(str(p.get(k, "")) for k in ("label", "short", "name", "placement")).upper()
    if "AI OFF" in text or text.strip() in ("OFF", "NONE"):
        return None
    for d in DEVICES:
        if d in text.split() or f"ON {d}" in text:
            return d
    return None


def offload_phases():
    """Per-placement results from offload_dashboard.json.

    Returns (doc, {device or 'off': result}, fmt) where fmt is 'placements' (new design: aggregated per placement,
    adjacent AI-off baselines, per-round lists in result['rounds']) or 'phases' (old single-pass design).
    """
    path = OFFLOAD_JSON[0]
    d = load(path.name, path.parent)
    out = {}
    placements = d.get("placements")
    if isinstance(placements, list) and placements:
        fmt, items = "placements", placements
    else:
        fmt, items = "phases", d.get("phases") or []
    for ph in items:
        if not isinstance(ph, dict):
            continue
        res = ph.get("result") or ph.get("results") or ph.get("summary") or ph
        if not isinstance(res, dict):
            continue
        dev = _placement_device(ph)
        if dev in DEVICES and str(ph.get("status", "ok")) in ("ok", "executed"):
            out.setdefault(dev, res)
        elif dev is None and "off" not in out:
            out["off"] = res  # (first) AI-off baseline
    if not any(k in out for k in DEVICES):
        raise Skip(f"{path.name}: no per-device {fmt} found")
    return d, out, fmt


def offload_metric(res, keys):
    """(value, [per-round values]) for the first key present in res; rounds come from res['rounds'][key]."""
    if not isinstance(res, dict):
        return None, []
    for k in keys:
        v = num(res.get(k))
        if v is not None:
            rounds = (res.get("rounds") or {}).get(k) if isinstance(res.get("rounds"), dict) else None
            rr = [num(x) for x in rounds] if isinstance(rounds, list) else []
            return v, [x for x in rr if x is not None]
    return None, []


# ----------------------------------------------------------------------------------------------- charts
@dataclass
class Section:
    name: str
    title: str
    takeaway: str
    sources: list
    numbers: list = field(default_factory=list)
    notes: list = field(default_factory=list)


CHARTS = {}


def chart(name, kind):
    def deco(fn):
        CHARTS[name] = (fn, kind)
        return fn
    return deco


@chart("latency_by_workload", "technical")
def c_latency(draw=True):
    _, rows = atlas_rows()
    work = []  # (label, {dev: (p50, p95)}, source)

    def add(label, getter, src):
        try:
            work.append((label, getter(), src))
        except Skip as e:
            warn(f"latency_by_workload: dropping '{label.splitlines()[0]}' ({e})")

    add("Vision detect\nYOLOX-tiny 416, INT8", lambda: vision_lat("detect"), "vision_side_by_side.json")
    add("Segmentation\nselfie seg 256", lambda: vision_lat("segment"), "vision_side_by_side.json")
    add("Depth\nDepth-Anything-V2-S 364", depth_lat, "depth_parallax.json")

    def atlas(model, batch):
        def g():
            out = {}
            for dev in DEVICES:
                r = atlas_get(rows, model, batch, "FP16", dev)
                if r and r.get("status") == "executed" and num(r.get("median_ms")) is not None:
                    out[dev] = (num(r["median_ms"]), num(r.get("p95_ms")))
            if not out:
                raise Skip(f"no executed {model} b{batch} FP16 rows")
            return out
        return g

    add("Atlas CONV-M\nbatch 32, FP16", atlas("CONV-M", 32), "perf_atlas.json")
    add("Atlas MLP-L\nbatch 1, FP16", atlas("MLP-L", 1), "perf_atlas.json")

    def llm():
        _, devs = llm_devices()
        out = {}
        for dev, rec in devs.items():
            t = llm_token_latency(rec)
            if t:
                out[dev] = t[:2]
        if not out:
            raise Skip("no token timings")
        return out
    add("LLM per-token\nPhi-3-mini INT4", llm, "llm_chat.json")
    if len(work) < 2:
        raise Skip("fewer than two workloads available")

    wins = [lbl for lbl, v, _ in work if min(v, key=lambda d: v[d][0]) == "NPU"]
    title = f"The NPU has the lowest median latency on {len(wins)} of {len(work)} workloads"
    nums = []
    for lbl, v, src in work:
        best = min(v, key=lambda d: v[d][0])
        nums.append(f"{lbl.replace(chr(10), ', ')} ({src}): " + "; ".join(
            f"{d} p50 {v[d][0]:.2f} ms / p95 {v[d][1]:.2f} ms" for d in DEVICES if d in v) + f" -> fastest {best}")
    sec = Section("latency_by_workload", title,
                  title + " (log scale; p50 bars, p95 whiskers; lower is better).",
                  sorted({s for *_, s in work}), nums,
                  ["LLM per-token = pooled inter-token intervals over the 4 prompts (TTFT excluded).",
                   "Vision/segmentation timings exclude decode, pre/post-processing and drawing."])
    if not draw:
        return sec

    fig = figure(title)
    ax = fig.add_axes([0.065, 0.17, 0.91, 0.59])
    n = len(work)
    bw = 0.26
    for i, (lbl, v, _) in enumerate(work):
        for j, dev in enumerate(DEVICES):
            if dev not in v:
                continue
            x = i + (j - 1) * bw
            p50, p95 = v[dev]
            ax.bar(x, p50, width=bw, color=DEVICE_HEX[dev], edgecolor=SURFACE, linewidth=2, zorder=3)
            if p95 is not None and p95 > p50:
                ax.plot([x, x], [p50, p95], color=INK, lw=1.6, zorder=4, solid_capstyle="butt")
                ax.plot([x - bw * 0.22, x + bw * 0.22], [p95, p95], color=INK, lw=1.6, zorder=4)
            top = max(p50, p95 or p50)
            ax.text(x, top * 1.12, fmt_ms(p50), ha="center", va="bottom", fontsize=12, color=INK2, zorder=5)
    ax.set_yscale("log")
    log_axis(ax)
    allv = [x for _, v, _ in work for d in v for x in v[d] if x]
    ax.set_ylim(min(allv) / 2.2, max(allv) * 3)
    ax.set_xticks(range(n), [w[0] for w in work], fontsize=14)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_xlim(-0.55, n - 0.45)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("latency per inference, ms (log scale)")
    handles = [Patch(color=DEVICE_HEX[d], label=d) for d in DEVICES] + [
        Line2D([], [], color=INK, lw=1.6, label="whisker = p95")]
    ax.legend(handles=handles, loc="upper left", ncol=4, fontsize=14, handlelength=1.4)
    ax.text(0.995, 0.985, "lower is better · bar = p50 (value printed) · log scale: each gridline step is "
            "×2 or ×2.5", transform=ax.transAxes, ha="right", va="top", fontsize=12, color=INK2)
    footer(fig, "Source: benchmarks/results/vision_side_by_side.json, depth_parallax.json, perf_atlas.json (FP16), "
                "llm_chat.json (inter-token interval, TTFT excluded). Vision timings exclude decode, pre/post-processing "
                "and drawing. Generated by scripts/make_charts.py.")
    save(fig, sec.name)
    return sec


@chart("llm_compare", "technical")
def c_llm(draw=True):
    d, devs = llm_devices()
    tok = {k: llm_summary(r, "tok_s_median", "tok_s") for k, r in devs.items()}
    ttft = {k: llm_summary(r, "ttft_ms_median", "ttft_ms") for k, r in devs.items()}
    cores = {k: llm_summary(r, "cpu_cores_busy_median", "cpu_cores_busy") for k, r in devs.items()}
    loads = {k: llm_load(r) for k, r in devs.items()}
    if not any(v is not None for v in tok.values()):
        raise Skip("llm_chat.json: no tok/s values")
    model = ((d.get("meta") or {}).get("model") or {}).get("hf_id", "Phi-3-mini INT4")
    npu, gpu, cpu = tok.get("NPU"), tok.get("GPU"), tok.get("CPU")
    if npu and gpu and cores.get("NPU") and cores.get("GPU") and npu >= 0.85 * gpu and cores["NPU"] < cores["GPU"]:
        title = (f"The NPU generates nearly as fast as the GPU ({npu:.1f} vs {gpu:.1f} tok/s) with "
                 f"{cores['GPU'] / cores['NPU']:.0f}\u00d7 less CPU; the GPU answers first")
    else:
        title = "LLM on three devices: speed, first-token time, CPU use and load time"
    nums = [f"{k}: {tok[k]:.2f} tok/s, TTFT {ttft[k]:.1f} ms, CPU cores busy {cores[k]:.2f} of 6, "
            f"first load {loads[k]['first']} s (labelled '{loads[k]['first_label']}'), cached load "
            + (f"{loads[k]['cached']} s" if loads[k]['cached'] is not None else "n/a")
            for k in DEVICES if k in devs]
    sec = Section("llm_compare", title, title + ".", ["llm_chat.json"], nums,
                  ["Medians over 4 prompts, greedy decoding, max 160 new tokens, devices run one at a time.",
                   "NPU first load (30.22 s) is recorded as cache='cached' but is really a cold compile: depth_parallax "
                   "wrote a blob to the shared models/npucache just before, so the 'blobs present' check misfired."])
    if not draw:
        return sec

    fig = figure(title)
    boxes = [[0.05, 0.15, 0.19, 0.53], [0.29, 0.15, 0.19, 0.53], [0.53, 0.15, 0.19, 0.53], [0.77, 0.15, 0.21, 0.53]]
    a1, a2, a3, a4 = [fig.add_axes(b) for b in boxes]
    device_bars(a1, tok, lambda v: f"{v:.1f}")
    a1.set_title("Decode speed\ntokens / s (higher is better)")
    device_bars(a2, ttft, lambda v: f"{v:.0f}", " ms")
    a2.set_title("Time to first token\nms (lower is better)")
    device_bars(a3, cores, lambda v: f"{v:.2f}", ymax=6 * 0.42)
    a3.set_title("CPU cores busy while generating\nof 6 logical CPUs (lower is better)")
    # load time: first (solid) vs cached (light) per device
    xs = np.arange(len(DEVICES))
    bw = 0.36
    top = max([v for L in loads.values() for v in (L["first"], L["cached"]) if v] or [1]) * 1.2
    for i, dev in enumerate(DEVICES):
        L = loads.get(dev)
        if not L:
            continue
        for k, (val, alpha, off) in enumerate(((L["first"], 1.0, -bw / 2), (L["cached"], 0.38, bw / 2))):
            if val is None:
                a4.text(i + off, top * 0.01, "\u2014", ha="center", va="bottom", fontsize=14, color=MUTED)
                continue
            a4.bar(i + off, val, width=bw, color=DEVICE_HEX[dev], alpha=alpha, edgecolor=SURFACE, linewidth=2,
                   zorder=3)
            a4.text(i + off, val + top * 0.015, f"{val:.1f}", ha="center", va="bottom", fontsize=13,
                    fontweight="semibold", color=INK)
    a4.set_xticks(xs, DEVICES)
    a4.set_ylim(0, top)
    a4.grid(axis="x", visible=False)
    a4.set_title("Model load time, s\nfirst start vs from compile cache")
    a4.legend(handles=[Patch(color=MUTED, label="first start (compiles)"),
                       Patch(color=MUTED, alpha=0.38, label="cached start")], loc="upper right", fontsize=12)
    footer(fig, f"Source: benchmarks/results/llm_chat.json. Model {model} (3.8B, INT4). Medians over 4 prompts, greedy, "
                "≤160 new tokens; CPU cores busy = process CPU time / wall time. NPU first load is a cold compile even "
                "though the JSON labels it 'cached' (shared models/npucache). \u2014 = CPU uses no compile cache.")
    save(fig, sec.name)
    return sec


def offload_summary():
    """Everything cpu_cost / simple_cpu_free need from offload_dashboard.json (either format)."""
    d, ph, fmt = offload_phases()
    s = {"fmt": fmt, "doc": d}
    for key, keys in (("ms", ["ai_proc_cpu_ms_per_frame", "ai_cpu_ms_per_frame", "cpu_ms_per_frame"]),
                      ("kept", ["fg_pct", "work_kept_pct"]),
                      ("fps", ["ai_fps", "fps"]),
                      ("sys", ["cpu_pct", "system_cpu_pct"])):
        s[key] = {k: offload_metric(ph.get(k), keys) for k in DEVICES}
    s["sys_off"] = offload_metric(ph.get("off"), ["cpu_pct", "system_cpu_pct"])[0]
    # "Work kept" is only trustworthy with adjacent AI-off baselines (new design). The old single-pass run drifted
    # (-20 % between its two baselines) and reports >100 % for every placement, so it is not plotted.
    kept_vals = [v for v, _ in s["kept"].values() if v is not None]
    s["kept_ok"] = bool(kept_vals) and (fmt == "placements" or max(kept_vals) <= 100.5)
    fg = (d.get("config") or {}).get("fg_workers")
    s["fg_desc"] = f"{int(num(fg))} foreground workers" if num(fg) else "foreground workers"
    s["baseline"] = d.get("baseline") if isinstance(d.get("baseline"), dict) else {}
    return s


def _vals(m):
    return {k: v for k, (v, _) in m.items() if v is not None}


def round_dots(ax, metric):
    """Per-round values as small ink dots over each device bar (devices in DEVICES order, bars at 0..n-1)."""
    devs = [d for d in DEVICES if metric.get(d, (None, []))[0] is not None]
    plotted = False
    for x, d in enumerate(devs):
        rr = metric[d][1]
        if len(rr) >= 2:
            offs = np.linspace(-0.12, 0.12, len(rr))
            ax.scatter(x + offs, rr, s=70, color=INK, edgecolor=SURFACE, linewidth=2, zorder=6)
            plotted = True
    return plotted


def _round_max(m):
    return {k: max(rr) for k, (_, rr) in m.items() if rr}


def _rounds_txt(m, k, f):
    return f" (rounds {', '.join(format(x, f) for x in m[k][1])})" if m[k][1] else ""


@chart("cpu_cost", "technical")
def c_cpu_cost(draw=True):
    panels, nums, srcs, notes = [], [], [], []
    off = None
    s = None
    try:
        s = offload_summary()
        srcs.append("offload_dashboard.json")
        ms, kept, fps = s["ms"], s["kept"], s["fps"]
        if _vals(ms):
            panels.append(("ms", ms))
            nums.append(f"Offload ({s['fmt']} format) AI-process CPU ms per AI frame: " + ", ".join(
                f"{k} {v:.2f}" + _rounds_txt(ms, k, ".2f") for k, v in _vals(ms).items()))
        if s["kept_ok"]:
            panels.append(("kept", kept))
            nums.append("Background work kept, % of AI-off baseline: " + ", ".join(
                f"{k} {v:.1f}" + _rounds_txt(kept, k, ".1f") for k, v in _vals(kept).items()))
        elif _vals(s["sys"]):
            off = s["sys_off"]
            panels.append(("sys", s["sys"]))
            nums.append("Offload system CPU % (old single-pass format): AI off " +
                        (f"{off:.1f}" if off is not None else "n/a") + ", " +
                        ", ".join(f"{k} {v:.2f}" for k, v in _vals(s["sys"]).items()))
            notes.append("Old-format offload file: 'work kept' (fg_pct) is not plotted because the single-pass run's "
                         "baseline drifted and every placement reports >100 % (" + ", ".join(
                             f"{k} {v:.1f}" for k, v in _vals(kept).items()) + ").")
        if _vals(fps):
            nums.append("AI FPS achieved (target 30): " + ", ".join(f"{k} {v:.2f}" for k, v in _vals(fps).items()))
        b = s["baseline"]
        if b:
            nums.append("Offload baseline: " + ", ".join(
                f"{k}={b[k]}" for k in ("mean_ops_s", "reference_ops_s", "drift_pct", "drift_first_to_last_pct",
                                        "spread_pct", "method") if k in b))
    except Skip as e:
        warn(f"cpu_cost: offload panels skipped ({e})")
        s = None
    try:
        _, devs = llm_devices()
        cores = {k: (llm_summary(devs[k], "cpu_cores_busy_median", "cpu_cores_busy") if k in devs else None, [])
                 for k in DEVICES}
        if _vals(cores):
            panels.append(("llm", cores))
            nums.append("LLM CPU cores busy (median of 4 prompts, of 6): " + ", ".join(
                f"{k} {v:.2f}" for k, v in _vals(cores).items()))
            srcs.append("llm_chat.json")
    except Skip as e:
        warn(f"cpu_cost: LLM panel skipped ({e})")
    if not panels:
        raise Skip("no CPU-cost inputs")

    def npu_best(m, low=True):
        v = _vals(m)
        return bool(v) and (min if low else max)(v, key=v.get) == "NPU"
    lowest = all(npu_best(m, kind != "kept") for kind, m in panels if kind != "sys")
    msv = _vals(panels[0][1]) if panels[0][0] == "ms" else {}
    if lowest and "NPU" in msv and "CPU" in msv:
        title = (f"NPU frees the CPU: {msv['NPU']:.0f} ms of CPU time per AI frame, vs {msv['CPU']:.0f} ms "
                 f"with the AI on the CPU")
    elif lowest:
        title = "NPU frees the CPU: the lowest CPU cost of any placement, in every test"
    else:
        title = "CPU cost of running AI on each device"
    notes.append("CPU ms per AI frame = CPU time of the AI process / AI frames processed; it includes "
                 "pre/post-processing, which runs on the CPU in every placement, so it is comparable even when AI FPS differs.")
    if s:
        notes.append("offload_dashboard.json is being redesigned; re-run scripts/make_charts.py after the final recording.")
    sec = Section("cpu_cost", title, title + ".", srcs, nums, notes)
    if not draw:
        return sec

    fig = figure(title)
    nP = len(panels)
    gap = 0.07
    wP = (0.92 - gap * (nP - 1)) / nP
    for i, (kind, m) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * (wP + gap), 0.17, wP, 0.51])
        vals = _vals(m)
        if kind == "ms":
            hi = max(list(vals.values()) + [x for _, rr in m.values() for x in rr])
            device_bars(ax, vals, lambda v: f"{v:.1f}", " ms", ymax=hi * 1.25, label_y=_round_max(m))
            ax.set_title("Video AI: CPU time per AI frame\nms (lower is better)")
            fv = _vals(s["fps"]) if s else {}
            if fv:
                devs = [d for d in DEVICES if d in vals]
                ax.set_xticks(range(len(devs)), [f"{d}\n{fv[d]:.0f} AI fps" if d in fv else d for d in devs])
        elif kind == "kept":
            hi = max(list(vals.values()) + [x for _, rr in m.values() for x in rr] + [100])
            device_bars(ax, vals, lambda v: f"{v:.0f}", " %", ymax=hi * 1.22, label_y=_round_max(m))
            ax.axhline(100, color=INK2, lw=1.4, zorder=4)
            ax.text(1.01, 100, "AI off\n= 100 %", transform=ax.get_yaxis_transform(), fontsize=12.5,
                    color=INK2, va="center", ha="left")
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title("Same test: background work kept\n% of its AI-off speed (higher is better)")
        elif kind == "sys":
            device_bars(ax, vals, lambda v: f"{v:.0f}", " %", ymax=118)
            ax.set_yticks([0, 25, 50, 75, 100])
            if off is not None:
                ax.axhline(off, color=INK2, lw=1.4, zorder=4)
                ax.text(1.01, off, f"AI off\n{off:.0f} %", transform=ax.get_yaxis_transform(), fontsize=12.5,
                        color=INK2, va="center", ha="left")
            ax.set_title("Same test: whole-system CPU busy, %\n(background work alone = line)")
        else:
            device_bars(ax, vals, lambda v: f"{v:.2f}", ymax=6 * 0.42)
            ax.set_title("Local chatbot (Phi-3-mini)\nCPU cores busy, of 6 (lower is better)")
        if round_dots(ax, m):
            ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=INK, label="one round"),
                               Patch(color=MUTED, label="bar = all rounds")], loc="upper left", fontsize=12.5)
    fg = s["fg_desc"] if s else "foreground workers"
    design = ("adjacent AI-off baselines, rounds in alternating order" if s and s["fmt"] == "placements"
              else "single pass, one AI-off baseline at each end")
    footer(fig, f"Source: benchmarks/results/offload_dashboard.json (selfie segmentation + YOLOX-tiny, 30 FPS target; pre/"
                f"post-processing on the CPU in every placement; {fg} as background load; {design}) and llm_chat.json "
                f"(median of 4 prompts).")
    save(fig, sec.name)
    return sec


@chart("compile_cost", "technical")
def c_compile(draw=True):
    groups = []  # (title, {dev: (first, cached, note_cached)})
    nums = []
    srcs = []
    try:
        _, devs = llm_devices()
        g = {}
        for k in DEVICES:
            if k in devs:
                L = llm_load(devs[k])
                g[k] = (L["first"], L["cached"], "no cache" if k == "CPU" else "n/a")
        groups.append(("LLM: Phi-3-mini INT4 (3.8B)", g))
        srcs.append("llm_chat.json")
    except Skip as e:
        warn(f"compile_cost: LLM skipped ({e})")
    try:
        dd = load("depth_parallax.json")
        g = {}
        for k in DEVICES:
            r = (dd.get("devices") or {}).get(k) or {}
            c = num(r.get("compile_s"))
            if c is None:
                continue
            cache = str(r.get("cache") or "")
            if cache.startswith("warm"):
                g[k] = (None, c, "")
            else:
                g[k] = (c, None, "no cache" if not cache else "not\nrecorded")
        groups.append(("Depth: Depth-Anything-V2-S 364", g))
        srcs.append("depth_parallax.json")
    except Skip as e:
        warn(f"compile_cost: depth skipped ({e})")
    try:
        vd = load("vision_side_by_side.json")
        g = {}
        for k in DEVICES:
            comp = (((vd.get("devices") or {}).get(k) or {}).get("compile")) or {}
            parts = [num((comp.get(t) or {}).get("compile_s")) for t in ("detect", "segment")]
            if all(p is not None for p in parts):
                g[k] = (sum(parts), None, "no cache\nused")
        groups.append(("Vision: YOLOX-tiny + selfie seg", g))
        srcs.append("vision_side_by_side.json")
    except Skip as e:
        warn(f"compile_cost: vision skipped ({e})")
    groups = [(t, g) for t, g in groups if g]
    if not groups:
        raise Skip("no compile timings")
    for t, g in groups:
        nums.append(f"{t}: " + "; ".join(
            f"{k} first " + ("n/a" if g[k][0] is None else f"{round(g[k][0], 3)} s") + ", cached "
            + ("n/a" if g[k][1] is None else f"{round(g[k][1], 3)} s") for k in DEVICES if k in g))
    llm = dict(groups[0][1]) if groups[0][0].startswith("LLM") else {}
    if llm.get("NPU") and llm["NPU"][0] and llm["NPU"][1]:
        title = (f"The NPU's first compile is the slowest ({llm['NPU'][0]:.0f} s for the LLM); "
                 f"the compile cache cuts it to {llm['NPU'][1]:.1f} s")
    else:
        title = "First compile vs cached load, per device"
    sec = Section("compile_cost", title, title + ".", srcs, nums,
                  ["Vision = detect + segment compile_s summed; that example uses no compile cache, so every start pays it.",
                   "Depth cached load was not recorded in the clean run (cache was cold).",
                   "LLM NPU first load is labelled 'cached' in the JSON but is a cold compile (see llm_compare)."])
    if not draw:
        return sec

    fig = figure(title)
    nG = len(groups)
    gap = 0.06
    wG = (0.9 - gap * (nG - 1)) / nG
    top = max(v for _, g in groups for f, c, _ in g.values() for v in (f, c) if v) * 1.18
    bw = 0.36
    ax0 = None
    for i, (t, g) in enumerate(groups):
        ax = fig.add_axes([0.07 + i * (wG + gap), 0.15, wG, 0.55], sharey=ax0)
        ax0 = ax0 or ax
        for j, dev in enumerate(DEVICES):
            if dev not in g:
                continue
            first, cached, note = g[dev]
            for val, alpha, off, empty in ((first, 1.0, -bw / 2, "n/a"), (cached, 0.38, bw / 2, note)):
                if val is None:
                    ax.text(j + off, top * 0.01, "\u2014", ha="center", va="bottom", fontsize=14, color=MUTED)
                    continue
                ax.bar(j + off, val, width=bw, color=DEVICE_HEX[dev], alpha=alpha, edgecolor=SURFACE, linewidth=2,
                       zorder=3)
                ax.text(j + off, val + top * 0.012, f"{val:.1f}" if val >= 1 else f"{val:.2f}", ha="center",
                        va="bottom", fontsize=13, fontweight="semibold", color=INK)
        ax.set_xticks(range(len(DEVICES)), DEVICES)
        ax.set_ylim(0, top)
        ax.grid(axis="x", visible=False)
        ax.set_title(t)
        if i == 0:
            ax.set_ylabel("seconds to compile / load (same scale in every panel)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    fig.axes[0].legend(handles=[Patch(color=MUTED, label="first start (compile)"),
                                Patch(color=MUTED, alpha=0.38, label="start from compile cache"),
                                Line2D([], [], ls="", label="\u2014  = no cached start (see note)")],
                       loc="upper right", fontsize=13)
    footer(fig, "Source: benchmarks/results/llm_chat.json (load_s), depth_parallax.json (compile_s), vision_side_by_side."
                "json (detect + segment compile_s). Vision uses no compile cache, so it pays this on every start; depth "
                "cached load not recorded in the clean run (\u2014 = no cached start). LLM NPU first load is a cold compile despite its 'cached' label.")
    save(fig, sec.name)
    return sec


@chart("npu_batch_bug", "technical")
def c_batch_bug(draw=True):
    d, rows = atlas_rows()
    batches = [b for b in ((d.get("meta") or {}).get("grid") or {}).get("batches", [1, 8, 32])]
    conv = ["CONV-XS", "CONV-S", "CONV-M"]
    mlp = ["MLP-S", "MLP-M", "MLP-L"]
    err = {}
    lat = {}
    fails = [r for r in rows if r.get("device") == "NPU" and r.get("status") == "failed_correctness"]
    tol = None
    for m in conv:
        for b in batches:
            r = atlas_get(rows, m, b, "FP16", "NPU")
            if r:
                err[(m, b)] = (num(r.get("rel_err")), r.get("status"))
                tol = tol or num(r.get("tolerance"))
    for m in mlp:
        for b in batches:
            r = atlas_get(rows, m, b, "FP16", "NPU")
            if r and r.get("status") == "executed":
                lat[(m, b)] = num(r.get("median_ms"))
    if not err and not lat:
        raise Skip("no NPU FP16 conv/MLP rows")
    l8, l32 = lat.get(("MLP-L", 8)), lat.get(("MLP-L", 32))
    fail_b = sorted({r.get("batch") for r in fails})
    title = (f"NPU batching bug: conv stacks at batch 8 return wrong results, and MLP-L at batch 8 takes "
             f"{l8:.1f} ms vs {l32:.1f} ms at batch 32" if l8 and l32 else "NPU batching: correctness and latency by batch")
    nums = ["NPU FP16 conv relative error vs numpy (tolerance %s): " % tol + "; ".join(
        f"{m} b{b} {e[0]:.6f} ({e[1]})" for (m, b), e in err.items() if e[0] is not None),
        f"NPU failed_correctness configs in the whole grid: {len(fails)} (" + ", ".join(
            f"{r['model']} b{r['batch']} {r['precision']}" for r in fails) + ")",
        "NPU FP16 MLP p50 ms: " + "; ".join(f"{m} b{b} {v:.4f}" for (m, b), v in lat.items() if v is not None)]
    sec = Section("npu_batch_bug", title, title + f" (still present on NPU driver {CTX.get('npu_drv')}).",
                  ["perf_atlas.json"], nums,
                  ["The atlas grid tests batch 1, 8 and 32 only. The wider 'batch 2-8 fail, batch 1 and >=12 correct' "
                   "finding is documented in docs/DRIVER_SETUP.md / examples/perf_atlas/README.md, not in this JSON.",
                   "The same 9 configs (3 conv sizes x FP32/FP16/INT8 at batch 8) fail with identical errors on the "
                   "preliminary driver 32.0.100.4724."])
    if not draw:
        return sec

    fig = figure(title)
    a1 = fig.add_axes([0.07, 0.2, 0.4, 0.48])
    a2 = fig.add_axes([0.57, 0.2, 0.4, 0.48])
    bw = 0.26
    for i, m in enumerate(conv):
        for j, b in enumerate(batches):
            e, st = err.get((m, b), (None, None))
            if e is None:
                continue
            x = i + (j - 1) * bw
            bad = st != "executed"
            a1.bar(x, e, width=bw, color=FAIL if bad else BATCH_RAMP.get(b, MUTED), edgecolor=SURFACE, linewidth=2,
                   zorder=3)
            a1.text(x, e * 1.15, ("wrong\noutput" if bad else f"{e:.4f}"), ha="center", va="bottom",
                    fontsize=12 if bad else 11, color=INK if bad else INK2, fontweight="semibold" if bad else "normal")
    a1.set_yscale("log")
    log_axis(a1)
    a1.set_ylim(1e-4, 3)
    if tol:
        a1.axhline(tol, color=INK, lw=1.4, zorder=4)
    a1.set_xticks(range(len(conv)), conv)
    a1.grid(axis="x", visible=False)
    a1.set_ylabel("relative error vs numpy (log scale)")
    a1.set_title("Correctness: conv stacks on the NPU, FP16")
    a1.legend(handles=[Patch(color=BATCH_RAMP[b], label=f"batch {b}") for b in batches if b in BATCH_RAMP] +
              [Patch(color=FAIL, label="ran, wrong output")] +
              ([Line2D([], [], color=INK, lw=1.4, label=f"pass threshold {tol:g}")] if tol else []),
              loc="upper left", ncol=3, fontsize=12.5)
    a1.text(0.0, -0.17, f"{len(fails)} NPU configs fail in the full grid: all conv sizes at batch "
            f"{', '.join(map(str, fail_b))}, in FP32, FP16 and INT8.", transform=a1.transAxes, fontsize=12.5,
            color=INK2, va="top")
    top = max(v for v in lat.values() if v) * 1.2
    for i, m in enumerate(mlp):
        for j, b in enumerate(batches):
            v = lat.get((m, b))
            if v is None:
                continue
            x = i + (j - 1) * bw
            a2.bar(x, v, width=bw, color=BATCH_RAMP.get(b, MUTED), edgecolor=SURFACE, linewidth=2, zorder=3)
            a2.text(x, v + top * 0.012, f"{v:.1f}", ha="center", va="bottom", fontsize=12.5,
                    fontweight="semibold" if b == 8 and m == "MLP-L" else "normal", color=INK)
    a2.set_ylim(0, top)
    a2.set_xticks(range(len(mlp)), mlp)
    a2.grid(axis="x", visible=False)
    a2.set_ylabel("p50 latency per inference, ms (whole batch)")
    a2.set_title("Latency: MLP stacks on the NPU, FP16")
    a2.legend(handles=[Patch(color=BATCH_RAMP[b], label=f"batch {b}") for b in batches if b in BATCH_RAMP],
              loc="upper left", fontsize=12.5)
    if l8 and l32:
        a2.text(0.0, -0.17, f"Batch 8 is the slow case: MLP-L b8 {l8:.1f} ms vs b32 {l32:.1f} ms "
                f"({l8 / l32:.1f}× slower for 4× less work).", transform=a2.transAxes, fontsize=12.5,
                color=INK2, va="top")
    footer(fig, f"Source: benchmarks/results/perf_atlas.json (synthetic seeded stacks, checked against numpy). Grid tests "
                f"batch {', '.join(map(str, batches))} only; the wider 'batch 2–8 wrong, 1 and ≥12 correct' "
                f"result is documented in examples/perf_atlas/README.md. Same failures on preliminary driver 32.0.100.4724.")
    save(fig, sec.name)
    return sec


@chart("driver_contention", "technical")
def c_contention(draw=True):
    _, clean = atlas_rows()
    _, pre = atlas_rows(ARCHIVE)
    pre_hw = try_load("hardware.json", ARCHIVE) or {}
    pdrv = {("NPU" if "NPU" in str(a.get("DeviceName")) else "GPU"): a.get("DriverVersion")
            for a in pre_hw.get("AcceleratorDrivers", []) or []}
    cfgs = [("CPU", "CONV-M", 32), ("CPU", "MLP-L", 32), ("GPU", "CONV-S", 32), ("GPU", "MLP-L", 1),
            ("NPU", "CONV-M", 32), ("NPU", "MLP-L", 1)]
    rows = []
    for dev, m, b in cfgs:
        a, c = atlas_get(pre, m, b, "FP16", dev), atlas_get(clean, m, b, "FP16", dev)
        pa, pc = num((a or {}).get("median_ms")), num((c or {}).get("median_ms"))
        if pa is not None and pc is not None:
            rows.append((dev, f"{m} b{b}", pa, pc))
    llm = []
    try:
        _, dc = llm_devices()
        _, dp = llm_devices(ARCHIVE)
        if "GPU" in dc and "GPU" in dp:
            a, c = llm_load(dp["GPU"])["first"], llm_load(dc["GPU"])["first"]
            if a and c:
                llm.append(("GPU", "LLM cold load", a, c))
    except Skip as e:
        warn(f"driver_contention: LLM load skipped ({e})")
    if not rows and not llm:
        raise Skip("no comparable preliminary/clean pairs")
    ratios = {d: [p / c for dd, _, p, c in rows if dd == d] for d in DEVICES}
    other = [r for d in ("CPU", "GPU") for r in ratios[d]]
    npu = ratios["NPU"]
    title = "Measurement conditions matter: CPU and GPU timings moved up to " \
            f"{max(other):.1f}× between runs; NPU timings barely moved" if other and npu else \
            "Preliminary vs clean run"
    nums = [f"{dev} {lbl} FP16 p50: preliminary {p:.4f} ms -> clean {c:.4f} ms ({p / c:.2f}x)" for dev, lbl, p, c in rows]
    nums += [f"{dev} {lbl}: preliminary {p:.2f} s -> clean {c:.2f} s ({p / c:.1f}x)" for dev, lbl, p, c in llm]
    sec = Section("driver_contention", title, title + ". Drivers AND machine load both changed, so the difference "
                  "cannot be attributed to the driver alone.", ["perf_atlas.json", "llm_chat.json",
                  "archive/20261111_npu32.0.100.4724_preliminary/{perf_atlas,llm_chat,hardware}.json"], nums,
                  [f"Preliminary: NPU driver {pdrv.get('NPU')}, graphics driver {pdrv.get('GPU')}, other jobs running "
                   f"(2026-11-11). Clean: NPU {CTX.get('npu_drv')}, graphics {CTX.get('gpu_drv')}, idle, sequential "
                   f"(2026-11-12).",
                   "LLM GPU cold load also depends on compile-cache state; both runs were labelled 'cold'."])
    if not draw:
        return sec

    sub = (f"Preliminary run: 2026-11-11, NPU driver {pdrv.get('NPU', '?')}, graphics driver {pdrv.get('GPU', '?')}, "
           f"other jobs running on the laptop.\nClean run: {CTX['date']}, NPU driver {CTX['npu_drv']}, graphics "
           f"driver {CTX['gpu_drv']}, idle, one device at a time.")
    fig = figure(title, sub=sub)
    a1 = fig.add_axes([0.2, 0.15, 0.47, 0.55])
    ys = np.arange(len(rows))[::-1]
    for y, (dev, lbl, p, c) in zip(ys, rows):
        a1.plot([c, p], [y, y], color=GRID, lw=5, zorder=2, solid_capstyle="round")
        a1.scatter([p], [y], s=150, color=PRELIM, edgecolor=SURFACE, linewidth=2, zorder=4)
        a1.scatter([c], [y], s=150, color=CLEAN, edgecolor=SURFACE, linewidth=2, zorder=5)
        a1.text(max(p, c) * 1.25, y, f"{p / c:.1f}×" + (" (dots overlap)" if abs(p / c - 1) < 0.1 else ""),
                va="center", fontsize=14, fontweight="semibold",
                color=INK)
    a1.set_xscale("log")
    log_axis(a1, "x")
    allv = [v for *_, p, c in rows for v in (p, c)]
    a1.set_xlim(min(allv) / 1.6, max(allv) * 3)
    a1.set_yticks(ys, [f"{dev}  {lbl}" for dev, lbl, *_ in rows], fontsize=14)
    a1.tick_params(axis="y", length=0, pad=30)
    a1.set_ylim(-0.7, len(rows) - 0.3)
    for t, (dev, *_) in zip(a1.get_yticklabels(), rows):
        t.set_color(INK)
    for y, (dev, *_) in zip(ys, rows):
        a1.add_patch(plt.Rectangle((-0.032, y - 0.22), 0.012, 0.44, transform=a1.get_yaxis_transform(),
                                   color=DEVICE_HEX[dev], clip_on=False))
    a1.grid(axis="y", visible=False)
    a1.set_xlabel("p50 latency, ms (log scale)")
    a1.set_title("Perf Atlas, FP16 (same configs in both runs)")
    a1.legend(handles=[Line2D([], [], marker="o", ls="", ms=12, color=PRELIM, label="preliminary"),
                       Line2D([], [], marker="o", ls="", ms=12, color=CLEAN, label="clean"),
                       Line2D([], [], ls="", label="label = preliminary / clean")], loc="lower right", fontsize=12.5)
    if llm:
        a2 = fig.add_axes([0.76, 0.15, 0.2, 0.53])
        dev, lbl, p, c = llm[0]
        a2.bar([0, 1], [p, c], width=0.6, color=[PRELIM, CLEAN], edgecolor=SURFACE, linewidth=2, zorder=3)
        for x, v in ((0, p), (1, c)):
            a2.text(x, v + p * 0.015, f"{v:.1f} s", ha="center", va="bottom", fontsize=15, fontweight="semibold")
        a2.set_xticks([0, 1], ["preliminary", "clean"])
        a2.set_ylim(0, p * 1.18)
        a2.grid(axis="x", visible=False)
        a2.set_title("LLM first load on the GPU, s\n(Phi-3-mini, cold compile cache)")
    footer(fig, "Source: benchmarks/results/perf_atlas.json and llm_chat.json vs benchmarks/results/archive/"
                "20261111_npu32.0.100.4724_preliminary/. Both the drivers and the machine load changed between the runs, "
                "so these gaps cannot be attributed to the driver alone.")
    save(fig, sec.name)
    return sec


# ------------------------------------------------------------------------------------- simple set (advisory board)
def simple_bars(fig, values, fmt, xmax, ax_box=(0.27, 0.2, 0.66, 0.55)):
    ax = fig.add_axes(list(ax_box))
    devs = [d for d in DEVICES if values.get(d) is not None]
    ys = np.arange(len(devs))[::-1]
    for y, d in zip(ys, devs):
        ax.barh(y, values[d], height=0.62, color=DEVICE_HEX[d], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(values[d] + xmax * 0.015, y, fmt(values[d]), va="center", ha="left", fontsize=32,
                fontweight="bold", color=INK)
    ax.set_yticks(ys, [PLAIN[d] for d in devs], fontsize=26, color=INK)
    ax.tick_params(axis="y", length=0, pad=18)
    ax.set_xlim(0, xmax)
    ax.set_xticks([])
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    return ax


def simple_header(fig, message):
    _fit(fig, message, 0.05, 0.80, 26, 22, 0.9, 0.045, color=INK2)


def simple_footer(fig, source):
    c = CTX
    footer(fig, f"Laptop: Intel Core Ultra 5 322 (Panther Lake) · NPU driver {c['npu_drv']} · graphics "
                f"driver {c['gpu_drv']} · measured {c['date']}.  {source}", simple=True)


@chart("simple_speed", "simple")
def s_speed(draw=True):
    v = vision_lat("frame")
    fps = {k: 1000.0 / p50 for k, (p50, _) in v.items()}
    if "NPU" not in fps or "CPU" not in fps:
        raise Skip("need NPU and CPU vision frame times")
    x = fps["NPU"] / fps["CPU"]
    msg = f"The AI chip handles {fps['NPU']:.0f} video frames a second: {x:.1f}× the main processor."
    sec = Section("simple_speed", "Same AI task, three chips", msg, ["vision_side_by_side.json"],
                  [f"Vision frame p50 (detect + segment): " + ", ".join(
                      f"{k} {v[k][0]:.4f} ms = {fps[k]:.1f} frames/s" for k in DEVICES if k in v),
                   f"NPU vs CPU: {x:.2f}x; NPU vs GPU: {fps['NPU'] / fps['GPU']:.2f}x" if "GPU" in fps else ""],
                  ["Frames/s = 1000 / median frame AI time; it counts only the AI step (no video decoding or drawing)."])
    if not draw:
        return sec
    fig = figure("Same AI task, three chips", simple=True)
    simple_header(fig, msg)
    simple_bars(fig, fps, lambda f: f"{f:.0f} frames / second", max(fps.values()) * 1.8)
    simple_footer(fig, "Task: find people and cars in street video, and cut out a person. AI step only. "
                       "Data: benchmarks/results/vision_side_by_side.json")
    save(fig, sec.name)
    return sec


@chart("simple_cpu_free", "simple")
def s_cpu_free(draw=True):
    # Preferred: offload "work kept" (only trustworthy in the redesigned, placements-format file). Fallback: LLM CPU use.
    try:
        s = offload_summary()
        kept = _vals(s["kept"]) if s["kept_ok"] else {}
    except Skip as e:
        warn(f"simple_cpu_free: offload data unusable ({e}); using the chatbot CPU share")
        s, kept = None, {}
    if "NPU" in kept and len(kept) >= 2:
        ms = _vals(s["ms"])
        worst = min(kept, key=kept.get)
        msg = (f"With live video AI on the NPU, your other work keeps {kept['NPU']:.0f}% of its speed "
               f"({kept[worst]:.0f}% on the {'main processor' if worst == 'CPU' else 'graphics chip'}).")
        sec = Section("simple_cpu_free", "The NPU leaves your processor free", msg, ["offload_dashboard.json"],
                      ["Background work kept, % of AI-off speed: " + ", ".join(
                          f"{k} {kept[k]:.1f}%" + _rounds_txt(s["kept"], k, ".1f") for k in DEVICES if k in kept),
                       "CPU ms per AI frame: " + ", ".join(f"{k} {ms[k]:.2f}" for k in DEVICES if k in ms)],
                      ["Work kept = foreground worker throughput with AI running / adjacent AI-off baseline."])
        if not draw:
            return sec
        fig = figure("The NPU leaves your processor free", simple=True)
        simple_header(fig, msg)
        simple_bars(fig, kept, lambda p: f"{p:.0f}% of normal speed", max(max(kept.values()), 100) * 1.65)
        simple_footer(fig, "Your other work = programs busy on every processor thread while AI effects run on live "
                           "video. Data: benchmarks/results/offload_dashboard.json")
        save(fig, "simple_cpu_free")
        return sec

    _, devs = llm_devices()
    n_cpu = 6
    cores = {k: llm_summary(devs[k], "cpu_cores_busy_median", "cpu_cores_busy") for k in DEVICES if k in devs}
    pct = {k: 100.0 * v / n_cpu for k, v in cores.items() if v is not None}
    if "NPU" not in pct:
        raise Skip("no NPU CPU-cores-busy value")
    msg = f"While the chatbot answers, the NPU keeps just {pct['NPU']:.0f}% of the processor busy."
    sec = Section("simple_cpu_free", "The NPU leaves your processor free", msg, ["llm_chat.json"],
                  ["LLM CPU cores busy (median of 4 prompts) -> share of 6 logical CPUs: " + ", ".join(
                      f"{k} {cores[k]:.2f} cores = {pct[k]:.1f}%" for k in DEVICES if k in pct)],
                  ["Share = cpu_cores_busy / 6 logical CPUs."])
    if not draw:
        return sec
    fig = figure("The NPU leaves your processor free", simple=True)
    simple_header(fig, msg)
    simple_bars(fig, pct, lambda p: f"{p:.0f}% of processor busy", 100)
    simple_footer(fig, "While a chatbot on the laptop writes its answer; share of the processor's 6 threads in use. "
                       "Data: benchmarks/results/llm_chat.json")
    save(fig, sec.name)
    return sec


@chart("simple_llm", "simple")
def s_llm(draw=True):
    _, devs = llm_devices()
    WPT = 0.75
    tok = {k: llm_summary(devs[k], "tok_s_median", "tok_s") for k in DEVICES if k in devs}
    wps = {k: v * WPT for k, v in tok.items() if v is not None}
    if not wps:
        raise Skip("no tok/s")
    best = max(wps, key=wps.get)
    msg = (f"No internet, no cloud: the laptop's own chips write up to about {wps[best]:.0f} words a second.")
    sec = Section("simple_llm", "A chatbot running entirely on the laptop", msg, ["llm_chat.json"],
                  ["Words/s = tok/s median x 0.75 (approximation): " + ", ".join(
                      f"{k} {tok[k]:.2f} tok/s ~ {wps[k]:.1f} words/s" for k in DEVICES if k in wps)],
                  ["0.75 words per token is a rule-of-thumb approximation for English text, not measured here."])
    if not draw:
        return sec
    fig = figure("A chatbot running entirely on the laptop", simple=True)
    simple_header(fig, msg)
    simple_bars(fig, wps, lambda w: f"≈ {w:.0f} words / second", max(wps.values()) * 1.85)
    simple_footer(fig, "Microsoft Phi-3-mini (3.8B) on the laptop, no cloud. Words/second ≈ 0.75 × tokens/"
                       "second (approximation). Data: benchmarks/results/llm_chat.json")
    save(fig, sec.name)
    return sec


# ----------------------------------------------------------------------------------------------- README
def write_readme(sections, skipped):
    lines = ["# Chart set: NPU showcase results", "",
             "Generated by `scripts/make_charts.py` from `benchmarks/results/*.json`. Do not edit by hand: re-run",
             "`.venv\\Scripts\\python.exe scripts/make_charts.py` (or `--only <name>`) after re-recording results.",
             "", f"Hardware: Intel Core Ultra 5 322 (Panther Lake); NPU 5 \"Intel AI Boost\", 46 TOPS INT8; Intel "
             f"Graphics, 2 Xe3 cores; CPU 2P + 4LP-E, 6 threads. NPU driver {CTX['npu_drv']}, graphics driver "
             f"{CTX['gpu_drv']}, OpenVINO {CTX['ov']}. Clean run {CTX['date']}: idle machine, on AC, devices run one at "
             "a time.", "",
             "All PNGs are 1920 x 1080. File names are a contract with the decks and the site; do not rename them.", ""]
    for kind, head in (("technical", "Technical set"), ("simple", "Simple set (advisory board)")):
        lines += [f"## {head}", ""]
        for s in sections:
            if CHARTS[s.name][1] != kind:
                continue
            lines += [f"### `{s.name}.png`", "", f"**Takeaway:** {s.takeaway}", "",
                      "**Data:** " + ", ".join(f"`benchmarks/results/{x}`" for x in s.sources), "", "**Numbers:**", ""]
            lines += [f"- {n}" for n in s.numbers if n]
            if s.notes:
                lines += ["", "**Notes:**", ""] + [f"- {n}" for n in s.notes]
            lines.append("")
    if skipped:
        lines += ["## Skipped in the last run", ""] + [f"- `{n}`: {why}" for n, why in skipped] + [""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="chart name (e.g. cpu_cost or cpu_cost.png); repeatable or comma-separated")
    ap.add_argument("--list", action="store_true", help="list chart names and exit")
    ap.add_argument("--offload-json", type=Path, help="read this file instead of benchmarks/results/"
                    "offload_dashboard.json (e.g. to test a new recording)")
    args = ap.parse_args(argv)
    if args.offload_json:
        OFFLOAD_JSON[0] = args.offload_json.resolve()
    if args.list:
        for n, (_, k) in CHARTS.items():
            print(f"{n:22} {k}")
        return 0
    only = {o.strip().removesuffix(".png") for arg in args.only for o in arg.split(",") if o.strip()}
    unknown = only - set(CHARTS)
    if unknown:
        ap.error(f"unknown chart(s): {', '.join(sorted(unknown))}; choose from {', '.join(CHARTS)}")
    CTX.update(context())
    OUT.mkdir(parents=True, exist_ok=True)
    sections, skipped = [], []
    for name, (fn, _) in CHARTS.items():
        draw = not only or name in only
        try:
            sec = fn(draw=draw)
            sections.append(sec)
            if draw:
                print(f"wrote media/charts/{name}.png")
        except Skip as e:
            skipped.append((name, str(e)))
            warn(f"{name}: skipped ({e})")
        except Exception as e:  # malformed input in an unexpected shape: skip, never crash the run
            skipped.append((name, f"{type(e).__name__}: {e}"))
            warn(f"{name}: skipped after {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
        finally:
            plt.close("all")
    write_readme(sections, skipped)
    print("wrote media/charts/README.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
