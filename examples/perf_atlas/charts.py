"""Matplotlib rendering for the NPU Performance Atlas: multi-panel dashboard + sweep-progress frames."""
from __future__ import annotations

import numpy as np
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from examples.common.hud import DEVICE_HEX  # noqa: E402

# Light, README-friendly chrome (reference palette tokens).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STATUS = {"executed": "#0ca30c", "failed_correctness": "#fab219", "failed": "#d03b3b", "skipped": "#c3c2b7"}
STATUS_GLYPH = {"executed": "✓", "failed_correctness": "≠", "failed": "✗", "skipped": "–"}
STATUS_LABEL = {"executed": "executed + correct", "failed_correctness": "ran, wrong output",
                "failed": "compile/infer failed", "skipped": "skipped"}
MARKER = {"NPU": "o", "GPU": "^", "CPU": "s"}
DEVICE_ORDER = ["NPU", "GPU", "CPU"]
PREC_ORDER = ["FP32", "FP16", "INT8"]
# Diverging: blue = NPU faster, red = NPU slower, neutral grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "npu_speedup", ["#b8373a", "#e34948", "#f4b5b0", "#f0efec", "#b7d3f6", "#3987e5", "#184f95"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "text.color": INK,
})


def _devices(rows):
    present = {r["device"] for r in rows}
    return [d for d in DEVICE_ORDER if d in present] + sorted(present - set(DEVICE_ORDER))


def _color(d):
    return DEVICE_HEX.get(d, MUTED)


def _ok(rows, **kw):
    return [r for r in rows if r.get("status") == "executed" and all(r.get(k) == v for k, v in kw.items())]


def _models(rows):
    seen = []
    for r in rows:
        if r["model"] not in seen:
            seen.append(r["model"])
    return seen


def _fmt_x(v):
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×" if v >= 0.95 else f"{v:.2f}×"


def _log_axis_fmt(axis):
    """Clean 1-2-5 log ticks with plain numbers (0.02, 0.5, 2, 10) instead of overlapping 10^x minors."""
    axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def _fmt_gf(v):
    return f"{v:.2g}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"


# --------------------------------------------------------------------------- latency vs work
BATCH_STYLE = {1: "-", 8: "--", 32: ":"}


def _line(rows, device, family, batch, precision):
    pts = sorted(_ok(rows, device=device, family=family, batch=batch, precision=precision),
                 key=lambda r: r["gflop_per_infer"])
    return (np.array([r["gflop_per_infer"] for r in pts]), np.array([r["median_ms"] for r in pts]),
            np.array([r["p95_ms"] for r in pts]), [r["model"] for r in pts])


def crossovers(rows, precision="FP16"):
    """Per (family, batch): GFLOP where the NPU p50 line crosses the CPU/GPU p50 line (log-log interpolation
    between consecutive model sizes that both devices executed)."""
    out = []
    for family in sorted({r["family"] for r in rows}):
        for batch in sorted({r["batch"] for r in rows}):
            nx, ny, _, nm = _line(rows, "NPU", family, batch, precision)
            for other in ("CPU", "GPU"):
                ox, oy, _, om = _line(rows, other, family, batch, precision)
                common = [m for m in nm if m in om]
                if len(common) < 1:
                    continue
                pts = [(nx[nm.index(m)], ny[nm.index(m)], oy[om.index(m)]) for m in common]
                d = [np.log(a / b) for _, a, b in pts]
                summary = {"family": family, "batch": batch, "other": other,
                           "npu_faster_at": [m for m, v in zip(common, d) if v < 0], "crossings": []}
                for (x0, a0, b0), (x1, a1, b1), d0, d1 in zip(pts, pts[1:], d, d[1:]):
                    if np.sign(d0) != np.sign(d1) and d0 != d1:
                        t = d0 / (d0 - d1)
                        gx = float(np.exp(np.log(x0) + t * (np.log(x1) - np.log(x0))))
                        gy = float(np.exp(np.log(a0) + t * (np.log(a1) - np.log(a0))))
                        summary["crossings"].append({"gflop": round(gx, 4), "latency_ms": round(gy, 4),
                                                     "npu_wins_above": bool(d1 < 0)})
                out.append(summary)
    return out


def _crossover_text(c):
    if c["crossings"]:
        parts = []
        for x in c["crossings"]:
            parts.append(f"NPU overtakes {c['other']} above {_fmt_gf(x['gflop'])} GF" if x["npu_wins_above"]
                         else f"{c['other']} overtakes NPU above {_fmt_gf(x['gflop'])} GF")
        return "; ".join(parts)
    n = len(c["npu_faster_at"])
    return f"NPU faster than {c['other']} at all sizes" if n else f"NPU slower than {c['other']} at all sizes"


def panel_latency(ax, rows, family, precision="FP16", annotate=True, devices=None, batches=None):
    devices = devices or _devices(rows)
    batches = batches or sorted({r["batch"] for r in rows})
    fam_rows = [r for r in rows if r["family"] == family]
    for d in devices:
        c = _color(d)
        for b in batches:
            x, p50, p95, _ = _line(fam_rows, d, family, b, precision)
            if not len(x):
                continue
            ax.fill_between(x, p50, p95, color=c, alpha=0.13, linewidth=0, zorder=1)
            ax.plot(x, p50, color=c, lw=2, ls=BATCH_STYLE.get(b, "-."), solid_capstyle="round", zorder=3)
            ax.plot(x, p50, ls="none", marker=MARKER.get(d, "o"), ms=7, color=c, mec=SURFACE, mew=1.5, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    _log_axis_fmt(ax.xaxis); _log_axis_fmt(ax.yaxis)
    ax.set_xlabel("work per inference (GFLOP = batch \u00d7 GFLOP/sample)")
    ax.grid(True, which="major")
    ax.margins(x=0.08)
    found = []
    if annotate:
        found = [c for c in crossovers(fam_rows, precision) if c["family"] == family]
        lines = []
        for c in found:
            for x in c["crossings"]:
                ax.plot(x["gflop"], x["latency_ms"], marker="o", ms=13, mfc="none", mec=INK, mew=1.4, zorder=6)
                ax.annotate(f"b{c['batch']}", (x["gflop"], x["latency_ms"]), xytext=(7, -12),
                            textcoords="offset points", fontsize=8.5, color=INK, zorder=6)
            lines.append(f"b{c['batch']}: {_crossover_text(c)}")
        if lines:
            ax.text(0.015, 0.985, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8.8, color=INK2,
                    linespacing=1.45, bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=GRID, alpha=0.92), zorder=7)
    handles = [Line2D([], [], color=_color(d), marker=MARKER.get(d, "o"), lw=2, ms=7, mec=SURFACE, label=d)
               for d in devices]
    handles.append(Patch(color=MUTED, alpha=0.25, label="p50 \u2192 p95"))
    handles += [Line2D([], [], color=INK2, lw=1.6, ls=BATCH_STYLE.get(b, "-."), label=f"batch {b}") for b in batches]
    ax.legend(handles=handles, loc="lower right", fontsize=9, ncol=2, columnspacing=1.0, handlelength=2.4)
    return found


def _speedup_grid(rows, other, precision):
    models = _models(rows)
    batches = sorted({r["batch"] for r in rows})
    grid = np.full((len(models), len(batches)), np.nan)
    why = {}
    for i, m in enumerate(models):
        for j, b in enumerate(batches):
            for d in ("NPU", other):
                bad = [r for r in rows if (r["model"], r["batch"], r["precision"], r["device"]) == (m, b, precision, d)
                       and r["status"] != "executed"]
                if bad and (i, j) not in why:
                    why[(i, j)] = f"{d}\n" + {"failed_correctness": "wrong output", "failed": "failed"}.get(
                        bad[0]["status"], bad[0]["status"])
            n = _ok(rows, model=m, batch=b, precision=precision, device="NPU")
            o = _ok(rows, model=m, batch=b, precision=precision, device=other)
            if n and o:
                grid[i, j] = o[0]["median_ms"] / n[0]["median_ms"]
    return models, batches, grid, why


def panel_heatmap(ax, rows, other, precision="FP16"):
    models, batches, grid, why = _speedup_grid(rows, other, precision)
    gpm = {r["model"]: r["gflop_per_sample"] for r in rows}
    lg = np.log2(grid)
    lim = max(1.0, float(np.nanmax(np.abs(lg))) if np.isfinite(lg).any() else 1.0)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    ax.imshow(np.ma.masked_invalid(lg), cmap=DIVERGING, norm=norm, aspect="auto")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(len(models)):
        for j in range(len(batches)):
            v = grid[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#f0efec", hatch="///", ec=AXIS, lw=0))
                ax.text(j, i, why.get((i, j), "n/a"), ha="center", va="center", color=INK2, fontsize=9.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="#f0efec", ec="none"))
            else:
                strong = abs(np.log2(v)) > lim * 0.55
                ax.text(j, i, _fmt_x(v), ha="center", va="center", fontsize=11.5, fontweight="semibold",
                        color="#ffffff" if strong else INK)
    ax.set_xticks(range(len(batches)), [f"batch {b}" for b in batches])
    ax.set_yticks(range(len(models)), [f"{m}\n{_fmt_gf(gpm[m])} GF/sample" for m in models], fontsize=9.5)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(batches)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
    ax.grid(True, which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"NPU speed-up vs {other}  ({precision}, p50)")
    ax.text(0, -0.08 if len(models) > 4 else -0.12, f"blue = NPU faster · red = {other} faster · "
            f"value = {other} p50 / NPU p50", transform=ax.transAxes, fontsize=9, color=INK2, va="top")
    return grid


def _grouped_bars(ax, groups, devices, values, log=False, fmt=lambda v: f"{v:.3g}", label_tips=True, ticks=None):
    n = len(devices)
    width = min(0.8 / max(n, 1), 0.26)
    for k, d in enumerate(devices):
        xs = np.arange(len(groups)) + (k - (n - 1) / 2) * (width + 0.02)
        vals = [values.get((g, d), np.nan) for g in groups]
        ax.bar(xs, vals, width=width, color=_color(d), label=d, zorder=3, edgecolor=SURFACE, linewidth=1)
        if label_tips:
            for x, v in zip(xs, vals):
                if np.isfinite(v):
                    ax.text(x, v, fmt(v), ha="center", va="bottom", fontsize=8.5, color=INK2, rotation=90 if n > 2 else 0,
                            rotation_mode="anchor" if False else "default")
                else:
                    ax.text(x, ax.get_ylim()[0] if log else 0, "✗", ha="center", va="bottom", fontsize=9, color=STATUS["failed"])
    ax.set_xticks(range(len(groups)), ticks or groups)
    if log:
        ax.set_yscale("log"); _log_axis_fmt(ax.yaxis)
    ax.grid(True, axis="y"); ax.grid(False, axis="x")
    ax.legend(loc="upper left", fontsize=10, ncol=n)


def panel_peak_gflops(ax, rows):
    devices = _devices(rows)
    vals, best = {}, {}
    for p in PREC_ORDER:
        for d in devices:
            ok = _ok(rows, precision=p, device=d)
            if ok:
                r = max(ok, key=lambda r: r["gflops_effective"])
                vals[(p, d)] = r["gflops_effective"]
                best[(p, d)] = r
    precs = [p for p in PREC_ORDER if any((p, d) in vals for d in devices)]
    ticks = [p + ("\n(NPU computes f16)" if p == "FP32" and "NPU" in devices else "") for p in precs]
    _grouped_bars(ax, precs, devices, vals, fmt=lambda v: f"{v:.0f}", ticks=ticks)
    ax.set_ylabel("best effective GFLOP/s (any config)")
    ax.set_title("Peak sustained compute, sync latency mode")
    top = max(vals.values()) if vals else 1
    ax.set_ylim(0, top * 1.28)
    nb = best.get(("FP16", "NPU"))
    if nb:
        ax.text(0.99, 0.90, f"NPU FP16 peak: {nb['model']} batch {nb['batch']}", transform=ax.transAxes, ha="right",
                va="top", fontsize=9, color=INK2)


def panel_compile(ax, rows):
    devices = _devices(rows)
    models = _models(rows)
    vals = {}
    for m in models:
        for d in devices:
            ok = [r["compile_s"] for r in rows if r["model"] == m and r["device"] == d and "compile_s" in r]
            if ok:
                vals[(m, d)] = float(np.median(ok))
    _grouped_bars(ax, models, devices, vals, log=True, fmt=lambda v: f"{v:.2g}s", label_tips=False,
                  ticks=[m.replace("-", "\n") for m in models])
    ax.set_ylabel("compile time (s, median over batch × precision)")
    ax.set_title("Compile cost (paid once per model/shape)")
    nm = [vals[(m, "NPU")] for m in models if (m, "NPU") in vals]
    if nm:
        m_hi = max((m for m in models if (m, "NPU") in vals), key=lambda m: vals[(m, "NPU")])
        ax.annotate(f"NPU {vals[(m_hi, 'NPU')]:.2g} s", (models.index(m_hi) + (devices.index('NPU') - (len(devices) - 1) / 2) * 0.28,
                    vals[(m_hi, "NPU")]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9.5,
                    color=INK, fontweight="semibold")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)


def panel_async(ax, rows, async_rows):
    devices = _devices(rows)
    cfgs = []
    for r in async_rows:
        key = (r["model"], r["batch"], r["precision"])
        if key not in cfgs:
            cfgs.append(key)
    if not cfgs:
        ax.text(0.5, 0.5, "no async runs", transform=ax.transAxes, ha="center", color=MUTED)
        return
    groups = [f"{m} b{b}\n{p}" for m, b, p in cfgs]
    vals = {}
    sync = {}
    for (m, b, p), g in zip(cfgs, groups):
        for d in devices:
            a = [r for r in async_rows if (r["model"], r["batch"], r["precision"], r["device"]) == (m, b, p, d)
                 and r.get("status") == "executed"]
            if a:
                vals[(g, d)] = a[0]["samples_per_s"]
            s = _ok(rows, model=m, batch=b, precision=p, device=d)
            if s:
                sync[(g, d)] = s[0]["samples_per_s"]
    _grouped_bars(ax, groups, devices, vals, log=True, label_tips=False)
    n = len(devices)
    width = min(0.8 / n, 0.26)
    for k, d in enumerate(devices):
        for gi, g in enumerate(groups):
            x = gi + (k - (n - 1) / 2) * (width + 0.02)
            if (g, d) in sync:
                ax.plot([x - width / 2, x + width / 2], [sync[(g, d)]] * 2, color=INK, lw=2, zorder=5,
                        solid_capstyle="butt")
            if (g, d) in vals:
                ax.text(x, vals[(g, d)] * 1.08, f"{vals[(g, d)]:.0f}", ha="center", va="bottom", fontsize=8.5,
                        color=INK2, rotation=90)
    ax.set_ylabel("samples / s (log)")
    ax.set_title("Throughput: async queue vs sync")
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=2)); l.append("sync (1 request)")
    ax.legend(h, l, loc="upper left", fontsize=9.5, ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] * 4)


def panel_compat(ax, rows):
    devices = _devices(rows)
    models = _models(rows)
    precs = [p for p in PREC_ORDER if any(r["precision"] == p for r in rows)]
    ylabels = [f"{d} {p}" for d in devices for p in precs]
    rank = {"executed": 0, "skipped": 1, "failed_correctness": 2, "failed": 3}
    ax.set_xlim(-0.5, len(models) - 0.5); ax.set_ylim(len(ylabels) - 0.5, -0.5)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i, (d, p) in enumerate((d, p) for d in devices for p in precs):
        for j, m in enumerate(models):
            cell = [r for r in rows if r["device"] == d and r["precision"] == p and r["model"] == m]
            if not cell:
                continue
            worst = max(cell, key=lambda r: rank.get(r["status"], 3))["status"]
            n_ok = sum(r["status"] == "executed" for r in cell)
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88, fc=STATUS.get(worst, MUTED), ec="none"))
            ax.text(j, i, f"{STATUS_GLYPH.get(worst, '?')} {n_ok}/{len(cell)}", ha="center", va="center",
                    fontsize=9, color="#0b0b0b" if worst in ("failed_correctness", "skipped") else "#ffffff",
                    fontweight="semibold")
    ax.set_xticks(range(len(models)), [m.replace("-", "\n") for m in models], fontsize=9)
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=9.5)
    ax.tick_params(length=0)
    ax.set_title("Compatibility & correctness  (n/m = batch sizes executed)")
    present = [s for s in ("executed", "failed_correctness", "failed", "skipped") if any(r["status"] == s for r in rows)]
    present = present or ["executed"]
    ax.legend(handles=[Patch(color=STATUS[s], label=f"{STATUS_GLYPH[s]} {STATUS_LABEL[s]}") for s in present],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)


# --------------------------------------------------------------------------- figures
def _subtitle(meta):
    dn = meta.get("device_names", {})
    hw = " \u00b7 ".join(f"{d}: {dn.get(d, d)}" for d in meta.get("devices", []))
    power = meta.get("power", {}).get("power_source", "unknown")
    return (f"{hw}  |  OpenVINO {meta.get('openvino_version', '?').split('-')[0]}  |  {meta.get('date', '')}  |  "
            f"power: {power}  |  grid: {meta.get('grid_name', '')}")


def _latency_pair(fig, cell, rows, precision="FP16", annotate=True, devices=None, batches=None):
    sub = cell.subgridspec(1, 2, wspace=0.08)
    ax1 = fig.add_subplot(sub[0])
    ax2 = fig.add_subplot(sub[1], sharey=ax1)
    found = []
    for ax, fam, label in ((ax1, "MLP", "MLP stacks (matmul, weight-bound at small batch)"),
                           (ax2, "CONV", "Conv stacks (3\u00d73 conv, compute-bound)")):
        found += panel_latency(ax, rows, fam, precision, annotate, devices, batches)
        ax.set_title(label, fontsize=12)
    ax1.set_ylabel(f"latency per inference, ms ({precision})")
    plt.setp(ax2.get_yticklabels(), visible=False)
    return ax1, ax2, found


def dashboard(data, path=None, dpi=100):
    rows, async_rows, meta = data["results"], data.get("async", []), data["meta"]
    fig = plt.figure(figsize=(24, 15), dpi=dpi)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.36, wspace=0.3,
                          left=0.045, right=0.985, top=0.855, bottom=0.07)
    fig.text(0.045, 0.962, "NPU Performance Atlas \u2014 where the Intel AI Boost NPU wins, and where it doesn\u2019t",
             fontsize=23, fontweight="bold", color=INK)
    fig.text(0.045, 0.934, _subtitle(meta), fontsize=11.5, color=INK2)
    n_ok = sum(r["status"] == "executed" for r in rows)
    fig.text(0.045, 0.912, f"{len(rows)} sync configurations ({n_ok} executed and verified against numpy), "
             f"{len(async_rows)} async runs. Synthetic seeded MLP & conv stacks; p50/p95 of prepared InferRequest calls; "
             f"INT8 = weight-only int8 with f16 activations. Not a TOPS measurement.", fontsize=10.5, color=INK2)
    ax1, _, found = _latency_pair(fig, gs[0, :2], rows, "FP16")
    fig.text(0.045, 0.882, "Latency vs work \u2014 FP16, line = p50, band = p50\u2192p95, circles = NPU crossover points",
             fontsize=13, fontweight="bold", color=INK)
    panel_heatmap(fig.add_subplot(gs[0, 2]), rows, "CPU")
    panel_heatmap(fig.add_subplot(gs[0, 3]), rows, "GPU")
    panel_peak_gflops(fig.add_subplot(gs[1, 0]), rows)
    panel_compile(fig.add_subplot(gs[1, 1]), rows)
    panel_async(fig.add_subplot(gs[1, 2]), rows, async_rows)
    panel_compat(fig.add_subplot(gs[1, 3]), rows)
    fig.text(0.045, 0.012, "Run: python -m examples.perf_atlas  \u00b7  data: " + meta.get("output", "") +
             "  \u00b7  timings are sensitive to power mode, thermals and concurrent NPU/GPU users.",
             fontsize=9.5, color=MUTED)
    if path:
        fig.savefig(path, facecolor=SURFACE)
    return fig, found


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def progress_frames(data, n_frames=36, size=(16, 9.6), dpi=80):
    """Frames of the latency-vs-work charts building up in sweep order (FP16 rows)."""
    rows = sorted(data["results"], key=lambda r: r.get("seq", 0))
    fp16 = [r for r in rows if r["precision"] == "FP16"]
    ok = _ok(fp16)
    if not ok:
        return []
    devices = _devices(rows)
    batches = sorted({r["batch"] for r in rows})
    xs = [r["gflop_per_infer"] for r in ok]
    ys = [r["median_ms"] for r in ok] + [r["p95_ms"] for r in ok]
    total = len(fp16)
    frames = []
    for k in np.unique(np.linspace(1, total, n_frames).astype(int)):
        sub = fp16[:k]
        fig = plt.figure(figsize=size, dpi=dpi)
        gs = fig.add_gridspec(1, 1, left=0.07, right=0.98, top=0.8, bottom=0.1)
        ax1, ax2, _ = _latency_pair(fig, gs[0], sub, "FP16", annotate=(k == total), devices=devices, batches=batches)
        for ax in (ax1, ax2):
            ax.set_ylim(min(ys) / 1.6, max(ys) * 2.5)
        fx = [r["gflop_per_infer"] for r in ok if r["family"] == "MLP"] or xs
        cx = [r["gflop_per_infer"] for r in ok if r["family"] == "CONV"] or xs
        ax1.set_xlim(min(fx) / 1.5, max(fx) * 1.5)
        ax2.set_xlim(min(cx) / 1.5, max(cx) * 1.5)
        last = sub[-1]
        fig.text(0.07, 0.94, "NPU Performance Atlas \u2014 sweep in progress", fontsize=22, fontweight="bold")
        fig.text(0.07, 0.895, f"FP16 configuration {k}/{total}  \u00b7  latest: {last['model']} batch {last['batch']} "
                 f"on {last['device']} \u2192 {last['status']}"
                 + (f", p50 {last['median_ms']:.3g} ms" if last["status"] == "executed" else ""),
                 fontsize=13, color=INK2)
        bar = fig.add_axes([0.07, 0.87, 0.91, 0.008]); bar.axis("off")
        bar.barh(0, 1, color=GRID, height=1); bar.barh(0, k / total, color=INK2, height=1)
        bar.set_xlim(0, 1)
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
    return frames
