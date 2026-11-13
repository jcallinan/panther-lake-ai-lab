"""Build the technical slide deck: docs/presentation/panther-lake-npu.pptx

Every number on the slides is read from benchmarks/results/*.json (clean, sequential runs) or
benchmarks/results/archive/*_preliminary/*.json (contended first runs) at build time, so re-running
this script after re-recording a demo or re-running a benchmark refreshes the deck.

Usage (from the repo root):
    .venv\\Scripts\\python.exe docs\\presentation\\build_tech_deck.py              # MP4 movies + poster frames
    .venv\\Scripts\\python.exe docs\\presentation\\build_tech_deck.py --media gif  # animated GIFs instead
    .venv\\Scripts\\python.exe docs\\presentation\\build_tech_deck.py --media poster  # still frames only (small file)

Charts come from media/charts/*.png (scripts/make_charts.py). A missing chart is drawn as a labelled
placeholder, so the deck always builds; re-run once the charts exist.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import statistics
import sys
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "benchmarks" / "results"
MEDIA = ROOT / "media"
CHARTS = MEDIA / "charts"
OUT_DEFAULT = Path(__file__).resolve().parent / "panther-lake-npu.pptx"
REPO_URL = "github.com/jcallinan/panther-lake-ai-lab"

# ----------------------------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------------------------
NPU, GPU, CPU = RGBColor(0x00, 0xAA, 0xF0), RGBColor(0x5A, 0xC8, 0x3C), RGBColor(0xF5, 0x78, 0x28)
DEV = {"NPU": NPU, "GPU": GPU, "CPU": CPU}
DARK_BG = RGBColor(0x0D, 0x15, 0x20)
DARK_CARD = RGBColor(0x17, 0x22, 0x31)
LIGHT_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xF1, 0xF4, 0xF8)
INK = RGBColor(0x14, 0x1C, 0x28)
MUTED = RGBColor(0x5B, 0x66, 0x75)
ON_DARK = RGBColor(0xF4, 0xF7, 0xFA)
ON_DARK_MUTED = RGBColor(0xA8, 0xB4, 0xC3)
WARN = RGBColor(0xD6, 0x3A, 0x3A)
LINE = RGBColor(0xD5, 0xDB, 0xE3)

HEAD = "Segoe UI Semibold"
BODY = "Segoe UI"
MONO = "Consolas"

SW, SH = 13.333, 7.5          # 16:9 canvas, inches
MX = 0.6                      # side margin
TITLE_Y, TITLE_H = 0.62, 0.85
KICKER_Y = 0.32
CONTENT_Y = 1.65


# ----------------------------------------------------------------------------------------------
# Data: everything on the slides comes from here
# ----------------------------------------------------------------------------------------------
def load(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def find_prelim_dir() -> Path | None:
    arch = RES / "archive"
    dirs = sorted(p for p in arch.glob("*preliminary*") if p.is_dir()) if arch.exists() else []
    return dirs[-1] if dirs else None


def f1(x):
    return f"{x:.1f}"


def f2(x):
    return f"{x:.2f}"


def fms(x):
    """Latency formatting: 3 significant-ish digits."""
    return f"{x:.0f}" if x >= 100 else (f"{x:.1f}" if x >= 10 else f"{x:.2f}")


class Data:
    def __init__(self):
        self.summary = load(RES / "record_all_summary.json")
        self.npu_drv = self.summary.get("npu_driver", "?")
        self.gfx_drv = self.summary.get("graphics_driver", "?")
        self.on_ac = self.summary.get("on_ac")
        self.atlas = load(RES / "perf_atlas.json")
        self.vision = load(RES / "vision_side_by_side.json")
        self.depth = load(RES / "depth_parallax.json")
        self.offload = load(RES / "offload_dashboard.json")
        self.llm = load(RES / "llm_chat.json")
        self.ov = self.atlas["meta"]["openvino_version"].split("-")[0]
        pdir = find_prelim_dir()
        self.prelim_dir = pdir
        self.p_atlas = load(pdir / "perf_atlas.json") if pdir and (pdir / "perf_atlas.json").exists() else None
        self.p_llm = load(pdir / "llm_chat.json") if pdir and (pdir / "llm_chat.json").exists() else None
        self.p_vision = load(pdir / "vision_side_by_side.json") if pdir and (pdir / "vision_side_by_side.json").exists() else None
        self.p_offload = load(pdir / "offload_dashboard.json") if pdir and (pdir / "offload_dashboard.json").exists() else None
        self.p_depth = load(pdir / "depth_parallax.json") if pdir and (pdir / "depth_parallax.json").exists() else None
        self.p_npu_drv = pdir.name.split("_npu")[-1].split("_")[0] if pdir and "_npu" in pdir.name else "older driver"

    # ---- perf atlas ---------------------------------------------------------------------
    @staticmethod
    def atlas_index(atlas):
        return {(r["model"], r["batch"], r["precision"], r["device"]): r for r in atlas["results"]}

    @staticmethod
    def atlas_stats(atlas):
        idx = Data.atlas_index(atlas)
        R = atlas["results"]
        wc = wg = n = 0
        for (m, b, p, d), r in idx.items():
            if d != "NPU" or r["status"] != "executed":
                continue
            c, g = idx[(m, b, p, "CPU")], idx[(m, b, p, "GPU")]
            if c["status"] == "executed" and g["status"] == "executed":
                n += 1
                wc += r["median_ms"] < c["median_ms"]
                wg += r["median_ms"] < g["median_ms"]
        bad = [r for r in R if r["status"] == "failed_correctness"]
        comp = {d: [r["compile_s"] for r in R if r["device"] == d and r.get("compile_s") is not None]
                for d in ("NPU", "GPU", "CPU")}
        peak = {d: max(((r.get("gflops_effective") or 0), r["model"], r["batch"], r["precision"])
                       for r in R if r["device"] == d and r["status"] == "executed") for d in ("NPU", "GPU", "CPU")}
        return {
            "n_total": len(R),
            "n_exec": sum(r["status"] == "executed" for r in R),
            "n_bad": len(bad),
            "bad": bad,
            "n_async": len(atlas.get("async", [])),
            "wins_cpu": wc, "wins_gpu": wg, "n_cmp": n,
            "compile_median": {d: statistics.median(v) for d, v in comp.items()},
            "compile_max": {d: max(v) for d, v in comp.items()},
            "peak": peak,
            "idx": idx,
        }

    def lat(self, model, batch, prec, dev, atlas=None):
        r = self.atlas_index(atlas or self.atlas)[(model, batch, prec, dev)]
        return r["median_ms"] if r["status"] == "executed" else None

    # ---- llm ----------------------------------------------------------------------------
    @staticmethod
    def llm_dev(llm, dev):
        for d in llm["devices"]:
            if d["device"] == dev:
                return d
        return None

    # ---- offload ------------------------------------------------------------------------
    def offload_phase(self, dev, off=None):
        """Per-placement result. New format: `placements` (mean over rounds); old format: first matching phase."""
        off = off or self.offload
        for key in ("placements", "phases"):
            for p in off.get(key) or []:
                if p.get("device") == dev and p.get("result", {}).get("ai_fps") is not None:
                    return p["result"]
        return None


# ----------------------------------------------------------------------------------------------
# Drawing helpers
# ----------------------------------------------------------------------------------------------
class Deck:
    def __init__(self, media_mode: str, tmp_posters: dict):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(SW), Inches(SH)
        self.blank = self.prs.slide_layouts[6]
        self.media_mode = media_mode
        self.posters = tmp_posters
        self.n = 0
        self.missing = []

    # -- slide ------------------------------------------------------------------------------
    def slide(self, dark=False, footer=True):
        s = self.prs.slides.add_slide(self.blank)
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = DARK_BG if dark else LIGHT_BG
        self.n += 1
        s._dark = dark
        if footer:
            c = ON_DARK_MUTED if dark else MUTED
            self.text(s, MX, SH - 0.45, 8, 0.3, "Panther Lake NPU  ·  measured on a Core Ultra 5 322 laptop",
                      size=10, color=c, name="footer")
            self.text(s, SW - MX - 1.0, SH - 0.45, 1.0, 0.3, str(self.n), size=10, color=c,
                      align=PP_ALIGN.RIGHT, name="slide number")
        return s

    def notes(self, s, text):
        s.notes_slide.notes_text_frame.text = text.strip()

    # -- text -------------------------------------------------------------------------------
    def text(self, s, x, y, w, h, content, size=16, color=None, bold=False, font=BODY, align=PP_ALIGN.LEFT,
             anchor=MSO_ANCHOR.TOP, name=None, line_spacing=None, space_after=None, bullets=False, italic=False):
        """content: str | list[paragraph]; paragraph: str | list[(text, opts)] | dict(runs=..., bullet=, size=...)"""
        tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        if name:
            tb.name = name
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = anchor
        color = color or (ON_DARK if getattr(s, "_dark", False) else INK)
        paras = content if isinstance(content, list) else [content]
        for i, para in enumerate(paras):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = align
            popts = {}
            if isinstance(para, dict):
                popts = para
                runs = para["runs"]
            else:
                runs = para
            if isinstance(runs, str):
                runs = [(runs, {})]
            psize = popts.get("size", size)
            if line_spacing:
                p.line_spacing = line_spacing
            sa = popts.get("space_after", space_after)
            if sa is not None:
                p.space_after = Pt(sa)
            if popts.get("space_before") is not None:
                p.space_before = Pt(popts["space_before"])
            if popts.get("bullet", bullets):
                self._bullet(p, psize, popts.get("bullet_color"))
            for t, o in runs:
                r = p.add_run()
                r.text = t
                fnt = r.font
                fnt.name = o.get("font", popts.get("font", font))
                fnt.size = Pt(o.get("size", psize))
                fnt.bold = o.get("bold", popts.get("bold", bold))
                fnt.italic = o.get("italic", italic)
                fnt.color.rgb = o.get("color", popts.get("color", color))
        return tb

    @staticmethod
    def _bullet(p, size, color=None):
        pPr = p._p.get_or_add_pPr()
        indent = int(Pt(size) * 1.1)
        pPr.set("marL", str(indent))
        pPr.set("indent", str(-indent))
        for tag in ("a:buClr", "a:buFont", "a:buChar", "a:buNone"):
            for el in pPr.findall(qn(tag)):
                pPr.remove(el)
        if color is not None:
            buClr = etree.SubElement(pPr, qn("a:buClr"))
            srgb = etree.SubElement(buClr, qn("a:srgbClr"))
            srgb.set("val", str(color))
        buFont = etree.SubElement(pPr, qn("a:buFont"))
        buFont.set("typeface", "Arial")
        buChar = etree.SubElement(pPr, qn("a:buChar"))
        buChar.set("char", "•")

    def kicker_title(self, s, kicker, title, kcolor=NPU):
        self.text(s, MX, KICKER_Y, SW - 2 * MX, 0.3, kicker.upper(), size=12, color=kcolor, font=HEAD,
                  name="kicker")
        self.text(s, MX, TITLE_Y, SW - 2 * MX, TITLE_H, title, size=34, font=HEAD, name="title",
                  anchor=MSO_ANCHOR.TOP)

    # -- shapes -----------------------------------------------------------------------------
    def box(self, s, x, y, w, h, fill, radius=0.08, line=None, name=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
        sh = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
        if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            sh.adjustments[0] = radius
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
        if line is None:
            sh.line.fill.background()
        else:
            sh.line.color.rgb = line
            sh.line.width = Pt(1.25)
        sh.shadow.inherit = False
        if name:
            sh.name = name
        sh.text_frame.text = ""
        return sh

    def pill(self, s, x, y, label, color, w=0.82, h=0.34, size=12, text_color=INK):
        sh = self.box(s, x, y, w, h, color, radius=0.5, name=f"pill {label}")
        tf = sh.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size, r.font.bold, r.font.name = Pt(size), True, HEAD
        r.font.color.rgb = text_color
        return sh

    def stat(self, s, x, y, w, value, label, color=None, vsize=40, lsize=14, lcolor=None, h_val=None):
        dark = getattr(s, "_dark", False)
        color = color or (ON_DARK if dark else INK)
        lcolor = lcolor or (ON_DARK_MUTED if dark else MUTED)
        hv = h_val or vsize / 72 * 1.25
        self.text(s, x, y, w, hv, value, size=vsize, color=color, font=HEAD, name="stat value")
        self.text(s, x, y + hv, w, 0.65, label, size=lsize, color=lcolor, name="stat label")

    # -- media ------------------------------------------------------------------------------
    def fit(self, path: Path, x, y, w, h):
        from PIL import Image
        with Image.open(path) as im:
            iw, ih = im.size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        return x + (w - pw) / 2, y + (h - ph) / 2, pw, ph

    def placeholder(self, s, x, y, w, h, label):
        dark = getattr(s, "_dark", False)
        b = self.box(s, x, y, w, h, DARK_CARD if dark else CARD, radius=0.04, line=LINE if not dark else None,
                     name="placeholder")
        tf = b.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = label
        r.font.size, r.font.name = Pt(16), BODY
        r.font.color.rgb = ON_DARK_MUTED if dark else MUTED
        self.missing.append(label)

    def image(self, s, path: Path, x, y, w, h, alt=None):
        if not path.exists():
            self.placeholder(s, x, y, w, h, f"Pending: {path.relative_to(ROOT).as_posix()}")
            return None
        px, py, pw, ph = self.fit(path, x, y, w, h)
        pic = s.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
        pic.name = path.stem
        if alt:
            pic._element.nvPicPr.cNvPr.set("descr", alt)
        return pic

    def demo_media(self, s, stem, x, y, w, h, alt):
        mp4 = MEDIA / "mp4" / f"{stem}.mp4"
        gif = MEDIA / f"{stem}.gif"
        mode = self.media_mode
        if mode == "mp4" and mp4.exists():
            poster = self.posters.get(stem)
            if poster is not None:
                px, py, pw, ph = self.fit_size(poster["size"], x, y, w, h)
                mv = s.shapes.add_movie(str(mp4), Inches(px), Inches(py), Inches(pw), Inches(ph),
                                        poster_frame_image=io.BytesIO(poster["png"]), mime_type="video/mp4")
                mv.name = f"{stem} video"
                mv._element.nvPicPr.cNvPr.set("descr", alt)
                return "mp4"
        if mode in ("mp4", "gif") and gif.exists():
            self.image(s, gif, x, y, w, h, alt)
            return "gif"
        poster = self.posters.get(stem)
        if poster is not None:
            px, py, pw, ph = self.fit_size(poster["size"], x, y, w, h)
            pic = s.shapes.add_picture(io.BytesIO(poster["png"]), Inches(px), Inches(py), Inches(pw), Inches(ph))
            pic._element.nvPicPr.cNvPr.set("descr", alt)
            return "poster"
        self.placeholder(s, x, y, w, h, f"Pending: media/mp4/{stem}.mp4")
        return None

    @staticmethod
    def fit_size(size, x, y, w, h):
        iw, ih = size
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        return x + (w - pw) / 2, y + (h - ph) / 2, pw, ph


def extract_posters(stems_fracs: dict) -> dict:
    """Grab one representative frame from each MP4 (cv2, else imageio)."""
    out = {}
    for stem, frac in stems_fracs.items():
        mp4 = MEDIA / "mp4" / f"{stem}.mp4"
        if not mp4.exists():
            continue
        frame_rgb = None
        try:
            import cv2
            cap = cv2.VideoCapture(str(mp4))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(n - 1, int(n * frac))))
            ok, frame = cap.read()
            cap.release()
            if ok:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            frame_rgb = None
        if frame_rgb is None:  # fallback: imageio-ffmpeg
            try:
                import numpy as np
                import imageio_ffmpeg
                gen = imageio_ffmpeg.read_frames(str(mp4))
                meta = next(gen)
                n = int(meta.get("duration", 0) * meta.get("fps", 0)) or 1
                target = int(n * frac)
                w, h = meta["size"]
                for i, raw in enumerate(gen):
                    if i >= target:
                        frame_rgb = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
                        break
                gen.close()
            except Exception:
                frame_rgb = None
        if frame_rgb is None:
            continue
        from PIL import Image
        im = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        out[stem] = {"png": buf.getvalue(), "size": im.size}
    return out


# ----------------------------------------------------------------------------------------------
# Slides
# ----------------------------------------------------------------------------------------------
def build(d: Data, deck: Deck):
    A = Data.atlas_stats(d.atlas)
    PA = Data.atlas_stats(d.p_atlas) if d.p_atlas else None
    drv_line = f"NPU driver {d.npu_drv} · graphics {d.gfx_drv} · OpenVINO {d.ov} · {'AC power' if d.on_ac else 'battery'}"
    SRC_CLEAN = f"benchmarks/results/*.json (clean sequential run via scripts/record_all.ps1; {drv_line})"
    SRC_PRELIM = (f"{d.prelim_dir.relative_to(ROOT).as_posix()}/ (preliminary, NPU driver {d.p_npu_drv}, other jobs running)"
                  if d.prelim_dir else "n/a")

    # ------------------------------------------------------------------ 1. Title
    s = deck.slide(dark=True, footer=False)
    deck.text(s, MX, 1.2, 9, 0.4, "DEVELOPER TALK  ·  ON-DEVICE AI", size=14, color=NPU, font=HEAD)
    deck.text(s, MX, 1.75, 11.5, 2.4,
              [[("What a 46-TOPS laptop NPU", {})], [("actually does", {"color": NPU})]],
              size=54, font=HEAD, line_spacing=0.95)
    deck.text(s, MX, 4.15, 11.5, 0.9,
              f"Measured AI on an Intel Core Ultra 5 322 (Panther Lake) with OpenVINO {d.ov}: "
              "vision, depth, offload and a local LLM on NPU vs GPU vs CPU",
              size=20, color=ON_DARK_MUTED)
    for i, dv in enumerate(("NPU", "GPU", "CPU")):
        deck.pill(s, MX + i * 1.0, 5.35, dv, DEV[dv])
    deck.text(s, MX, 6.35, 11, 0.4, f"Jeremy Callinan  ·  {REPO_URL}", size=16, color=ON_DARK_MUTED)
    deck.notes(s, f"""
Welcome. This talk is about what the NPU in a mainstream Panther Lake laptop actually does when you run real AI on it, measured rather than quoted from a spec sheet.
Test machine: Lenovo IdeaPad 5 2-in-1 15IPH11, Intel Core Ultra 5 322, OpenVINO {d.ov}.
Every number in this deck is regenerated from the JSON results in the repo by docs/presentation/build_tech_deck.py.
Colour code used throughout: NPU blue, GPU green, CPU orange.
Sources: README.md; {SRC_CLEAN}.
""")

    # ------------------------------------------------------------------ 2. Why NPUs
    s = deck.slide()
    deck.kicker_title(s, "Motivation", "Why put AI on an NPU at all?")
    llm_npu, llm_cpu = Data.llm_dev(d.llm, "NPU"), Data.llm_dev(d.llm, "CPU")
    off_cpu, off_npu = d.offload_phase("CPU"), d.offload_phase("NPU")
    cards = [
        ("1", "Always-on AI", "Background blur, detection, transcription and assistants run continuously. "
         "They need an engine that is not your CPU or your game's GPU."),
        ("2", "Keep the CPU free",
         f"Measured: the 3.8B LLM decodes with {llm_npu['summary']['cpu_cores_busy_median']:.2f} CPU cores busy on the NPU, "
         f"vs {llm_cpu['summary']['cpu_cores_busy_median']:.2f} on the CPU."),
        ("3", "Efficiency", "Intel positions the NPU as the low-power engine. Not measured yet: this laptop "
         "reports power only on battery, so this talk makes no power claim."),
    ]
    cw, gap = (SW - 2 * MX - 2 * 0.4) / 3, 0.4
    for i, (num, head, body) in enumerate(cards):
        x = MX + i * (cw + gap)
        deck.box(s, x, CONTENT_Y + 0.2, cw, 4.6, CARD, radius=0.05, name=f"card {head}")
        circ = deck.box(s, x + 0.4, CONTENT_Y + 0.6, 0.75, 0.75, NPU, shape=MSO_SHAPE.OVAL, name="number")
        tf = circ.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.size, r.font.name, r.font.bold = num, Pt(24), HEAD, True
        r.font.color.rgb = INK
        deck.text(s, x + 0.4, CONTENT_Y + 1.65, cw - 0.8, 0.6, head, size=22, font=HEAD)
        deck.text(s, x + 0.4, CONTENT_Y + 2.35, cw - 0.8, 2.3, body, size=16, color=MUTED, line_spacing=1.1)
    deck.notes(s, f"""
Three reasons people care about NPUs.
1) Always-on workloads: camera effects, live captions, local assistants. They run for hours, so they should not occupy the CPU or the GPU.
2) Keeping the CPU free. Measured here: Phi-3-mini int4 decoding keeps {llm_npu['summary']['cpu_cores_busy_median']:.2f} CPU cores busy on the NPU versus {llm_cpu['summary']['cpu_cores_busy_median']:.2f} cores when run on the CPU (median over 4 prompts). In the offload demo the AI process used {off_npu['ai_proc_cpu_ms_per_frame']:.1f} ms of CPU per frame with inference on the NPU versus {off_cpu['ai_proc_cpu_ms_per_frame']:.1f} ms with inference on the CPU.
3) Efficiency: this is Intel's positioning (vendor claim). We have NOT measured power: on AC the laptop's WMI BatteryStatus.DischargeRate reads 0 and no per-device power counter is readable without admin. Battery-power runs are on the next-steps list.
Sources: benchmarks/results/llm_chat.json (devices[].summary.cpu_cores_busy_median); benchmarks/results/offload_dashboard.json (phases[].result.ai_proc_cpu_ms_per_frame); examples/offload_dashboard/README.md (telemetry table).
""")

    # ------------------------------------------------------------------ 3. Hardware
    s = deck.slide()
    deck.kicker_title(s, "The hardware", "Core Ultra 5 322: three engines, one memory bus")
    blocks = [
        ("CPU", "6 cores / 6 threads", ["2 P-cores, up to 4.4 GHz", "4 LP E-cores, 3.3 GHz", "No Hyper-Threading", "Intel 18A compute tile"]),
        ("GPU", "2 Xe3 cores", ["18 TOPS INT8 peak (vendor)", "\"Intel Graphics\", not Arc", "2.3 GHz max", "OpenVINO: \"Intel(R) Graphics\""]),
        ("NPU", "NPU 5 \"Intel AI Boost\"", ["46 TOPS INT8 peak (vendor)", "Computes in FP16 / INT8", "Static shapes are the main path", "OpenVINO: \"Intel(R) AI Boost\""]),
    ]
    bw, bg = (SW - 2 * MX - 2 * 0.35) / 3, 0.35
    for i, (dv, head, lines) in enumerate(blocks):
        x = MX + i * (bw + bg)
        y = CONTENT_Y + 0.05
        deck.box(s, x, y, bw, 3.55, CARD, radius=0.05, name=f"{dv} block")
        deck.pill(s, x + 0.35, y + 0.3, dv, DEV[dv])
        deck.text(s, x + 0.35, y + 0.82, bw - 0.7, 0.5, head, size=19, font=HEAD)
        deck.text(s, x + 0.35, y + 1.42, bw - 0.7, 2.0, [{"runs": l, "bullet": True, "space_after": 3} for l in lines],
                  size=15, color=MUTED)
    # memory bar
    my = CONTENT_Y + 3.85
    mem = deck.box(s, MX, my, SW - 2 * MX, 0.95, INK, radius=0.12, name="memory")
    deck.text(s, MX + 0.4, my + 0.14, SW - 2 * MX - 0.8, 0.7,
              [[("Shared memory  ", {"font": HEAD, "color": NPU}),
                ("DDR5-5600, dual channel: about 90 GB/s theoretical, shared by all three engines "
                 "(the SoC supports DDR5-6400)", {"color": ON_DARK})]],
              size=17, anchor=MSO_ANCHOR.MIDDLE)
    deck.text(s, MX, my + 1.12, SW - 2 * MX, 0.35,
              "Vendor = Intel ARK / Quick Reference Guide figures, not measured here. 90 GB/s is derived (2 × 64-bit × 5600 MT/s).",
              size=12, color=MUTED, italic=True)
    deck.notes(s, """
Test laptop: Lenovo IdeaPad 5 2-in-1 15IPH11 (machine type 83UL) with the Core Ultra 5 322, the lowest Core Ultra 5 tier.
CPU: 6C/6T = 2 P-cores (Cougar Cove, 2.5 / 4.4 GHz) + 4 LP E-cores (1.9 / 3.3 GHz), no Hyper-Threading. CPU tile on Intel 18A. (Intel ARK)
GPU: "Intel Graphics" with 2 Xe3 cores at 2.3 GHz, 18 TOPS INT8 peak. NOT Arc-branded: Intel's Quick Reference Guide footnote reserves the Arc brand for configs meeting a minimum Xe-core count. Some third-party listings wrongly say 4 Xe cores or 50 NPU TOPS; Intel's own pages say 2 Xe and 46 TOPS.
NPU: NPU 5, 46 TOPS INT8 peak (vendor claim, ARK). OpenVINO on this unit reports NPU optimization capabilities FP16 and INT8 only (local observation); the NPU plugin docs say HW computation precision is FP16.
Platform TOPS for this SKU is not published; NPU+GPU = 64 TOPS is derived arithmetic, and the family's "up to 180" applies to top H SKUs only.
Memory: this laptop ships DDR5-5600 SO-DIMMs (Lenovo PSREF), below the SoC's DDR5-6400 ceiling. ~90 GB/s theoretical is derived in examples/llm_chat/README.md. Matters for LLM decode, which is bandwidth-bound.
Sources: docs/PANTHER_LAKE_REFERENCE.md (At a glance table, sections 1-2); examples/llm_chat/README.md ("Memory bandwidth sets the speed").
""")

    # ------------------------------------------------------------------ 4. Software stack
    s = deck.slide()
    deck.kicker_title(s, "Software stack", "OpenVINO everywhere, with NPU-specific rules")
    layers = [
        ("Your app: Python examples in this repo", CARD, INK),
        (f"OpenVINO GenAI {d.ov}  (LLMPipeline, streaming)", CARD, INK),
        (f"OpenVINO Runtime {d.ov}  (compile_model, InferRequest)", CARD, INK),
    ]
    lx, lw = MX, 6.6
    y = CONTENT_Y + 0.05
    for label, fill, col in layers:
        b = deck.box(s, lx, y, lw, 0.72, fill, radius=0.12, name="layer")
        deck.text(s, lx + 0.3, y, lw - 0.6, 0.72, label, size=16, color=col, anchor=MSO_ANCHOR.MIDDLE)
        y += 0.88
    pw3 = (lw - 2 * 0.2) / 3
    for i, dv in enumerate(("CPU", "GPU", "NPU")):
        x = lx + i * (pw3 + 0.2)
        deck.box(s, x, y, pw3, 0.72, DEV[dv], radius=0.12, name=f"{dv} plugin")
        deck.text(s, x, y, pw3, 0.72, f"{dv} plugin", size=16, font=HEAD, color=INK, align=PP_ALIGN.CENTER,
                  anchor=MSO_ANCHOR.MIDDLE)
    y += 0.88
    deck.box(s, lx, y, lw, 0.95, INK, radius=0.12, name="drivers")
    deck.text(s, lx + 0.3, y, lw - 0.6, 0.95,
              [[("Drivers  ", {"font": HEAD, "color": NPU}), (f"NPU {d.npu_drv}", {"color": ON_DARK})],
               [(f"Graphics {d.gfx_drv}", {"color": ON_DARK})]],
              size=15, anchor=MSO_ANCHOR.MIDDLE)
    rx = lx + lw + 0.6
    rw = SW - MX - rx
    deck.text(s, rx, CONTENT_Y + 0.05, rw, 0.45, "NPU rules of the road", size=20, font=HEAD)
    rules = [
        ("Static shapes. ", "The compiler builds one graph per input shape. Bounded dynamic shapes are preview-only."),
        ("FP16 compute. ", "FP32 models run in f16; INT8 weights are supported."),
        ("LLMs need symmetric int4. ", "Use channel‑wise *‑int4‑cw‑ov models. An asymmetric int4 model hung while compiling."),
        ("Cache compiled blobs. ", "Set CACHE_DIR: first compile takes seconds to minutes."),
        ("Driver ≥ 32.0.100.5540. ", "OpenVINO 2026.4 known issue CVS-233822: crashes with older NPU drivers."),
    ]
    deck.text(s, rx, CONTENT_Y + 0.65, rw, 4.6,
              [{"runs": [(a, {"font": HEAD, "color": INK}), (b, {"color": MUTED})], "bullet": True, "space_after": 8,
                "bullet_color": NPU} for a, b in rules], size=15)
    deck.notes(s, f"""
Everything runs through OpenVINO {d.ov} (PyPI 2026.4.1) and OpenVINO GenAI for the LLM. One code path, three device plugins: the only thing that changes is the device string.
NPU-specific rules, all from the OpenVINO NPU device docs and GenAI-on-NPU guide (cited in docs/PANTHER_LAKE_REFERENCE.md section 3):
- Static shapes are the standard path; bounded dynamic shapes are preview, NPU40XX+, validated on vision models.
- "Computation precision for the HW is FP16." So the Perf Atlas FP32 rows run at f16 on the NPU.
- LLMs on NPU: export with symmetric INT4 (--sym --weight-format int4 --group-size -1). The repo's Qwen3-0.6B INT4_ASYM (group 128) hung for over 20 minutes compiling for the NPU; Phi-3-mini int4-cw-ov works.
- CACHE_DIR / ov::cache_dir is supported and essential (see compile-cost slide).
- OpenVINO 2026.4 release notes: known issue CVS-233822, possible Windows crashes on Core Ultra Series 3 with older NPU drivers; recommends NPU driver 32.0.100.5540 or later.
Drivers used for every clean result: {drv_line}.
Sources: docs/PANTHER_LAKE_REFERENCE.md §3; docs/DRIVER_SETUP.md; examples/llm_chat/README.md ("Why symmetric int4 matters"); benchmarks/results/record_all_summary.json.
""")

    # ------------------------------------------------------------------ 5. Methodology
    s = deck.slide()
    deck.kicker_title(s, "Methodology", "Trust, but verify: how every number was measured")
    steps = [
        ("Verify the device", "Read EXECUTION_DEVICES from the compiled model. Never fall back silently."),
        ("Verify the answer", "Compare outputs to a numpy float32 reference. Wrong output is reported, not timed."),
        ("Time only infer()", "Prepared InferRequest, input bound once, warm-up first. Report p50 and p95."),
        ("Run one thing at a time", "Devices and demos run sequentially on an idle machine, on AC power."),
        ("Record the setup", "Driver versions, OpenVINO version and power source go into every JSON."),
    ]
    sy = CONTENT_Y + 0.05
    for i, (h, b) in enumerate(steps):
        y = sy + i * 0.98
        c = deck.box(s, MX, y + 0.08, 0.58, 0.58, NPU, shape=MSO_SHAPE.OVAL, name="step")
        tf = c.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.size, r.font.name, r.font.bold = str(i + 1), Pt(18), HEAD, True
        r.font.color.rgb = INK
        deck.text(s, MX + 0.85, y, 5.4, 0.4, h, size=18, font=HEAD)
        deck.text(s, MX + 0.85, y + 0.4, 5.4, 0.55, b, size=14, color=MUTED)
    cx, cw2 = 7.15, SW - MX - 7.15
    deck.box(s, cx, CONTENT_Y + 0.05, cw2, 4.75, INK, radius=0.04, name="code")
    code = [
        ('cm = core.compile_model(model, "NPU", {', ON_DARK),
        ('    "PERFORMANCE_HINT": "LATENCY",', ON_DARK),
        ('    "CACHE_DIR": "models/npucache"})', ON_DARK),
        ('assert cm.get_property(', ON_DARK),
        ('    "EXECUTION_DEVICES") == ["NPU"]', ON_DARK),
        ("", ON_DARK),
        ("req = cm.create_infer_request()", ON_DARK),
        ("req.set_input_tensor(ov.Tensor(x))", ON_DARK),
        ("for _ in range(3): req.infer()  # warm-up", ON_DARK_MUTED),
        ("t0 = perf_counter(); req.infer()", ON_DARK),
        ("dt = perf_counter() - t0  # only this", ON_DARK_MUTED),
        ("", ON_DARK),
        ("err = max|out - ref| / max|ref|  # numpy", ON_DARK_MUTED),
    ]
    deck.text(s, cx + 0.35, CONTENT_Y + 0.35, cw2 - 0.6, 4.3,
              [{"runs": [(t if t else " ", {"color": c})]} for t, c in code], size=14, font=MONO, line_spacing=1.05)
    deck.notes(s, f"""
Five rules behind every number.
1) Verify the device: OpenVINO can be configured to fall back (AUTO/HETERO); we compile for an explicit device and record EXECUTION_DEVICES from the compiled model. Examples never substitute another device. The LLM pipeline does not expose EXECUTION_DEVICES, so we use host CPU-cores-busy as evidence (0.1 cores on NPU vs ~2 on CPU).
2) Verify the answer: Perf Atlas compares each output to a numpy float32 reference built from the effective weights; tolerance 2e-3 for f32 compute, 1.5e-2 for f16. This is how we found the NPU batch bug.
3) Timing: ovutil.time_request uses a prepared InferRequest with the input tensor bound once; timed region is request.infer() only; 3 warm-up runs, then adaptive 10-200 runs (~1 s per config). We report p50 and p95.
4) Sequential: scripts/record_all.ps1 re-ran every benchmark and demo one at a time on an idle machine. The first, preliminary runs overlapped with other jobs, and that changed the conclusions (see "measurement conditions" slide).
5) Every JSON records OpenVINO version, device names and power source; drivers are logged in docs/DRIVER_SETUP.md and record_all_summary.json.
The code on the right is a simplified sketch of the pattern, not a verbatim excerpt.
Sources: examples/perf_atlas/README.md ("What it measures"); examples/llm_chat/README.md; README.md (NPU showcase examples); docs/DRIVER_SETUP.md.
""")

    # ------------------------------------------------------------------ 6. Perf Atlas overview
    s = deck.slide()
    deck.kicker_title(s, "Performance Atlas", f"{A['n_total']} configurations, one dashboard")
    deck.image(s, MEDIA / "perf_atlas.png", MX, CONTENT_Y - 0.05, 8.75, 5.15,
               alt="Perf Atlas dashboard: latency vs work, speed-up heatmaps, peak compute, compile cost, throughput, correctness")
    rx = MX + 8.75 + 0.4
    rw = SW - MX - rx
    deck.text(s, rx, CONTENT_Y + 0.1, rw, 0.4, "The grid", size=20, font=HEAD)
    deck.text(s, rx, CONTENT_Y + 0.6, rw, 1.9, [
        {"runs": "6 synthetic models: 3 MLPs, 3 conv stacks", "bullet": True, "space_after": 4},
        {"runs": "batch 1, 8, 32", "bullet": True, "space_after": 4},
        {"runs": "FP32, FP16, INT8 weights", "bullet": True, "space_after": 4},
        {"runs": "× NPU, GPU, CPU", "bullet": True},
    ], size=15, color=MUTED)
    deck.stat(s, rx, CONTENT_Y + 2.65, rw, f"{A['n_exec']}", "executed and verified against numpy", color=INK, vsize=36)
    deck.stat(s, rx, CONTENT_Y + 3.95, rw, f"{A['n_bad']}", "ran but returned the wrong answer (all NPU)", color=WARN, vsize=36)
    deck.notes(s, f"""
The Performance Atlas is the controlled experiment behind the demos: synthetic OpenVINO models built in-process with ov.opset13 from seeded weights (rng 2026), no downloads.
Grid: 6 models (MLP-S/M/L: 0.017-0.201 GFLOP/sample; CONV-XS/S/M: 0.058-1.81 GFLOP/sample) × batch 1, 8, 32 × FP32/FP16/INT8 × 3 devices = {A['n_total']} sync configs, plus {A['n_async']} async throughput runs.
Result of the clean run: {A['n_exec']} executed and matched the numpy reference; {A['n_bad']} returned the wrong answer, all on the NPU (next slides).
INT8 here is weight-only (i8 const -> Convert -> Multiply), activations in f16. It is not a full-INT8 calibrated model, so it does not exercise the NPU's int8 MAC arrays fully. Effective GFLOP/s is analytic FLOPs / latency; it is not TOPS.
Panels: latency vs work (log-log), NPU speed-up vs CPU and vs GPU heatmaps, peak effective compute, compile cost, async vs sync throughput, compatibility & correctness grid.
Sources: benchmarks/results/perf_atlas.json (meta, results[].status, async); media/perf_atlas.png; examples/perf_atlas/README.md.
""")

    # ------------------------------------------------------------------ 7. Where NPU wins / loses
    s = deck.slide()
    deck.kicker_title(s, "Performance Atlas", "Where the NPU wins, and where it doesn't")
    deck.image(s, CHARTS / "latency_by_workload.png", MX, CONTENT_Y - 0.05, 8.3, 5.4,
               alt="Latency by workload, NPU vs GPU vs CPU")
    rx = MX + 8.3 + 0.45
    rw = SW - MX - rx
    cm_n, cm_g, cm_c = (d.lat("CONV-M", 32, "FP16", x) for x in ("NPU", "GPU", "CPU"))
    deck.stat(s, rx, CONTENT_Y + 0.05, rw, f"{A['wins_cpu']} / {A['n_cmp']}", "configs where the NPU beats the CPU", color=NPU, vsize=36)
    deck.stat(s, rx, CONTENT_Y + 1.35, rw, f"{A['wins_gpu']} / {A['n_cmp']}", "configs where the NPU beats the GPU", color=GPU, vsize=36)
    deck.text(s, rx, CONTENT_Y + 2.75, rw, 2.6, [
        {"runs": [("Big work wins: ", {"font": HEAD, "color": INK}),
                  (f"CONV-M batch 32 takes {fms(cm_n)} ms on NPU, {fms(cm_g)} on GPU, {fms(cm_c)} on CPU.", {})],
         "space_after": 8},
        {"runs": [("Small work loses: ", {"font": HEAD, "color": INK}),
                  ("the GPU wins small convs and MLP-S at batch 1, and every batch-8 MLP.", {})]},
    ], size=15, color=MUTED)
    mlpl1 = [d.lat("MLP-L", 1, "FP16", x) for x in ("NPU", "GPU", "CPU")]
    deck.notes(s, f"""
Headline from the clean run: across the {A['n_cmp']} configs where all three devices returned a correct answer, the NPU had the lowest p50 vs CPU in {A['wins_cpu']} and vs GPU in {A['wins_gpu']}. So: a clear win over the CPU, roughly a coin-flip against the 2-core iGPU.
Where it wins: lots of work per call. CONV-M FP16 batch 32: NPU {fms(cm_n)} ms, GPU {fms(cm_g)} ms, CPU {fms(cm_c)} ms. Peak effective compute: NPU {A['peak']['NPU'][0]:.0f} GFLOP/s ({A['peak']['NPU'][1]} b{A['peak']['NPU'][2]} {A['peak']['NPU'][3]}), GPU {A['peak']['GPU'][0]:.0f}, CPU {A['peak']['CPU'][0]:.0f}. Effective GFLOP/s, not TOPS.
MLP-L FP16 batch 1: NPU {fms(mlpl1[0])} ms, GPU {fms(mlpl1[1])} ms, CPU {fms(mlpl1[2])} ms.
Where it loses: tiny work, where dispatch overhead dominates (CONV-XS b1 FP16: NPU {fms(d.lat('CONV-XS',1,'FP16','NPU'))} ms vs GPU {fms(d.lat('CONV-XS',1,'FP16','GPU'))} ms), and batch 8 (next slide).
Chart: media/charts/latency_by_workload.png (scripts/make_charts.py) spans six real and synthetic workloads; its takeaway per media/charts/README.md: the NPU has the lowest median latency on 5 of 6 workloads (vision detect, segmentation, depth, CONV-M b32, MLP-L b1); the GPU is fastest per LLM token (41.8 vs 42.5 ms).
Sources: benchmarks/results/perf_atlas.json results[] median_ms; win counts computed by this build script over configs where NPU, GPU and CPU all executed.
""")

    # ------------------------------------------------------------------ 8. Batch-8 cliff (native chart)
    s = deck.slide()
    _n8, _n32 = d.lat("MLP-L", 8, "FP16", "NPU"), d.lat("MLP-L", 32, "FP16", "NPU")
    deck.kicker_title(s, "Performance Atlas", f"The batch-8 cliff: same model, {_n8 / _n32:.0f}× slower")
    cd = CategoryChartData()
    cd.categories = ["batch 1", "batch 8", "batch 32"]
    vals = {}
    for dv in ("NPU", "GPU", "CPU"):
        vals[dv] = [d.lat("MLP-L", b, "FP16", dv) for b in (1, 8, 32)]
        cd.add_series(dv, [round(v, 1) if v is not None else None for v in vals[dv]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(MX), Inches(CONTENT_Y), Inches(8.2), Inches(5.0), cd)
    gf.name = "MLP-L FP16 latency chart"
    ch = gf.chart
    ch.font.name, ch.font.size = BODY, Pt(14)
    ch.font.color.rgb = MUTED
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.TOP
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(14)
    plot = ch.plots[0]
    plot.gap_width = 60
    plot.overlap = -10
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = '0.0', False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(13)
    dl.font.color.rgb = INK
    for ser, dv in zip(plot.series, ("NPU", "GPU", "CPU")):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = DEV[dv]
    va = ch.value_axis
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.has_title = True
    va.axis_title.text_frame.text = "p50 latency, ms (lower is better)"
    va.axis_title.text_frame.paragraphs[0].runs[0].font.size = Pt(13)
    va.axis_title.text_frame.paragraphs[0].runs[0].font.bold = False
    va.axis_title.text_frame.paragraphs[0].runs[0].font.color.rgb = MUTED
    ch.category_axis.format.line.color.rgb = LINE
    ch.category_axis.tick_labels.font.size = Pt(14)
    rx = MX + 8.2 + 0.5
    rw = SW - MX - rx
    n1, n8, n32 = vals["NPU"]
    deck.text(s, rx, CONTENT_Y + 0.1, rw, 0.4, "MLP-L, FP16, 100.7 M params", size=18, font=HEAD)
    deck.stat(s, rx, CONTENT_Y + 0.7, rw, f"{fms(n8)} ms", f"NPU at batch 8, vs {fms(n32)} ms at batch 32", color=NPU, vsize=40)
    deck.text(s, rx, CONTENT_Y + 2.35, rw, 2.6, [
        {"runs": f"Batch 8 is slower than batch 32 on the NPU for every MLP. The GPU and CPU scale normally.",
         "space_after": 8},
        {"runs": [("Lesson: ", {"font": HEAD, "color": INK}),
                  ("benchmark the exact shapes you ship. On this NPU, pad to batch 32 or stay at batch 1.", {})]},
    ], size=15, color=MUTED)
    p_n8 = d.lat("MLP-L", 8, "FP16", "NPU", d.p_atlas) if d.p_atlas else None
    p_n32 = d.lat("MLP-L", 32, "FP16", "NPU", d.p_atlas) if d.p_atlas else None
    deck.notes(s, f"""
Native chart, data from the clean Perf Atlas run: MLP-L FP16 p50 latency.
NPU: b1 {fms(vals['NPU'][0])} ms, b8 {fms(vals['NPU'][1])} ms, b32 {fms(vals['NPU'][2])} ms.
GPU: b1 {fms(vals['GPU'][0])}, b8 {fms(vals['GPU'][1])}, b32 {fms(vals['GPU'][2])} ms.
CPU: b1 {fms(vals['CPU'][0])}, b8 {fms(vals['CPU'][1])}, b32 {fms(vals['CPU'][2])} ms.
So batch 8 on the NPU is {n8 / n32:.1f}× slower than batch 32 even though batch 32 does 4× the work. Same pattern on MLP-S (NPU b8 {fms(d.lat('MLP-S',8,'FP16','NPU'))} vs b32 {fms(d.lat('MLP-S',32,'FP16','NPU'))} ms) and MLP-M (b8 {fms(d.lat('MLP-M',8,'FP16','NPU'))} vs b32 {fms(d.lat('MLP-M',32,'FP16','NPU'))} ms).
It is not a contention artefact: the preliminary run on NPU driver {d.p_npu_drv} measured {fms(p_n8) if p_n8 else 'n/a'} ms vs {fms(p_n32) if p_n32 else 'n/a'} ms, and the clean run on driver {d.npu_drv} reproduces it. docs/DRIVER_SETUP.md: "NPU MLPs at batch 8 are still about 7× slower than at batch 32".
Hypothesis only (not verified): the NPU compiler picks a poor tiling for this shape. Worth reporting to Intel together with the correctness bug.
Sources: benchmarks/results/perf_atlas.json; {SRC_PRELIM}; docs/DRIVER_SETUP.md driver history.
""")

    # ------------------------------------------------------------------ 9. Correctness bug
    s = deck.slide()
    deck.kicker_title(s, "Correctness", "NPU bug: batched convs return sample 0 for everyone", kcolor=WARN)
    deck.image(s, CHARTS / "npu_batch_bug.png", MX, CONTENT_Y - 0.05, 7.6, 5.3,
               alt="NPU batch bug: batch 2-8 conv outputs replicate sample 0")
    rx = MX + 7.6 + 0.45
    rw = SW - MX - rx
    diag = [r.get("batch_diagnosis") or {} for r in A["bad"]]
    e_own = max(x.get("err_samples_1plus_vs_own_ref", 0) for x in diag) if diag else 0
    e_s0 = max(x.get("err_samples_1plus_vs_sample0_ref", 0) for x in diag) if diag else 0
    rep = all(x.get("outputs_replicate_sample0") for x in diag) if diag else False
    bad_models = sorted({r["model"] for r in A["bad"]})
    bad_batches = sorted({r["batch"] for r in A["bad"]})
    deck.text(s, rx, CONTENT_Y + 0.05, rw, 4.9, [
        {"runs": [("What: ", {"font": HEAD, "color": INK}),
                  (f"all {A['n_bad']} NPU conv configs at batch {', '.join(map(str, bad_batches))} "
                   f"({len(bad_models)} models × 3 precisions) fail the numpy check.", {})], "space_after": 10},
        {"runs": [("Why: ", {"font": HEAD, "color": INK}),
                  (f"samples 1..n match sample 0's reference (error {e_s0:.4f}), not their own (error {e_own:.3f}).", {})],
         "space_after": 10},
        {"runs": [("Scope: ", {"font": HEAD, "color": INK}),
                  ("a batch scan found batch 2–8 wrong; batch 1 and ≥ 12 correct. MLPs unaffected.", {})], "space_after": 10},
        {"runs": [("Status: ", {"font": HEAD, "color": WARN}),
                  (f"persists on NPU driver {d.npu_drv}. It fails silently: no error, just wrong numbers.", {})]},
    ], size=15, color=MUTED)
    deck.notes(s, f"""
This is the most important finding for developers: the NPU returned confidently wrong answers with no error.
Clean run ({d.npu_drv}): {A['n_bad']} NPU configs with status failed_correctness: {', '.join(bad_models)} at batch {', '.join(map(str, bad_batches))}, in FP32, FP16 and INT8.
batch_diagnosis in the JSON: outputs_replicate_sample0 = {rep}; max error of samples 1+ vs their own reference = {e_own:.4f}; vs sample 0's reference = {e_s0:.5f}. In other words, every element of the batch gets sample 0's output.
Batch scan (done on the first driver, documented in examples/perf_atlas/README.md and docs/DRIVER_SETUP.md): batch 2-8 wrong; batch 1 and >= 12 correct. Only batch 8 is in the clean grid; the 2-8 range comes from that scan.
Driver history: found on NPU driver 32.0.100.4724; after updating to {d.npu_drv} "the same 9 NPU configs give failed_correctness" (docs/DRIVER_SETUP.md).
Workarounds: run batch 1 (most real-time apps do), or batch >= 12, and keep a correctness check in CI when drivers change. Next step: minimal repro + report to Intel.
Chart: media/charts/npu_batch_bug.png (scripts/make_charts.py); exact numbers in media/charts/README.md.
Sources: benchmarks/results/perf_atlas.json results[status=failed_correctness].batch_diagnosis; docs/DRIVER_SETUP.md; examples/perf_atlas/README.md.
""")

    # ------------------------------------------------------------------ demo slide helper
    def demo_slide(kicker, title, stem, alt, rows, row_unit, takeaway, notes, foot=None, poster_only_hint=None):
        s = deck.slide(dark=True)
        deck.kicker_title(s, kicker, title)
        vx, vy, vw, vh = MX, CONTENT_Y, 8.35, 8.35 * 9 / 16
        kind = deck.demo_media(s, stem, vx, vy, vw, vh, alt)
        rx = vx + vw + 0.45
        rw = SW - MX - rx
        deck.text(s, rx, CONTENT_Y - 0.02, rw, 0.35, row_unit, size=13, color=ON_DARK_MUTED)
        ry = CONTENT_Y + 0.42
        for dv, val, sub in rows:
            deck.pill(s, rx, ry + 0.12, dv, DEV[dv], w=0.78, h=0.34, size=12)
            deck.text(s, rx + 0.98, ry - 0.02, rw - 0.98, 0.6, val, size=30, font=HEAD, color=DEV[dv])
            if sub:
                deck.text(s, rx + 0.98, ry + 0.55, rw - 0.98, 0.3, sub, size=12, color=ON_DARK_MUTED)
            ry += 0.98
        deck.box(s, rx, ry + 0.05, rw, SH - 0.75 - (ry + 0.05), DARK_CARD, radius=0.06, name="takeaway")
        deck.text(s, rx + 0.25, ry + 0.25, rw - 0.5, SH - 0.95 - (ry + 0.25), takeaway, size=15, color=ON_DARK,
                  line_spacing=1.05)
        if foot:
            deck.text(s, vx, vy + vh + 0.12, vw, 0.3, foot, size=11, color=ON_DARK_MUTED)
        deck.notes(s, notes + f"\nMedia on slide: {kind or 'placeholder'} (media/mp4/{stem}.mp4, fallback media/{stem}.gif). Click the video to play.")
        return s

    # ------------------------------------------------------------------ 10. Vision demo
    V = d.vision["devices"]
    fr = {k: V[k]["latency_ms"]["frame"]["median_ms"] for k in V}
    det = {k: V[k]["latency_ms"]["detect"]["median_ms"] for k in V}
    seg = {k: V[k]["latency_ms"]["segment"]["median_ms"] for k in V}
    p95 = {k: V[k]["latency_ms"]["frame"]["p95_ms"] for k in V}
    demo_slide(
        "Demo 1 · Vision side-by-side", "Same frames, three devices, detect + segment",
        "vision_side_by_side", "Street footage processed by YOLOX-Tiny and selfie segmentation on NPU, GPU and CPU",
        [(k, f"{fms(fr[k])} ms", f"p95 {fms(p95[k])} ms") for k in ("NPU", "GPU", "CPU")],
        "p50 per frame, detect + segment",
        [{"runs": [("NPU is fastest here: ", {"font": HEAD}),
                   (f"{fr['GPU'] / fr['NPU']:.1f}× the GPU and {fr['CPU'] / fr['NPU']:.1f}× the CPU on INT8 YOLOX-Tiny at batch 1.", {})]}],
        f"""
Demo: the same 1280x720 street clip goes through YOLOX-Tiny INT8 (416x416) + MediaPipe selfie segmentation (256x256) on each device; panels show each device's own detections and masks, with EXECUTION_DEVICES on screen.
Mode: {d.vision['mode']} — each device runs alone on the same pre-processed frame, order rotates every frame. Timing excludes decode, pre-processing, NMS and drawing.
Clean run, {d.vision['frames']} frames: per-frame p50 NPU {fms(fr['NPU'])} ms, GPU {fms(fr['GPU'])} ms, CPU {fms(fr['CPU'])} ms (p95 {fms(p95['NPU'])} / {fms(p95['GPU'])} / {fms(p95['CPU'])}).
Detection alone: {fms(det['NPU'])} / {fms(det['GPU'])} / {fms(det['CPU'])} ms. Segmentation: {fms(seg['NPU'])} / {fms(seg['GPU'])} / {fms(seg['CPU'])} ms.
Compile: NPU detect {V['NPU']['compile']['detect']['compile_s']:.2f} s, GPU detect {V['GPU']['compile']['detect']['compile_s']:.2f} s, CPU detect {V['CPU']['compile']['detect']['compile_s']:.2f} s.
Pipeline FPS ({d.vision['pipeline_fps']:.1f}) is the whole loop running all three devices one after another plus drawing, not one device's rate.
Outputs differ slightly between devices (precision differences); the demo shows that as is.
Sources: benchmarks/results/vision_side_by_side.json; examples/vision_side_by_side/README.md. Footage: "DiagonalCrosswalkYongeDundas.webm" by Raysonho, CC0 1.0, Wikimedia Commons.""",
        foot="Footage: \"DiagonalCrosswalkYongeDundas\" by Raysonho, CC0 1.0, via Wikimedia Commons")

    # ------------------------------------------------------------------ 11. Depth demo
    DP = d.depth["devices"]
    dm = {k: DP[k]["benchmark"]["median_ms"] for k in DP}
    live = d.depth["live"]
    demo_slide(
        "Demo 2 · Depth Parallax", "Flat video in, orbiting 3D out",
        "depth_parallax", "Forest clip turned into an orbiting 3D point cloud using NPU depth",
        [(k, f"{fms(dm[k])} ms", f"{DP[k]['benchmark']['per_s_at_median']:.0f} inferences/s") for k in ("NPU", "GPU", "CPU")],
        "Depth Anything V2 Small, 364×364, median",
        [{"runs": [("NPU does depth, CPU does 3D: ", {"font": HEAD}),
                   (f"{live['end_to_end_fps']:.0f} FPS end to end. The bottleneck is the CPU renderer "
                    f"({live['host_ms_per_frame']['median_ms']:.0f} ms/frame), not the NPU.", {})]}],
        f"""
Demo: every frame goes through Depth Anything V2 Small on the NPU; the CPU lifts it to a point cloud and re-renders from an orbiting virtual camera.
Startup benchmark, same model and input, clean run: NPU {fms(dm['NPU'])} ms, GPU {fms(dm['GPU'])} ms, CPU {fms(dm['CPU'])} ms median (p95 NPU {fms(DP['NPU']['benchmark']['p95_ms'])} ms). NPU is {dm['CPU'] / dm['NPU']:.1f}× faster than the CPU and {dm['GPU'] / dm['NPU']:.2f}× the GPU.
Live run ({live['frames']} frames): NPU model latency p50 {fms(live['model_latency_ms']['median_ms'])} ms (InferRequest.latency, overlapped with host work); host work {fms(live['host_ms_per_frame']['median_ms'])} ms/frame of which parallax render {fms(live['parallax_render_ms']['median_ms'])} ms; end-to-end {live['end_to_end_fps']:.1f} FPS including recording overhead.
Inference of frame N+1 overlaps rendering of frame N, so the demo is CPU-render bound. That is the point: the NPU leaves the CPU free to do the graphics.
Compile: NPU cold {DP['NPU']['compile_s']:.1f} s ({DP['NPU'].get('cache')}); see compile-cost slide for cached numbers.
Caveat: relative monocular depth, plausible not metric.
Sources: benchmarks/results/depth_parallax.json; examples/depth_parallax/README.md. Footage: "Fern Canyon in Prairie Creek Redwoods State Park" by Thomas Farley, CC0 1.0.""",
        foot="Footage: \"Fern Canyon in Prairie Creek Redwoods State Park\" by Thomas Farley, CC0 1.0, via Wikimedia Commons")

    # ------------------------------------------------------------------ 12. Offload demo
    O = {k: d.offload_phase(k) for k in ("CPU", "GPU", "NPU")}
    drift = (d.offload.get("baseline") or {}).get("drift_pct")
    cfg = d.offload.get("config", {})
    noisy = drift is not None and abs(drift) > 10
    rounds = cfg.get("rounds")
    fg = lambda k: f"{O[k]['fg_pct']:.0f}%" if O[k].get("fg_pct") is not None else "n/a"  # noqa: E731
    pre = lambda k: f"{O[k]['prepost_ms']:.1f}" if O[k].get("prepost_ms") is not None else "n/a"  # noqa: E731
    pO = d.offload_phase("CPU", d.p_offload) if d.p_offload else None
    take = [{"runs": [("Same 30 FPS target, very different CPU bill: ", {"font": HEAD}),
                      (f"{O['NPU']['ai_proc_cpu_ms_per_frame']:.1f} ms of CPU per frame with the NPU vs "
                       f"{O['CPU']['ai_proc_cpu_ms_per_frame']:.1f} ms on the CPU.", {})]}]
    demo_slide(
        "Demo 3 · Offload dashboard", "Always-on AI vs your foreground work",
        "offload_dashboard", "Dashboard: blur + detection at 30 FPS placed on CPU, GPU, then NPU while CPU-bound work runs",
        [(k, f"{O[k]['ai_fps']:.1f} FPS", f"{O[k]['ai_proc_cpu_ms_per_frame']:.1f} ms of CPU per frame")
         for k in ("NPU", "GPU", "CPU")],
        "AI pipeline rate (target 30 FPS)",
        take,
        f"""
Demo: one continuous AI workload (selfie segmentation + YOLOX-tiny on a 640x360 stream, paced to {cfg.get('target_fps', 30):.0f} FPS) placed on CPU, GPU and NPU, while {cfg.get('fg_workers', '?')} foreground processes run zlib compression as stand-in "user work". Phases {cfg.get('phase_seconds', '?')} s{f", {rounds} rounds (values are means over rounds)" if rounds else ''}, first {cfg.get('settle', '?')} s of each phase excluded.
Clean run ({d.offload.get('timestamp', '?')}):
- AI on CPU: {O['CPU']['ai_fps']:.1f} FPS, inference {fms(O['CPU'].get('infer_ms') or 0)} ms, AI process {O['CPU']['ai_proc_cpu_ms_per_frame']:.1f} ms CPU/frame ({O['CPU'].get('ai_proc_cpu_pct', 0):.1f}% of system CPU), foreground work {fg('CPU')} of baseline.
- AI on GPU: {O['GPU']['ai_fps']:.1f} FPS, inference {fms(O['GPU'].get('infer_ms') or 0)} ms, {O['GPU']['ai_proc_cpu_ms_per_frame']:.1f} ms CPU/frame ({O['GPU'].get('ai_proc_cpu_pct', 0):.1f}%), foreground {fg('GPU')}.
- AI on NPU: {O['NPU']['ai_fps']:.1f} FPS, inference {fms(O['NPU'].get('infer_ms') or 0)} ms, {O['NPU']['ai_proc_cpu_ms_per_frame']:.1f} ms CPU/frame ({O['NPU'].get('ai_proc_cpu_pct', 0):.1f}%), foreground {fg('NPU')}.
Pre/post (resize, blend, NMS, drawing) stays on the CPU in every placement: ~{pre('NPU')} ms/frame. Offloading removes the inference cost, not all CPU cost.
{'CAVEAT: the two AI-off baselines differ by ' + f'{drift:.0f}' + '%, so the foreground-% column is within noise in this run; the CPU-ms-per-frame numbers are the robust signal. ' if noisy else ''}{f"The preliminary (contended) run showed a starker picture: AI on CPU reached only {pO['ai_fps']:.1f} FPS and foreground work fell to {pO['fg_pct']:.0f}%, because other demos were using 24-49% of the CPU." if pO else ''}
This demo is scheduled to be re-recorded; rebuilding the deck picks up the new MP4 and JSON automatically.
Sources: benchmarks/results/offload_dashboard.json (placements[] or phases[] .result, baseline); {SRC_PRELIM}offload_dashboard.json; examples/offload_dashboard/README.md.""")

    # ------------------------------------------------------------------ 13. CPU cost chart
    s = deck.slide()
    deck.kicker_title(s, "Offload", "What the CPU pays for the same AI workload")
    deck.image(s, CHARTS / "cpu_cost.png", MX, CONTENT_Y - 0.05, 8.3, 5.4, alt="CPU cost of AI per device")
    rx = MX + 8.3 + 0.45
    rw = SW - MX - rx
    ratio = O["CPU"]["ai_proc_cpu_ms_per_frame"] / O["NPU"]["ai_proc_cpu_ms_per_frame"]
    deck.stat(s, rx, CONTENT_Y + 0.05, rw, f"{ratio:.0f}× less", "CPU time per AI frame with the NPU vs the CPU", color=NPU, vsize=40)
    lc = Data.llm_dev(d.llm, "CPU")["summary"]["cpu_cores_busy_median"]
    lg = Data.llm_dev(d.llm, "GPU")["summary"]["cpu_cores_busy_median"]
    ln = Data.llm_dev(d.llm, "NPU")["summary"]["cpu_cores_busy_median"]
    deck.stat(s, rx, CONTENT_Y + 1.55, rw, f"{ln:.2f} cores", f"busy during LLM decode on the NPU (GPU {lg:.2f}, CPU {lc:.2f})", color=NPU, vsize=40)
    deck.text(s, rx, CONTENT_Y + 3.2, rw, 1.9,
              "On a 6-thread laptop CPU, that headroom is the difference between AI that is invisible and AI you can feel.",
              size=15, color=MUTED)
    deck.notes(s, f"""
The quantitative core of the offload story, from the clean runs.
Offload demo, AI process CPU time per frame: CPU placement {O['CPU']['ai_proc_cpu_ms_per_frame']:.1f} ms, GPU {O['GPU']['ai_proc_cpu_ms_per_frame']:.1f} ms, NPU {O['NPU']['ai_proc_cpu_ms_per_frame']:.1f} ms (ratio CPU/NPU = {ratio:.1f}×). Includes OpenVINO runtime threads; in the NPU case it is almost all CPU-side pre/post-processing.
LLM decode, host CPU cores busy (median of 4 prompts): NPU {ln:.2f}, GPU {lg:.2f}, CPU {lc:.2f}.
The Core Ultra 5 322 has only 2 P-cores + 4 LP E-cores and no SMT, so any CPU-side AI takes a big share. Bigger chips would show smaller relative losses.
Chart: media/charts/cpu_cost.png (scripts/make_charts.py); exact numbers in media/charts/README.md.
Sources: benchmarks/results/offload_dashboard.json; benchmarks/results/llm_chat.json.
""")

    # ------------------------------------------------------------------ 14. LLM demo
    L = {k: Data.llm_dev(d.llm, k) for k in ("NPU", "GPU", "CPU")}
    LS = {k: L[k]["summary"] for k in L}
    meta = d.llm["meta"]
    demo_slide(
        "Demo 4 · Local LLM", f"Phi-3-mini int4 (3.8B) chatting on the NPU",
        "llm_chat", "Replay of the same prompt on NPU, GPU and CPU at measured token timings",
        [(k, f"{LS[k]['tok_s_median']:.1f} tok/s", f"TTFT {LS[k]['ttft_ms_median']:.0f} ms · {LS[k]['cpu_cores_busy_median']:.2f} cores busy")
         for k in ("NPU", "GPU", "CPU")],
        "decode speed, median of 4 prompts",
        [{"runs": [("NPU ≈ GPU on decode, GPU wins first token. ", {"font": HEAD}),
                   ("The NPU's edge is that it leaves the CPU almost idle.", {})]}],
        f"""
Demo: {meta['model']['hf_id']} ({meta['model']['params']}, {meta['model']['weight_mode']}, group size {meta['model']['weight_group_size']}, {meta['model']['weights_gb']} GB weights) via OpenVINO GenAI LLMPipeline. The video is a replay of SEQUENTIAL runs at measured per-token timing, not a live concurrent race.
Clean run, greedy, up to {meta['max_new_tokens']} new tokens, medians over 4 prompts:
- NPU: {LS['NPU']['tok_s_median']:.1f} tok/s, TTFT {LS['NPU']['ttft_ms_median']:.0f} ms, {LS['NPU']['cpu_cores_busy_median']:.2f} cores busy.
- GPU: {LS['GPU']['tok_s_median']:.1f} tok/s, TTFT {LS['GPU']['ttft_ms_median']:.0f} ms, {LS['GPU']['cpu_cores_busy_median']:.2f} cores busy.
- CPU: {LS['CPU']['tok_s_median']:.1f} tok/s, TTFT {LS['CPU']['ttft_ms_median']:.0f} ms, {LS['CPU']['cpu_cores_busy_median']:.2f} cores busy.
Note the change from the preliminary run (other jobs running, driver {d.p_npu_drv}): there the NPU looked fastest ({Data.llm_dev(d.p_llm,'NPU')['summary']['tok_s_median']:.1f} vs GPU {Data.llm_dev(d.p_llm,'GPU')['summary']['tok_s_median']:.1f} tok/s). On an idle machine the GPU catches up. The NPU number barely moved; the NPU is the most consistent device.
NPU TTFT is ~0.7 s regardless of prompt length (16-102 tokens): consistent with a fixed-shape static prefill graph (MAX_PROMPT_LEN={meta['npu_config']['MAX_PROMPT_LEN']}, MIN_RESPONSE_LEN={meta['npu_config']['MIN_RESPONSE_LEN']}).
Sources: benchmarks/results/llm_chat.json (devices[].summary, devices[].prompts[]); {SRC_PRELIM}llm_chat.json; examples/llm_chat/README.md.""")

    # ------------------------------------------------------------------ 15. LLM numbers
    s = deck.slide()
    deck.kicker_title(s, "Local LLM", "Decode is a memory-bandwidth problem")
    deck.image(s, CHARTS / "llm_compare.png", MX, CONTENT_Y - 0.05, 7.9, 5.4, alt="LLM decode speed, TTFT and CPU load per device")
    rx = MX + 7.9 + 0.45
    rw = SW - MX - rx
    deck.text(s, rx, CONTENT_Y + 0.05, rw, 0.4, "Back-of-envelope ceiling", size=20, font=HEAD)
    deck.box(s, rx, CONTENT_Y + 0.6, rw, 1.45, CARD, radius=0.08, name="formula")
    deck.text(s, rx + 0.3, CONTENT_Y + 0.6, rw - 0.6, 1.45, [
        [(f"90 GB/s ÷ {meta['model']['weights_gb']:.1f} GB/token", {"font": MONO, "size": 15})],
        [("≈ 40 tok/s ceiling", {"font": HEAD, "color": NPU, "size": 22})],
    ], size=16, anchor=MSO_ANCHOR.MIDDLE)
    deck.text(s, rx, CONTENT_Y + 2.3, rw, 2.9, [
        {"runs": "Every token reads all the weights, and all three engines share the same DDR5-5600.", "bullet": True,
         "space_after": 6},
        {"runs": f"Measured decode: {LS['CPU']['tok_s_median']:.0f} to {max(LS['NPU']['tok_s_median'], LS['GPU']['tok_s_median']):.0f} tok/s, all within 2× of each other.",
         "bullet": True, "space_after": 6},
        {"runs": "Prefill (TTFT) is compute-bound: there the GPU leads clearly.", "bullet": True},
    ], size=15, color=MUTED)
    deck.notes(s, f"""
Why the devices land close together on decode: each generated token reads essentially all ~{meta['model']['weights_gb']} GB of int4 weights + scales. Dual-channel DDR5-5600 gives ~90 GB/s theoretical, shared by NPU, GPU and CPU, so ~40 tok/s is a ceiling for any device; real kernels reach part of that. (Derivation from examples/llm_chat/README.md.)
Measured decode (clean): NPU {LS['NPU']['tok_s_median']:.1f}, GPU {LS['GPU']['tok_s_median']:.1f}, CPU {LS['CPU']['tok_s_median']:.1f} tok/s.
TTFT: NPU {LS['NPU']['ttft_ms_median']:.0f} ms, GPU {LS['GPU']['ttft_ms_median']:.0f} ms, CPU {LS['CPU']['ttft_ms_median']:.0f} ms (median; CPU TTFT for the 102-token prompt was {[p['ttft_ms'] for p in L['CPU']['prompts'] if p['prompt_tokens'] >= 100][0]:.0f} ms).
Model format matters on NPU: symmetric channel-wise int4 works; the repo's Qwen3-0.6B INT4_ASYM (group 128) hung for >20 min compiling on the NPU (it runs on CPU at ~31 tok/s and on GPU). Output-quality caveat: channel-wise int4 produces occasional glitches in Phi-3-mini outputs.
Chart: media/charts/llm_compare.png (scripts/make_charts.py); exact numbers in media/charts/README.md.
Sources: benchmarks/results/llm_chat.json; examples/llm_chat/README.md ("Memory bandwidth sets the speed", "Why symmetric int4 matters").
""")

    # ------------------------------------------------------------------ 16. Compile cost
    s = deck.slide()
    deck.kicker_title(s, "Compile cost", "The NPU's hidden tax: compile once, then cache")
    deck.image(s, CHARTS / "compile_cost.png", MX, CONTENT_Y - 0.05, 7.9, 5.4, alt="Compile time per device")
    rx = MX + 7.9 + 0.45
    rw = SW - MX - rx
    pL_npu = Data.llm_dev(d.p_llm, "NPU") if d.p_llm else None
    p_npu_cold = pL_npu["load"].get("cold_s") if pL_npu else None
    # Clean run: the NPU's first load is a cold compile even though the JSON labels it 'cached'
    # (depth_parallax had just written an unrelated blob to the shared cache dir; see media/charts/README.md).
    npu_cold = L["NPU"]["load"].get("cold_s") or L["NPU"]["load"]["first"]["load_s"]
    npu_cached = L["NPU"]["load"].get("cached_s")
    deck.stat(s, rx, CONTENT_Y + 0.05, rw, f"{npu_cold:.0f} s → {npu_cached:.1f} s",
              "LLM load on the NPU: first compile → from cache", color=NPU, vsize=36)
    deck.stat(s, rx, CONTENT_Y + 1.45, rw, f"{DP['NPU']['compile_s']:.0f} s",
              f"cold NPU compile, depth model (GPU {DP['GPU']['compile_s']:.0f} s, CPU {DP['CPU']['compile_s']:.0f} s)",
              color=NPU, vsize=36)
    deck.text(s, rx, CONTENT_Y + 2.95, rw, 2.2, [
        {"runs": [("Set CACHE_DIR. ", {"font": HEAD, "color": INK}),
                  ("Ship or pre-warm the blob, show a splash on first run.", {})], "space_after": 8},
        {"runs": [("Clear caches after a driver update. ", {"font": HEAD, "color": INK}),
                  ("Blobs are tied to the compiler version.", {})]},
    ], size=15, color=MUTED)
    pdp = d.p_depth["devices"] if d.p_depth else None
    deck.notes(s, f"""
Compile time is where the NPU pays. Perf Atlas (clean): median compile_model NPU {A['compile_median']['NPU']:.2f} s (max {A['compile_max']['NPU']:.1f} s), GPU {A['compile_median']['GPU']:.2f} s (max {A['compile_max']['GPU']:.1f} s), CPU {A['compile_median']['CPU']:.3f} s.
Depth model, cold compile (clean run): NPU {DP['NPU']['compile_s']:.1f} s, GPU {DP['GPU']['compile_s']:.1f} s, CPU {DP['CPU']['compile_s']:.1f} s.{f" From cache (preliminary run): NPU {pdp['NPU']['compile_s']:.2f} s, GPU {pdp['GPU']['compile_s']:.2f} s." if pdp else ''}
LLM (Phi-3-mini int4), clean run: NPU first load {npu_cold:.1f} s, then {npu_cached} s from cache. {'The JSON labels that first load cache=cached, but it was really a cold compile (a shared models/npucache made the blobs-present check misfire; fixed in llm.py). ' if L['NPU']['load']['first']['cache'] == 'cached' else 'The JSON correctly labels that first load as a cold compile. '}Earlier cold compiles: {p_npu_cold if p_npu_cold else 'n/a'} s in the preliminary run (driver {d.p_npu_drv}), and 204 s in an even earlier contended run (examples/llm_chat/README.md). GPU cold {L['GPU']['load'].get('cold_s')} s and cached {L['GPU']['load'].get('cached_s')} s in the clean run, vs {Data.llm_dev(d.p_llm,'GPU')['load'].get('cold_s') if d.p_llm else 'n/a'} s cold in the preliminary run. CPU needs no compile ({L['CPU']['load'].get('no_cache_s')} s load).
Practice: CACHE_DIR, pre-warm at install time, splash screen on first run. Compile caches from the old driver were moved aside when updating drivers (docs/DRIVER_SETUP.md).
Chart: media/charts/compile_cost.png (scripts/make_charts.py); exact numbers in media/charts/README.md.
Sources: benchmarks/results/perf_atlas.json (compile_s); benchmarks/results/depth_parallax.json; benchmarks/results/llm_chat.json (load); {SRC_PRELIM}.
""")

    # ------------------------------------------------------------------ 17. Measurement conditions
    s = deck.slide()
    deck.kicker_title(s, "Measurement conditions", "Same laptop, different answers: contention matters")
    deck.image(s, CHARTS / "driver_contention.png", MX, CONTENT_Y - 0.05, 7.9, 5.4,
               alt="Preliminary (contended, old driver) vs clean (idle, new driver) results")
    rx = MX + 7.9 + 0.45
    rw = SW - MX - rx
    c_old = d.lat("CONV-M", 32, "FP16", "CPU", d.p_atlas) if d.p_atlas else None
    c_new = d.lat("CONV-M", 32, "FP16", "CPU")
    rows = []
    if PA:
        rows.append((f"{PA['wins_gpu']} → {A['wins_gpu']}", f"of {A['n_cmp']} configs where NPU beats GPU"))
    if c_old:
        rows.append((f"{c_old:.0f} → {c_new:.0f} ms", "CONV-M batch 32 on the CPU"))
    if d.p_llm:
        rows.append((f"{Data.llm_dev(d.p_llm,'GPU')['summary']['tok_s_median']:.1f} → {LS['GPU']['tok_s_median']:.1f}",
                     "GPU LLM decode, tok/s"))
    y = CONTENT_Y + 0.05
    for v, lab in rows:
        deck.stat(s, rx, y, rw, v, lab, color=INK, vsize=30)
        y += 1.25
    deck.text(s, rx, y + 0.05, rw, 1.2, "Preliminary → clean. Driver also changed, so the two effects are confounded.",
              size=13, color=MUTED, italic=True)
    vp = d.p_vision["devices"] if d.p_vision else None
    deck.notes(s, f"""
Cautionary tale. The first full runs (archived in {d.prelim_dir.relative_to(ROOT).as_posix() if d.prelim_dir else 'archive'}) were taken while other jobs used the NPU, GPU and CPU, on NPU driver {d.p_npu_drv}. The clean runs were sequential on an idle machine with NPU driver {d.npu_drv} / graphics {d.gfx_drv}. Both on AC.
What changed (preliminary -> clean):
- Perf Atlas NPU wins vs CPU: {PA['wins_cpu'] if PA else 'n/a'} -> {A['wins_cpu']} of {A['n_cmp']}; vs GPU: {PA['wins_gpu'] if PA else 'n/a'} -> {A['wins_gpu']}. Contention slowed CPU and GPU more than the NPU, flattering the NPU.
- CONV-M b32 FP16 on CPU: {f1(c_old) if c_old else 'n/a'} -> {f1(c_new)} ms. MLP-L b1 FP16 on GPU: {fms(d.lat('MLP-L',1,'FP16','GPU',d.p_atlas)) if d.p_atlas else 'n/a'} -> {fms(d.lat('MLP-L',1,'FP16','GPU'))} ms.
- LLM GPU decode: {Data.llm_dev(d.p_llm,'GPU')['summary']['tok_s_median'] if d.p_llm else 'n/a'} -> {LS['GPU']['tok_s_median']} tok/s; GPU cold load {Data.llm_dev(d.p_llm,'GPU')['load'].get('cold_s') if d.p_llm else 'n/a'} -> {L['GPU']['load'].get('cold_s')} s. NPU decode: {Data.llm_dev(d.p_llm,'NPU')['summary']['tok_s_median'] if d.p_llm else 'n/a'} -> {LS['NPU']['tok_s_median']} tok/s (stable).
- Vision frame p50 GPU: {fms(vp['GPU']['latency_ms']['frame']['median_ms']) if vp else 'n/a'} -> {fms(fr['GPU'])} ms; NPU: {fms(vp['NPU']['latency_ms']['frame']['median_ms']) if vp else 'n/a'} -> {fms(fr['NPU'])} ms.
- Offload, AI on CPU: {pO['ai_fps'] if pO else 'n/a'} -> {O['CPU']['ai_fps']} FPS.
Not changed: the NPU batch 2-8 conv bug and the batch-8 slowdown.
Caveat: driver and contention changed together, so we cannot attribute each delta to one cause. The NPU numbers were the most stable across both.
Chart: media/charts/driver_contention.png (scripts/make_charts.py); exact numbers in media/charts/README.md.
Sources: {SRC_CLEAN}; {SRC_PRELIM}; docs/DRIVER_SETUP.md driver history.
""")

    # ------------------------------------------------------------------ 18. Practical guidance
    s = deck.slide()
    deck.kicker_title(s, "Practical guidance", "Which engine should run your model?")
    cols = [
        ("NPU", "Use the NPU for", [
            "Always-on, steady-rate vision/audio at batch 1",
            "Large static batches (batch 32 dense or conv work)",
            "LLM decode when the CPU must stay free",
            "Anything that runs for hours on a laptop",
        ]),
        ("GPU", "Use the GPU for", [
            "Lowest time-to-first-token (prefill)",
            "Tiny models and small convs at batch 1",
            "Batch 2–8, where the NPU is slow or wrong",
            "Dynamic shapes and models that change often",
        ]),
        ("CPU", "Use the CPU for", [
            "One-shot runs where compile time dominates",
            "Ops or formats the NPU rejects (asymmetric int4)",
            "The correctness reference for the others",
            "Not sustained AI: it has only 6 threads",
        ]),
    ]
    cw, gap = (SW - 2 * MX - 2 * 0.35) / 3, 0.35
    for i, (dv, head, items) in enumerate(cols):
        x = MX + i * (cw + gap)
        deck.box(s, x, CONTENT_Y + 0.05, cw, 4.9, CARD, radius=0.05, name=f"{dv} guidance")
        deck.pill(s, x + 0.35, CONTENT_Y + 0.35, dv, DEV[dv])
        deck.text(s, x + 0.35, CONTENT_Y + 0.85, cw - 0.7, 0.5, head, size=20, font=HEAD)
        deck.text(s, x + 0.35, CONTENT_Y + 1.5, cw - 0.7, 3.3,
                  [{"runs": it, "bullet": True, "space_after": 10, "bullet_color": DEV[dv]} for it in items],
                  size=15, color=MUTED)
    deck.notes(s, f"""
Rules of thumb from this laptop's data, not universal laws.
NPU: vision at batch 1 (vision demo NPU {fms(fr['NPU'])} ms vs GPU {fms(fr['GPU'])} ms per frame; depth {fms(dm['NPU'])} vs {fms(dm['GPU'])} ms); batch-32 dense/conv (CONV-M b32 {fms(cm_n)} ms vs GPU {fms(cm_g)} ms); LLM decode with {LS['NPU']['cpu_cores_busy_median']:.2f} cores busy.
GPU: TTFT {LS['GPU']['ttft_ms_median']:.0f} ms vs NPU {LS['NPU']['ttft_ms_median']:.0f} ms; wins small convs at batch 1 (CONV-XS b1 {fms(d.lat('CONV-XS',1,'FP16','GPU'))} vs NPU {fms(d.lat('CONV-XS',1,'FP16','NPU'))} ms) and every batch-8 MLP; NPU batch 2-8 conv is wrong; NPU wants static shapes.
CPU: compile median {A['compile_median']['CPU']:.3f} s vs NPU {A['compile_median']['NPU']:.2f} s; ran asymmetric-int4 Qwen3-0.6B (~31 tok/s per README) which hung compiling on the NPU; useful as reference. But sustained AI on the CPU costs {O['CPU']['ai_proc_cpu_ms_per_frame']:.0f} ms CPU per frame in the offload demo.
Sources: all results JSONs listed on earlier slides; examples/llm_chat/README.md for Qwen3.
""")

    # ------------------------------------------------------------------ 19. Developer checklist
    s = deck.slide()
    deck.kicker_title(s, "Developer checklist", "Seven things to do before you ship on the NPU")
    checks = [
        ("Assert EXECUTION_DEVICES", "A device string is a request, not proof."),
        ("Set CACHE_DIR", "And pre-warm it; cold NPU compiles take seconds to minutes."),
        ("Pick shapes deliberately", "Static shapes; batch 1 or ≥ 12. Benchmark the exact shape you ship."),
        ("Check outputs, not just speed", "Keep a reference comparison in CI, re-run it on every driver update."),
        ("Use symmetric int4 for LLMs", "Channel‑wise int4: export with ‑‑sym, or use *‑int4‑cw‑ov."),
        ("Update the NPU driver", f"≥ 32.0.100.5540 for OpenVINO 2026.4 (CVS-233822)."),
        ("Measure on an idle machine", "Record driver, power source and what else is running."),
    ]
    half = 4
    colw = (SW - 2 * MX - 0.5) / 2
    for i, (h, b) in enumerate(checks):
        col, row = (0, i) if i < half else (1, i - half)
        x = MX + col * (colw + 0.5)
        y = CONTENT_Y + 0.05 + row * 1.22
        c = deck.box(s, x, y + 0.05, 0.5, 0.5, NPU, shape=MSO_SHAPE.OVAL, name="check")
        tf = c.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.size, r.font.bold, r.font.name = "✓", Pt(18), True, "Segoe UI Symbol"
        r.font.color.rgb = INK
        deck.text(s, x + 0.75, y, colw - 0.8, 0.45, h, size=18, font=HEAD)
        deck.text(s, x + 0.75, y + 0.45, colw - 0.8, 0.65, b, size=15, color=MUTED)
    deck.notes(s, f"""
The actionable summary for developers. Each item maps to a finding:
1) EXECUTION_DEVICES: every example records it; GenAI LLMPipeline doesn't expose it, so use CPU-cores-busy as evidence.
2) CACHE_DIR: NPU LLM first compile {npu_cold:.1f} s vs {npu_cached} s from cache (clean run).
3) Shapes: batch-8 cliff (MLP-L {fms(n8)} vs {fms(n32)} ms at b32) and the batch 2-8 conv bug.
4) Outputs: the bug fails silently; it was caught only by the numpy check. It persisted across driver {d.p_npu_drv} -> {d.npu_drv}.
5) Symmetric int4: Qwen3-0.6B INT4_ASYM hung >20 min compiling on NPU; Phi-3-mini int4-cw works.
6) Driver: OpenVINO 2026.4 release notes, CVS-233822.
7) Idle machine: preliminary vs clean changed the NPU-vs-GPU win count from {PA['wins_gpu'] if PA else 'n/a'} to {A['wins_gpu']} of {A['n_cmp']}.
Sources: docs/DRIVER_SETUP.md; docs/PANTHER_LAKE_REFERENCE.md §3; examples/*/README.md; results JSONs.
""")

    # ------------------------------------------------------------------ 20. Limitations & next steps
    s = deck.slide()
    deck.kicker_title(s, "Limitations & next steps", "What this does not show yet")
    lim = [
        "One laptop, one SKU, Windows 11, AC power only",
        "No power or energy numbers yet (no AC power counter)",
        "Perf Atlas models are synthetic; INT8 is on weights only",
        "Offload foreground % is noisy between runs",
        "Driver and contention changed together between runs",
    ]
    nxt = [
        "Battery runs: energy per inference and per token",
        "More real models: full-INT8 (NNCF), speech, VLMs",
        "Linux: ivpu driver + linux-npu-driver on Ubuntu",
        "Minimal repro of the batch 2–8 bug, reported to Intel",
        "Re-test asymmetric-int4 LLM compile on new drivers",
    ]
    colw = (SW - 2 * MX - 0.5) / 2
    for i, (head, items, col) in enumerate((("Limitations", lim, WARN), ("Next steps", nxt, NPU))):
        x = MX + i * (colw + 0.5)
        deck.box(s, x, CONTENT_Y + 0.05, colw, 4.9, CARD, radius=0.05, name=head)
        deck.text(s, x + 0.4, CONTENT_Y + 0.35, colw - 0.8, 0.5, head, size=22, font=HEAD, color=col)
        deck.text(s, x + 0.4, CONTENT_Y + 1.05, colw - 0.8, 3.8,
                  [{"runs": it, "bullet": True, "space_after": 10, "bullet_color": col} for it in items], size=16, color=INK)
    deck.notes(s, f"""
Be upfront about limits.
- Single machine: Lenovo IdeaPad 5 2-in-1 15IPH11 with the lowest Core Ultra 5 tier (2 Xe3 cores, 6 threads). Bigger Panther Lake SKUs have up to 12 Xe3 cores and more CPU, so GPU/CPU comparisons will shift.
- Power: WMI BatteryStatus.DischargeRate is 0 on AC; no per-device package power counter readable without admin (examples/offload_dashboard/README.md telemetry table). Planned: battery runs for energy per inference / per token.
- Perf Atlas: synthetic MLP/conv stacks; INT8 = weight-only, no activation quantization, so the NPU's int8 MAC arrays are not fully exercised.
- Offload: foreground-% depends on baseline drift ({f'{drift:.0f}%' if drift is not None else 'see JSON'} in the current run).
- Preliminary vs clean runs differ in both driver ({d.p_npu_drv} -> {d.npu_drv}) and contention.
Next: Linux (NPU 5 in ivpu since Linux 6.13; intel/linux-npu-driver v1.38.0 verifies Panther Lake on Ubuntu 24.04/26.04; OpenVINO 2026.4 adds Ubuntu 26.04 NPU support) — docs/PANTHER_LAKE_REFERENCE.md §6. Report the NPU batch bug to Intel with a minimal repro. Re-test Qwen3-0.6B INT4_ASYM NPU compile hang.
Sources: docs/PANTHER_LAKE_REFERENCE.md; docs/DRIVER_SETUP.md; examples/*/README.md caveats.
""")

    # ------------------------------------------------------------------ 21. Resources
    s = deck.slide()
    deck.kicker_title(s, "Resources", "Run it yourself")
    deck.box(s, MX, CONTENT_Y + 0.05, 6.7, 4.9, INK, radius=0.04, name="quickstart")
    qs = [
        ("git clone https://" + REPO_URL, ON_DARK),
        ("py -3 -m venv .venv", ON_DARK),
        (".venv\\Scripts\\python.exe -m pip install \\", ON_DARK),
        ("    -r requirements.txt -r requirements-examples.txt", ON_DARK),
        (".venv\\Scripts\\python.exe scripts\\fetch_models.py", ON_DARK),
        ("", ON_DARK),
        ("# one demo", ON_DARK_MUTED),
        (".venv\\Scripts\\python.exe -m examples.perf_atlas", ON_DARK),
        ("# everything, sequentially", ON_DARK_MUTED),
        ("scripts\\record_all.ps1", ON_DARK),
    ]
    deck.text(s, MX + 0.3, CONTENT_Y + 0.35, 6.2, 4.4,
              [{"runs": [(t if t else " ", {"color": c})]} for t, c in qs], size=12, font=MONO, line_spacing=1.1)
    links = [
        ("This project", REPO_URL),
        ("OpenVINO NPU device", "docs.openvino.ai › … › npu-device"),
        ("GenAI on NPU", "docs.openvino.ai › … › inference-with-genai-on-npu"),
        ("Pre-converted models", "huggingface.co/OpenVINO  (look for *-int4-cw-ov)"),
        ("Intel NPU driver", "intel.com › download › intel-npu-driver-windows"),
        ("Core Ultra 5 322 spec", "Intel ARK, SKU 245716"),
    ]
    rx = MX + 6.7 + 0.5
    rw = SW - MX - rx
    deck.text(s, rx, CONTENT_Y + 0.05, rw, 4.9,
              [{"runs": [(a + "\n", {"font": HEAD, "color": INK, "size": 16}), (b, {"color": MUTED, "size": 14})],
                "space_after": 12} for a, b in links], size=15)
    deck.notes(s, f"""
Everything is open source (MIT): https://{REPO_URL}
Quick start is from README.md. scripts/record_all.ps1 re-runs every benchmark and re-records all media one at a time; this deck is rebuilt with .venv\\Scripts\\python.exe docs\\presentation\\build_tech_deck.py.
Full links:
- OpenVINO NPU device: https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html
- GenAI on NPU: https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html
- OpenVINO release notes (CVS-233822): https://docs.openvino.ai/releasenotes
- Hugging Face OpenVINO org: https://huggingface.co/OpenVINO
- Intel NPU driver: https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
- Intel graphics driver: https://www.intel.com/content/www/us/en/download/785597/intel-arc-graphics-windows.html
- Core Ultra 5 322 ARK: https://www.intel.com/content/www/us/en/products/sku/245716/intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html
Hardware facts and their sources: docs/PANTHER_LAKE_REFERENCE.md. Driver history: docs/DRIVER_SETUP.md.
""")

    # ------------------------------------------------------------------ 22. Q&A
    s = deck.slide(dark=True, footer=False)
    deck.text(s, MX, 2.0, 11, 1.2, "Questions?", size=60, font=HEAD)
    deck.text(s, MX, 3.35, 11.5, 1.4, [
        {"runs": [("The NPU wins on steady, static work and keeps the CPU free. ", {"color": ON_DARK}),
                  ("Verify the device, verify the output, cache the compile.", {"color": NPU})]},
    ], size=22)
    for i, dv in enumerate(("NPU", "GPU", "CPU")):
        deck.pill(s, MX + i * 1.0, 5.0, dv, DEV[dv])
    deck.text(s, MX, 6.35, 11, 0.4, f"{REPO_URL}  ·  {drv_line}", size=14, color=ON_DARK_MUTED)
    deck.notes(s, f"""
Recap in one sentence: on this Core Ultra 5 322, the NPU is the right home for steady, static-shape, always-on work and for keeping the CPU free; the 2-core iGPU is competitive and wins on prefill and small/odd shapes; and you must verify device, outputs and compile caching.
Likely questions:
- Power/battery? Not measured yet (AC only); planned.
- Why not ONNX Runtime / Windows ML / Ollama? OpenVINO first for direct NPU control; others are on the project plan (README goals).
- Is the bug reported? Not yet; next step is a minimal repro.
- Bigger SKUs? Expect a stronger GPU (up to 12 Xe3) and slightly higher NPU TOPS (47-50).
Sources: docs/PANTHER_LAKE_REFERENCE.md §1; README.md.
""")


# ----------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--media", choices=("mp4", "gif", "poster"), default="mp4",
                    help="demo media: embedded MP4 movies (default), animated GIFs, or still poster frames")
    ap.add_argument("--output", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args(argv)

    d = Data()
    posters = extract_posters({"vision_side_by_side": 0.45, "depth_parallax": 0.5,
                               "offload_dashboard": 0.75, "llm_chat": 0.85})
    deck = Deck(args.media, posters)
    build(d, deck)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deck.prs.save(str(args.output))
    for stem in ("vision_side_by_side", "depth_parallax", "offload_dashboard", "llm_chat"):
        mp4, js = MEDIA / "mp4" / f"{stem}.mp4", RES / f"{stem}.json"
        if mp4.exists() and js.exists() and mp4.stat().st_mtime - js.stat().st_mtime > 300:
            print(f"warning: media/mp4/{stem}.mp4 is newer than benchmarks/results/{stem}.json; "
                  "the video and the slide numbers may come from different runs")
    mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({deck.n} slides, {mb:.1f} MB, media={args.media})")
    if deck.missing:
        print("placeholders (re-run when these exist):")
        for m in deck.missing:
            print("  -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
