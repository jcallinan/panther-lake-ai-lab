"""Build the simplified advisory-board deck.

Output: docs/presentation/panther-lake-npu-advisory-board.pptx

Audience: the University of Pittsburgh at Bradford advisory board (mostly
non-technical). Plain language, one idea per slide, big type and visuals.

Re-run any time; media and numbers are picked up fresh on every run:
    .venv\\Scripts\\python.exe docs\\presentation\\build_board_deck.py

* Numbers are read from benchmarks/results/*.json and rounded for a lay
  audience ("about 4x faster"). Nothing is hard-coded except fallbacks.
* Demo media: media/<name>.gif is used if present (animates in PowerPoint);
  otherwise media/mp4/<name>.mp4 is embedded as a movie with a poster frame;
  otherwise a labelled placeholder is drawn.
* Charts: media/charts/simple_*.png if present; otherwise a native
  PowerPoint chart (speed) or a placeholder.

No university, Intel or Lenovo logos or official branding are used.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "presentation" / "panther-lake-npu-advisory-board.pptx"
MEDIA = ROOT / "media"
CHARTS = MEDIA / "charts"
RESULTS = ROOT / "benchmarks" / "results"
REPO_URL = "github.com/jcallinan/panther-lake-ai-lab"

# ---------------------------------------------------------------- palette
INK = RGBColor(0x1C, 0x20, 0x33)      # deep ink: dark slides + body text
MUTED = RGBColor(0x4A, 0x50, 0x68)    # secondary text on white
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xF3, 0xF4, 0xF7)     # soft card tint on white slides
AMBER = RGBColor(0xF2, 0xB3, 0x3D)    # warm accent (dark slides)
SOFT = RGBColor(0xC9, 0xCD, 0xDB)     # light text on dark slides
NPU = RGBColor(0x00, 0xAA, 0xF0)
GPU = RGBColor(0x5A, 0xC8, 0x3C)
CPU = RGBColor(0xF5, 0x78, 0x28)
FONT = "Calibri"
EMOJI_FONT = "Segoe UI Emoji"

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.6)


# ---------------------------------------------------------------- numbers
def _load(name: str) -> dict | None:
    p = RESULTS / f"{name}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def times(a: float | None, b: float | None) -> str:
    """'about 4x' style ratio a/b for a lay audience."""
    if not a or not b:
        return "much"
    r = a / b
    if r < 1.15:
        return "about as"
    if r < 1.75:
        return "a bit"
    return f"about {round(r)}×"


def facts() -> dict:
    f: dict = {}
    v = _load("vision_side_by_side")
    if v:
        lat = {d: v["devices"][d]["latency_ms"]["frame"]["median_ms"]
               for d in ("NPU", "GPU", "CPU") if v["devices"].get(d, {}).get("latency_ms")}
        f["vision_ms"] = lat
    d = _load("depth_parallax")
    if d:
        f["depth_ms"] = {k: x["benchmark"]["median_ms"]
                         for k, x in d["devices"].items() if x.get("benchmark")}
    l = _load("llm_chat")
    if l:
        f["llm"] = {x["device"]: x["summary"] for x in l["devices"] if x.get("summary")}
    o = _load("offload_dashboard")
    if o:
        f["offload"] = {p["device"]: p["result"] for p in o.get("phases", [])
                        if p.get("device") and p.get("result")}
    return f


F = facts()


def vision_line() -> str:
    ms = F.get("vision_ms", {})
    return f"{times(ms.get('CPU'), ms.get('NPU')).capitalize()} faster than the main processor"


def depth_line() -> str:
    ms = F.get("depth_ms", {})
    return f"{times(ms.get('CPU'), ms.get('NPU')).capitalize()} faster than the main processor"


# ---------------------------------------------------------------- helpers
def set_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def text(slide, x, y, w, h, runs, *, size=24, color=INK, bold=False,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font=FONT, name=None,
         margin=0, space_after=0):
    """Add a text box. `runs` is a str or a list of paragraphs; a paragraph is
    a str or a list of (text, {overrides}) tuples."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0)
    paras = runs if isinstance(runs, list) else [runs]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if space_after:
            p.space_after = Pt(space_after)
        parts = para if isinstance(para, list) else [(para, {})]
        for t, o in parts:
            r = p.add_run()
            r.text = t
            r.font.name = o.get("font", font)
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.italic = o.get("italic", False)
            r.font.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill, *, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
        radius=0.08, line=None, name=None):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if name:
        s.name = name
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(3)
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if s.has_text_frame:
        s.text_frame.text = ""
    return s


def title(slide, txt, *, color=INK, y=Inches(0.45), size=40, align=PP_ALIGN.LEFT):
    """Fill the layout's title placeholder (keeps slide titles accessible)."""
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = MARGIN, y, SLIDE_W - 2 * MARGIN, Inches(1.0)
    tf = t.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = txt
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = color
    return t


def fit(img_w, img_h, x, y, w, h):
    """Largest box with the image's aspect inside (x, y, w, h), centred."""
    s = min(w / img_w, h / img_h)
    nw, nh = int(img_w * s), int(img_h * s)
    return x + (w - nw) // 2, y + (h - nh) // 2, nw, nh


def placeholder(slide, x, y, w, h, label):
    box(slide, x, y, w, h, CARD, name="Media placeholder")
    text(slide, x, y, w, h, label.split("\n"), size=24, color=MUTED, align=PP_ALIGN.CENTER,
         anchor=MSO_ANCHOR.MIDDLE, margin=0.3)


_tmp = Path(tempfile.mkdtemp(prefix="board_deck_"))


def poster_for(mp4: Path) -> Path | None:
    try:
        import cv2
        cap = cv2.VideoCapture(str(mp4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 3))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        out = _tmp / f"{mp4.stem}_poster.png"
        cv2.imwrite(str(out), frame)
        return out
    except Exception:
        return None


def media(slide, stem, x, y, w, h, alt):
    """GIF (animates) > MP4 movie with poster > placeholder."""
    from PIL import Image
    gif = MEDIA / f"{stem}.gif"
    mp4 = MEDIA / "mp4" / f"{stem}.mp4"
    if gif.exists():
        with Image.open(gif) as im:
            iw, ih = im.size
        px, py, pw, ph = fit(iw, ih, x, y, w, h)
        pic = slide.shapes.add_picture(str(gif), px, py, pw, ph)
        pic.name = alt
        _alt(pic, alt)
        return pic
    if mp4.exists():
        poster = poster_for(mp4)
        iw, ih = (16, 9)
        if poster:
            with Image.open(poster) as im:
                iw, ih = im.size
        px, py, pw, ph = fit(iw, ih, x, y, w, h)
        mv = slide.shapes.add_movie(str(mp4), px, py, pw, ph,
                                    poster_frame_image=str(poster) if poster else None,
                                    mime_type="video/mp4")
        mv.name = alt
        return mv
    placeholder(slide, x, y, w, h, f"Demo video goes here\n(media/{stem}.gif)")
    return None


# The simple_*.png charts are full 16:9 "chart slides" with their own title,
# subtitle and a long footnote. The deck supplies its own title and takeaway,
# so crop to the bars. Adjust if the chart layout in make_charts changes.
CHART_CROP_TOP, CHART_CROP_BOTTOM = 0.22, 0.15


def chart_image(slide, name, x, y, w, h, alt):
    from PIL import Image
    p = CHARTS / f"{name}.png"
    if not p.exists():
        return None
    with Image.open(p) as im:
        iw, ih = im.size
    ih_c = ih * (1 - CHART_CROP_TOP - CHART_CROP_BOTTOM)
    px, py, pw, ph = fit(iw, ih_c, x, y, w, h)
    pic = slide.shapes.add_picture(str(p), px, py, pw, ph)
    pic.crop_top, pic.crop_bottom = CHART_CROP_TOP, CHART_CROP_BOTTOM
    pic.name = alt
    _alt(pic, alt)
    return pic


def _alt(shape, alt):
    shape._element.nvPicPr.cNvPr.set("descr", alt)


def notes(slide, txt):
    slide.notes_slide.notes_text_frame.text = txt


def chip(slide, x, y, size, color, label, name):
    """A simple 'chip' tile: rounded square with little pins on each side."""
    pin_len, pin_w, n = Inches(0.16), Inches(0.07), 4
    gap = (size - Inches(0.5)) // (n - 1)
    for i in range(n):
        off = Inches(0.25) + i * gap - pin_w // 2
        for (px, py, pw, ph) in (
            (x + off, y - pin_len, pin_w, pin_len),            # top
            (x + off, y + size, pin_w, pin_len),               # bottom
            (x - pin_len, y + off, pin_len, pin_w),            # left
            (x + size, y + off, pin_len, pin_w),               # right
        ):
            box(slide, px, py, pw, ph, MUTED, shape=MSO_SHAPE.RECTANGLE)
    box(slide, x, y, size, size, color, radius=0.12, name=f"{name} chip")
    text(slide, x, y, size, size, label, size=40, bold=True, color=INK,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name=f"{name} label")


# ---------------------------------------------------------------- slides
def build() -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H
    layout = prs.slide_layouts[5]  # "Title Only"

    def new(bg=WHITE):
        s = prs.slides.add_slide(layout)
        set_bg(s, bg)
        return s

    content_top = Inches(1.7)
    W = SLIDE_W - 2 * MARGIN

    # 1. Title ---------------------------------------------------------------
    s = new(INK)
    title(s, "AI on Everyday Laptops", color=WHITE, y=Inches(2.0), size=54)
    text(s, MARGIN, Inches(3.1), W, Inches(0.8),
         "A student-ready lab project", size=36, color=AMBER)
    # three small device dots as the deck's motif
    for i, c in enumerate((CPU, GPU, NPU)):
        box(s, MARGIN + Inches(0.55) * i, Inches(4.35), Inches(0.4), Inches(0.4), c,
            shape=MSO_SHAPE.OVAL, name=f"Motif dot {i + 1}")
    text(s, MARGIN, Inches(5.4), W, Inches(0.6),
         "Prepared for the Pitt-Bradford Advisory Board", size=26, color=SOFT)
    text(s, MARGIN, Inches(6.0), W, Inches(0.6),
         "Presenter name  ·  Date", size=22, color=SOFT, name="Presenter line")
    notes(s, "Good morning, everyone. Thank you for having me. In the next ten minutes I want to "
             "show you a small lab project that puts artificial intelligence on ordinary, everyday "
             "laptops, and why I think it is a good fit for our students.\n\n"
             "[Presenter: replace 'Presenter name · Date' on the slide with your name and the meeting date.]")

    # 2. What is an NPU --------------------------------------------------------
    s = new()
    title(s, "New laptops have a chip built just for AI")
    size = Inches(2.0)
    col_w = W // 3
    items = [
        (CPU, "CPU", "The generalist", "Runs everything"),
        (GPU, "GPU", "The graphics pro", "Pictures, video, games"),
        (NPU, "NPU", "The AI specialist", "Does AI work efficiently"),
    ]
    for i, (c, lab, head, sub) in enumerate(items):
        cx = MARGIN + col_w * i
        chip(s, cx + (col_w - size) // 2, Inches(2.05), size, c, lab, lab)
        text(s, cx, Inches(4.55), col_w, Inches(0.6), head, size=30, bold=True,
             align=PP_ALIGN.CENTER)
        text(s, cx, Inches(5.2), col_w, Inches(0.6), sub, size=24, color=MUTED,
             align=PP_ALIGN.CENTER)
    text(s, MARGIN, Inches(6.3), W, Inches(0.6),
         [[("NPU", {"bold": True}), (" = neural processing unit: an AI engine inside the laptop", {})]],
         size=24, color=MUTED, align=PP_ALIGN.CENTER)
    notes(s, "Every laptop has a main processor, the CPU. Think of it as the generalist: it can do "
             "anything, but it is busy running everything. Most laptops also have a graphics chip, the "
             "GPU, which is great at pictures and video. What is new is the third chip, the NPU, or "
             "neural processing unit. It is a specialist built only for AI, designed to do that one "
             "job efficiently. Our test laptop is an ordinary entry-level model with all three.\n\n"
             "If asked 'Is this exotic hardware?': No. It is a standard consumer laptop from 2026; "
             "these AI chips are now common in new laptops.")

    # 3. Why it matters -------------------------------------------------------
    s = new()
    title(s, "Why it matters")
    cards = [
        ("\U0001F512", "Private", "AI runs on the laptop. Nothing goes to the cloud."),
        ("\U0001F50B", "Efficient", "A specialist chip built to do AI with less power."),
        ("⚡", "Responsive", "The main processor stays free for your work."),
    ]
    gap = Inches(0.4)
    cw = (W - 2 * gap) // 3
    for i, (icon, head, body) in enumerate(cards):
        cx = MARGIN + (cw + gap) * i
        box(s, cx, content_top, cw, Inches(4.9), CARD, name=f"{head} card")
        circ = box(s, cx + (cw - Inches(1.3)) // 2, content_top + Inches(0.45),
                   Inches(1.3), Inches(1.3), WHITE, shape=MSO_SHAPE.OVAL, name=f"{head} icon circle")
        text(s, circ.left, circ.top, circ.width, circ.height, icon, size=40, font=EMOJI_FONT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, name=f"{head} icon")
        text(s, cx, content_top + Inches(2.05), cw, Inches(0.7), head, size=32, bold=True,
             align=PP_ALIGN.CENTER)
        text(s, cx + Inches(0.35), content_top + Inches(2.85), cw - Inches(0.7), Inches(1.8),
             body, size=24, color=MUTED, align=PP_ALIGN.CENTER)
    notes(s, "Why should anyone care? Three reasons. First, privacy: when AI runs on the laptop "
             "itself, your photos, documents and conversations never leave the machine. Second, "
             "efficiency: the AI chip is designed to do this work with less power than the main "
             "processor. Third, responsiveness: when AI runs on its own chip, the rest of the "
             "computer stays quick for whatever you are actually doing.\n\n"
             "If asked 'Why not just use the cloud?': Cloud AI is powerful, but it costs money per use, "
             "needs an internet connection, and means sending your data to someone else; local AI "
             "avoids all three.\n\n"
             "Honesty note: we have not yet measured battery life directly; 'efficient' describes "
             "what the chip is designed for.")

    # 4-6. Demos ---------------------------------------------------------------
    def demo(head, stem, alt, takeaway, sub, note_txt):
        s = new()
        title(s, head)
        media(s, stem, MARGIN, content_top - Inches(0.1), W, Inches(4.45), alt)
        text(s, MARGIN, Inches(6.15), W, Inches(0.55), takeaway, size=30, bold=True,
             align=PP_ALIGN.CENTER)
        if sub:
            text(s, MARGIN, Inches(6.7), W, Inches(0.5), sub, size=22, color=MUTED,
                 align=PP_ALIGN.CENTER)
        notes(s, note_txt)
        return s

    vms = F.get("vision_ms", {})
    demo("Seeing: spotting people and objects", "vision_side_by_side",
         "The same street video processed on three chips side by side, with people and objects outlined",
         "On the AI chip: " + vision_line().lower(),
         "Same video, same AI, three chips side by side",
         "This is a live street scene. The laptop finds the people, cars and other objects in every "
         "frame, and separates people from the background, the same trick video-call apps use to blur "
         "your room. We run the exact same AI on all three chips side by side. On the AI chip, each "
         f"frame takes about {vms.get('NPU', 0):.0f} thousandths of a second, versus about "
         f"{vms.get('CPU', 0):.0f} on the main processor, so {times(vms.get('CPU'), vms.get('NPU'))} faster. "
         "It is also a bit faster than the graphics chip.\n\n"
         "If asked 'Does it need the internet?': No. Everything you see runs on the laptop itself.")

    dms = F.get("depth_ms", {})
    demo("Understanding depth: 3D from flat video", "depth_parallax",
         "An ordinary forest video turned into a slowly rotating 3D scene using AI depth estimation",
         "On the AI chip: " + depth_line().lower(),
         "The AI guesses how far away everything is, frame by frame",
         "Here the AI looks at an ordinary flat video and estimates how far away every point is: the "
         "ferns are close, the canyon is far. With that, the laptop rebuilds the scene in 3D and we can "
         "gently swing the camera around. Estimating depth is the heavy part. The AI chip does it in "
         f"about {dms.get('NPU', 0):.0f} thousandths of a second per frame, versus about "
         f"{dms.get('CPU', 0):.0f} on the main processor, roughly {times(dms.get('CPU'), dms.get('NPU')).replace('about ', '')} "
         "faster, which leaves the main processor free to draw the 3D view.\n\n"
         "If asked 'What is this useful for?': The same idea shows up in robotics, accessibility tools "
         "and augmented reality.")

    llm = F.get("llm", {})
    ln, lg, lc = (llm.get(k, {}) for k in ("NPU", "GPU", "CPU"))
    demo("Talking: a private chatbot on the laptop", "llm_chat",
         "A chatbot answering the same question on three chips, replayed at measured speed",
         "No internet, no cloud: the conversation never leaves the laptop",
         "The graphics chip is just as fast; the AI chip leaves the main processor almost idle",
         "This is a small chatbot, similar in spirit to ChatGPT but much smaller, running entirely on "
         "the laptop. We asked the same questions on all three chips and replay them at the speed they "
         "actually ran. To be honest about it: the graphics chip is about as fast as the AI chip here "
         f"(roughly {round(lg.get('tok_s_median', 0) * 0.75)} versus {round(ln.get('tok_s_median', 0) * 0.75)} "
         "words a second, faster than most people read), and it starts answering a little sooner. The main "
         "processor alone is noticeably slower. The AI chip's advantage is that it keeps the main processor "
         "almost completely free while it talks, so the rest of the laptop stays responsive.\n\n"
         "If asked about privacy: Nothing typed into this chatbot leaves the machine; once the model is "
         "downloaded, it runs entirely on the laptop with no internet connection needed.")

    # 7. Chart ---------------------------------------------------------------
    s = new()
    title(s, "Same task, three chips")
    cx, cy, cw_, ch_ = MARGIN, content_top - Inches(0.1), W, Inches(4.75)
    pic = chart_image(s, "simple_speed", cx, cy, cw_, ch_,
                      "Bar chart comparing how fast the same AI task runs on the AI chip, graphics chip and main processor")
    if pic is None and vms:
        data = CategoryChartData()
        order = [k for k in ("NPU", "GPU", "CPU") if k in vms]
        names = {"NPU": "AI chip (NPU)", "GPU": "Graphics chip (GPU)", "CPU": "Main processor (CPU)"}
        data.categories = [names[k] for k in order]
        data.add_series("Video frames per second", [round(1000 / vms[k]) for k in order])
        gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, cx + Inches(0.5), cy,
                                cw_ - Inches(1.0), ch_, data)
        gf.name = "Speed chart"
        ch = gf.chart
        ch.has_legend = False
        ch.font.size = Pt(24)
        ch.font.name = FONT
        ch.font.color.rgb = INK
        plot = ch.plots[0]
        plot.gap_width = 45
        plot.has_data_labels = True
        plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
        plot.data_labels.font.size = Pt(24)
        plot.data_labels.font.bold = True
        plot.data_labels.number_format = '0" frames/s"'
        plot.data_labels.number_format_is_linked = False
        cols = {"NPU": NPU, "GPU": GPU, "CPU": CPU}
        for i, k in enumerate(order):
            pt = plot.series[0].points[i]
            pt.format.fill.solid()
            pt.format.fill.fore_color.rgb = cols[k]
        ch.category_axis.reverse_order = True
        ch.category_axis.format.line.fill.background()
        ch.value_axis.maximum_scale = max(1000 / vms[k] for k in order) * 1.3
        ch.value_axis.minimum_scale = 0
        ch.value_axis.visible = False
        ch.value_axis.has_major_gridlines = False
    elif pic is None:
        placeholder(s, cx, cy, cw_, ch_, "Chart goes here\n(media/charts/simple_speed.png)")
    text(s, MARGIN, Inches(6.55), W, Inches(0.55),
         f"Video AI on the AI chip: {times(vms.get('CPU'), vms.get('NPU'))} faster than the main processor",
         size=26, bold=True, align=PP_ALIGN.CENTER)
    notes(s, "Here is the same comparison as a simple chart. Same AI, same video, same laptop, three "
             "different chips. Longer bars are better. The AI chip comes out in front, the graphics chip "
             "is a solid second, and the main processor is well behind. These are our own measurements on "
             "one laptop; the exact numbers depend on the software version, and every result and the code "
             "that produced it is published so anyone can check it.\n\n"
             "If asked 'Is the AI chip always fastest?': No. For some jobs, like the chatbot, the graphics "
             "chip keeps pace; mapping where each chip wins is part of the research.")

    # 8. Students & teaching -------------------------------------------------
    s = new()
    title(s, "What this means for our students")
    rows = [
        (NPU, "\U0001F4BB", "Hands-on AI skills", "on affordable, everyday laptops"),
        (GPU, "\U0001F501", "A lab anyone can repeat", "step-by-step, free and open to all"),
        (CPU, "\U0001F4DD", "A path to research", "a paper draft is underway"),
    ]
    rh, rgap = Inches(1.35), Inches(0.3)
    for i, (c, icon, head, sub) in enumerate(rows):
        ry = content_top + (rh + rgap) * i
        box(s, MARGIN, ry, W, rh, CARD, name=f"{head} row")
        circ = box(s, MARGIN + Inches(0.3), ry + (rh - Inches(1.0)) // 2, Inches(1.0), Inches(1.0), c,
                   shape=MSO_SHAPE.OVAL, name=f"{head} icon circle")
        text(s, circ.left, circ.top, circ.width, circ.height, icon, size=32, font=EMOJI_FONT,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, MARGIN + Inches(1.65), ry, W - Inches(1.9), rh,
             [[(head, {"bold": True, "size": 32}), ("  —  " + sub, {"color": MUTED})]],
             size=26, anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "So what does this mean for us? First, students can learn practical AI skills on the kind of "
             "laptop they might actually own, not on expensive servers. Second, the whole lab is written "
             "down step by step and shared openly, so a class can repeat every experiment and check the "
             "results for themselves. Third, it is a genuine research project: there is a draft paper and a "
             "plan for submitting to a conference. Nothing has been accepted yet, but it is a real "
             "opportunity for students to be part of published work.\n\n"
             "If asked 'What does it cost?': The software is free and open source; the hardware is an "
             "ordinary entry-level laptop, so the main cost is time.")

    # 9. Next steps --------------------------------------------------------------
    s = new()
    title(s, "Next steps: how you could help")
    steps = [
        ("1", "Student time", "independent studies or research assistants"),
        ("2", "More laptops", "to compare different models and brands"),
        ("3", "Connections", "partners and places to share what students build"),
    ]
    cw = (W - 2 * gap) // 3
    for i, (num, head, body) in enumerate(steps):
        cx = MARGIN + (cw + gap) * i
        box(s, cx, content_top, cw, Inches(4.3), CARD, name=f"Step {num} card")
        circ = box(s, cx + Inches(0.4), content_top + Inches(0.45), Inches(1.1), Inches(1.1), AMBER,
                   shape=MSO_SHAPE.OVAL, name=f"Step {num} circle")
        text(s, circ.left, circ.top, circ.width, circ.height, num, size=40, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
        text(s, cx + Inches(0.4), content_top + Inches(1.9), cw - Inches(0.8), Inches(0.7), head,
             size=30, bold=True)
        text(s, cx + Inches(0.4), content_top + Inches(2.7), cw - Inches(0.8), Inches(1.9), body,
             size=24, color=MUTED)
    notes(s, "[Presenter: these are suggestions; edit them to match what you actually want to ask for.]\n\n"
             "Where could this go next? Three things would help most. First, student time: an independent "
             "study or a research assistant could run and extend these experiments. Second, a few more "
             "laptops, so we can compare different models and see whether the results hold up. Third, "
             "connections: partners in the community or industry, and places where students can show and "
             "share what they build.\n\n"
             "If asked 'How much would this need?': A modest start is possible with existing equipment; "
             "[Presenter: add a specific figure or request if you have one].")

    # 10. Thank you --------------------------------------------------------------
    s = new(INK)
    title(s, "Thank you", color=WHITE, y=Inches(2.0), size=54)
    text(s, MARGIN, Inches(3.1), W, Inches(0.8), "Questions?", size=40, color=AMBER)
    for i, c in enumerate((CPU, GPU, NPU)):
        box(s, MARGIN + Inches(0.55) * i, Inches(4.3), Inches(0.4), Inches(0.4), c,
            shape=MSO_SHAPE.OVAL, name=f"Motif dot {i + 1}")
    text(s, MARGIN, Inches(5.4), W, Inches(0.6), "Everything is open and free to explore:", size=24, color=SOFT)
    text(s, MARGIN, Inches(5.95), W, Inches(0.6), REPO_URL, size=28, bold=True, color=WHITE)
    notes(s, "Thank you. I am happy to take any questions.\n\n"
             "Quick answers to likely questions:\n"
             "- Cost? The software is free and open source; it runs on an ordinary entry-level laptop.\n"
             "- Privacy? Everything runs on the laptop; no data is sent to the cloud.\n"
             "- Why not the cloud? Local AI works offline, has no per-use fees, and keeps data on the device.\n"
             "- Is the AI chip always best? No; the graphics chip keeps pace on some jobs. Finding where each "
             "chip wins is part of the research.\n"
             "- Battery life? Not measured yet; it is on the to-do list.\n\n"
             "Two backup slides follow if someone asks for more detail.")

    # Backup A: responsiveness ----------------------------------------------
    off = F.get("offload", {})
    oc, on_ = off.get("CPU", {}), off.get("NPU", {})
    a, b = oc.get("ai_proc_cpu_ms_per_frame"), on_.get("ai_proc_cpu_ms_per_frame")
    threads = 6  # logical CPUs on the Core Ultra 5 322 (6C/6T)
    pct = {k: round(100 * llm.get(k, {}).get("cpu_cores_busy_median", 0) / threads) for k in ("NPU", "GPU", "CPU")}
    s = new()
    title(s, "Backup: the AI chip keeps the laptop free")
    pic = chart_image(s, "simple_cpu_free", MARGIN, content_top - Inches(0.1), W, Inches(4.75),
                      "Bar chart: while the chatbot answers, the AI chip keeps about 2 percent of the main processor "
                      "busy, the graphics chip about 16 percent and the main processor alone about 29 percent")
    if pic is not None:
        caption = "While the chatbot answers, the main processor stays almost idle on the AI chip"
    else:
        media(s, "offload_dashboard", MARGIN, content_top - Inches(0.1), W, Inches(4.75),
              "Dashboard showing always-on video AI moved from the main processor to the graphics chip and then the AI chip")
        caption = (f"Always-on video AI uses {times(a, b)} less of the main processor on the AI chip"
                   if a and b else "Always-on AI on the AI chip leaves the main processor free")
    text(s, MARGIN, Inches(6.55), W, Inches(0.55), caption, size=26, bold=True, align=PP_ALIGN.CENTER)
    notes(s, "Backup slide, use only if asked. While the chatbot writes its answer, how much of the main "
             f"processor is busy? About {pct['NPU']} percent when the AI chip does the work, about {pct['GPU']} "
             f"percent with the graphics chip, and about {pct['CPU']} percent when the main processor does it "
             "alone. That spare capacity is what keeps the laptop responsive."
             + ("\n\nA second experiment points the same way: running always-on video AI (background blur plus "
                "object detection) on the AI chip instead of the main processor cut the main-processor effort it "
                f"needed by roughly {times(a, b).replace('about ', '')}. That measurement is being re-recorded, so "
                "treat it as preliminary." if a and b else ""))

    # Backup B: chatbot ------------------------------------------------------
    def words(d):  # tokens/s -> approx. words/s (1 token ~ 0.75 words)
        return round(d.get("tok_s_median", 0) * 0.75)
    s = new()
    title(s, "Backup: the chatbot on three chips")
    pic = chart_image(s, "simple_llm", MARGIN, content_top - Inches(0.1), W, Inches(4.75),
                      f"Bar chart of chatbot writing speed: about {words(ln)} words per second on the AI chip, "
                      f"{words(lg)} on the graphics chip and {words(lc)} on the main processor")
    if pic is None:
        stats = [
            (NPU, "AI chip", ln), (GPU, "Graphics chip", lg), (CPU, "Main processor", lc)]
        cw = (W - 2 * gap) // 3
        for i, (col, lab, d) in enumerate(stats):
            cx = MARGIN + (cw + gap) * i
            box(s, cx, content_top, cw, Inches(4.5), CARD, name=f"{lab} stat card")
            box(s, cx + (cw - Inches(0.6)) // 2, content_top + Inches(0.35), Inches(0.6), Inches(0.6), col,
                shape=MSO_SHAPE.OVAL)
            text(s, cx, content_top + Inches(1.1), cw, Inches(0.6), lab, size=28, bold=True,
                 align=PP_ALIGN.CENTER)
            text(s, cx, content_top + Inches(1.8), cw, Inches(1.0), f"≈ {words(d)}",
                 size=60, bold=True, align=PP_ALIGN.CENTER)
            text(s, cx, content_top + Inches(2.8), cw, Inches(0.5), "words per second", size=22,
                 color=MUTED, align=PP_ALIGN.CENTER)
    text(s, MARGIN, Inches(6.55), W, Inches(0.55),
         "AI chip and graphics chip: about the same speed, faster than you can read",
         size=26, bold=True, align=PP_ALIGN.CENTER)
    notes(s, "Backup slide, use only if asked. The chatbot writes at about the same speed on the AI chip and the "
             f"graphics chip, roughly {words(ln)} and {words(lg)} words a second, which is faster than most people "
             f"read. The main processor alone manages about {words(lc)}. The graphics chip also starts replying a "
             "little sooner. The AI chip's real advantage is the previous backup slide: it leaves the main "
             "processor almost idle. (Words per second is an approximation: the model counts word-pieces, and "
             "we convert at about three quarters of a word each.)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    return OUT


if __name__ == "__main__":
    out = build()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("facts used:", json.dumps(F.get("vision_ms")), json.dumps(F.get("depth_ms")))
