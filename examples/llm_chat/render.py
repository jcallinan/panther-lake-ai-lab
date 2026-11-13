"""Race-replay video and static chart from benchmarks/results/llm_chat.json.

The devices were benchmarked one after another (never concurrently). The video replays each run side by side,
revealing every token at the timestamp it was actually streamed, measured from that run's generate() call.
"""
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..common import MEDIA_DIR
from ..common.hud import DEVICE_HEX
from ..common.media import GIF_MAX_BYTES, Recorder

W, H = 1280, 720
FPS = 20
BG = (11, 15, 20)
PANEL = (16, 21, 28)
BORDER = (44, 52, 62)
INK = (230, 237, 243)
MUTED = (139, 148, 158)
DIM = (88, 96, 106)
ORDER = ["NPU", "GPU", "CPU"]
DEVICE_LABEL = {"NPU": "Intel AI Boost NPU", "GPU": "Xe3 integrated GPU", "CPU": "Core Ultra 5 322 CPU"}

FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu"), Path("/Library/Fonts")]


def _font(names, size):
    for d in FONT_DIRS:
        for n in names:
            p = d / n
            if p.exists():
                return ImageFont.truetype(str(p), size)
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size)


MONO = ["CascadiaMono.ttf", "consola.ttf", "DejaVuSansMono.ttf"]
MONO_B = ["CascadiaMono.ttf", "consolab.ttf", "DejaVuSansMono-Bold.ttf"]
SANS = ["segoeui.ttf", "DejaVuSans.ttf"]
SANS_B = ["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"]


def hex_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


class Fonts:
    def __init__(self):
        self.title = _font(SANS_B, 27)
        self.sub = _font(SANS, 15)
        self.small = _font(SANS, 13)
        self.label = _font(SANS_B, 12)
        self.head = _font(SANS_B, 21)
        self.head_small = _font(SANS, 13)
        self.mono = _font(MONO, 15)
        self.mono_val = _font(MONO_B, 22)
        self.mono_clock = _font(MONO_B, 20)
        self.mono_prompt = _font(MONO, 15)
        self.card_head = _font(SANS_B, 22)
        self.card = _font(SANS, 16)
        self.card_b = _font(SANS_B, 16)
        self.card_val = _font(MONO_B, 18)


def wrap(text: str, cols: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, cols, drop_whitespace=True, break_long_words=True,
                                   replace_whitespace=False) or [""])
    # collapse runs of blank lines (Phi-3 often emits "\n\n")
    out = []
    for ln in lines:
        if ln == "" and out and out[-1] == "":
            continue
        out.append(ln)
    return out


def load_runs(doc: dict, prompt_id: str) -> list[dict]:
    runs = []
    devs = {d["device"]: d for d in doc["devices"] if d.get("status") == "executed"}
    for name in ORDER + [d for d in devs if d not in ORDER]:
        d = devs.get(name)
        if not d:
            continue
        p = next((p for p in d["prompts"] if p["id"] == prompt_id), None)
        if not p:
            continue
        runs.append({"device": name, "color": hex_rgb(DEVICE_HEX.get(name, "#C8C8C8")), "rec": d, "p": p,
                     "times": np.asarray(p["token_times_ms"]) / 1000.0, "texts": p["token_text"],
                     "summary": d.get("summary", {}), "load": d.get("load", {})})
    return runs


class Race:
    def __init__(self, doc: dict, prompt_id: str):
        self.doc = doc
        self.f = Fonts()
        self.runs = load_runs(doc, prompt_id)
        if not self.runs:
            raise SystemExit(f"no executed runs with prompt '{prompt_id}' in the bench JSON")
        self.prompt = next(p for p in doc["prompts"] if p["id"] == prompt_id)
        self.end_t = max(r["times"][-1] for r in self.runs)
        finish = sorted(self.runs, key=lambda r: r["times"][-1])
        for i, r in enumerate(finish):
            r["place"] = i + 1
        n = len(self.runs)
        self.mx, self.gap, self.top, self.bottom = 32, 16, 150, 662
        self.pw = (W - 2 * self.mx - (n - 1) * self.gap) // n
        self.cw = self.f.mono.getlength("M")
        self.cols = int((self.pw - 32) // self.cw)
        self.line_h = 20

    # ------------------------------------------------------------------------------------------ drawing
    def header(self, d: ImageDraw.ImageDraw, t: float):
        f, m = self.f, self.doc["meta"]
        d.text((self.mx, 16), "Local LLM chat on the NPU: Phi-3-mini int4 on Panther Lake", font=f.title, fill=INK)
        ver = m["versions"]["openvino_genai"].split("-")[0]
        d.text((self.mx, 55), f"Replay of sequential runs at measured per-token timing  ·  greedy decoding  ·  "
                              f"OpenVINO GenAI {ver}  ·  same prompt, same model", font=f.sub, fill=MUTED)
        clock = f"t = {max(t, 0):5.2f} s"
        tw = f.mono_clock.getlength(clock)
        d.rounded_rectangle((W - self.mx - tw - 24, 18, W - self.mx, 50), 8, fill=PANEL, outline=BORDER)
        d.text((W - self.mx - tw - 12, 22), clock, font=f.mono_clock, fill=INK)
        # prompt bar
        d.rounded_rectangle((self.mx, 88, W - self.mx, 136), 10, fill=(22, 27, 34), outline=BORDER)
        d.text((self.mx + 14, 92), "PROMPT", font=f.label, fill=MUTED)
        ptxt = self.prompt["text"]
        cols = int((W - 2 * self.mx - 28) // f.mono_prompt.getlength("M"))
        if len(ptxt) > cols:
            ptxt = ptxt[: cols - 1] + "…"
        d.text((self.mx + 14, 110), ptxt, font=f.mono_prompt, fill=INK)

    def panel(self, img: Image.Image, d: ImageDraw.ImageDraw, i: int, r: dict, t: float):
        f = self.f
        x0 = self.mx + i * (self.pw + self.gap)
        x1, y0, y1 = x0 + self.pw, self.top, self.bottom
        times, n = r["times"], len(r["times"])
        k = int(np.searchsorted(times, t, side="right"))  # tokens visible
        done = k >= n
        col = r["color"]
        d.rounded_rectangle((x0, y0, x1, y1), 12, fill=PANEL, outline=col if done else BORDER, width=2 if done else 1)
        # coloured header
        d.rounded_rectangle((x0, y0, x1, y0 + 46), 12, fill=col)
        d.rectangle((x0, y0 + 30, x1, y0 + 46), fill=col)
        dark = (8, 12, 16)
        d.text((x0 + 16, y0 + 9), r["device"], font=f.head, fill=dark)
        sub = DEVICE_LABEL.get(r["device"], "")
        d.text((x0 + 16 + f.head.getlength(r["device"]) + 10, y0 + 16), sub, font=f.head_small, fill=dark)
        if done:
            badge = {1: "1st", 2: "2nd", 3: "3rd"}.get(r["place"], f"{r['place']}th")
            bw = f.label.getlength(badge) + 18
            d.rounded_rectangle((x1 - bw - 12, y0 + 12, x1 - 12, y0 + 34), 11, fill=dark)
            d.text((x1 - bw - 3, y0 + 15), badge, font=f.label, fill=col)
        # metric cells
        my = y0 + 56
        ttft = r["p"]["ttft_ms"] if k > 0 else None
        if k >= 3:
            rate = (k - 1) / max(times[k - 1] - times[0], 1e-6)
        else:
            rate = None
        cells = [("TTFT", f"{ttft:.0f} ms" if ttft is not None else (f"{t * 1000:.0f}…" if t > 0 else "–")),
                 ("DECODE tok/s", f"{rate:.1f}" if rate else "–"),
                 ("TOKENS", f"{k}")]
        cw = (self.pw - 32) / 3
        for j, (lab, val) in enumerate(cells):
            cx = x0 + 16 + j * cw
            d.text((cx, my), lab, font=f.label, fill=MUTED)
            d.text((cx, my + 16), val, font=f.mono_val, fill=INK if (lab != "TTFT" or ttft is not None) else DIM)
        d.line((x0 + 16, my + 52, x1 - 16, my + 52), fill=BORDER, width=1)
        # body
        by0, by1 = my + 64, y1 - 40
        max_lines = (by1 - by0) // self.line_h
        text = "".join(r["texts"][:k]).lstrip()
        lines = wrap(text, self.cols) if text else []
        if k == 0:
            dots = "." * (1 + int(t * 4) % 3) if t > 0 else ""
            d.text((x0 + 16, by0), f"prefill{dots}", font=f.mono, fill=DIM)
        else:
            shown = lines[-max_lines:]
            for li, ln in enumerate(shown):
                d.text((x0 + 16, by0 + li * self.line_h), ln, font=f.mono, fill=INK)
            if not done and int(t * 2.5) % 2 == 0:
                last = shown[-1] if shown else ""
                cx = x0 + 16 + f.mono.getlength(last) + 2
                cy = by0 + (len(shown) - 1) * self.line_h
                d.rectangle((cx, cy + 2, cx + 8, cy + 17), fill=col)
        # footer: progress strip + finish time
        fy = y1 - 26
        d.rounded_rectangle((x0 + 16, fy, x1 - 16, fy + 6), 3, fill=(32, 38, 46))
        frac = k / n
        if frac > 0:
            d.rounded_rectangle((x0 + 16, fy, x0 + 16 + (self.pw - 32) * frac, fy + 6), 3, fill=col)
        if done:
            msg = f"done  {n} tokens in {times[-1]:.2f} s"
            d.text((x0 + 16, fy - 20), msg, font=f.small, fill=col)

    def footer(self, d: ImageDraw.ImageDraw):
        m = self.doc["meta"]["model"]
        txt = (f"{m.get('base_model', m['name'])}  ·  {m.get('weight_mode', 'int4')} channel-wise weights  ·  "
               f"{m.get('license', '')} licence  ·  Core Ultra 5 322, 2×DDR5-5600  ·  TTFT includes tokenize + prefill")
        d.text((self.mx, H - 44), txt, font=self.f.small, fill=DIM)

    def summary_card(self, img: Image.Image, alpha: float):
        """Final overlay: medians over all bench prompts per device."""
        over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        a = int(255 * alpha)
        d.rectangle((0, 0, W, H), fill=(5, 8, 12, int(170 * alpha)))
        cx0, cx1, cy0 = 120, W - 120, 182
        rows = self.runs
        cy1 = cy0 + 116 + 62 * len(rows)
        d.rounded_rectangle((cx0, cy0, cx1, cy1), 16, fill=(18, 23, 30, a), outline=(*BORDER, a), width=1)
        f = self.f
        nprom = len(self.doc["prompts"])
        d.text((cx0 + 28, cy0 + 20), "Summary", font=f.card_head, fill=(*INK, a))
        d.text((cx0 + 28 + f.card_head.getlength("Summary") + 14, cy0 + 27),
               f"medians over {nprom} prompts, greedy, up to {self.doc['meta']['max_new_tokens']} new tokens",
               font=f.card, fill=(*MUTED, a))
        heads = [("DEVICE", 0), ("DECODE tok/s", 150), ("TTFT", 520), ("LOAD cold / cached", 640),
                 ("HOST CPU busy", 860)]
        hy = cy0 + 70
        for lab, dx in heads:
            d.text((cx0 + 28 + dx, hy), lab, font=f.label, fill=(*MUTED, a))
        top_rate = max(r["summary"].get("tok_s_median", 0) for r in rows) or 1
        for i, r in enumerate(rows):
            y = hy + 30 + i * 62
            s, ld, col = r["summary"], r["load"], r["color"]
            d.rounded_rectangle((cx0 + 28, y + 2, cx0 + 36, y + 30), 3, fill=(*col, a))
            d.text((cx0 + 48, y + 4), r["device"], font=f.card_b, fill=(*INK, a))
            rate = s.get("tok_s_median", 0)
            bw = 270 * rate / top_rate
            d.rounded_rectangle((cx0 + 178, y + 6, cx0 + 178 + max(bw, 6), y + 26), 4, fill=(*col, a))
            d.text((cx0 + 178 + bw + 10, y + 3), f"{rate:.1f}", font=f.card_val, fill=(*INK, a))
            d.text((cx0 + 548, y + 3), f"{s.get('ttft_ms_median', 0):.0f} ms", font=f.card_val, fill=(*INK, a))
            if ld.get("cached_s") is not None:
                cold = f"{ld['cold_s']:.0f} s" if ld.get("cold_s") else "n/a"
                load = f"{cold} / {ld['cached_s']:.1f} s"
            else:
                load = f"{ld.get('first', {}).get('load_s', 0):.1f} s (no compile)"
            d.text((cx0 + 668, y + 3), load, font=f.card_val, fill=(*INK, a))
            d.text((cx0 + 888, y + 3), f"{s.get('cpu_cores_busy_median', 0):.1f} cores", font=f.card_val,
                   fill=(*INK, a))
        note = ("Decode speed is bounded by DDR5-5600 memory bandwidth: every token streams ~2 GB of int4 weights.")
        d.text((cx0 + 28, cy1 - 34), note, font=f.small, fill=(*MUTED, a))
        img.alpha_composite(over)

    def frame(self, t: float, card_alpha: float = 0.0) -> np.ndarray:
        img = Image.new("RGBA", (W, H), (*BG, 255))
        d = ImageDraw.Draw(img)
        self.header(d, min(t, self.end_t))
        for i, r in enumerate(self.runs):
            self.panel(img, d, i, r, t)
        self.footer(d)
        if card_alpha > 0:
            self.summary_card(img, card_alpha)
        return np.asarray(img.convert("RGB"))


def write_text_gif(frames, path: Path, fps: float, width: int, max_bytes: int = GIF_MAX_BYTES) -> Path:
    """GIF tuned for terminal-style text: one shared palette, no dithering.

    media.write_gif quantizes every frame independently (with dithering), so static pixels change between frames
    and the size explodes. With one shared, undithered palette, unchanged pixels stay identical, and Pillow then
    stores only the changed rectangle of each frame (new tokens, counters, clock). Width shrinks until it fits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        def prep(f):
            im = Image.fromarray(f)
            return im.resize((width, round(im.height * width / im.width)), Image.LANCZOS) if im.width > width else im
        sample = [prep(frames[i]) for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
        strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
        for i, im in enumerate(sample):
            strip.paste(im, (0, i * im.height))
        pal = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        images = [prep(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
        images[0].save(path, save_all=True, append_images=images[1:], loop=0, duration=round(1000 / fps),
                       optimize=False, disposal=1)
        if path.stat().st_size <= max_bytes or width < 640:
            return path
        width = int(width * 0.88)


def render_video(doc: dict, prompt_id: str, name: str = "llm_chat", preview: Path | None = None) -> dict:
    race = Race(doc, prompt_id)
    rec = Recorder(name, fps=FPS)
    lead, tail_race, hold, fade = 0.6, 1.2, 5.0, 0.6
    total = lead + race.end_t + tail_race + hold
    for i in range(int(total * FPS)):
        t = i / FPS - lead
        after = t - race.end_t - tail_race
        alpha = 0.0 if after < 0 else min(1.0, after / fade)
        rec.add(race.frame(t, alpha))
    if preview:
        Image.fromarray(race.frame(race.end_t * 0.45)).save(preview.with_name(preview.stem + "_mid.png"))
        Image.fromarray(race.frame(race.end_t + 10, 1.0)).save(preview.with_name(preview.stem + "_end.png"))
    out = rec.save(gif=False)  # mp4 at full rate via the shared Recorder; GIF via the text-tuned writer
    step = 2  # 20 fps -> 10 fps GIF
    out["gif"] = str(write_text_gif(rec.frames[::step], MEDIA_DIR / f"{name}.gif", FPS / step, W))
    return out


# ------------------------------------------------------------------------------------------------- chart
def render_chart(doc: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devs = [d for name in ORDER for d in doc["devices"] if d["device"] == name and d.get("status") == "executed"]
    names = [d["device"] for d in devs]
    cols = [DEVICE_HEX[n] for n in names]
    ink, muted, grid = "#1F2328", "#59636E", "#E6E8EB"
    plt.rcParams.update({"font.family": ["Segoe UI"] if Path("C:/Windows/Fonts/segoeui.ttf").exists() else ["DejaVu Sans"], "font.size": 10.5, "axes.edgecolor": grid,
                         "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": muted,
                         "axes.titlecolor": ink, "axes.titleweight": "bold", "axes.titlesize": 12})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), facecolor="white")
    x = np.arange(len(devs))

    def bars(ax, key, title, unit, fmt):
        med = [d["summary"][key] for d in devs]
        ax.bar(x, med, width=0.56, color=cols, zorder=2)
        pkey = {"ttft_ms_median": "ttft_ms", "tok_s_median": "tok_s", "cpu_cores_busy_median": "cpu_cores_busy"}[key]
        ymax = max(max(p.get(pkey) or 0 for d in devs for p in d["prompts"]), max(med)) * 1.18
        for i, d in enumerate(devs):
            vals = [p[pkey] for p in d["prompts"] if p.get(pkey) is not None]
            ax.scatter(np.full(len(vals), i) + np.linspace(-0.16, 0.16, len(vals)), vals, s=18, color=ink,
                       alpha=0.5, zorder=3, linewidths=0)
            # median label beside the bar top, so per-prompt dots never push it away from its bar
            ax.text(i + 0.31, med[i], fmt.format(med[i]), ha="left", va="center", color=ink,
                    fontsize=10.5, fontweight="bold")
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xticks(x, names)
        ax.set_xlim(-0.5, len(devs) - 0.5 + 0.45)
        ax.set_ylim(0, ymax)

    bars(axes[0], "tok_s_median", "Decode speed (higher is better)", "tokens / s", "{:.1f}")
    bars(axes[1], "ttft_ms_median", "Time to first token", "ms (tokenize + prefill)", "{:.0f} ms")
    bars(axes[2], "cpu_cores_busy_median", "Host CPU busy while generating", "CPU cores (process time / wall)",
         "{:.1f}")
    ax = axes[3]
    w = 0.36
    for i, d in enumerate(devs):
        ld = d["load"]
        cold = ld.get("cold_s") or ld.get("no_cache_s") or ld["first"]["load_s"]
        cached = ld.get("cached_s")
        ax.bar(i - w / 2 if cached is not None else i, cold, width=w, color=cols[i], zorder=2)
        ax.text(i - w / 2 if cached is not None else i, cold * 1.12, f"{cold:.0f} s" if cold >= 10 else f"{cold:.1f} s",
                ha="center", va="bottom", fontsize=9.5, color=ink)
        if cached is not None:
            ax.bar(i + w / 2, cached, width=w, color=cols[i], alpha=0.45, hatch="///", edgecolor="white", zorder=2)
            ax.text(i + w / 2, cached * 1.12, f"{cached:.1f} s", ha="center", va="bottom", fontsize=9.5, color=ink)
    ax.set_yscale("log")
    ax.set_title("Pipeline load: cold compile vs cached", loc="left")
    ax.set_ylabel("seconds (log)")
    ax.set_xticks(x, names)
    top = max((d["load"].get("cold_s") or d["load"]["first"]["load_s"]) for d in devs)
    ax.set_ylim(0.5, top * 14)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#8C959F", label="cold (first compile)"),
                       Patch(facecolor="#8C959F", alpha=0.45, hatch="///", edgecolor="white", label="cached blob")],
              frameon=False, fontsize=9, loc="upper right")
    for a in axes:
        a.grid(axis="y", color=grid, zorder=0)
        a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    m = doc["meta"]
    fig.suptitle(f"Phi-3-mini-4k-instruct ({m['model'].get('weight_mode', 'int4')}, channel-wise) on Core Ultra 5 322"
                 f"  ·  OpenVINO GenAI {m['versions']['openvino_genai'].split('-')[0]}  ·  "
                 f"{len(doc['prompts'])} prompts, greedy, ≤{m['max_new_tokens']} new tokens  ·  bar = median, dot = one prompt",
                 x=0.01, ha="left", fontsize=12.5, color=ink, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    return path


def render_all(json_path: Path, prompt_id: str = "npu_vs_gpu", preview: Path | None = None):
    doc = json.loads(Path(json_path).read_text(encoding="utf-8"))
    chart = render_chart(doc, MEDIA_DIR / "llm_chat.png")
    out = render_video(doc, prompt_id, preview=preview)
    out["png"] = str(chart)
    for k, v in out.items():
        size = Path(v).stat().st_size / 1e6
        print(f"{k}: {v} ({size:.2f} MB)")
    return out
