"""Build the static GitHub Pages site into site/_build/.

    python site/build.py              # local: extract numbers from benchmarks/results/*.json,
                                      # refresh site/data/results.json, then build
    python site/build.py --from-data  # CI: build only from the committed site/data/results.json

benchmarks/results/ is gitignored, so CI never sees it. Run the plain build locally after new
results (or new charts) and commit site/data/results.json together with media/.

Standard library only. Templates are string.Template files in site/templates/.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import re
import shutil
import statistics
import sys
from pathlib import Path
from string import Template

SITE = Path(__file__).resolve().parent
ROOT = SITE.parent
RESULTS = ROOT / "benchmarks" / "results"
MEDIA = ROOT / "media"
CHARTS = MEDIA / "charts"
DATA_FILE = SITE / "data" / "results.json"
REPO = "https://github.com/jcallinan/panther-lake-ai-lab"
MP4_LIMIT = 10 * 1024 * 1024
DEVICES = ("NPU", "GPU", "CPU")

ARK = ("https://www.intel.com/content/www/us/en/products/sku/245716/"
       "intel-core-ultra-5-processor-322-12m-cache-up-to-4-40-ghz/specifications.html")
QRG = ("https://cdrdv2-public.intel.com/871380/Intel%20Core%20Ultra%20Series%203%20Processors"
       "%20-%20Quick%20Reference%20Guide%20v1.pdf")
PSREF = ("https://psref.lenovo.com/syspool/Sys/PDF/IdeaPad/IdeaPad_5_2_in_1_15IPH11/"
         "IdeaPad_5_2_in_1_15IPH11_Spec.pdf")
CNC = "https://old.chipsandcheese.com/2025/10/14/panther-lakes-reveal-at-itt-2025/"

# Chart files produced by scripts/make_charts.py (exact names). Captions are fallbacks only;
# the numbers in them come from the data, and media/charts/README.md is rendered below them.
CHARTS_SPEC = [
    ("latency_by_workload", "Latency by workload, NPU vs GPU vs CPU",
     "Median latency of each real workload on each device (lower is better)."),
    ("llm_compare", "Local LLM: decode speed, time to first token and CPU load",
     "Phi-3-mini int4 on each device, run one device at a time."),
    ("cpu_cost", "CPU cost of always-on AI",
     "Host CPU time the AI process uses per frame for each placement."),
    ("compile_cost", "One-off compile cost",
     "Model compile time per device; the NPU pays the most, which is why compile caches matter."),
    ("npu_batch_bug", "NPU batch 2-8 convolution correctness bug",
     "Batched convolutions on the NPU at batch 2-8 return sample 0's output for every element."),
    ("driver_contention", "Driver update vs a busy machine",
     "Preliminary contended runs on the old driver compared with clean sequential runs on the new one."),
]


# --------------------------------------------------------------------------- extraction

def _load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as e:
        print(f"warning: cannot read {p.name}: {e}", file=sys.stderr)
        return None


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def _lat(d):
    """Keep only the summary stats of a latency block."""
    if not isinstance(d, dict):
        return None
    return {k: _r(d.get(k), 3) for k in ("median_ms", "p95_ms", "min_ms", "max_ms")} | {"runs": d.get("runs")}


def _short_ov(v):
    return (v or "").split("-")[0] or None


def extract() -> dict:
    """Pull the numbers the site needs from benchmarks/results/*.json. No paths, names or IDs."""
    out: dict = {"extracted": _dt.date.today().isoformat()}
    env: dict = {}
    summ = _load("record_all_summary.json") or {}
    env["npu_driver"] = summ.get("npu_driver")
    env["graphics_driver"] = summ.get("graphics_driver")
    env["on_ac"] = summ.get("on_ac")
    env["recorded"] = (summ.get("finished") or "")[:10] or None

    # ---- perf atlas
    pa = _load("perf_atlas.json")
    if pa:
        meta = pa.get("meta", {})
        env["openvino"] = _short_ov(meta.get("openvino_version"))
        env["device_names"] = meta.get("device_names")
        env["power"] = (meta.get("power") or {}).get("power_source")
        rows = pa.get("results", [])
        idx = {(r["model"], r["batch"], r["precision"], r["device"]): r for r in rows}
        status = {}
        for r in rows:
            status.setdefault(r["device"], {}).setdefault(r["status"], 0)
            status[r["device"]][r["status"]] += 1
        wins = {}
        for other in ("CPU", "GPU"):
            w = n = 0
            for (m, b, p, d), r in idx.items():
                if d != "NPU" or r["status"] != "executed":
                    continue
                o = idx.get((m, b, p, other))
                if o and o["status"] == "executed":
                    n += 1
                    w += r["median_ms"] < o["median_ms"]
            wins[other] = {"npu_wins": w, "comparable": n}
        picks = {}
        for m, b in [("MLP-S", 1), ("MLP-L", 1), ("MLP-L", 8), ("MLP-L", 32), ("CONV-XS", 1),
                     ("CONV-M", 1), ("CONV-M", 32)]:
            for p in ("FP16", "INT8"):
                key = f"{m} b{b} {p}"
                picks[key] = {}
                for d in DEVICES:
                    r = idx.get((m, b, p, d))
                    if r:
                        picks[key][d] = {"status": r["status"], "median_ms": _r(r.get("median_ms"), 3),
                                         "p95_ms": _r(r.get("p95_ms"), 3),
                                         "gflops": _r(r.get("gflops_effective"), 1)}
        bad = [{"model": r["model"], "batch": r["batch"], "precision": r["precision"],
                "replicates_sample0": (r.get("batch_diagnosis") or {}).get("outputs_replicate_sample0")}
               for r in rows if r["device"] == "NPU" and r["status"] == "failed_correctness"]
        compile_s, peak = {}, {}
        for d in DEVICES:
            cs = [r["compile_s"] for r in rows if r["device"] == d and r.get("compile_s") is not None]
            if cs:
                compile_s[d] = {"median_s": _r(statistics.median(cs), 2), "max_s": _r(max(cs), 1)}
            gf = [r["gflops_effective"] for r in rows
                  if r["device"] == d and r["status"] == "executed" and r.get("gflops_effective")]
            if gf:
                peak[d] = _r(max(gf), 0)
        asyn = {}
        for r in pa.get("async", []):
            if r.get("status") == "executed":
                asyn.setdefault(f'{r["model"]} b{r["batch"]} {r["precision"]}', {})[r["device"]] = _r(r.get("samples_per_s"), 1)
        out["perf_atlas"] = {
            "date": meta.get("date"), "configs": len(rows),
            "grid": {k: (meta.get("grid") or {}).get(k) for k in ("models", "batches", "precisions")},
            "sweep_s": _r(meta.get("sweep_seconds"), 0), "status": status, "wins": wins,
            "picks": picks, "wrong_output": bad, "compile": compile_s, "peak_gflops": peak, "async": asyn,
        }

    # ---- vision
    v = _load("vision_side_by_side.json")
    if v:
        env.setdefault("openvino", _short_ov(v.get("openvino_version")))
        out["vision"] = {
            "date": (v.get("date") or "")[:10], "mode": v.get("mode"), "frames": v.get("frames"),
            "pipeline_fps": _r(v.get("pipeline_fps"), 1),
            "devices": {d: {"status": x.get("status"),
                            "execution_devices": x.get("execution_devices"),
                            "detect": _lat((x.get("latency_ms") or {}).get("detect")),
                            "segment": _lat((x.get("latency_ms") or {}).get("segment")),
                            "frame": _lat((x.get("latency_ms") or {}).get("frame")),
                            "compile_s": {k: _r(c.get("compile_s"), 2) for k, c in (x.get("compile") or {}).items()}}
                        for d, x in (v.get("devices") or {}).items()},
        }

    # ---- depth
    dp = _load("depth_parallax.json")
    if dp:
        live = dp.get("live") or {}
        out["depth"] = {
            "date": (dp.get("meta") or {}).get("date"),
            "input": (dp.get("meta") or {}).get("input_shape"),
            "devices": {d: {"status": x.get("status"), "execution_devices": x.get("execution_devices"),
                            "compile_s": _r(x.get("compile_s"), 2), "cache": x.get("cache"),
                            "bench": _lat(x.get("benchmark"))}
                        for d, x in (dp.get("devices") or {}).items()},
            "live": {"device": live.get("device"), "fps": _r(live.get("end_to_end_fps"), 1),
                     "frames": live.get("frames"), "model": _lat(live.get("model_latency_ms")),
                     "host": _lat(live.get("host_ms_per_frame")), "render": _lat(live.get("parallax_render_ms"))},
        }

    # ---- offload
    of = _load("offload_dashboard.json")
    if of:
        phases = []
        for ph in of.get("phases", []):
            r = ph.get("result") or {}
            phases.append({"label": ph.get("label"), "device": ph.get("device"), "status": ph.get("status"),
                           **{k: _r(r.get(k), 2) for k in ("ai_fps", "fg_pct", "ai_proc_cpu_pct", "cpu_pct",
                                                             "ai_proc_cpu_ms_per_frame", "infer_ms", "prepost_ms",
                                                             "ui_lag_p95_ms", "other_procs_cpu_pct",
                                                             "npu_util", "gpu_util")}})
        cfg = of.get("config") or {}
        out["offload"] = {
            "date": (of.get("timestamp") or "")[:10], "phase_s": cfg.get("phase_seconds"),
            "fg_workers": cfg.get("fg_workers"), "target_fps": cfg.get("target_fps"),
            "drift_pct": _r((of.get("baseline") or {}).get("drift_pct"), 1),
            "execution_devices": {d: sorted({e for m in x.values() for e in (m.get("execution_devices") or [])})
                                  for d, x in (of.get("compile") or {}).items()},
            "phases": phases,
        }

    # ---- llm
    lm = _load("llm_chat.json")
    if lm:
        meta = lm.get("meta") or {}
        env["openvino_genai"] = _short_ov((meta.get("versions") or {}).get("openvino_genai"))
        env["python"] = (meta.get("versions") or {}).get("python")
        devs = {}
        for x in lm.get("devices", []):
            s = x.get("summary") or {}
            load = x.get("load") or {}
            ps = x.get("prompts") or []
            devs[x["device"]] = {
                "status": x.get("status"), "tok_s": _r(s.get("tok_s_median"), 1),
                "tok_s_range": [_r(min(p["tok_s"] for p in ps), 1), _r(max(p["tok_s"] for p in ps), 1)] if ps else None,
                "ttft_ms": _r(s.get("ttft_ms_median"), 0), "cores_busy": _r(s.get("cpu_cores_busy_median"), 2),
                "load_cached_s": _r(load.get("cached_s"), 1), "load_cold_s": _r(load.get("cold_s"), 1),
                "load_nocache_s": _r(load.get("no_cache_s"), 1),
            }
        m = meta.get("model") or {}
        out["llm"] = {"date": (meta.get("created") or "")[:10],
                      "model": {k: m.get(k) for k in ("hf_id", "params", "weight_mode", "license", "weights_gb")},
                      "max_new_tokens": meta.get("max_new_tokens"), "prompts": len(lm.get("prompts", [])),
                      "devices": devs}

    env.setdefault("recorded", (out.get("perf_atlas") or {}).get("date"))
    out["env"] = env
    return out


# --------------------------------------------------------------------------- tiny markdown

def _inline(s: str, link_fix) -> str:
    codes: list[str] = []

    def keep(m):
        codes.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]+)`", keep, s)
    s = html.escape(s, quote=False)
    s = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)",
               lambda m: f'<img src="{html.escape(link_fix(m.group(2)))}" alt="{m.group(1)}" loading="lazy">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)",
               lambda m: f'<a href="{html.escape(link_fix(m.group(2)))}">{m.group(1)}</a>', s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], s)


def markdown(text: str, link_fix=lambda u: u, shift: int = 1) -> str:
    """Small Markdown subset: headings, paragraphs, lists, tables, code fences, quotes, rules."""
    lines = text.replace("\r\n", "\n").split("\n")
    out, i, para = [], 0, []

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para), link_fix) + "</p>")
            para.clear()

    while i < len(lines):
        ln = lines[i]
        st = ln.strip()
        if st.startswith("```"):
            flush()
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                buf.append(lines[j])
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i = j + 1
            continue
        if not st:
            flush(); i += 1; continue
        m = re.match(r"(#{1,6})\s+(.*)", st)
        if m:
            flush()
            lvl = min(6, len(m.group(1)) + shift)
            out.append(f"<h{lvl}>{_inline(m.group(2), link_fix)}</h{lvl}>")
            i += 1; continue
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", st):
            flush(); out.append("<hr>"); i += 1; continue
        if st.startswith("|") and i + 1 < len(lines) and re.fullmatch(r"\|?\s*:?-{2,}.*", lines[i + 1].strip()):
            flush()
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]
            head = cells(st)
            i += 2
            body = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body.append(cells(lines[i])); i += 1
            t = ['<div class="table-wrap"><table><thead><tr>']
            t += [f'<th scope="col">{_inline(h, link_fix)}</th>' for h in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{_inline(c, link_fix)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            out.append("".join(t)); continue
        if st.startswith(">"):
            flush()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append("<blockquote>" + markdown("\n".join(buf), link_fix, shift) + "</blockquote>")
            continue
        lm = re.match(r"([-*+]|\d+[.)])\s+(.*)", st)
        if lm:
            flush()
            tag = "ol" if lm.group(1)[0].isdigit() else "ul"
            items = []
            while i < len(lines):
                s2 = lines[i].strip()
                mm = re.match(r"([-*+]|\d+[.)])\s+(.*)", s2)
                if mm:
                    items.append([mm.group(2)])
                elif s2 and lines[i].startswith((" ", "\t")) and items:
                    items[-1].append(s2)
                else:
                    break
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(' '.join(it), link_fix)}</li>" for it in items) + f"</{tag}>")
            continue
        para.append(st)
        i += 1
    flush()
    return "\n".join(out)


def _repo_link_fixer(base_dir: str):
    """Rewrite relative links in a repo Markdown file: chart images stay local, others go to GitHub."""
    def fix(u: str) -> str:
        if re.match(r"[a-z]+:|#|/", u):
            return u
        if base_dir == "media/charts" and u.lower().endswith(".png") and "/" not in u:
            return f"media/charts/{u}"
        path = (Path(base_dir) / u).as_posix()
        parts = []
        for seg in path.split("/"):
            if seg == "..":
                parts and parts.pop()
            elif seg not in ("", "."):
                parts.append(seg)
        return f"{REPO}/blob/main/{'/'.join(parts)}"
    return fix


# --------------------------------------------------------------------------- helpers

E = html.escape


def fmt(x, unit="", nd=None):
    if x is None:
        return "n/a"
    if nd is None:
        nd = 0 if abs(x) >= 100 else (1 if abs(x) >= 10 else 2)
    s = f"{x:,.{nd}f}"
    return f"{s}{unit}"


def ratio(a, b):
    if not a or not b:
        return None
    return b / a


def bars(values: dict, unit: str, nd=None, higher_better=False) -> str:
    """Mini horizontal bar comparison. Values are always printed, so colour is never the only cue."""
    vals = {d: values.get(d) for d in DEVICES if values.get(d) is not None}
    if not vals:
        return '<p class="muted small">No data.</p>'
    top = max(vals.values()) or 1
    li = []
    for d in DEVICES:
        if d not in vals:
            continue
        w = max(2.0, 100.0 * vals[d] / top)
        li.append(f'<li class="{d.lower()}"><span class="chip">{d}</span>'
                  f'<span class="track" aria-hidden="true"><span class="fill" style="width:{w:.1f}%;display:block"></span></span>'
                  f'<span class="val">{fmt(vals[d], unit, nd)}</span></li>')
    note = "higher is better" if higher_better else "lower is better"
    return f'<ul class="bars" aria-label="{note}">{"".join(li)}</ul><p class="sub" style="margin:.4rem 0 0">{note}</p>'


def g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def media_tag(name: str, alt: str, lazy=True, cls="") -> str:
    attrs = 'loading="lazy" decoding="async"' if lazy else 'fetchpriority="high" decoding="async"'
    c = f' class="{cls}"' if cls else ""
    return f'<img src="media/{name}" alt="{E(alt)}" {attrs}{c}>'


# --------------------------------------------------------------------------- pages

NAV = [("index.html", "Home"), ("demos.html", "Demos"), ("results.html", "Results"),
       ("hardware.html", "Hardware"), ("get-started.html", "Get started")]


class Site:
    def __init__(self, data: dict, out: Path):
        self.d = data
        self.out = out
        self.env = data.get("env", {})
        self.mp4s: set[str] = set()
        self.charts: dict[str, bool] = {}
        self.base = Template((SITE / "templates" / "base.html").read_text(encoding="utf-8"))
        self.warnings: list[str] = []

    # ---- numbers used in several places
    def depth_ms(self):
        return {d: g(self.d, "depth", "devices", d, "bench", "median_ms") for d in DEVICES}

    def detect_ms(self):
        return {d: g(self.d, "vision", "devices", d, "detect", "median_ms") for d in DEVICES}

    def frame_ms(self):
        return {d: g(self.d, "vision", "devices", d, "frame", "median_ms") for d in DEVICES}

    def llm(self, key):
        return {d: g(self.d, "llm", "devices", d, key) for d in DEVICES}

    def offload(self, key):
        res = {}
        for ph in g(self.d, "offload", "phases", default=[]) or []:
            if ph.get("device") in DEVICES and ph.get("status", "ok") == "ok":
                res[ph["device"]] = ph.get(key)
        return res

    def bug_summary(self):
        bad = g(self.d, "perf_atlas", "wrong_output", default=[]) or []
        models = sorted({b["model"] for b in bad})
        batches = sorted({b["batch"] for b in bad})
        return len(bad), models, batches

    def caveat_line(self):
        n, models, _ = self.bug_summary()
        bug = (f" On NPU driver {self.env.get('npu_driver')}, batched convolutions still return wrong results "
               f"({n} batch-8 configs in the sweep; batch 2-8 is affected)." if n else "")
        return ("One laptop, mostly synthetic or small models, each device measured alone on AC power. "
                "The NPU does not win everywhere: the iGPU is faster on tiny batch-1 convs, on batch-8 MLPs and "
                "on LLM decode speed and time to first token." + bug)

    # ---- page shell
    def page(self, fname: str, title: str, desc: str, content: str):
        nav = "\n".join(
            f'        <li><a href="{h}"{" aria-current=\"page\"" if h == fname else ""}>{t}</a></li>' for h, t in NAV)
        full = self.base.substitute(
            title=E(title), description=E(desc), nav=nav, content=content, repo=REPO,
            results_date=E(self.env.get("recorded") or "n/a"),
            npu_driver=E(self.env.get("npu_driver") or "n/a"),
            graphics_driver=E(self.env.get("graphics_driver") or "n/a"),
            openvino=E(self.env.get("openvino") or "n/a"),
            built=_dt.date.today().isoformat())
        (self.out / fname).write_text(full, encoding="utf-8")

    # ---- home
    def home(self):
        dm, vm = self.depth_ms(), self.detect_ms()
        tok, cores = self.llm("tok_s"), self.llm("cores_busy")
        cpu_ms = self.offload("ai_proc_cpu_ms_per_frame")
        wins = g(self.d, "perf_atlas", "wins", default={})
        stats = []
        if any(dm.values()):
            x = ratio(dm.get("NPU"), dm.get("CPU"))
            stats.append(("Depth estimation", "Depth Anything V2 Small, one 364x364 frame, p50",
                          f'{fmt(dm.get("NPU"), " ms")} on the NPU' + (f", {x:.0f}x faster than the CPU" if x else ""),
                          bars(dm, " ms")))
        if any(vm.values()):
            x = ratio(vm.get("NPU"), vm.get("GPU"))
            stats.append(("Object detection", "YOLOX-Tiny INT8, 416x416, p50",
                          f'{fmt(vm.get("NPU"), " ms")} on the NPU' + (f", {x:.1f}x faster than the iGPU" if x else ""),
                          bars(vm, " ms")))
        if any(cores.values()):
            stats.append(("Local LLM, CPU left free", "Phi-3-mini int4: CPU cores kept busy while generating",
                          f'{fmt(tok.get("NPU"), " tok/s", 1)} on the NPU using {fmt(cores.get("NPU"), " cores")}',
                          bars(cores, " cores")))
        if any(cpu_ms.values()):
            x = ratio(cpu_ms.get("NPU"), cpu_ms.get("CPU"))
            stats.append(("Always-on video AI", "Blur + detection at 30 FPS: CPU ms used per frame",
                          f'{fmt(cpu_ms.get("NPU"), " ms")} of CPU per frame on the NPU' + (f", {x:.0f}x less than on the CPU" if x else ""),
                          bars(cpu_ms, " ms")))
        cards = "".join(f'<section class="stat" aria-label="{E(t)}"><h3>{E(t)}</h3><p class="sub">{E(s)}</p>'
                        f'<p class="big">{E(b)}</p>{bar}</section>' for t, s, b, bar in stats)
        wtxt = ""
        if wins:
            wtxt = (f'<p>In a {g(self.d, "perf_atlas", "configs")}-configuration synthetic sweep, the NPU was faster than the CPU in '
                    f'{wins["CPU"]["npu_wins"]} of {wins["CPU"]["comparable"]} comparable configs and faster than the iGPU in '
                    f'{wins["GPU"]["npu_wins"]} of {wins["GPU"]["comparable"]}. <a href="results.html">See where it loses.</a></p>')
        content = f"""
<section class="hero">
  <div>
    <h1>What can a laptop NPU actually do?</h1>
    <p class="lede">Reproducible experiments on an Intel Core Ultra 5 322 (Panther Lake) laptop. Every workload runs on the
    <span class="chip npu">NPU</span>, the <span class="chip gpu">GPU</span> and the <span class="chip cpu">CPU</span>,
    reports the device OpenVINO actually used, and checks its own answers.</p>
    <div class="cta"><a class="btn primary" href="demos.html">See the demos</a><a class="btn" href="results.html">Read the results</a><a class="btn" href="get-started.html">Run it yourself</a></div>
  </div>
  <figure>
    {media_tag("vision_side_by_side.gif", "Street video processed three times side by side: object boxes and background blur on NPU, GPU and CPU, each panel showing its own latency", lazy=False)}
    <figcaption>The same street clip through detection and segmentation on NPU, GPU and CPU at once.</figcaption>
  </figure>
</section>

<h2>Headline numbers</h2>
<p class="muted small">Medians from clean sequential runs, one device at a time, recorded {E(self.env.get("recorded") or "")} on NPU driver {E(self.env.get("npu_driver") or "n/a")}.</p>
<div class="stats">{cards}</div>
<p class="caveat"><strong>Caveat:</strong> {E(self.caveat_line())}</p>
{wtxt}

<h2>What is here</h2>
<div class="grid2">
  <div><h3><a href="demos.html">Five demos</a></h3><p>A 162-config performance atlas, live vision, depth-to-3D, an offload dashboard and a local LLM chat, each with a recording and a one-line run command.</p></div>
  <div><h3><a href="results.html">Results and method</a></h3><p>Charts, the full atlas dashboard, how each number was measured, and the known problems, including an NPU correctness bug.</p></div>
  <div><h3><a href="hardware.html">Hardware</a></h3><p>What the Core Ultra 5 322 is, with vendor claims and independent measurements kept apart.</p></div>
  <div><h3><a href="get-started.html">Get started</a></h3><p>Clone, install Python 3.12 and the drivers, fetch the models, run a demo.</p></div>
</div>
"""
        self.page("index.html", "Panther Lake AI Lab",
                  "NPU vs GPU vs CPU experiments on an Intel Core Ultra 5 322 laptop, with verified execution devices.",
                  content)

    # ---- demos
    def demo_list(self):
        pa = self.d.get("perf_atlas", {})
        wins = pa.get("wins", {})
        n, _, _ = self.bug_summary()
        p1 = g(pa, "picks", "MLP-L b1 FP16", default={})
        p8 = g(pa, "picks", "MLP-L b8 FP16", default={})
        p32 = g(pa, "picks", "MLP-L b32 FP16", default={})
        dm, vm = self.depth_ms(), self.detect_ms()
        tok, ttft, cores = self.llm("tok_s"), self.llm("ttft_ms"), self.llm("cores_busy")
        cpu_ms, fps = self.offload("ai_proc_cpu_ms_per_frame"), self.offload("ai_fps")
        live = g(self.d, "depth", "live", default={})

        def ms(dct, d):
            return fmt(g(dct, d, "median_ms"), " ms")

        return [
            ("perf_atlas", "Performance Atlas",
             "NPU Performance Atlas animation: latency curves and heatmaps for MLP and convolution models building up into a seven-panel dashboard",
             f"A {pa.get('configs', 162)}-configuration sweep of synthetic MLP and convolution models (size x batch x precision x device), "
             "timed at p50/p95 and checked against numpy. It maps where the NPU wins, where it loses and where it is wrong.",
             (f"NPU faster than the CPU in {g(wins, 'CPU', 'npu_wins')}/{g(wins, 'CPU', 'comparable')} and the iGPU in "
              f"{g(wins, 'GPU', 'npu_wins')}/{g(wins, 'GPU', 'comparable')} comparable configs. MLP-L FP16: {ms(p1, 'NPU')} at batch 1, "
              f"{ms(p8, 'NPU')} at batch 8, {ms(p32, 'NPU')} at batch 32. {n} NPU configs gave wrong output." if wins else "n/a"),
             "python -m examples.perf_atlas"),
            ("vision_side_by_side", "Vision side by side",
             "Three panels of the same street video with detection boxes and blurred background, labelled NPU, GPU and CPU, plus a latency comparison panel",
             "Object detection (YOLOX-Tiny INT8) plus person segmentation and background blur on the same frames, on all three devices, "
             "each panel showing its own p50/p95 and EXECUTION_DEVICES.",
             f"Detection p50: NPU {fmt(vm.get('NPU'), ' ms')}, GPU {fmt(vm.get('GPU'), ' ms')}, CPU {fmt(vm.get('CPU'), ' ms')}.",
             "python -m examples.vision_side_by_side"),
            ("depth_parallax", "Depth Parallax",
             "A forest creek video beside its colourised depth map and a re-projected 3D view that slowly orbits",
             "Monocular depth (Depth Anything V2 Small) on every frame on the NPU; the CPU turns each frame into a point cloud "
             "and re-renders it from an orbiting camera.",
             f"364x364 depth p50: NPU {fmt(dm.get('NPU'), ' ms')}, GPU {fmt(dm.get('GPU'), ' ms')}, CPU {fmt(dm.get('CPU'), ' ms')}. "
             f"Live demo: {fmt(live.get('fps'), ' FPS', 1)} end to end, CPU-render bound.",
             "python -m examples.depth_parallax"),
            ("offload_dashboard", "Offload dashboard",
             "Dashboard with a processed video of a person, live tiles and line charts for foreground work, AI CPU use and AI frame rate per placement",
             "Always-on background blur plus detection at a 30 FPS target, placed on the CPU, then the GPU, then the NPU, while a "
             "fixed CPU-bound job runs in the foreground.",
             f"CPU time per AI frame: NPU {fmt(cpu_ms.get('NPU'), ' ms')}, GPU {fmt(cpu_ms.get('GPU'), ' ms')}, CPU {fmt(cpu_ms.get('CPU'), ' ms')} "
             f"(AI FPS {fmt(fps.get('NPU'), '', 1)} / {fmt(fps.get('GPU'), '', 1)} / {fmt(fps.get('CPU'), '', 1)}).",
             "python -m examples.offload_dashboard"),
            ("llm_chat", "LLM chat",
             "Three chat transcripts filling in side by side at their measured token timing for NPU, GPU and CPU",
             "Phi-3-mini (3.8B, int4) chatting on the NPU with OpenVINO GenAI; the recording replays sequential runs on all three "
             "devices at their measured per-token timing.",
             f"Decode: NPU {fmt(tok.get('NPU'), ' tok/s', 1)} with {fmt(cores.get('NPU'), ' CPU cores')} busy; GPU {fmt(tok.get('GPU'), ' tok/s', 1)} "
             f"({fmt(cores.get('GPU'), ' cores')}); CPU {fmt(tok.get('CPU'), ' tok/s', 1)} ({fmt(cores.get('CPU'), ' cores')}). "
             f"Time to first token: GPU {fmt(ttft.get('GPU'), ' ms')}, NPU {fmt(ttft.get('NPU'), ' ms')}.",
             "python -m examples.llm_chat"),
        ]

    def demos(self):
        cards = []
        for slug, name, alt, what, key, cmd in self.demo_list():
            mp4 = f'<a href="media/mp4/{slug}.mp4">MP4 video</a>' if slug in self.mp4s else ""
            cards.append(f"""<article class="card" id="{slug}">
  <figure>{media_tag(slug + ".gif", alt)}</figure>
  <div class="body">
    <h2>{E(name)}</h2>
    <p>{E(what)}</p>
    <div class="key"><strong>Key numbers:</strong> {E(key)}</div>
    <pre><code>.venv\\Scripts\\{E(cmd.replace("python ", "python.exe "))}</code></pre>
    <p class="links"><a href="{REPO}/tree/main/examples/{slug}">Code and README</a>{mp4}</p>
  </div>
</article>""")
        content = f"""
<h1>Demos</h1>
<p class="lede">Each demo runs the same workload on NPU, GPU and CPU, reports the device that actually executed it
(<code>EXECUTION_DEVICES</code>) and never falls back silently. The animations are recordings from the test laptop.</p>
<p class="small muted">Setup first: see <a href="get-started.html">Get started</a>. Numbers are medians from runs recorded {E(self.env.get("recorded") or "")}.</p>
<div class="cards">{"".join(cards)}</div>
"""
        self.page("demos.html", "Demos · Panther Lake AI Lab",
                  "Five NPU vs GPU vs CPU demos: performance atlas, vision, depth parallax, offload dashboard, LLM chat.",
                  content)

    # ---- results
    def _table(self, head, rows, numcols=()):
        th = "".join(f'<th scope="col"{" class=\"num\"" if i in numcols else ""}>{h}</th>' for i, h in enumerate(head))
        tr = "".join("<tr>" + "".join(
            (f'<th scope="row">{c}</th>' if i == 0 else f'<td{" class=\"num\"" if i in numcols else ""}>{c}</td>')
            for i, c in enumerate(r)) + "</tr>" for r in rows)
        return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'

    def _chip(self, d):
        return f'<span class="chip {d.lower()}">{d}</span>'

    def results(self):
        d = self.d
        pa = d.get("perf_atlas", {})
        wins = pa.get("wins", {})
        n_bad, bad_models, bad_batches = self.bug_summary()
        picks = pa.get("picks", {})

        # charts
        notes_html = ""
        takeaways = {}
        readme = CHARTS / "README.md"
        if readme.exists():
            md = readme.read_text(encoding="utf-8")
            # Takeaway line per chart section: "### `name.png`" ... "**Takeaway:** text"
            for m in re.finditer(r"^###\s+`?([\w-]+)\.png`?\s*$(.*?)(?=^#{2,3}\s|\Z)", md, re.M | re.S):
                t = re.search(r"\*\*Takeaway:\*\*\s*(.+)", m.group(2))
                if t:
                    takeaways[m.group(1)] = re.sub(r"[`*]", "", t.group(1)).strip()
            # Only the technical set belongs on this page; the simplified set is for slide decks.
            md = re.split(r"^##\s+Simple set", md, maxsplit=1, flags=re.M)[0]
            notes_html = markdown(md, _repo_link_fixer("media/charts"), shift=1)
        figs = []
        for stem, title, cap in CHARTS_SPEC:
            cap = takeaways.get(stem, cap)
            if self.charts.get(stem):
                figs.append(f'<figure class="chart" id="chart-{stem}"><h3>{E(title)}</h3>'
                            f'<a href="media/charts/{stem}.png"><img src="media/charts/{stem}.png" alt="Chart: {E(title)}. {E(cap)}" loading="lazy" decoding="async"></a>'
                            f'<figcaption>{E(cap)}</figcaption></figure>')
            else:
                figs.append(f'<figure class="chart" id="chart-{stem}"><h3>{E(title)}</h3><div class="placeholder">Chart not generated yet '
                            f'(<code>media/charts/{stem}.png</code>).</div></figure>')

        # atlas table
        rows = []
        for key in ("MLP-S b1 FP16", "MLP-L b1 FP16", "MLP-L b1 INT8", "MLP-L b8 FP16", "MLP-L b32 FP16",
                    "CONV-XS b1 FP16", "CONV-M b1 FP16", "CONV-M b32 FP16"):
            pk = picks.get(key)
            if not pk:
                continue
            cells = [E(key)]
            for dv in DEVICES:
                x = pk.get(dv) or {}
                if x.get("status") == "executed":
                    cells.append(f'{fmt(x.get("median_ms"), "")} / {fmt(x.get("p95_ms"), "")}')
                else:
                    cells.append(E(x.get("status", "n/a").replace("_", " ")))
            rows.append(cells)
        atlas_tbl = self._table(["Config"] + [f"{self._chip(x)} p50 / p95 ms" for x in DEVICES], rows, (1, 2, 3))
        comp = pa.get("compile", {})
        comp_tbl = self._table(["Device", "Median compile", "Slowest compile", "Peak effective GFLOP/s"],
                               [[self._chip(x), fmt(g(comp, x, "median_s"), " s"), fmt(g(comp, x, "max_s"), " s"),
                                 fmt(g(pa, "peak_gflops", x), "", 0)] for x in DEVICES], (1, 2, 3))

        # vision / depth / llm / offload tables
        vis = d.get("vision", {}).get("devices", {})
        vis_tbl = self._table(["Device", "Detect p50 / p95", "Segment p50 / p95", "Frame p50", "EXECUTION_DEVICES"],
                              [[self._chip(x),
                                f'{fmt(g(vis, x, "detect", "median_ms"))} / {fmt(g(vis, x, "detect", "p95_ms"))} ms',
                                f'{fmt(g(vis, x, "segment", "median_ms"))} / {fmt(g(vis, x, "segment", "p95_ms"))} ms',
                                fmt(g(vis, x, "frame", "median_ms"), " ms"),
                                E(", ".join(sorted({e for v in (g(vis, x, "execution_devices") or {}).values() for e in v})))]
                               for x in DEVICES if x in vis], (1, 2, 3))
        dep = d.get("depth", {}).get("devices", {})
        dep_tbl = self._table(["Device", "Depth p50 / p95", "Compile (cold)", "EXECUTION_DEVICES"],
                              [[self._chip(x), f'{fmt(g(dep, x, "bench", "median_ms"))} / {fmt(g(dep, x, "bench", "p95_ms"))} ms',
                                fmt(g(dep, x, "compile_s"), " s"), E(", ".join(g(dep, x, "execution_devices") or []))]
                               for x in DEVICES if x in dep], (1, 2))
        lm = d.get("llm", {}).get("devices", {})

        def load(x):
            v = g(lm, x)
            if not v:
                return "n/a"
            if v.get("load_nocache_s") is not None:
                return f'{fmt(v["load_nocache_s"], " s", 1)} (no compile)'
            return f'{fmt(v.get("load_cached_s"), " s", 1)} cached'
        llm_tbl = self._table(["Device", "Decode tok/s (range)", "Time to first token", "CPU cores busy", "Load"],
                              [[self._chip(x), f'{fmt(g(lm, x, "tok_s"), "", 1)} ({fmt((g(lm, x, "tok_s_range") or [None])[0], "", 1)}-{fmt((g(lm, x, "tok_s_range") or [None, None])[1], "", 1)})',
                                fmt(g(lm, x, "ttft_ms"), " ms"), fmt(g(lm, x, "cores_busy")), load(x)]
                               for x in DEVICES if x in lm], (1, 2, 3))
        off = d.get("offload", {})
        off_rows = []
        for ph in off.get("phases", []):
            dv = ph.get("device")
            off_rows.append([self._chip(dv) if dv else E(ph.get("label") or ""), fmt(ph.get("ai_fps"), "", 1) if dv else "-",
                             fmt(ph.get("ai_proc_cpu_ms_per_frame"), " ms") if dv else "-",
                             fmt(ph.get("infer_ms"), " ms") if dv else "-",
                             fmt(ph.get("ai_proc_cpu_pct"), " %", 1), fmt(ph.get("ui_lag_p95_ms"), " ms")])
        off_tbl = self._table(["Phase", "AI FPS", "CPU ms per AI frame", "Inference ms", "AI process CPU (% of system)", "UI lag p95"],
                              off_rows, (1, 2, 3, 4, 5))
        drift = off.get("drift_pct")

        mlp1, mlp8, mlp32 = (g(picks, k, "NPU", "median_ms") for k in ("MLP-L b1 FP16", "MLP-L b8 FP16", "MLP-L b32 FP16"))
        gpu8 = g(picks, "MLP-L b8 FP16", "GPU", "median_ms")
        cm32 = {x: g(picks, "CONV-M b32 FP16", x, "median_ms") for x in DEVICES}
        cxs = {x: g(picks, "CONV-XS b1 FP16", x, "median_ms") for x in DEVICES}

        findings = []
        if wins:
            findings.append(f'The NPU was faster than the CPU in <strong>{wins["CPU"]["npu_wins"]} of {wins["CPU"]["comparable"]}</strong> comparable configs and faster than the iGPU in <strong>{wins["GPU"]["npu_wins"]} of {wins["GPU"]["comparable"]}</strong> (all precisions; configs where the NPU gave wrong output are excluded).')
        if all(cm32.values()):
            findings.append(f'Big batched convs are the NPU\'s best case: CONV-M at batch 32 (FP16) takes {fmt(cm32["NPU"], " ms")} on the NPU, {fmt(cm32["GPU"], " ms")} on the iGPU and {fmt(cm32["CPU"], " ms")} on the CPU.')
        if mlp8 and mlp32:
            findings.append(f'Batch 8 is a pathological shape for the NPU compiler: MLP-L FP16 takes {fmt(mlp8, " ms")} at batch 8 but {fmt(mlp32, " ms")} at batch 32 and {fmt(mlp1, " ms")} at batch 1; the iGPU does batch 8 in {fmt(gpu8, " ms")}.')
        if all(cxs.values()):
            findings.append(f'Tiny models favour the iGPU: CONV-XS at batch 1 runs in {fmt(cxs["GPU"], " ms")} on the iGPU vs {fmt(cxs["NPU"], " ms")} on the NPU, where fixed dispatch overhead dominates.')
        if n_bad:
            findings.append(f'<strong>Correctness bug:</strong> {n_bad} NPU configs ({", ".join(bad_models)} at batch {", ".join(map(str, bad_batches))}, every precision) ran but returned sample 0\'s result for every batch element. They are shown as wrong output and not timed. MLPs at the same batch size are correct.')
        if comp:
            findings.append(f'Compile time is the NPU\'s up-front cost: median {fmt(g(comp, "NPU", "median_s"), " s")} (slowest {fmt(g(comp, "NPU", "max_s"), " s")}) vs {fmt(g(comp, "GPU", "median_s"), " s")} on the GPU and {fmt(g(comp, "CPU", "median_s"), " s")} on the CPU. Use a compile cache.')
        tok, ttft = self.llm("tok_s"), self.llm("ttft_ms")
        if tok.get("NPU") and tok.get("GPU"):
            findings.append(f'LLM decode is memory-bandwidth bound, so the devices are close: {fmt(tok["NPU"], " tok/s", 1)} on the NPU, {fmt(tok["GPU"], " tok/s", 1)} on the iGPU, {fmt(tok.get("CPU"), " tok/s", 1)} on the CPU. The NPU\'s advantage is the idle CPU; the iGPU has far better time to first token ({fmt(ttft.get("GPU"), " ms")} vs {fmt(ttft.get("NPU"), " ms")}).')

        content = f"""
<h1>Results</h1>
<p class="lede">Clean sequential runs (one device at a time, nothing else running) recorded {E(self.env.get("recorded") or "")} with
NPU driver {E(self.env.get("npu_driver") or "n/a")}, graphics driver {E(self.env.get("graphics_driver") or "n/a")} and
OpenVINO {E(self.env.get("openvino") or "n/a")}, on AC power.</p>
<nav class="toc" aria-label="On this page"><ul>
<li><a href="#findings">Key findings</a></li><li><a href="#atlas">Performance Atlas dashboard</a></li>
<li><a href="#charts">Charts</a></li><li><a href="#workloads">Real workloads</a></li>
<li><a href="#method">Methodology</a></li><li><a href="#caveats">Caveats</a></li></ul></nav>

<h2 id="findings">Key findings</h2>
<ul>{"".join(f"<li>{f}</li>" for f in findings)}</ul>

<h2 id="atlas">Performance Atlas dashboard</h2>
<p>{pa.get("configs", "n/a")} configurations: {E(", ".join(g(pa, "grid", "models") or []))} at batch {E(", ".join(map(str, g(pa, "grid", "batches") or [])))} in {E(", ".join(g(pa, "grid", "precisions") or []))}, on each device.
INT8 here is weight-only compression with FP16 activations, not a calibrated full-INT8 model.</p>
<figure class="chart"><a href="media/perf_atlas.png"><img src="media/perf_atlas.png" alt="Seven-panel Performance Atlas dashboard: latency vs work for MLP and conv models per device, NPU speed-up heatmaps vs CPU and GPU, peak compute, compile cost, async throughput, and a compatibility grid marking wrong NPU outputs at batch 8" loading="lazy" decoding="async"></a>
<figcaption>Full-resolution dashboard. How to read each panel: <a href="{REPO}/tree/main/examples/perf_atlas#how-to-read-the-dashboard">perf_atlas README</a>.</figcaption></figure>
{atlas_tbl}
{comp_tbl}

<h2 id="charts">Charts</h2>
{"".join(figs)}
{f'<details><summary>Exact numbers and data sources for every chart</summary><div class="md-notes">{notes_html}</div></details>' if notes_html else ""}

<h2 id="workloads">Real workloads</h2>
<h3>Vision: YOLOX-Tiny INT8 detection + selfie segmentation</h3>
<p class="small muted">{g(d, "vision", "frames", default="n/a")} frames, {E(g(d, "vision", "mode", default="") or "")} mode (device order rotates per frame). Timings are <code>infer()</code> only.</p>
{vis_tbl}
<h3>Depth: Depth Anything V2 Small, 364x364</h3>
{dep_tbl}
<h3>LLM: Phi-3-mini-4k-instruct int4 (channel-wise symmetric)</h3>
<p class="small muted">Medians over {g(d, "llm", "prompts", default="n/a")} prompts, greedy decoding, up to {g(d, "llm", "max_new_tokens", default="n/a")} new tokens. OpenVINO GenAI does not expose EXECUTION_DEVICES, so host CPU cores busy is the evidence of where the work ran.</p>
{llm_tbl}
<figure class="chart"><img src="media/llm_chat.png" alt="Bar charts of LLM decode speed, time to first token, host CPU load and load time for NPU, GPU and CPU" loading="lazy" decoding="async"></figure>
<h3>Offload: always-on AI next to foreground work</h3>
<p class="small muted">{off.get("phase_s", "n/a")} s phases, {off.get("fg_workers", "n/a")} foreground zlib workers, 30 FPS target. The foreground-throughput measure is left out: the start and end baselines differed by {fmt(drift, " %", 1)}, too much to compare placements. CPU ms per AI frame (which includes the CPU-side pre/post-processing every placement pays) is the robust measure.</p>
{off_tbl}
<figure class="chart"><img src="media/offload_dashboard.png" alt="Offload dashboard summary: bars of foreground work retained and AI frame rate for each AI placement" loading="lazy" decoding="async"></figure>

<h2 id="method">Methodology</h2>
<ul>
<li><strong>Execution device verified, never assumed.</strong> Every compiled model's <code>EXECUTION_DEVICES</code> is recorded. If a device fails to compile, the run is marked failed; nothing silently falls back to another device.</li>
<li><strong>Correctness checked.</strong> The atlas compares each output with a numpy float32 reference using the effective (fp16-rounded or dequantized) weights: <code>max|out - ref| / max|ref|</code>, tolerance 2e-3 for f32 compute and 1.5e-2 for f16. Wrong answers are reported as <em>wrong output</em> and never timed.</li>
<li><strong>p50 and p95.</strong> A prepared <code>InferRequest</code> with its input bound once; only <code>infer()</code> is timed, after warm-up runs. The atlas adapts the run count to about 1 s per config (10 to 200 runs).</li>
<li><strong>Sequential, isolated runs.</strong> All results come from one <code>scripts/record_all.ps1</code> pass that runs each benchmark alone; devices within a demo run one after another. Earlier contended runs on the old driver are archived and excluded.</li>
<li><strong>AC power</strong>, power source recorded in each result file. Driver versions are recorded with every run.</li>
<li><strong>Compile time kept separate</strong> from inference; NPU and GPU compile caches are reported as cold or warm.</li>
</ul>

<h2 id="caveats">Caveats</h2>
<ul>
<li><strong>NPU batch 2-8 convolution bug, still present on driver {E(self.env.get("npu_driver") or "n/a")}.</strong> Batched conv stacks return sample 0's output for every element. A batch-size scan on the previous driver (32.0.100.4724) found batch 2-8 affected and batch 1 and 12 or more correct; on this driver the same {n_bad} batch-8 configs still fail (the sweep tests batch 1, 8 and 32). MLPs are unaffected. Check your outputs if you batch convolutions on this NPU.</li>
<li><strong>Synthetic models in the atlas.</strong> Dense MLP and 3x3 conv stacks show the regimes, but real networks add attention, norms and layout changes. The demos use real but small models.</li>
<li><strong>One laptop.</strong> A 6-thread CPU and a 2-core iGPU, the smallest Panther Lake configuration, with DDR5-5600. Other SKUs, drivers and firmware will differ.</li>
<li><strong>Effective GFLOP/s is not TOPS.</strong> Vendor TOPS are INT8 peaks; INT8 in the atlas is weight-only.</li>
<li><strong>Power was not measured</strong> (Windows reports discharge rate only on battery), so no efficiency claims are made.</li>
<li>LLM output quality: channel-wise int4 Phi-3-mini shows occasional glitches, and devices can produce different text.</li>
</ul>
<p class="small">Extracted numbers behind this page: <a href="data/results.json">data/results.json</a>.</p>
"""
        self.page("results.html", "Results · Panther Lake AI Lab",
                  "Charts, the 162-config performance atlas, methodology and caveats for NPU vs GPU vs CPU on a Core Ultra 5 322.",
                  content)

    # ---- hardware
    def hardware(self):
        t = Template((SITE / "templates" / "hardware.html").read_text(encoding="utf-8"))
        content = t.substitute(repo=REPO, ark=ARK, qrg=QRG, psref=PSREF, cnc=CNC, ref_date="2026-11-11",
                               npu_driver=E(self.env.get("npu_driver") or "n/a"),
                               graphics_driver=E(self.env.get("graphics_driver") or "n/a"),
                               openvino=E(self.env.get("openvino") or "n/a"),
                               openvino_genai=E(self.env.get("openvino_genai") or "n/a"),
                               python=E(self.env.get("python") or "3.12"))
        self.page("hardware.html", "Hardware · Panther Lake AI Lab",
                  "Intel Core Ultra 5 322 (Panther Lake) specs with sources, vendor claims labelled, and driver versions.",
                  content)

    # ---- get started
    def get_started(self):
        runs = "\n".join(f".venv\\Scripts\\python.exe -m examples.{s}" for s, *_ in self.demo_list())
        content = f"""
<h1>Get started</h1>
<p class="lede">Everything runs on Windows 11 with an Intel Core Ultra NPU. The examples were developed on a Core Ultra 5 322; other Core Ultra chips should work but are untested.</p>

<h2>1. Drivers</h2>
<p>Install current Intel drivers first: NPU driver <strong>32.0.100.5540 or later</strong> (OpenVINO 2026.4 needs it on Core Ultra Series 3) and graphics driver 32.0.101.9175 or later.
Step-by-step instructions and download links: <a href="{REPO}/blob/main/docs/DRIVER_SETUP.md">docs/DRIVER_SETUP.md</a>.</p>

<h2>2. Clone and install (PowerShell)</h2>
<p>You need Python 3.12 (<code>winget install -e --id Python.Python.3.12</code> also installs the <code>py</code> launcher).</p>
<pre><code>git clone {REPO}.git
cd panther-lake-ai-lab
py -3.12 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt -r requirements-examples.txt</code></pre>

<h2>3. Check your devices</h2>
<pre><code>.venv\\Scripts\\python.exe scripts\\detect_devices.py</code></pre>
<p>You should see <code>CPU</code>, <code>GPU</code> and <code>NPU</code>. A device being listed does not prove a model will compile or run correctly on it; the demos check that themselves.</p>

<h2>4. Fetch the models</h2>
<pre><code>.venv\\Scripts\\python.exe scripts\\fetch_models.py               # about 2 GB, mostly the LLM
.venv\\Scripts\\python.exe scripts\\fetch_models.py --only vision # or depth, llm</code></pre>
<p>Models go to <code>models/</code>. Stock clips download to <code>assets/clips/</code> on first use. Both are gitignored.</p>

<h2>5. Run the demos</h2>
<pre><code>{E(runs)}</code></pre>
<p>Each one opens a live window (press <kbd>q</kbd> or <kbd>Esc</kbd> to quit) and writes a results JSON to <code>benchmarks/results/</code>. Useful flags:</p>
<ul>
<li><code>--devices NPU CPU</code> limits the devices; <code>--no-display --record</code> records a GIF and MP4 headless.</li>
<li><code>python -m examples.perf_atlas --quick</code> runs a 1.5-minute subset of the sweep.</li>
<li><code>python -m examples.llm_chat --bench</code> runs the fixed LLM benchmark on all three devices.</li>
<li>The first NPU compile can take from seconds to a few minutes; compiled blobs are cached under <code>models/</code>.</li>
</ul>
<p>To re-run every benchmark and re-record all media one at a time: <code>scripts\\record_all.ps1</code>.</p>

<h2>6. Rebuild this site</h2>
<pre><code>.venv\\Scripts\\python.exe site\\build.py     # refreshes site/data/results.json from your results
cd site\\_build; python -m http.server 8000</code></pre>
<p>Publish your own numbers only with the driver versions and power state recorded, and review hardware inventories for personal details first.</p>
"""
        self.page("get-started.html", "Get started · Panther Lake AI Lab",
                  "Clone, install Python 3.12 and Intel drivers, fetch models and run the NPU demos.", content)

    def not_found(self):
        self.page("404.html", "Not found · Panther Lake AI Lab", "Page not found.",
                  '<h1>Page not found</h1><p><a href="index.html">Back to the home page</a>.</p>')


# --------------------------------------------------------------------------- main

def copy_assets(site: Site, out: Path):
    shutil.copytree(SITE / "static", out / "static")
    (out / "media").mkdir()
    for p in sorted(MEDIA.glob("*.gif")) + sorted(MEDIA.glob("*.png")):
        shutil.copy2(p, out / "media" / p.name)
    for stem, *_ in CHARTS_SPEC:
        src = CHARTS / f"{stem}.png"
        site.charts[stem] = src.exists()
        if src.exists():
            (out / "media" / "charts").mkdir(exist_ok=True)
            shutil.copy2(src, out / "media" / "charts" / src.name)
        else:
            site.warnings.append(f"chart missing: media/charts/{stem}.png")
    # MP4s are gitignored, so they only exist in local builds. Copy small ones only.
    mp4dir = MEDIA / "mp4"
    if mp4dir.exists():
        for p in sorted(mp4dir.glob("*.mp4")):
            if p.stat().st_size < MP4_LIMIT:
                (out / "media" / "mp4").mkdir(exist_ok=True)
                shutil.copy2(p, out / "media" / "mp4" / p.name)
                site.mp4s.add(p.stem)
    (out / "data").mkdir()
    shutil.copy2(DATA_FILE, out / "data" / "results.json")
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-data", action="store_true",
                    help="use the committed site/data/results.json instead of benchmarks/results (CI)")
    ap.add_argument("--out", type=Path, default=SITE / "_build")
    a = ap.parse_args(argv)

    if not a.from_data:
        if (RESULTS / "perf_atlas.json").exists():
            data = extract()
            DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            DATA_FILE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
            print(f"wrote {DATA_FILE.relative_to(ROOT)}")
        else:
            print("benchmarks/results not found; falling back to site/data/results.json", file=sys.stderr)
    if not DATA_FILE.exists():
        print("error: site/data/results.json is missing; run build.py locally after record_all.ps1", file=sys.stderr)
        return 1
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))

    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    site = Site(data, out)
    copy_assets(site, out)
    site.home(); site.demos(); site.results(); site.hardware(); site.get_started(); site.not_found()

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({total / 1e6:.1f} MB, {sum(1 for _ in out.rglob('*.html'))} pages, "
          f"mp4: {', '.join(sorted(site.mp4s)) or 'none'})")
    for w in site.warnings:
        print("warning:", w, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
