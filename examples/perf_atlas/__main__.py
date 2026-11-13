"""NPU Performance Atlas: sweep synthetic OpenVINO models over CPU / GPU / NPU and render a dashboard.

Run:   .venv\\Scripts\\python.exe -m examples.perf_atlas            (full grid, ~10-15 min)
       .venv\\Scripts\\python.exe -m examples.perf_atlas --quick    (small grid, < 2 min)
       .venv\\Scripts\\python.exe -m examples.perf_atlas --plot-only benchmarks/results/perf_atlas.json

Writes the full results JSON to --output, media/<stem>.png + media/<stem>.gif + media/mp4/<stem>.mp4,
progress to stderr and a JSON summary to stdout. Devices are never substituted: a device that is not
enumerated is recorded as "skipped", a compile/infer error as "failed", a wrong answer as "failed_correctness".
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

from examples.common import MEDIA_DIR, REPO_ROOT, RESULTS_DIR, ovutil
from examples.perf_atlas import sweep

FULL = dict(name="full", models=list(sweep.SPECS), batches=[1, 8, 32], precisions=list(sweep.PRECISIONS),
            budget_s=1.0, max_runs=200, async_budget_s=2.0,
            async_cfgs=[("MLP-L", 1, "FP16"), ("MLP-L", 1, "INT8"), ("MLP-L", 32, "FP16"),
                        ("CONV-M", 1, "FP16"), ("CONV-M", 32, "FP16")])
QUICK = dict(name="quick", models=["MLP-S", "MLP-M", "CONV-XS"], batches=[1, 8], precisions=list(sweep.PRECISIONS),
             budget_s=0.25, max_runs=60, async_budget_s=0.75, async_cfgs=[("MLP-M", 8, "FP16")])


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def render(data: dict, stem: str, gif: bool = True) -> dict:
    from examples.common.media import write_gif, write_mp4
    from examples.perf_atlas import charts
    import matplotlib.pyplot as plt
    from PIL import Image

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    png = MEDIA_DIR / f"{stem}.png"
    fig, found = charts.dashboard(data, png, dpi=100)
    out = {"png": _rel(png)}
    if gif:
        frames = charts.progress_frames(data)
        if frames:
            h, w = frames[0].shape[:2]
            final = Image.fromarray(charts.fig_to_rgb(fig)).convert("RGB")
            final = final.resize((w, round(final.height * w / final.width)), Image.LANCZOS)
            canvas = Image.new("RGB", (w, max(h, final.height)), charts.SURFACE)
            canvas.paste(final, (0, (canvas.height - final.height) // 2))
            final_arr = np.asarray(canvas.resize((w, h), Image.LANCZOS))
            seq = frames + [frames[-1]] * 8 + [final_arr] * 30
            out["gif"] = _rel(write_gif(seq, MEDIA_DIR / f"{stem}.gif", fps=6, width=960))
            out["gif_mb"] = round((MEDIA_DIR / f"{stem}.gif").stat().st_size / 1e6, 2)
            out["mp4"] = _rel(write_mp4(seq, MEDIA_DIR / "mp4" / f"{stem}.mp4", fps=6))
    plt.close(fig)
    out["crossovers_fp16"] = [{**c, "summary": charts._crossover_text(c)} for c in found]
    return out


def summarize(data: dict) -> dict:
    rows = data["results"]
    by_status = {}
    for r in rows:
        key = f"{r['device']}:{r['status']}"
        by_status[key] = by_status.get(key, 0) + 1
    wins = {}
    for other in ("CPU", "GPU"):
        ratios = []
        for r in rows:
            if r["device"] != "NPU" or r["status"] != "executed":
                continue
            o = [x for x in rows if x["device"] == other and x["status"] == "executed" and
                 (x["model"], x["batch"], x["precision"]) == (r["model"], r["batch"], r["precision"])]
            if o:
                ratios.append((round(o[0]["median_ms"] / r["median_ms"], 3), r["model"], r["batch"], r["precision"]))
        if ratios:
            ratios.sort()
            wins[other] = {"npu_faster_in": sum(x[0] > 1 for x in ratios), "of": len(ratios),
                           "best_npu_speedup": ratios[-1], "worst_npu_speedup": ratios[0]}
    peak = {}
    for d in {r["device"] for r in rows}:
        ok = [r for r in rows if r["device"] == d and r["status"] == "executed"]
        if ok:
            b = max(ok, key=lambda r: r["gflops_effective"])
            peak[d] = {"gflops": b["gflops_effective"], "at": f"{b['model']} b{b['batch']} {b['precision']}"}
    return {"status_counts": by_status, "npu_vs": wins, "peak_effective_gflops": peak,
            "async": [{k: r.get(k) for k in ("model", "batch", "precision", "device", "status", "nireq",
                                             "samples_per_s", "error")} for r in data.get("async", [])]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    p.add_argument("--quick", action="store_true", help="small grid, < 2 minutes")
    p.add_argument("--output", type=Path, default=None,
                   help="results JSON (default benchmarks/results/perf_atlas.json, or perf_atlas_quick.json with --quick)")
    p.add_argument("--plot-only", type=Path, metavar="JSON", help="render charts/media from an existing results JSON")
    p.add_argument("--no-gif", action="store_true", help="skip the animated GIF / MP4")
    a = p.parse_args()

    if a.plot_only:
        data = json.loads(a.plot_only.read_text(encoding="utf-8"))
        media = render(data, a.plot_only.stem, gif=not a.no_gif)
        print(json.dumps({"media": media, **summarize(data)}, indent=2))
        return

    grid = QUICK if a.quick else FULL
    output = a.output or RESULTS_DIR / ("perf_atlas_quick.json" if a.quick else "perf_atlas.json")
    devices = [d.upper() for d in a.devices]
    meta = {
        "tool": "examples.perf_atlas", "grid_name": grid["name"], "date": dt.date.today().isoformat(),
        "openvino_version": ov.__version__, "available_devices": list(ovutil.core().available_devices),
        "devices": devices, "device_names": sweep.device_names(), "power": sweep.power_state(),
        "seed": sweep.SEED, "grid": {k: v for k, v in grid.items() if k != "name"},
        "models": {n: {**s.__dict__, "gflop_per_sample": s.gflop_per_sample, "params_m": s.params_m}
                   for n, s in sweep.SPECS.items() if n in grid["models"]},
        "precision_notes": {
            "FP32": "fp32 weights; CPU/GPU INFERENCE_PRECISION_HINT=f32; NPU default (computes f16)",
            "FP16": "fp16 weights (+Convert); INFERENCE_PRECISION_HINT=f16 on all devices",
            "INT8": "weight-only symmetric per-channel int8 (i8->Convert->Multiply), f16 activations; "
                    "no activation quantization / FakeQuantize; NNCF not used",
        },
        "output": _rel(output),
    }
    sweep.log(f"perf_atlas {grid['name']} grid on {devices} | OpenVINO {ov.__version__} | power {meta['power']}")
    t0 = time.perf_counter()
    async_set = {tuple(c) for c in grid["async_cfgs"]}
    results, async_rows = sweep.run_sweep(
        [sweep.SPECS[m] for m in grid["models"]], grid["batches"], grid["precisions"], devices,
        grid["budget_s"], grid["max_runs"], async_set, grid["async_budget_s"])
    meta["sweep_seconds"] = round(time.perf_counter() - t0, 1)
    data = {"meta": meta, "results": results, "async": async_rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=1), encoding="utf-8")
    sweep.log(f"sweep done in {meta['sweep_seconds']} s -> {output}; rendering charts...")
    media = render(data, output.stem, gif=not a.no_gif)
    print(json.dumps({"output": _rel(output), "sweep_seconds": meta["sweep_seconds"], "media": media,
                      **summarize(data)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
