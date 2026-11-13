"""Local LLM chat on the Intel AI Boost NPU (Phi-3-mini int4, OpenVINO GenAI).

    python -m examples.llm_chat                       # interactive streaming chat on the NPU
    python -m examples.llm_chat --device CPU          # ... or on CPU / GPU
    python -m examples.llm_chat --bench               # fixed prompts on NPU, CPU, GPU (sequentially)
    python -m examples.llm_chat --bench --cold        # ... clearing the NPU/GPU compile caches first
    python -m examples.llm_chat --render              # race-replay video + chart from the bench JSON
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

from ..common import RESULTS_DIR, ovutil
from . import llm

PROMPTS = [
    {"id": "npu_vs_gpu", "title": "Explain NPU vs GPU",
     "text": "Explain to a high-school student the difference between an NPU and a GPU in a laptop, "
             "in one short paragraph."},
    {"id": "haiku", "title": "Haiku",
     "text": "Write a haiku about a laptop chip called Panther Lake."},
    {"id": "summary", "title": "Summarize",
     "text": "Summarize in two sentences: Modern laptop processors combine three kinds of compute engines. "
             "The CPU handles general tasks and quick decisions, the integrated GPU handles graphics and highly "
             "parallel math, and the NPU is a dedicated low-power accelerator for neural networks. Running AI "
             "features such as background blur, transcription or a local chatbot on the NPU keeps the CPU and "
             "GPU free and can greatly extend battery life, although large models are still limited by memory "
             "bandwidth."},
    {"id": "python", "title": "Python function",
     "text": "Write a short Python function that returns the n-th Fibonacci number iteratively. Code only."},
]
DEFAULT_JSON = RESULTS_DIR / "llm_chat.json"


# --------------------------------------------------------------------------------------------- interactive chat
def chat(model_dir: Path, device: str, max_new_tokens: int):
    tty = sys.stdout.isatty()
    if tty and sys.platform == "win32":
        import os

        os.system("")  # enable ANSI escape processing in legacy Windows consoles
    bold, dim, reset = ("\033[1m", "\033[2m", "\033[0m") if tty else ("", "", "")
    print(f"Loading {model_dir.name} on {device} ...", flush=True)
    if device == "NPU" and not llm.cache_blobs("NPU"):
        print("  First run on the NPU: the model is compiled for the NPU once (1-4 minutes on this laptop)\n"
              f"  and cached in {llm.CACHE_DIRS['NPU']}. Later starts load the cached blob in seconds.", flush=True)
    pipe, info = llm.load(model_dir, device)
    print(f"  loaded in {info['load_s']:.1f} s (compile cache: {info['cache']})", flush=True)
    tok = pipe.get_tokenizer()
    t0 = time.perf_counter()  # first inference allocates buffers / JITs kernels; keep it out of the first reply
    pipe.generate(llm.chat_prompt(tok, [{"role": "user", "content": "Hi"}]), llm.generation_config(1))
    print(f"  warm-up {time.perf_counter() - t0:.1f} s", flush=True)
    print("Type a message. Commands: /reset clears history, /exit quits.\n", flush=True)
    history: list[dict] = []
    limit = llm.NPU_MAX_PROMPT_LEN - 16 if device == "NPU" else 3000
    while True:
        try:
            user = input(f"{bold}you>{reset} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            history.clear()
            print("(history cleared)\n")
            continue
        history.append({"role": "user", "content": user})
        prompt = llm.chat_prompt(tok, history)
        n_prompt = llm.count_tokens(tok, prompt)
        while n_prompt > limit and len(history) > 1:  # drop oldest turns to fit the NPU's static prompt window
            history = history[2:] if len(history) > 2 else history[-1:]
            prompt = llm.chat_prompt(tok, history)
            n_prompt = llm.count_tokens(tok, prompt)
        if n_prompt > limit:
            print(f"(message is {n_prompt} tokens; the {device} pipeline accepts at most {limit})\n")
            history.pop()
            continue
        print(f"{bold}{device.lower()}>{reset} ", end="", flush=True)
        streamer = llm.TimedStreamer(tok, on_text=lambda s: print(s, end="", flush=True))
        streamer.start()
        try:
            pipe.generate(prompt, llm.generation_config(max_new_tokens), streamer)
        except KeyboardInterrupt:
            print(" [interrupted]")
        st = streamer.stats()
        history.append({"role": "assistant", "content": streamer.text.strip()})
        print(f"\n{dim}[{device} | prompt {n_prompt} tok | TTFT {st.get('ttft_ms', 0):.0f} ms | "
              f"{st.get('tok_s', 0):.1f} tok/s | {st['new_tokens']} tok | "
              f"host CPU {st['cpu_cores_busy']:.1f} cores busy]{reset}\n", flush=True)


# ------------------------------------------------------------------------------------------------------ bench
def bench_device(model_dir: Path, device: str, max_new_tokens: int, cold: bool, previous: dict | None) -> dict:
    rec = {"device": device, "status": "skipped"}
    if not ovutil.available(device):
        rec["reason"] = "device not enumerated"
        return rec
    try:
        rec["full_device_name"] = ovutil.core().get_property(device, "FULL_DEVICE_NAME")
    except Exception:
        pass
    if cold:
        n = llm.clear_cache(device)
        llm.log(f"[{device}] cleared {n} cached blob(s)")
    try:
        llm.log(f"[{device}] loading (first load) ...")
        pipe, info1 = llm.load(model_dir, device)
        llm.log(f"[{device}] first load {info1['load_s']} s, cache={info1['cache']}")
        rec["load"] = {"first": info1}
        if info1["cache"] == "cold":
            rec["load"]["cold_s"] = info1["load_s"]
        elif previous and previous.get("load", {}).get("cold_s"):
            rec["load"]["cold_s"] = previous["load"]["cold_s"]
            rec["load"]["cold_from"] = previous.get("load", {}).get("cold_from") or previous.get("run_at")
        if device in llm.CACHE_DIRS:  # reload to time a warm-cache start
            llm.unload(pipe)
            pipe, info2 = llm.load(model_dir, device)
            rec["load"]["cached_s"] = info2["load_s"]
            llm.log(f"[{device}] reload from cache {info2['load_s']} s")
        else:
            rec["load"]["cached_s"] = None
        rec["load"]["no_cache_s"] = info1["load_s"] if device not in llm.CACHE_DIRS else None
        rec["peak_rss_gb_after_load"] = llm.peak_rss_gb()
        rec["run_at"] = datetime.now().isoformat(timespec="seconds")
        tok = pipe.get_tokenizer()
        warm = llm.TimedStreamer(tok)
        warm.start()
        pipe.generate(llm.chat_prompt(tok, [{"role": "user", "content": "Say hello."}]),
                      llm.generation_config(8), warm)
        rec["prompts"] = []
        for p in PROMPTS:
            prompt = llm.chat_prompt(tok, [{"role": "user", "content": p["text"]}])
            streamer = llm.TimedStreamer(tok)
            streamer.start()
            result = pipe.generate([prompt], llm.generation_config(max_new_tokens), streamer)
            st = streamer.stats()
            pm = result.perf_metrics
            entry = {"id": p["id"], "prompt_tokens": int(pm.get_num_input_tokens()), **st,
                     "genai_ttft_ms": round(pm.get_ttft().mean, 1),
                     "genai_tpot_ms": round(pm.get_tpot().mean, 2),
                     "genai_tok_s": round(pm.get_throughput().mean, 2),
                     "text": streamer.text.strip(),
                     "token_times_ms": [round(e["t"] * 1000, 1) for e in streamer.events],
                     "token_text": [e["text"] for e in streamer.events]}
            rec["prompts"].append(entry)
            llm.log(f"[{device}] {p['id']:<11} prompt {entry['prompt_tokens']:>3} tok | TTFT {st.get('ttft_ms')} ms | "
                    f"{st.get('tok_s')} tok/s | {st['new_tokens']} tok | CPU {st['cpu_cores_busy']} cores")
        rec["peak_rss_gb"] = llm.peak_rss_gb()
        rec["status"] = "executed"
        # LLMPipeline does not expose the compiled model's EXECUTION_DEVICES. The NPU/GPU pipelines raise on
        # compile failure rather than falling back, and cpu_cores_busy shows where the work actually ran.
        rec["execution_device_note"] = "requested device; genai does not expose EXECUTION_DEVICES (see cpu_cores_busy)"
        llm.unload(pipe)
    except Exception as exc:
        rec.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:1500])
        llm.log(f"[{device}] FAILED: {exc}")
    return rec


def power_state() -> dict:
    try:
        from ..perf_atlas.sweep import power_state as ps

        return ps()
    except Exception:
        return {}


def summarize(rec: dict) -> dict:
    import statistics

    ps = rec.get("prompts") or []
    if not ps:
        return {}
    return {"ttft_ms_median": round(statistics.median(p["ttft_ms"] for p in ps), 1),
            "tok_s_median": round(statistics.median(p["tok_s"] for p in ps if p.get("tok_s")), 2),
            "cpu_cores_busy_median": round(statistics.median(p["cpu_cores_busy"] for p in ps), 2)}


def bench(model_dir: Path, devices: list[str], max_new_tokens: int, cold: bool, out: Path):
    previous = {}
    if out.exists():
        try:
            previous = {d["device"]: d for d in json.loads(out.read_text(encoding="utf-8")).get("devices", [])}
        except Exception:
            previous = {}
    doc = {"meta": {"created": datetime.now().isoformat(timespec="seconds"), "machine": platform.processor(),
                    "versions": llm.versions(), "model": llm.model_meta(model_dir), "power": power_state(),
                    "max_new_tokens": max_new_tokens, "decoding": "greedy",
                    "timing": "TTFT = generate() call to first streamed token (tokenize + prefill); "
                              "tok/s = (n-1)/(t_last - t_first); devices run sequentially, never concurrently",
                    "npu_config": {"MAX_PROMPT_LEN": llm.NPU_MAX_PROMPT_LEN,
                                   "MIN_RESPONSE_LEN": llm.NPU_MIN_RESPONSE_LEN}},
           "prompts": PROMPTS, "devices": []}
    # devices not re-run keep their previous results (same prompts / token budget only), so one device can be redone
    if previous and all(d.get("prompts") is None or [q["id"] for q in d["prompts"]] == [q["id"] for q in PROMPTS]
                        for d in previous.values()):
        doc["devices"] = [d for name, d in previous.items() if name not in devices]
        doc["meta"]["merged_from_previous"] = [d["device"] for d in doc["devices"]]
    for dev in devices:
        rec = bench_device(model_dir, dev, max_new_tokens, cold and dev in ("NPU", "GPU"), previous.get(dev))
        rec["summary"] = summarize(rec)
        doc["devices"].append(rec)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")  # save after each device
    summary = {d["device"]: {"status": d["status"], **d.get("summary", {}),
                             "load_cold_s": d.get("load", {}).get("cold_s"),
                             "load_cached_s": d.get("load", {}).get("cached_s"),
                             "load_s": d.get("load", {}).get("first", {}).get("load_s")} for d in doc["devices"]}
    print(json.dumps(summary, indent=1))
    llm.log(f"wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m examples.llm_chat", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="NPU", help="device for interactive chat (NPU, CPU, GPU)")
    ap.add_argument("--model", type=Path, default=llm.DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default 256 chat / 160 bench")
    ap.add_argument("--bench", action="store_true", help="run the fixed prompt set on --devices")
    ap.add_argument("--devices", nargs="+", default=["NPU", "CPU", "GPU"])
    ap.add_argument("--cold", action="store_true", help="clear NPU/GPU compile caches first to time a cold start")
    ap.add_argument("--render", action="store_true", help="render race-replay video + chart from the bench JSON")
    ap.add_argument("--race-prompt", default="npu_vs_gpu", help="prompt id shown in the race replay")
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args(argv)
    devices = [d.upper() for d in args.devices]
    if args.bench:
        n = args.max_new_tokens or 160
        if "NPU" in devices and n > llm.NPU_MIN_RESPONSE_LEN:
            sys.exit(f"--max-new-tokens must be <= {llm.NPU_MIN_RESPONSE_LEN} for the NPU pipeline")
        bench(args.model, devices, n, args.cold, args.json)
    if args.render:
        from . import render

        render.render_all(args.json, args.race_prompt)
    if not (args.bench or args.render):
        n = args.max_new_tokens or 256
        chat(args.model, args.device.upper(), n)


if __name__ == "__main__":
    main()
