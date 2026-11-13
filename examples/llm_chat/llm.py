"""OpenVINO GenAI helpers: pipeline loading with compile-cache bookkeeping, and a timestamping token streamer."""
import ctypes
import gc
import re
import sys
import time
from pathlib import Path

import openvino_genai as og

from ..common import MODELS_DIR

DEFAULT_MODEL = MODELS_DIR / "phi-3-mini-4k-instruct-int4-cw-ov"
MODEL_INFO = {
    "phi-3-mini-4k-instruct-int4-cw-ov": {
        "hf_id": "OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov",
        "base_model": "microsoft/Phi-3-mini-4k-instruct",
        "license": "MIT",
        "params": "3.8B",
    },
}
# Static shapes the NPU pipeline is compiled for. Prompt (incl. chat history) must fit MAX_PROMPT_LEN.
NPU_MAX_PROMPT_LEN = 1024
NPU_MIN_RESPONSE_LEN = 256
CACHE_DIRS = {"NPU": MODELS_DIR / "npucache", "GPU": MODELS_DIR / "gpucache"}


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


def model_meta(model_dir: Path) -> dict:
    """Model id, licence and the NNCF weight-compression parameters recorded in the IR's rt_info."""
    meta = {"name": model_dir.name, **MODEL_INFO.get(model_dir.name, {})}
    xml = model_dir / "openvino_model.xml"
    try:
        with open(xml, "rb") as f:
            f.seek(max(0, xml.stat().st_size - 20000))
            tail = f.read().decode("utf-8", "ignore")
        nncf = re.search(r"<nncf>(.*?)</nncf>", tail, re.S)
        if nncf:
            for key in ("mode", "group_size", "ratio", "backup_mode"):
                m = re.search(rf'<{key} value="([^"]*)"', nncf.group(1))
                if m:
                    meta[f"weight_{key}"] = m.group(1)
    except OSError:
        pass
    bin_path = model_dir / "openvino_model.bin"
    if bin_path.exists():
        meta["weights_gb"] = round(bin_path.stat().st_size / 1e9, 2)
    return meta


def versions() -> dict:
    import openvino as ov

    return {"openvino": ov.get_version(), "openvino_genai": og.__version__, "python": sys.version.split()[0]}


def pipeline_config(device: str) -> dict:
    cfg = {}
    if device == "NPU":
        cfg.update(MAX_PROMPT_LEN=NPU_MAX_PROMPT_LEN, MIN_RESPONSE_LEN=NPU_MIN_RESPONSE_LEN)
    if device in CACHE_DIRS:
        CACHE_DIRS[device].mkdir(parents=True, exist_ok=True)
        cfg["CACHE_DIR"] = str(CACHE_DIRS[device])
    return cfg


def cache_blobs(device: str) -> set:
    d = CACHE_DIRS.get(device)
    return {p.name for p in d.glob("*.blob")} if d and d.exists() else set()


def clear_cache(device: str) -> int:
    d = CACHE_DIRS.get(device)
    n = 0
    for p in (d.glob("*.blob") if d and d.exists() else []):
        p.chmod(0o666)
        p.unlink()
        n += 1
    return n


def peak_rss_gb() -> float | None:
    """Peak working set of this process (Windows), in GB. Includes the model weights mapped in."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return round(getattr(info, "peak_wset", info.rss) / 1e9, 2)
    except Exception:
        return None


def load(model_dir: Path, device: str) -> tuple:
    """Load an LLMPipeline. Returns (pipe, info) where info says whether the compile cache was cold or warm."""
    before = cache_blobs(device)
    start = time.perf_counter()
    pipe = og.LLMPipeline(str(model_dir), device, **pipeline_config(device))
    load_s = time.perf_counter() - start
    if device not in CACHE_DIRS:
        state = "no-cache"
    else:
        state = "cached" if before else "cold"  # blobs present before load => served from cache
    return pipe, {"device": device, "load_s": round(load_s, 2), "cache": state,
                  "cache_dir": str(CACHE_DIRS[device]) if device in CACHE_DIRS else None,
                  "config": {k: v for k, v in pipeline_config(device).items() if k != "CACHE_DIR"}}


def unload(pipe):
    del pipe
    gc.collect()


class TimedStreamer(og.StreamerBase):
    """Records a perf_counter timestamp for every generated token and emits decoded text deltas.

    Decodes the whole response so far each step (skip_special_tokens=True) and emits only the new suffix, which
    handles SentencePiece leading spaces correctly. A trailing U+FFFD (incomplete UTF-8) is held back until the
    next token completes it.
    """

    def __init__(self, tokenizer, on_text=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.ids: list[int] = []
        self.events: list[dict] = []  # {"t": seconds since start(), "id": token id, "text": emitted delta}
        self.text = ""
        self.t0 = None
        self.cpu0 = None

    def start(self):
        self.t0 = time.perf_counter()
        self.cpu0 = time.process_time()

    def _push(self, token: int):
        now = time.perf_counter()
        self.ids.append(int(token))
        full = self.tokenizer.decode(self.ids, skip_special_tokens=True)
        delta = ""
        if not full.endswith("�") and full.startswith(self.text):
            delta, self.text = full[len(self.text):], full
        elif not full.endswith("�"):  # decode changed earlier text (rare); resync
            delta, self.text = full[len(self.text):] if len(full) > len(self.text) else "", full
        self.events.append({"t": now - self.t0, "id": int(token), "text": delta})
        if delta and self.on_text:
            self.on_text(delta)

    def write(self, token):
        if isinstance(token, (list, tuple)):
            for t in token:
                self._push(t)
        else:
            self._push(token)
        return og.StreamingStatus.RUNNING

    def end(self):
        pass

    def stats(self) -> dict:
        """TTFT (from generate() call, incl. tokenization+prefill), decode tok/s, CPU cores used during the call."""
        wall = time.perf_counter() - self.t0
        cpu = time.process_time() - self.cpu0
        ev = self.events
        out = {"new_tokens": len(ev), "wall_s": round(wall, 3), "cpu_cores_busy": round(cpu / wall, 2) if wall else None}
        if ev:
            out["ttft_ms"] = round(ev[0]["t"] * 1000, 1)
            if len(ev) > 1 and ev[-1]["t"] > ev[0]["t"]:
                out["tok_s"] = round((len(ev) - 1) / (ev[-1]["t"] - ev[0]["t"]), 2)
        return out


def generation_config(max_new_tokens: int) -> og.GenerationConfig:
    cfg = og.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False  # greedy: identical prompt -> comparable token counts across devices
    cfg.apply_chat_template = False  # we apply the model's own template explicitly
    return cfg


def chat_prompt(tokenizer, messages: list[dict]) -> str:
    """Render messages with the model's built-in chat template (Phi-3: <|user|> ... <|end|><|assistant|>)."""
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def count_tokens(tokenizer, text: str) -> int:
    return int(tokenizer.encode(text, add_special_tokens=False).input_ids.get_shape()[-1])
