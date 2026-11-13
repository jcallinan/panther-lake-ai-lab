# Local LLM chat on the NPU

A 3.8B-parameter chat model (Microsoft Phi-3-mini, int4) running **entirely on the Intel AI Boost NPU** of a
Core Ultra 5 322 (Panther Lake) laptop, with OpenVINO GenAI. Tokens stream into the terminal as they are generated,
and a benchmark mode runs the same prompts on the NPU, CPU and iGPU so the three can be compared.

![Race replay: the same prompt on NPU, GPU and CPU, replayed at measured per-token timing](../../media/llm_chat.gif)

The video is a **replay of sequential runs at measured timing**. Each device ran on its own, one after another,
never concurrently. Every token appears at the moment it was actually streamed, measured from that run's
`generate()` call. Static chart: [`media/llm_chat.png`](../../media/llm_chat.png).

![Decode speed, time to first token, host CPU load and load time per device](../../media/llm_chat.png)

## How to run

```powershell
.venv\Scripts\python.exe -m examples.llm_chat                    # interactive chat on the NPU
.venv\Scripts\python.exe -m examples.llm_chat --device CPU       # ... or CPU / GPU
.venv\Scripts\python.exe -m examples.llm_chat --bench            # 4 fixed prompts on NPU, CPU, GPU (sequential)
.venv\Scripts\python.exe -m examples.llm_chat --bench --cold     # clear NPU/GPU compile caches first (times a cold start)
.venv\Scripts\python.exe -m examples.llm_chat --render           # video + chart from benchmarks/results/llm_chat.json
.venv\Scripts\python.exe -m examples.llm_chat --bench --devices NPU CPU --render
```

The model must be in `models/phi-3-mini-4k-instruct-int4-cw-ov` (about 2.1 GB), for example:

```powershell
.venv\Scripts\huggingface-cli.exe download OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov --local-dir models\phi-3-mini-4k-instruct-int4-cw-ov
```

**First run on the NPU.** The NPU compiles the model once, which took about 1 to 3.5 minutes on this laptop
(see below). The compiled blob is cached in `models/npucache/` (gitignored, about 240 MB), and later starts load in
about 2 to 3 s. The chat prints whether the cache was cold or warm, and how long loading took. GPU uses
`models/gpucache/` the same way. CPU needs no compile step.

**Chat commands:** `/reset` clears the history, `/exit` quits. Ctrl+C stops a reply. After each reply a stats line
shows the device, prompt tokens, time to first token (TTFT), decode tok/s, generated tokens and how many host CPU
cores were busy.

## What it measures

* The prompt is formatted with the model's **own chat template** (`Tokenizer.apply_chat_template`, giving Phi-3's
  `<|user|> … <|end|><|assistant|>`). Special tokens are stripped when decoding (`skip_special_tokens=True`).
* A `StreamerBase` subclass timestamps **every token** (`time.perf_counter`) and decodes the text incrementally.
* **TTFT** runs from the `generate()` call to the first streamed token, so it includes tokenization and prefill.
  **Decode tok/s** is `(n − 1) / (t_last − t_first)`. OpenVINO GenAI's own `perf_metrics` values (TTFT, TPOT,
  throughput) are recorded alongside.
* **Host CPU busy** is the process CPU time divided by wall time during `generate()`. It is the evidence for *where*
  the work ran. `LLMPipeline` does not expose the compiled model's `EXECUTION_DEVICES`. The NPU pipeline fails
  instead of falling back, and its run keeps about 0.1 CPU cores busy, against about 2 cores for the CPU run.
* `--bench`: greedy decoding, up to 160 new tokens, one warm-up generation, 4 prompts (explain NPU vs GPU to a
  student, a haiku about Panther Lake, summarize a paragraph, a Python Fibonacci function). For each device the
  pipeline is loaded twice: once for the cold or cached first load, then a reload to time a warm cache start.
* `benchmarks/results/llm_chat.json` holds versions, model id, licence and quantization (read from the IR's
  `rt_info`), power source, load times (cold and cached), per-prompt TTFT, tok/s, prompt tokens, the output text,
  and per-token timestamps and text.
* `--render` writes `media/llm_chat.gif` (< 5 MB, full 1280×720, one shared palette so only changed pixels are
  stored), `media/mp4/llm_chat.mp4` (gitignored) and `media/llm_chat.png`. `--race-prompt` selects which prompt
  the video replays.

## Results on this machine

Run on 2026-11-11 with OpenVINO / GenAI 2026.4.1 and NPU driver 32.0.100.4724, on AC power. Values are medians over
the 4 prompts. Other agents were using the NPU, GPU and CPU on the same machine at the time.

| Device | Decode tok/s | TTFT | Load: cold compile | Load: cached | Host CPU busy |
|---|---|---|---|---|---|
| **NPU** (AI Boost) | **22.6** (22.5–23.1) | 693 ms | 62 s | 3.1 s | **0.2 cores** |
| GPU (Xe3 iGPU) | 17.9 (16.4–18.5) | **133 ms** | 228 s | 2.3 s | 0.9 cores |
| CPU | 11.5 (8.9–13.7) | 666 ms (1.9 s for a 102-token prompt) | n/a | 1.8 s (no compile) | 1.4 cores |

* **The NPU decodes fastest and leaves the CPU nearly idle.** Its decode speed barely moves between prompts. In the
  race replay it finishes 160 tokens in 7.8 s, against 8.0 s for the GPU (147 tokens) and 18.4 s for the CPU.
* **The GPU has by far the best TTFT** (prefill is compute-bound). The NPU's TTFT stays at about 0.7 s whatever the
  prompt length (16 to 102 tokens), which is consistent with the NPU running a fixed-shape (static) prefill graph.
* **The compile cache is essential.** A cold load takes 62 s on the NPU (204 s in an earlier, more contended first
  run) and 228 s on the GPU. Both load from cache in 2 to 3 s.
* The CPU varies most because it competes with everything else on the machine. An earlier, quieter run gave
  16.1 tok/s and 447 ms TTFT. A first full run produced a 0.22 tok/s outlier (the process was stalled) and was
  redone with `--bench --devices CPU`, which keeps the other devices' results.
* For comparison, the asymmetric-int4 Qwen3-0.6B (6× fewer parameters) gets about 31 tok/s on the CPU. It cannot
  run on the NPU (see below).

## Why symmetric int4 matters on the NPU

The NPU LLM pipeline (NPUW) compiles the model into fixed-shape prefill and decode graphs, and it supports int4
weights only in **symmetric** form: channel-wise (`group_size = -1`, as in this model, `int4_sym` per the IR's
NNCF `rt_info`) or symmetric grouped. This repo's `qwen3-0.6b-int4-ov` is `INT4_ASYM` with group size 128. It runs on
CPU (31 tok/s) and GPU, but its NPU compile hung for over 20 minutes. Look for OpenVINO-org models whose names end in
`-int4-cw-ov`, or export with `optimum-cli export openvino --weight-format int4 --sym --group-size -1`.
(The upstream model card says `int4_asym`, but the IR's own `rt_info` says `int4_sym`, `group_size -1`.)

The NPU pipeline is compiled for `MAX_PROMPT_LEN=1024` and `MIN_RESPONSE_LEN=256`. In chat mode the oldest turns
are dropped when the templated history would exceed 1024 tokens.

## Memory bandwidth sets the speed

Generating each token reads essentially all the weights: about 2.0 GB of int4 weights plus scales. Two channels of
DDR5-5600 give a theoretical peak of about 90 GB/s, and the NPU, iGPU and CPU all share it. That puts a ceiling of
roughly 40 tok/s on any device, and real kernels reach only part of the peak. So the three devices come out within
about 2× of each other on decode, unlike compute-bound vision models. The NPU's advantages here are that it keeps
the CPU free (about 0.1 cores busy) and runs at low power. TTFT (prefill) is compute-bound, so the gaps there are
larger.

## Caveats

* **Concurrent load.** Other demos used the NPU and GPU while these numbers were taken, so treat them as
  provisional. Power state (AC or battery, Windows power mode) changes them a lot, and the JSON records `meta.power`.
* **Output quality.** Channel-wise int4 costs Phi-3-mini some accuracy. Even under greedy decoding, the outputs show
  glitches (e.g. `that'Delta-Eigen-…"`, `range(2, n + libc_a_1, 1)`). Different devices can produce different text
  for the same prompt because their kernels accumulate in different precisions. Group-wise int4 or int8 weights
  improve quality at the cost of speed and size.
* Cold-compile time varies (62 s to 204 s for the same model in separate runs here). The NPU driver and the
  contention on the machine both affect it. `--cold` deletes only this example's cached blobs.
* Licence: Phi-3-mini-4k-instruct is **MIT** (Microsoft). The OpenVINO IR is from
  [OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov](https://huggingface.co/OpenVINO/Phi-3-mini-4k-instruct-int4-cw-ov).
