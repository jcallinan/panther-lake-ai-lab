# Offload dashboard: always-on AI vs. your foreground work

**Story:** run always-on AI on the NPU and your CPU stays free for the work you are actually doing.

This demo runs one continuous AI workload: video-call style background blur plus object detection on a video
stream, paced to 30 FPS. It places that workload on the **CPU**, the **GPU** and the **NPU** in turn. The whole time, a
CPU-bound "user work" job keeps every CPU thread busy in the foreground. A live dashboard shows how fast the AI runs,
how much of your own work still gets done, and how much CPU time each AI frame costs.

![Offload dashboard: AI on CPU, GPU and NPU, separated by AI-off phases, then a summary](../../media/offload_dashboard.gif)

Static summary: [`media/offload_dashboard.png`](../../media/offload_dashboard.png) · raw numbers:
`benchmarks/results/offload_dashboard.json`.

## Run

```powershell
.venv\Scripts\python.exe -m examples.offload_dashboard                       # live window, ~3 min
.venv\Scripts\python.exe -m examples.offload_dashboard --no-display --record  # headless, writes GIF + MP4
.venv\Scripts\python.exe -m examples.offload_dashboard --phase-seconds 6 --rounds 1   # quick check (~1 min)
```

| Flag | Default | Meaning |
|---|---|---|
| `--phase-seconds` | 15 | length of each AI phase; the first `--settle` (2) seconds of every phase are left out of the stats |
| `--baseline-seconds` | ⅔ of `--phase-seconds` (10) | length of each AI-off phase |
| `--rounds` | 2 | how many times to run each placement. The order alternates: CPU→GPU→NPU, then NPU→GPU→CPU |
| `--devices` | `CPU GPU NPU` | AI placements to run |
| `--fg-workers` | logical CPUs (6) | foreground worker processes; the default keeps every thread busy |
| `--no-interleave` | off | use only one AI-off phase at the start and one at the end (the older design) |
| `--record` | off | writes `media/offload_dashboard.gif` (sped up to about 20 s, under 5 MB) and `media/mp4/offload_dashboard.mp4` (real time) |
| `--no-display` | off | headless (no OpenCV window) |
| `--output` | `benchmarks/results/offload_dashboard.json` | config, compile info, per-placement and per-phase results, every 0.5 s sample |
| `--target-fps` | 30 | pacing target for the AI pipeline |
| `--clip` | `person`, else `street` | stock clip from `examples/common/clips.py`, or a path. Falls back to a synthetic scene |
| `--snapshot-dir` | none | save one live frame from the NPU phase and the summary frame as PNG files |

Press `q` or `Esc` in the window to stop early.

## Design: why it measures it this way

An earlier version kept 2 of the 6 threads free (4 foreground workers) and used a single baseline at the start and at
the end. In that version, AI on the CPU simply ran on the idle threads. Foreground throughput did not drop: it read
121 %, while system CPU went from 74 % to 99 %. Meanwhile the baseline itself drifted −20 % over the run as the chip
heated up. So the throughput number neither showed the real cost nor stayed stable. The current design fixes both:

1. **The foreground job uses every thread** (`--fg-workers` = logical CPUs). Any CPU time the AI uses now has to come
   out of your work, as it would during a build, an export or a big recalculation. (Before the pre/post-processing was
   optimised, this load starved even the NPU pipeline to about 20 FPS. With about 2–4 ms of CPU-side work per frame it
   now holds 28–30 FPS.)
2. **Every AI phase sits between two AI-off phases**: off, CPU, off, GPU, off, NPU, off, and so on. "Your work kept" is
   the AI phase's foreground ops/s divided by the **mean of the AI-off phases just before and just after it**. This
   cancels the slow thermal and power-limit drift.
3. **Two rounds in alternating order**: CPU→GPU→NPU, then NPU→GPU→CPU. No placement always lands in the hottest part
   of the run. The results are averaged per placement, and the static chart shows each round as a dot.
4. **CPU time per AI frame is the primary metric.** It is the AI process's CPU seconds divided by the frames it
   produced, including OpenVINO's runtime threads. It needs no baseline and barely depends on background noise. "Your
   work kept" is the secondary metric: it is what the user feels, but it moves with everything else on the machine.

If you'd rather see the cost as **free CPU** than as throughput, run with `--fg-workers 4`. With two threads idle,
AI on the CPU then shows up as system CPU rising (74 % → 99 % in the earlier run), while foreground throughput stays
flat.

## How to read the dashboard

* **Header:** the run order. Pills show the AI placements and small squares show the AI-off phases.
* **Left:** the processed video, with a badge for the current device. Below it are tiles for the last ~2 s: your
  work (% of the most recent AI-off phase), AI FPS, system CPU, and UI lag. The table underneath shows each
  placement, averaged over the rounds finished so far: AI FPS, your work kept, AI process CPU %, CPU ms per frame, and
  how busy the accelerator is.
* **Right:** line charts across the whole run, coloured by placement (grey = AI off, orange = CPU, green = GPU,
  blue = NPU):
  1. your work (dashed line = 100 %)
  2. CPU used by the AI process
  3. AI FPS (dashed line = 30 FPS target)
  4. how busy the NPU and iGPU are (busiest engine)
* **End card:** "foreground work kept", "AI FPS" and "CPU time per AI frame" for each placement, plus a one-line
  takeaway for each.

## Methodology

* **AI workload** (`workers.ai_process`, its own process). For each 640×360 frame:
  * MediaPipe selfie segmentation at 256×256 and YOLOX-tiny INT8 at 416×416, both Apache-2.0, called synchronously
    one after the other.
  * Input normalisation (u8→f32, BGR→RGB, mean/scale) is built into both models with OpenVINO `PrePostProcessor`,
    so it runs on whichever device runs the model.
  * The CPU still resizes the inputs, blurs the background and blends it with the mask (`cv2.blendLinear`), runs NMS,
    and draws the boxes. This happens in **every** placement and is reported as `prepost_ms` (wall time).
  * Frames are paced to `--target-fps`. If a frame runs late, the pipeline does not try to catch up. The clip is
    decoded into memory up front.
  * All models are compiled on every device **before** the foreground load starts. A device that fails to compile is
    reported as `compile failed / not available`, and its phases are skipped. Its work is **never** run on another
    device instead. The JSON records `EXECUTION_DEVICES` for every compiled model.
* **Foreground "user work"** (`workers.foreground_worker`): one process per logical CPU, each repeatedly running
  `zlib.compress(level 6)` on a fixed 384 KB buffer. Throughput is counted in ops/s from shared counters.
* **Responsiveness probe**: one process asks for 2 ms sleeps and records how late it wakes up. The p95 lateness per
  0.5 s window is shown as "UI lag". This number is noisy.
* **Aggregation**: per phase, means of 0.5 s samples, leaving out the first 2 s; per-frame costs are weighted by frame
  count. Per placement, the mean over its phases (`placements[*].result`). The per-round values are kept in
  `placements[*].result.rounds`.

## Telemetry: what was and wasn't measurable

| Metric | Source | Status on this laptop |
|---|---|---|
| System CPU % | `psutil.cpu_percent` | works |
| Per-process CPU (AI / foreground / dashboard / everything else) | `psutil` `cpu_times` deltas | works. "Everything else" = system minus our processes; it shows how much noise there was |
| NPU utilisation | PDH `\GPU Engine(*)\Utilization Percentage`, read through `ctypes` (~1 ms per sample) | **works**. The NPU (Intel AI Boost) appears as its own adapter LUID with one engine. That engine is typed `Neural` with NPU driver 32.0.100.5540 and was typed `Compute` with the earlier 32.0.100.4724 driver. Both are recognised: the NPU is the adapter whose engines are all `Compute`/`Neural` and which has no 3D, video or copy engines |
| iGPU utilisation | same counter, the adapter that has `VideoDecode` engines | **works**. OpenVINO GPU work shows up on the iGPU's `Neural` engine; the busiest engine is reported |
| Utilisation from our AI process only | same counter, filtered by `pid_` in the instance name | works (`npu_util_ai_proc`, `gpu_util_ai_proc`) |
| Power | WMI `root\wmi BatteryStatus.DischargeRate`, via one long-lived PowerShell loop | **only on battery**. On AC the value is 0, so the dashboard says "on AC - power not measured". No per-device power counter (CPU/GPU/NPU package power) was found that can be read without admin rights |

LUIDs change between boots, so adapters are identified by their engine types, and re-checked on every sample until
both are found. Counter instances only appear once a process has opened the adapter. The JSON records the
LUID→engine-type map (`telemetry.accelerators`).

## Caveats

* **6-thread CPU.** The Core Ultra 5 322 has 2 P-cores and 4 LP E-cores, with no SMT. CPU-side AI takes a big share of a
  small CPU, and OpenVINO's CPU threads compete with your work for the two P-cores. Bigger chips will show smaller
  losses.
* **Shared power budget.** CPU, iGPU and NPU share one package power limit. When all CPU threads are busy, the iGPU
  clocks down: AI on the GPU then reaches only about 19–22 FPS while the iGPU reads 80–85 % busy. Even with AI on the
  NPU, your work can lose a few percent, from the ~2 % CPU the AI process uses plus shared power and memory bandwidth.
  The NPU mostly frees **CPU time**. It does not make the AI completely free.
* **Pre/post still runs on the CPU** with the NPU and GPU too: about 2 ms of resize, blend, NMS and drawing per frame
  when the machine is idle, and 3–5 ms when every thread is busy.
* **Accelerator % is not compute load.** The PDH engine counter shows how much of the time the engine had work
  queued. NPU ≈ 6 % at 30 FPS means it is idle between frames; it is not a measure of TOPS used.
* **AC vs. battery.** Turbo and power limits differ, and power is only reported on battery.
* **Noise.** Anything else running on the machine takes CPU from the foreground workers. Check
  `other_procs_cpu_pct`, `baseline.spread_pct` and the per-round dots before quoting a number.

## Results on this laptop

> The numbers below come from a **short development run**: `--phase-seconds 6` (4 s AI-off phases), 2 rounds,
> 6 foreground workers, on AC, NPU driver 32.0.100.5540, graphics driver 32.0.101.9175. Other agents were building
> charts at the same time and used 5–35 % of the CPU (`other_procs_cpu_pct`). The AI-off phases varied by 39 %
> (`baseline.spread_pct`). The default 15 s / 10 s run on an idle machine (`scripts/record_all.ps1`) regenerates the
> GIF, PNG and JSON; refresh this table from `placements` in that JSON.

| Placement | AI FPS (target 30) | Your work kept (round 1 / 2) | AI process CPU (% of system) | CPU ms per AI frame | pre/post ms | Accelerator busy |
|---|---|---|---|---|---|---|
| **AI on CPU** | **12.7** | **80 %** (77 / 82) | 10.7 % | **50.2** | 5.1 | – |
| AI on GPU | 19.2 | 81 % (77 / 86) | 2.6 % | 8.2 | 4.3 | iGPU 82 % |
| **AI on NPU** | **28.3** | **94 %** (93 / 95) | 1.9 % | **4.0** | 4.4 | NPU 6 % |

**Takeaways**

* **CPU:** with every thread busy, AI on the CPU reaches only about 13 of the 30 FPS. It costs about 50 ms of CPU per
  frame, and your work loses about a fifth of its throughput.
* **GPU:** CPU cost per frame is low (about 8 ms), but because the iGPU shares the power budget with a fully loaded
  CPU it only manages about 19 FPS.
* **NPU:** holds about 28–30 FPS for about 4 ms of CPU per frame, roughly **12× less CPU than AI on the CPU**. Your
  work keeps about 94 %. That remaining cost is CPU-side pre/post-processing plus shared power and memory, not
  inference.
