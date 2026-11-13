"""Cheap Windows telemetry for the offload dashboard.

* NPU / iGPU engine utilisation: the PDH counter `\\GPU Engine(*)\\Utilization Percentage` (the same source as Task
  Manager). Instances look like `pid_123_luid_0x0_0x1049B_phys_0_eng_0_engtype_Compute`. On this laptop the NPU is
  exposed as its own adapter LUID with a single engine, typed `Compute` (older driver) or `Neural` (NPU driver
  32.0.100.5540); the Xe3 iGPU is the adapter that also has
  `VideoDecode` engines (OpenVINO GPU work shows up on its engine type `Neural`/`Compute`). Adapters are identified by
  that engine-type signature at start-up; LUIDs change between boots, so nothing is hard-coded.
  Utilisation = per engine, summed across processes (capped at 100), then the busiest engine of the adapter.
* Battery discharge: `root\\wmi BatteryStatus.DischargeRate` (mW) via one long-lived PowerShell loop, only when the
  machine is on battery. On AC the field reads 0 and power is reported as not measured.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import re
import subprocess
import sys
import threading

_INST = re.compile(r"pid_(\d+)_luid_(0x[0-9A-Fa-f]+_0x[0-9A-Fa-f]+)_phys_\d+_eng_(\d+)_engtype_(.*)$")


class _FMT(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("pad", wt.DWORD), ("doubleValue", ctypes.c_double)]


class _ITEM(ctypes.Structure):
    _fields_ = [("szName", wt.LPWSTR), ("FmtValue", _FMT)]


class GpuEngines:
    """PDH wildcard query over all GPU/NPU engines. `sample()` costs ~1 ms."""

    PDH_FMT_DOUBLE = 0x200

    def __init__(self):
        self.ok = False
        self.error = None
        self.npu_luid = self.gpu_luid = None
        self.adapters = {}
        if sys.platform != "win32":
            self.error = "not Windows"
            return
        try:
            self.pdh = ctypes.WinDLL("pdh")
            self.q, self.c = wt.HANDLE(), wt.HANDLE()
            if self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.q)):
                raise OSError("PdhOpenQueryW failed")
            r = self.pdh.PdhAddEnglishCounterW(self.q, r"\GPU Engine(*)\Utilization Percentage", None,
                                               ctypes.byref(self.c))
            if r:
                raise OSError(f"PdhAddEnglishCounterW failed 0x{r & 0xffffffff:08x}")
            self.pdh.PdhCollectQueryData(self.q)
            self._identify()
            self.ok = True
        except Exception as exc:  # telemetry is optional; report why it is missing
            self.error = f"{type(exc).__name__}: {exc}"

    def _raw(self) -> list[tuple[str, float]]:
        self.pdh.PdhCollectQueryData(self.q)
        size, n = wt.DWORD(0), wt.DWORD(0)
        self.pdh.PdhGetFormattedCounterArrayW(self.c, self.PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n), None)
        if not size.value:
            return []
        buf = (ctypes.c_byte * size.value)()
        if self.pdh.PdhGetFormattedCounterArrayW(self.c, self.PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(n),
                                                 buf):
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_ITEM))
        return [(items[i].szName, items[i].FmtValue.doubleValue) for i in range(n.value)]

    NPU_ENGINE_TYPES = {"Compute", "Neural"}

    def _identify(self, raw=None):
        """Classify adapters by engine-type signature.

        iGPU = the adapter with VideoDecode engines. NPU = a compute-only adapter (no 3D / video / copy engines) whose
        engines are all `Compute` or `Neural`. The NPU driver names that engine `Compute` on 32.0.100.4xxx-era drivers
        and `Neural` on 32.0.100.5540, so both are accepted.
        """
        for name, _ in (raw if raw is not None else self._raw()):
            m = _INST.match(name)
            if m:
                self.adapters.setdefault(m.group(2), set()).add(m.group(4) or "(unnamed)")
        for luid, types in self.adapters.items():
            if "VideoDecode" in types:
                self.gpu_luid = luid
            elif types and types <= self.NPU_ENGINE_TYPES:
                self.npu_luid = luid

    def sample(self, pid: int | None = None) -> dict:
        """{'NPU': %, 'GPU': %, 'GPU_engine': busiest iGPU engine type, 'NPU_proc'/'GPU_proc': same for `pid` only}.

        Values are None when the adapter wasn't identified.
        """
        per_engine, per_engine_pid = {}, {}
        raw = self._raw()
        if self.npu_luid is None or self.gpu_luid is None:  # instances appear once a process opens the adapter
            self._identify(raw)
        for name, v in raw:
            m = _INST.match(name)
            if m:
                key = (m.group(2), m.group(3), m.group(4))
                per_engine[key] = per_engine.get(key, 0.0) + v
                if pid is not None and int(m.group(1)) == pid:
                    per_engine_pid[key] = per_engine_pid.get(key, 0.0) + v
        out = {"NPU": None, "GPU": None, "GPU_engine": None, "NPU_proc": None, "GPU_proc": None}
        for dev, luid in (("NPU", self.npu_luid), ("GPU", self.gpu_luid)):
            if luid is None:
                continue
            for suffix, table in (("", per_engine), ("_proc", per_engine_pid)):
                vals = {k: min(v, 100.0) for k, v in table.items() if k[0] == luid}
                k = max(vals, key=vals.get) if vals else None
                out[dev + suffix] = vals[k] if k else 0.0
                if dev == "GPU" and not suffix and k:
                    out["GPU_engine"] = k[2] or "(unnamed)"
        return out

    def describe(self) -> dict:
        return {"ok": self.ok, "error": self.error, "npu_luid": self.npu_luid, "gpu_luid": self.gpu_luid,
                "adapters": {k: sorted(v) for k, v in self.adapters.items()},
                "counter": r"\GPU Engine(*)\Utilization Percentage"}


class BatteryPower:
    """Battery discharge rate in watts when on battery; `None` on AC (power not measured)."""

    PS = ("while ($true) { $b = Get-CimInstance -Namespace root\\wmi -ClassName BatteryStatus | Select-Object -First 1;"
          " Write-Output \"$($b.PowerOnline) $($b.Discharging) $($b.DischargeRate)\"; Start-Sleep -Milliseconds 1000 }")

    def __init__(self):
        self.watts = None
        self.on_battery = None
        self.note = ""
        self.proc = None
        try:
            import psutil
            bat = psutil.sensors_battery()
        except Exception:
            bat = None
        if bat is None:
            self.note = "no battery reported - power not measured"
            return
        self.on_battery = not bat.power_plugged
        if not self.on_battery:
            self.note = "on AC - power not measured"
            return
        self.note = "battery DischargeRate (WMI root\\wmi BatteryStatus)"
        self.proc = subprocess.Popen(["powershell", "-NoProfile", "-Command", self.PS], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parts = line.split()
            if len(parts) == 3:
                online, discharging, rate = parts
                if online == "True":
                    self.on_battery, self.watts = False, None
                else:
                    self.on_battery = True
                    self.watts = int(rate) / 1000.0 if discharging == "True" else None

    def close(self):
        if self.proc:
            self.proc.kill()
