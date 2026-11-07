# Project Board — Panther Lake AI Lab

This Markdown board is version-controlled. [GitHub Issues](https://github.com/jcallinan/panther-lake-ai-lab/issues) hold the actionable tasks; this file is the planning snapshot. Update this board when work changes status.

## Backlog

| Issue | Workstream | Phase |
|---|---|---|
| [#1](../issues/1) | Contribution and reproducibility standards | 0 |
| [#4](../issues/4) | Ollama benchmark | 2 |
| [#5](../issues/5) | llama.cpp backends | 2 |
| [#6](../issues/6) | ONNX Runtime / OpenVINO EP | 3 |
| [#7](../issues/7) | Windows ML providers | 3 |
| [#8](../issues/8) | Intel AI Playground | 3 |
| [#9](../issues/9) | Model test matrix | 4 |
| [#10](../issues/10) | Power and thermal telemetry | 4 |
| [#11](../issues/11) | Memory upgrade comparison | 5 |
| [#12](../issues/12) | Results dashboard | 5 |

## Ready

- [#2 Hardware inventory and driver baseline](../issues/2)
- [#3 OpenVINO CPU/GPU/NPU execution validation](../issues/3)

## In progress

_None yet; awaiting first on-device results._

## Blocked

_None recorded._

## Review

_None._

## Done

- Public repository created
- Initial hardware inventory and OpenVINO scripts committed
- [Full project plan](PROJECT_PLAN.md) committed
- Twelve issues created

## Workflow

**Backlog → Ready → In progress → Review → Done**; move to **Blocked** when prerequisites fail. Each experiment must include reproducible commands, model/driver/runtime versions, observed device, measurements and sanitized results. Close an issue only after corresponding changes are committed.

**Note:** This is a Git-tracked Kanban representation, not a GitHub Projects v2 board. A native Projects board must be created separately in GitHub.
