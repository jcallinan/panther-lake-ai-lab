# Compatibility matrix

Statuses: **not tested**, **detected**, **compiled**, **executed**, **failed**. A runtime installing successfully does not establish NPU or GPU inference.

| Runtime | CPU | GPU | NPU | Notes |
|---|---|---|---|---|
| OpenVINO | Not tested | Not tested | Not tested | Device discovery first |
| ONNX Runtime | Not tested | Not tested | Not tested | Compare available providers |
| Windows ML | Not tested | Not tested | Not tested | Provider-specific |
| llama.cpp | Not tested | Not tested | Not tested | Backend-specific |
| Ollama | Not tested | Not tested | Not tested | Do not assume NPU support |
| Intel AI Playground | Not tested | Not tested | Not tested | Verify hardware requirements |

Every result should include runtime and driver versions, model name, model precision, backend, requested device, actual device, execution outcome and reproduction steps.
