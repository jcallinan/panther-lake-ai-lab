"""Discover OpenVINO CPU, GPU and NPU devices."""
import json
from openvino import Core

def main():
    core = Core()
    devices = []
    for device in core.available_devices:
        item = {"device": device}
        for prop in ("FULL_DEVICE_NAME", "OPTIMIZATION_CAPABILITIES"):
            try:
                item[prop.lower()] = str(core.get_property(device, prop))
            except Exception as exc:
                item[prop.lower()] = f"Unavailable: {type(exc).__name__}"
        devices.append(item)
    print(json.dumps({"devices": devices}, indent=2))

if __name__ == "__main__":
    main()
