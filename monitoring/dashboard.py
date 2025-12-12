import pynvml
from datetime import datetime
from .fps_meter import FPSMeter
from .logger import setup_logger, log_json


class Dashboard:
    def __init__(self, window_size=100, gpu_index=0):
        self.meter = FPSMeter(window_size=window_size)
        self.logger = setup_logger()
        self.gpu_available = False
        self.gpu_handle = None

        # Initialize GPU Monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_available = True
        except Exception as e:
            print(f"⚠️ NVML Init Failed. GPU stats disabled: {e}")

    def start_recording(self):
        """Wrapper for meter start."""
        return self.meter.start()

    def stop_recording(self, start_time):
        """Wrapper for meter stop."""
        return self.meter.stop(start_time)

    def _get_gpu_stats(self):
        """Fetches hardware stats from NVIDIA Driver."""
        if not self.gpu_available:
            return {"util": 0, "mem": 0}

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            return {
                "util_pct": util.gpu,
                "mem_used_mb": round(mem.used / 1024 ** 2, 1)
            }
        except:
            return {"util": 0, "mem": 0}

    def capture_snapshot(self):
        """
        Aggregates FPS, Latency, and GPU stats into a single JSON log event.
        Returns the dict for API response usage.
        """
        gpu_stats = self._get_gpu_stats()
        lat_stats = self.meter.get_latency_statistics()
        fps = self.meter.get_fps()

        snapshot = {
            "timestamp": datetime.utcnow().isoformat(),
            "performance": {
                "fps": round(fps, 2),
                "latency": lat_stats
            },
            "system": {
                "gpu_util": gpu_stats["util_pct"],
                "gpu_mem": gpu_stats["mem_used_mb"]
            }
        }

        # Log to console/file
        log_json(self.logger, snapshot)

        return snapshot