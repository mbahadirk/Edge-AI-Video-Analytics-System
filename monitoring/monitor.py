import time
import json
import logging
import numpy as np
import pynvml
from collections import deque
from datetime import datetime

# Configure standard logging to output JSON
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("perf_monitor")


class PerformanceMonitor:
    def __init__(self, window_size=100, gpu_index=0):
        self.window_size = window_size
        self.latency_buffer = deque(maxlen=window_size)
        self.request_timestamps = deque(maxlen=window_size)

        # Initialize GPU Monitoring (NVML)
        self.gpu_available = False
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            self.gpu_available = True
        except Exception as e:
            print(f"⚠️ Warning: NVML Init failed. GPU metrics disabled. {e}")

    def start_inference(self):
        """Call this exactly before model.infer()"""
        return time.perf_counter()

    def stop_inference(self, start_time):
        """Call this exactly after model.infer()"""
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_buffer.append(latency_ms)
        self.request_timestamps.append(time.time())
        return latency_ms

    def _get_gpu_stats(self):
        """Fetches raw GPU memory and utilization."""
        if not self.gpu_available:
            return {"gpu_util": 0, "mem_used_mb": 0}

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                "gpu_util_pct": util.gpu,
                "mem_used_mb": round(mem.used / 1024 ** 2, 1),
                "mem_total_mb": round(mem.total / 1024 ** 2, 1)
            }
        except:
            return {"gpu_util": 0, "mem_used_mb": 0}

    def _calculate_fps(self):
        """Calculates FPS based on the timestamps in the current window."""
        if len(self.request_timestamps) < 2:
            return 0.0

        # Time difference between the oldest and newest request in the buffer
        duration_sec = self.request_timestamps[-1] - self.request_timestamps[0]
        if duration_sec == 0:
            return 0.0

        return len(self.request_timestamps) / duration_sec

    def _calculate_percentiles(self):
        """Calculates p50 (Median), p90, and p95 latency."""
        if not self.latency_buffer:
            return {"p50": 0, "p90": 0, "p95": 0}

        data = np.array(self.latency_buffer)
        return {
            "p50": round(np.percentile(data, 50), 2),
            "p90": round(np.percentile(data, 90), 2),
            "p95": round(np.percentile(data, 95), 2),
            "avg": round(np.mean(data), 2)
        }

    def log_snapshot(self):
        """Generates a structured JSON log of current performance state."""
        gpu_stats = self._get_gpu_stats()
        lat_stats = self._calculate_percentiles()
        fps = self._calculate_fps()

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "perf": {
                "fps": round(fps, 2),
                "latency_ms": lat_stats
            },
            "resource": {
                "gpu_util_pct": gpu_stats["gpu_util_pct"],
                "gpu_mem_used_mb": gpu_stats["mem_used_mb"]
            }
        }

        # Log as valid JSON string
        logger.info(json.dumps(log_entry))
        return log_entry