import time
import numpy as np
from collections import deque


class FPSMeter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.latency_buffer = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)

    def start(self):
        """Start the timer for a single inference."""
        return time.perf_counter()

    def stop(self, start_time):
        """Stop the timer and record the latency."""
        latency_ms = (time.perf_counter() - start_time) * 1000
        self.latency_buffer.append(latency_ms)
        self.timestamps.append(time.time())
        return latency_ms

    def get_fps(self):
        """Calculates current FPS based on the window."""
        if len(self.timestamps) < 2:
            return 0.0

        duration = self.timestamps[-1] - self.timestamps[0]
        if duration == 0:
            return 0.0

        return len(self.timestamps) / duration

    def get_latency_statistics(self):
        """Returns P50, P90, P95, and Avg latency."""
        if not self.latency_buffer:
            return {"p50": 0, "p90": 0, "p95": 0, "avg": 0}

        data = np.array(self.latency_buffer)
        return {
            "p50": round(np.percentile(data, 50), 2),
            "p90": round(np.percentile(data, 90), 2),
            "p95": round(np.percentile(data, 95), 2),
            "avg": round(np.mean(data), 2)
        }