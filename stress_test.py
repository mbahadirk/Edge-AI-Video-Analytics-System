# stress_test.py
import requests
import time
import cv2
import os
import threading

# --- CONFIG ---
API_URL = "http://localhost:8000/detect"
IMAGE_PATH = "test_image.jpg"  # Make sure this file exists next to the script!
NUM_REQUESTS = 50  # How many requests to send
CONCURRENCY = 1  # Sequential (1) or Parallel (>1)


def check_image():
    """Create a dummy image if one doesn't exist."""
    if not os.path.exists(IMAGE_PATH):
        print(f"⚠️ {IMAGE_PATH} not found. Creating a dummy black image...")
        dummy = cv2.np.zeros((640, 640, 3), dtype='uint8')
        cv2.imwrite(IMAGE_PATH, dummy)
        print("✅ Dummy image created.")


def send_request(session, img_data):
    try:
        start = time.perf_counter()
        # Send image as multipart/form-data
        resp = session.post(API_URL, files={"file": img_data})
        lat = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            return lat, True
        else:
            print(f"Error: {resp.status_code} - {resp.text}")
            return lat, False
    except Exception as e:
        print(f"Request failed: {e}")
        return 0, False


def run_test():
    check_image()

    # Pre-load image into memory to test ONLY API speed (not disk speed)
    with open(IMAGE_PATH, "rb") as f:
        img_data = f.read()

    print(f"🚀 Starting Stress Test: {NUM_REQUESTS} requests...")

    session = requests.Session()
    latencies = []
    success_count = 0

    start_total = time.perf_counter()

    for i in range(NUM_REQUESTS):
        lat, success = send_request(session, img_data)
        latencies.append(lat)
        if success:
            success_count += 1
        print(f"Request {i + 1}/{NUM_REQUESTS}: {lat:.2f}ms")

    total_time = time.perf_counter() - start_total
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    fps = NUM_REQUESTS / total_time

    print("\n" + "=" * 30)
    print("📊 RESULTS")
    print("=" * 30)
    print(f"Successful Requests: {success_count}/{NUM_REQUESTS}")
    print(f"Total Time:          {total_time:.2f}s")
    print(f"Average Latency:     {avg_lat:.2f}ms")
    print(f"Throughput (FPS):    {fps:.2f}")
    print("=" * 30)


if __name__ == "__main__":
    run_test()