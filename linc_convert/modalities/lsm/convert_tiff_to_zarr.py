import time
import psutil
from threading import Thread
from linc_convert.modalities.lsm.mosaic import convert


# Shared dictionary to store metrics
metrics = {"cpu": [], "memory": []}


def monitor_usage():
    """Continuously monitors CPU and memory usage."""
    process = psutil.Process()
    while True:
        cpu_percent = process.cpu_percent(interval=0.1)  # Short interval for responsiveness
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
        metrics["cpu"].append(cpu_percent)
        metrics["memory"].append(memory_mb)
        time.sleep(0.1)  # Adjust monitoring frequency as needed


def main():
    # Start monitoring in a separate thread
    monitor_thread = Thread(target=monitor_usage, daemon=True)
    monitor_thread.start()

    # Start the timer
    start_time = time.time()
    print("Starting execution...")

    # Execute the convert function
    convert(
        inp="./",
        out="",
        shard=["auto"],
        chunk=[32]
    )

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Execution finished.")

    # Collect metrics
    avg_cpu = sum(metrics["cpu"]) / len(metrics["cpu"])
    max_memory = max(metrics["memory"])

    print(f"\nExecution Time: {elapsed_time:.2f} seconds")
    print(f"Average CPU Usage: {avg_cpu:.2f}%")
    print(f"Peak Memory Usage: {max_memory:.2f} MB")


if __name__ == "__main__":
    main()