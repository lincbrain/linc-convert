import time
import psutil
from linc_convert.modalities.lsm.mosaic import convert


def monitor_usage():
    """Monitors CPU and memory usage."""
    process = psutil.Process()
    cpu_percent = process.cpu_percent(interval=0.1)  # Measure CPU over a small interval
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Resident Set Size in MB
    return cpu_percent, memory_mb


def main():
    # Start monitoring
    start_time = time.time()
    start_cpu_percent, start_memory_mb = monitor_usage()

    print(f"Initial CPU Usage: {start_cpu_percent}%")
    print(f"Initial Memory Usage: {start_memory_mb:.2f} MB")

    # Execute the convert function
    convert(
        inp="./",
        out="",
        shard=["auto"],
        chunk=[32]
    )

    # End monitoring
    end_time = time.time()
    end_cpu_percent, end_memory_mb = monitor_usage()

    elapsed_time = end_time - start_time
    print(f"\nExecution Time: {elapsed_time:.2f} seconds")
    print(f"Final CPU Usage: {end_cpu_percent}%")
    print(f"Final Memory Usage: {end_memory_mb:.2f} MB")


if __name__ == "__main__":
    main()