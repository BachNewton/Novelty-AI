"""
Hardware Monitor - Tracks CPU and GPU utilization.

Provides real-time hardware metrics for the training dashboard.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import psutil


@dataclass
class CPUStats:
    """CPU utilization statistics."""
    total_percent: float
    per_core_percent: List[float]
    core_count: int


@dataclass
class GPUStats:
    """GPU utilization statistics."""
    utilization_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    name: str
    available: bool = True


@dataclass
class HardwareStats:
    """Combined hardware statistics."""
    cpu: CPUStats
    gpu: Optional[GPUStats]
    timestamp: float = field(default_factory=time.time)


class HardwareMonitor:
    """
    Monitors CPU and GPU utilization in real-time.

    Runs in a background thread to avoid blocking training.
    """

    def __init__(self, update_interval: float = 1.0):
        """
        Initialize the hardware monitor.

        Args:
            update_interval: Seconds between updates
        """
        self.update_interval = update_interval
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Latest stats
        self._latest_stats: Optional[HardwareStats] = None
        self._stats_lock = threading.Lock()

        # GPU monitoring setup
        self._gpu_available = False
        self._gpu_type: Optional[str] = None  # "nvidia" or "amd"
        self._setup_gpu_monitoring()

    def _setup_gpu_monitoring(self):
        """Set up GPU monitoring based on available hardware."""
        # Try NVIDIA first
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self._gpu_available = True
                self._gpu_type = "nvidia"
                return
        except (ImportError, Exception):
            pass

        # Skip WMI for AMD - it can cause hangs on some systems
        # Just mark as unavailable for now
        self._gpu_available = False
        self._gpu_type = None

    def start(self):
        """Start the monitoring thread."""
        if self.thread is not None and self.thread.is_alive():
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        while self.running:
            try:
                stats = self._collect_stats()
                with self._stats_lock:
                    self._latest_stats = stats
            except Exception as e:
                # Don't crash on monitoring errors
                pass

            time.sleep(self.update_interval)

    def _collect_stats(self) -> HardwareStats:
        """Collect current hardware statistics."""
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(interval=None, percpu=True)

        cpu_stats = CPUStats(
            total_percent=cpu_percent,
            per_core_percent=per_core,
            core_count=len(per_core),
        )

        # GPU stats
        gpu_stats = self._get_gpu_stats()

        return HardwareStats(cpu=cpu_stats, gpu=gpu_stats)

    def _get_gpu_stats(self) -> Optional[GPUStats]:
        """Get GPU statistics based on available hardware."""
        if not self._gpu_available:
            return None

        if self._gpu_type == "nvidia":
            return self._get_nvidia_stats()
        elif self._gpu_type == "amd":
            return self._get_amd_stats()

        return None

    def _get_nvidia_stats(self) -> Optional[GPUStats]:
        """Get NVIDIA GPU stats using GPUtil."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None

            gpu = gpus[0]  # Use first GPU
            return GPUStats(
                utilization_percent=gpu.load * 100,
                memory_used_mb=gpu.memoryUsed,
                memory_total_mb=gpu.memoryTotal,
                memory_percent=(gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                name=gpu.name,
                available=True,
            )
        except Exception:
            return GPUStats(
                utilization_percent=0,
                memory_used_mb=0,
                memory_total_mb=0,
                memory_percent=0,
                name="NVIDIA GPU",
                available=False,
            )

    def _get_amd_stats(self) -> Optional[GPUStats]:
        """Get AMD GPU stats using WMI."""
        try:
            import wmi
            w = wmi.WMI(namespace="root\\cimv2")

            gpu_name = "AMD GPU"
            for gpu in w.Win32_VideoController():
                if "AMD" in gpu.Name or "Radeon" in gpu.Name:
                    gpu_name = gpu.Name
                    break

            # WMI doesn't provide detailed GPU utilization easily
            # Try to get memory info at least
            memory_total = 0
            memory_used = 0

            # Note: Getting AMD GPU utilization on Windows requires
            # additional tools or AMD's ADL SDK. For now, we'll show
            # that AMD GPU is being used but may not have detailed stats.

            return GPUStats(
                utilization_percent=0,  # Not easily available
                memory_used_mb=memory_used,
                memory_total_mb=memory_total,
                memory_percent=0,
                name=gpu_name,
                available=True,
            )
        except Exception:
            return GPUStats(
                utilization_percent=0,
                memory_used_mb=0,
                memory_total_mb=0,
                memory_percent=0,
                name="AMD GPU",
                available=False,
            )

    def get_stats(self) -> Optional[HardwareStats]:
        """Get the latest hardware statistics."""
        with self._stats_lock:
            return self._latest_stats

    def get_cpu_bars(self, max_width: int = 100) -> List[Dict[str, Any]]:
        """
        Get CPU core utilization as bar data.

        Args:
            max_width: Maximum bar width in pixels

        Returns:
            List of dicts with 'core', 'percent', 'width' keys
        """
        stats = self.get_stats()
        if stats is None:
            return []

        bars = []
        for i, percent in enumerate(stats.cpu.per_core_percent):
            bars.append({
                "core": i,
                "percent": percent,
                "width": int(percent / 100 * max_width),
            })

        return bars

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information as a dictionary."""
        stats = self.get_stats()
        if stats is None or stats.gpu is None:
            return {
                "available": False,
                "name": "No GPU",
                "utilization": 0,
                "memory_percent": 0,
                "memory_used": 0,
                "memory_total": 0,
            }

        return {
            "available": stats.gpu.available,
            "name": stats.gpu.name,
            "utilization": stats.gpu.utilization_percent,
            "memory_percent": stats.gpu.memory_percent,
            "memory_used": stats.gpu.memory_used_mb,
            "memory_total": stats.gpu.memory_total_mb,
        }

    def is_gpu_available(self) -> bool:
        """Check if GPU monitoring is available."""
        return self._gpu_available

    def get_gpu_type(self) -> Optional[str]:
        """Get the type of GPU being monitored."""
        return self._gpu_type
