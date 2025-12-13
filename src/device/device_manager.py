"""
Device Manager - Automatic GPU detection with graceful fallback.

Detection Priority:
1. NVIDIA CUDA (RTX 4070 via eGPU)
2. DirectML (AMD Radeon 780M integrated)
3. CPU (fallback)
"""
import torch
from typing import Optional, Literal
from dataclasses import dataclass

DeviceType = Literal["cuda", "directml", "cpu"]


@dataclass
class DeviceInfo:
    """Information about the detected compute device."""
    device: torch.device
    device_type: DeviceType
    name: str
    memory_gb: Optional[float] = None


class DeviceManager:
    """
    Manages device selection for PyTorch operations.

    Automatically detects available GPUs and selects the best option:
    - NVIDIA CUDA (highest priority)
    - AMD DirectML (fallback for AMD GPUs on Windows)
    - CPU (always available fallback)
    """

    def __init__(
        self,
        preferred: Optional[DeviceType] = None,
        force_cpu: bool = False
    ):
        """
        Initialize the device manager.

        Args:
            preferred: Force a specific device type ("cuda", "directml", "cpu")
            force_cpu: If True, always use CPU regardless of GPU availability
        """
        self.preferred = preferred
        self.force_cpu = force_cpu
        self._device_info: Optional[DeviceInfo] = None
        self._directml_available: Optional[bool] = None

    def _check_directml(self) -> bool:
        """Check if DirectML is available."""
        if self._directml_available is not None:
            return self._directml_available
        try:
            import torch_directml
            self._directml_available = True
            return True
        except ImportError:
            self._directml_available = False
            return False

    def _get_directml_device(self) -> torch.device:
        """Get the DirectML device."""
        import torch_directml
        return torch_directml.device()

    def detect_device(self) -> DeviceInfo:
        """
        Detect and return the best available compute device.

        Returns:
            DeviceInfo containing the device and metadata
        """
        if self._device_info is not None:
            return self._device_info

        # Force CPU if requested
        if self.force_cpu:
            self._device_info = DeviceInfo(
                device=torch.device("cpu"),
                device_type="cpu",
                name="CPU"
            )
            return self._device_info

        # Check for preferred device first
        if self.preferred == "cuda" and torch.cuda.is_available():
            return self._create_cuda_device()
        elif self.preferred == "directml" and self._check_directml():
            return self._create_directml_device()
        elif self.preferred == "cpu":
            return self._create_cpu_device()

        # Auto-detection: Priority 1 - NVIDIA CUDA
        if torch.cuda.is_available():
            return self._create_cuda_device()

        # Auto-detection: Priority 2 - DirectML (AMD GPU)
        if self._check_directml():
            return self._create_directml_device()

        # Auto-detection: Priority 3 - CPU fallback
        return self._create_cpu_device()

    def _create_cuda_device(self) -> DeviceInfo:
        """Create CUDA device info."""
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        self._device_info = DeviceInfo(
            device=device,
            device_type="cuda",
            name=gpu_name,
            memory_gb=round(memory_gb, 1)
        )
        return self._device_info

    def _create_directml_device(self) -> DeviceInfo:
        """Create DirectML device info."""
        device = self._get_directml_device()
        self._device_info = DeviceInfo(
            device=device,
            device_type="directml",
            name="DirectML (AMD GPU)",
            memory_gb=None  # DirectML doesn't expose memory info easily
        )
        return self._device_info

    def _create_cpu_device(self) -> DeviceInfo:
        """Create CPU device info."""
        self._device_info = DeviceInfo(
            device=torch.device("cpu"),
            device_type="cpu",
            name="CPU"
        )
        return self._device_info

    def get_device(self) -> torch.device:
        """Get the PyTorch device object."""
        return self.detect_device().device

    def get_device_type(self) -> DeviceType:
        """Get the device type string."""
        return self.detect_device().device_type

    def to_device(self, tensor_or_model):
        """
        Move tensor or model to the detected device.

        Args:
            tensor_or_model: A PyTorch tensor or model

        Returns:
            The tensor/model on the selected device
        """
        return tensor_or_model.to(self.get_device())

    def print_device_info(self):
        """Print information about the detected device."""
        info = self.detect_device()
        print(f"[Device Manager] Using: {info.name}")
        print(f"[Device Manager] Type: {info.device_type}")
        if info.memory_gb:
            print(f"[Device Manager] Memory: {info.memory_gb} GB")

    def get_summary(self) -> str:
        """Get a one-line summary of the device."""
        info = self.detect_device()
        if info.memory_gb:
            return f"{info.name} ({info.memory_gb} GB)"
        return info.name
