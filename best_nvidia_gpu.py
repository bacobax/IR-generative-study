import subprocess
import re
from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu_pct: int

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


def _run_nvidia_smi() -> str:
    """
    Runs nvidia-smi with a stable, parse-friendly query format.
    Raises RuntimeError if nvidia-smi isn't available or fails.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as e:
        raise RuntimeError("nvidia-smi not found in PATH.") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"nvidia-smi failed: {e.output.strip()}") from e


def _parse_nvidia_smi_csv(text: str) -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    # Each line: index, uuid, name, mem_used, mem_total, util
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            # Skip unexpected lines rather than crashing
            continue

        try:
            idx = int(parts[0])
            uuid = parts[1]
            name = parts[2]
            mem_used = int(parts[3])
            mem_total = int(parts[4])
            util = int(parts[5])
        except ValueError:
            continue

        gpus.append(
            GpuInfo(
                index=idx,
                uuid=uuid,
                name=name,
                memory_used_mb=mem_used,
                memory_total_mb=mem_total,
                utilization_gpu_pct=util,
            )
        )
    return gpus


def get_least_used_cuda_gpu(
    *,
    prefer: str = "memory",              # "memory" or "util"
    min_free_mb: int = 0,                # filter GPUs with at least this much free VRAM
    return_type: str = "torch",          # "torch" or "index"
) -> Optional[object]:
    """
    If CUDA is available, returns the least-used GPU based on `nvidia-smi`.
    Otherwise returns None.

    Selection rules:
      - Filters out GPUs with free VRAM < min_free_mb
      - Ranking primary key depends on `prefer`:
          * "memory": lowest memory.used (MB), then lowest utilization.gpu (%)
          * "util":   lowest utilization.gpu (%), then lowest memory.used (MB)
      - Returns either torch.device("cuda:<idx>") or the integer idx (return_type)

    Notes:
      - Uses nvidia-smi; does not require NVML bindings.
      - If torch isn't installed and return_type="torch", raises RuntimeError.
    """
    # Quick CUDA availability check (works even if torch is present but CUDA isn't)
    try:
        import torch
        has_torch = True
    except Exception:
        has_torch = False

    if has_torch:
        if not torch.cuda.is_available():
            return None
    else:
        # No torch: we can still pick a GPU, but the user asked "if cuda device is available"
        # We'll interpret that as "CUDA driver + nvidia-smi present"; if nvidia-smi fails, return None.
        pass

    try:
        smi_out = _run_nvidia_smi()
    except RuntimeError:
        return None

    gpus = _parse_nvidia_smi_csv(smi_out)
    if not gpus:
        return None

    # Apply free memory filter
    gpus = [g for g in gpus if g.memory_free_mb >= min_free_mb]
    if not gpus:
        return None

    if prefer not in {"memory", "util"}:
        raise ValueError("prefer must be 'memory' or 'util'")
    if return_type not in {"torch", "index"}:
        raise ValueError("return_type must be 'torch' or 'index'")

    if prefer == "memory":
        gpus.sort(key=lambda g: (g.memory_used_mb, g.utilization_gpu_pct, -g.memory_total_mb, g.index))
    else:  # prefer == "util"
        gpus.sort(key=lambda g: (g.utilization_gpu_pct, g.memory_used_mb, -g.memory_total_mb, g.index))

    best = gpus[0]

    if return_type == "index":
        return best.index

    if not has_torch:
        raise RuntimeError("torch is not installed; use return_type='index' instead.")

    return torch.device(f"cuda:{best.index}"), smi_out.strip()


# Example:
# dev = get_least_used_cuda_gpu(prefer="memory", min_free_mb=2048, return_type="torch")
# if dev is not None:
#     print("Using:", dev)
# else:
#     print("No CUDA available")

if __name__ == "__main__":
    
    dev, smi_out = get_least_used_cuda_gpu(prefer="memory", min_free_mb=2048, return_type="torch")
    if dev is not None:
        print("Using:", dev)
        print("nvidia-smi output:\n", smi_out)
    else:
        print("No suitable CUDA GPU available")