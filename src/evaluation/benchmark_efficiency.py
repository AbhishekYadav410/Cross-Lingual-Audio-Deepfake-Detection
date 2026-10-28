"""Efficiency and latency benchmarking suite (RQ3).

Profiles model parameters, checkpoint size, inference latency (CPU/GPU),
and processing throughput per second.
"""

from typing import Dict, Any
import time
import os
import torch
import numpy as np


def benchmark_model_efficiency(
    model: torch.nn.Module,
    input_shape: tuple = (1, 1, 80, 94),
    device: str = "cpu",
    n_warmup: int = 20,
    n_iterations: int = 100,
) -> Dict[str, Any]:
    """Profiles memory, parameter count, and inference latency."""
    device_obj = torch.device(device)
    model = model.to(device_obj)
    model.eval()

    dummy_input = torch.randn(*input_shape, device=device_obj)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Warmup runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_iterations):
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    mean_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    throughput = float(1000.0 / mean_latency) if mean_latency > 0 else 0.0

    return {
        "device": device,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "mean_latency_ms": round(mean_latency, 3),
        "std_latency_ms": round(std_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "throughput_samples_per_sec": round(throughput, 1),
    }
