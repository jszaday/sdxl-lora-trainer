"""Tests for logging utilities."""

import torch

from lora_trainer.logging import collect_gpu_metrics, log_perf_metrics


class DummyWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, scalar_value, global_step))


def test_collect_gpu_metrics_cpu_only(monkeypatch):
    """When CUDA is unavailable, metrics should be empty."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert collect_gpu_metrics() == {}


def test_log_perf_metrics_records_scalars(monkeypatch):
    """log_perf_metrics should write step/image throughput and gpu metrics."""
    writer = DummyWriter()

    # Force predictable gpu metrics
    fake_metrics = {"gpu_utilization": 50.0, "gpu_mem_used_bytes": 123.0}
    monkeypatch.setattr(
        "lora_trainer.logging.collect_gpu_metrics",
        lambda device=None: fake_metrics,
    )

    log_perf_metrics(
        writer=writer,
        global_step=10,
        step_time=0.5,
        effective_batch_size=4,
        device="cpu",
    )

    recorded = {tag: val for tag, val, step in writer.scalars if step == 10}
    assert recorded["perf/step_time_sec"] == 0.5
    assert recorded["perf/images_per_sec"] == 8.0
    # GPU metrics bubbled through
    assert recorded["perf/gpu_utilization"] == 50.0
    assert recorded["perf/gpu_mem_used_bytes"] == 123.0
