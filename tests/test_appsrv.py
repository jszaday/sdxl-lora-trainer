from types import SimpleNamespace

import pytest

from lora_trainer import appsrv


def test_default_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(appsrv.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(appsrv.torch.backends, "mps", SimpleNamespace(is_available=lambda: True))

    assert appsrv._default_device() == "cuda"


def test_default_device_uses_mps_before_cpu(monkeypatch):
    monkeypatch.setattr(appsrv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(appsrv.torch.backends, "mps", SimpleNamespace(is_available=lambda: True))

    assert appsrv._default_device() == "mps"


def test_default_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(appsrv.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(appsrv.torch.backends, "mps", SimpleNamespace(is_available=lambda: False))

    assert appsrv._default_device() == "cpu"


def test_mps_dtype_honors_requested_fp16():
    assert appsrv._dtype_for_device("fp16", "mps") == appsrv.torch.float16


def test_cuda_dtype_honors_requested_precision():
    assert appsrv._dtype_for_device("fp16", "cuda") == appsrv.torch.float16


def test_trt_backend_rejects_non_cuda_device(monkeypatch):
    args = SimpleNamespace(
        backend="trt",
        device="mps",
        precision="fp16",
    )
    monkeypatch.setattr(appsrv, "parse_args", lambda: args)

    with pytest.raises(ValueError, match="TensorRT backend requires a CUDA device"):
        appsrv.main()
