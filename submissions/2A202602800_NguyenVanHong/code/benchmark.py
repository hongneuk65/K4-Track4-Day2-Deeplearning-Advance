import time
import copy
import numpy as np
import torch

from inference import forward_method


def bench(fn, warmup=10, iters=50, sync=None):
    if warmup < 10 or iters < 50:
        raise ValueError("Cần warmup >= 10 và iters >= 50")

    for _ in range(warmup):
        fn()

    times = []

    for _ in range(iters):
        if sync is not None:
            sync()

        start = time.perf_counter()
        fn()

        if sync is not None:
            sync()

        times.append((time.perf_counter() - start) * 1000)

    return {
        "p50": float(np.percentile(times, 50)),
        "p95": float(np.percentile(times, 95)),
        "p99": float(np.percentile(times, 99)),
        "mean": float(np.mean(times)),
        "n": iters,
    }


def method_latency(
    model,
    method,
    batch_size,
    device,
    temperature=1.0,
):
    model.eval()

    # Five-crop nhận ảnh 256; từng crop đưa vào model là 224.
    size = 256 if method == "I02" else 224

    images = torch.randn(
        batch_size, 3, size, size,
        device=device,
    )

    def predict():
        return forward_method(
            model, images, method, temperature
        )

    sync = (
        torch.cuda.synchronize
        if torch.device(device).type == "cuda"
        else None
    )

    measured = bench(
        predict,
        warmup=10,
        iters=50,
        sync=sync,
    )

    return {
        **measured,
        "batch": batch_size,
        "input_size": size,
        "model_input_size": 224,
        "dtype": "AMP" if method == "I04" else "FP32",
        "gpu": (
            torch.cuda.get_device_name(0)
            if torch.device(device).type == "cuda"
            else "CPU"
        ),
        "torch": torch.__version__,
        "images_per_s": batch_size / (measured["p50"] / 1000),
        "preprocessing_included": False,
        "view_operations_included": True,
        "fused_bn": False,
    }


def latency_report(model, batch_size, img_size, dtype="fp32", device="cuda",
                   warmup=10, iters=100, k_views=1):
    """Forward-only latency; copy model so FP16 does not mutate the caller."""
    device = torch.device(device)
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError(dtype)
    if dtype == "fp16" and device.type != "cuda":
        raise ValueError("FP16 benchmark requires CUDA")
    if not isinstance(k_views, int) or k_views < 1:
        raise ValueError("k_views must be a positive integer")
    measured_model = copy.deepcopy(model).to(device).float().eval()
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        measured_model.half()
        images = images.half()

    @torch.inference_mode()
    def forward():
        with torch.autocast(device.type, enabled=dtype == "amp"):
            outputs = [measured_model(images).float().softmax(1)
                       for _ in range(k_views)]
            return torch.stack(outputs).mean(0)

    result = bench(forward, warmup, iters,
                   torch.cuda.synchronize if device.type == "cuda" else None)
    return dict(result, batch=batch_size, img_size=img_size, dtype=dtype,
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                torch=torch.__version__, images_per_s=batch_size/(result["p50"]/1000),
                preprocessing_included=False, k_views=k_views)


def tta_latency(model, k_views, **kw):
    """Measure K forwards and probability aggregation, not K * single latency.

    Synthetic equal-resolution views; notebook method_latency measures actual
    horizontal-flip/five-crop transformations for the reported experiments.
    """
    return latency_report(model, k_views=k_views, **kw)
