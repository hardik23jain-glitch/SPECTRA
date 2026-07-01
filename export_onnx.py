"""
ONNX export — Master Plan Phase 4.4: export both stage generators to ONNX for
TensorRT compilation / fast inference deployment.

Usage:
    python -m scripts.export_onnx --sr-checkpoint checkpoints/stage1_sr.pt \
        --color-checkpoint checkpoints/stage2_color.pt --out-dir onnx_export --size 256
"""
import argparse
import os
import time

import torch

from models.sr_model import ESRGANGenerator
from models.colorization_model import UNetColorizationGenerator


def export_sr(checkpoint, out_path, size=128, scale=2):
    model = ESRGANGenerator(scale=scale, num_rrdb=4)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1, 1, size, size)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["ir_input"], output_names=["ir_enhanced"],
        dynamic_axes={"ir_input": {0: "batch", 2: "height", 3: "width"},
                      "ir_enhanced": {0: "batch", 2: "height", 3: "width"}},
        opset_version=17,
    )
    print(f"Exported SR model to {out_path}")


def export_colorization(checkpoint, out_path, size=256):
    model = UNetColorizationGenerator()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    dummy_ir = torch.randn(1, 1, size, size)
    dummy_cond = torch.rand(1, 2)
    torch.onnx.export(
        model, (dummy_ir, dummy_cond), out_path,
        input_names=["ir_enhanced", "cond"], output_names=["rgb_output"],
        dynamic_axes={"ir_enhanced": {0: "batch"}, "cond": {0: "batch"}, "rgb_output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported colorization model to {out_path}")


def benchmark_inference_time(sr_onnx_path, color_onnx_path, size=256, n_runs=20):
    """Benchmarks per-tile inference latency using onnxruntime, as required by the
    Master Plan's 'inference time per tile' scalability metric (Phase 1, 3.3 / evaluation params)."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping latency benchmark. pip install onnxruntime")
        return None

    import numpy as np
    sr_sess = ort.InferenceSession(sr_onnx_path)
    color_sess = ort.InferenceSession(color_onnx_path)

    ir_in = np.random.randn(1, 1, size // 2, size // 2).astype(np.float32)
    cond_in = np.random.rand(1, 2).astype(np.float32)

    # warmup
    for _ in range(3):
        sr_out = sr_sess.run(None, {"ir_input": ir_in})[0]
        color_sess.run(None, {"ir_enhanced": sr_out, "cond": cond_in})

    start = time.time()
    for _ in range(n_runs):
        sr_out = sr_sess.run(None, {"ir_input": ir_in})[0]
        color_sess.run(None, {"ir_enhanced": sr_out, "cond": cond_in})
    elapsed = (time.time() - start) / n_runs
    print(f"Average per-tile inference time ({size}x{size} output): {elapsed*1000:.2f} ms")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr-checkpoint", required=True)
    parser.add_argument("--color-checkpoint", required=True)
    parser.add_argument("--out-dir", default="onnx_export")
    parser.add_argument("--size", type=int, default=128, help="IR input patch size before SR upscaling")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sr_out = os.path.join(args.out_dir, "stage1_sr.onnx")
    color_out = os.path.join(args.out_dir, "stage2_color.onnx")

    export_sr(args.sr_checkpoint, sr_out, size=args.size)
    export_colorization(args.color_checkpoint, color_out, size=args.size * 2)

    if args.benchmark:
        benchmark_inference_time(sr_out, color_out, size=args.size * 2)

    print("\nNext step for production: compile these ONNX graphs with TensorRT, e.g.:")
    print(f"  trtexec --onnx={sr_out} --saveEngine=stage1_sr.engine --fp16")
    print(f"  trtexec --onnx={color_out} --saveEngine=stage2_color.engine --fp16")


if __name__ == "__main__":
    main()
