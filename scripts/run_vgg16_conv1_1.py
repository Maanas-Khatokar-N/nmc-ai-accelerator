#!/usr/bin/env python3
"""
Run the real VGG-16 CONV1-1 layer (224x224x3 -> 224x224x64) on the
NMC RTL cluster using the repository's paper-style 4-bank tiling flow.

Run from the nmc-ai-accelerator repository root:
    python scripts/run_vgg16_conv1_1.py --image path/to/image.jpg

Default paper-style tiling:
    n_if=4, tc=1, tk=16, policy=spatial
    4x4 spatial tiles; 16 output-channel tiles (4 channels/tile);
    4 banks execute four jobs concurrently; all 64 rounds are run.

This drives the existing cluster_tb.v using generated SRAM/program HEX files.
The NMC RTL computes the INT8 convolution and returns INT32 accumulators.
The script applies the pretrained model's bias and ReLU in Python for the
dequantized activation reference; it does NOT yet exercise the RTL postproc
unit for this VGG16 layer.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import vgg16, VGG16_Weights

# Import the repository's existing planner, data/program generator and checker.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tiling_plan as tp
import tiled_layer_golden as tiled


def quantize_symmetric_per_tensor(x):
    """Float tensor/array -> signed INT8 and one scale, zero point = 0."""
    x = np.asarray(x, dtype=np.float32)
    peak = float(np.max(np.abs(x)))
    scale = peak / 127.0 if peak > 0 else 1.0
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale


def quantize_weights_per_output_channel(w):
    """[Cout,Cin,Kh,Kw] float weights -> per-output-channel symmetric INT8."""
    w = np.asarray(w, dtype=np.float32)
    peak = np.max(np.abs(w), axis=(1, 2, 3))
    scales = np.where(peak > 0, peak / 127.0, 1.0).astype(np.float32)
    q = np.clip(
        np.rint(w / scales[:, None, None, None]), -127, 127
    ).astype(np.int8)
    return q, scales


def read_stitched_output(plan, rounds, npass):
    """Read each completed tile file and stitch the full [Cout,H,W] tensor."""
    L = plan.layer
    got = np.zeros((L.Cout, L.OH, L.OW), dtype=np.int32)
    have = np.zeros(got.shape, dtype=bool)

    for r, round_jobs in enumerate(rounds):
        for b, job in enumerate(round_jobs):
            if job is None:
                continue
            filename = f"sim/data/r{r}_p{npass - 1}_b{b}_of.hex"
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Missing NMC output tile: {filename}")

            words = np.asarray(
                [int(line.strip(), 16) for line in open(filename) if line.strip()],
                dtype=np.uint32,
            )
            tile = words.view(np.int32).reshape(job.nk, job.oh, job.ow)
            got[
                job.k0:job.k1,
                job.oy0:job.oy1,
                job.ox0:job.ox1,
            ] = tile
            have[
                job.k0:job.k1,
                job.oy0:job.oy1,
                job.ox0:job.ox1,
            ] = True

    coverage = float(have.mean())
    return got, coverage


def main():
    ap = argparse.ArgumentParser(
        description="Run pretrained VGG16 CONV1-1 on the NMC RTL cluster."
    )
    ap.add_argument("--image", required=True, help="Input RGB image; VGG16 preprocessing is applied.")
    ap.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Run only the first N tiling rounds (0 = all 64 rounds).",
    )
    ap.add_argument(
        "--weights-file",
        default=None,
        help="Optional local torchvision VGG16 state_dict (.pth); otherwise pretrained weights are downloaded.",
    )
    ap.add_argument("--seed", type=int, default=1, help="Unused placeholder for reproducibility.")
    args = ap.parse_args()

    root = Path.cwd()
    required = [
        root / "rtl",
        root / "tb/integration/cluster_tb.v",
        root / "scripts/tiling_plan.py",
        root / "scripts/tiled_layer_golden.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(
            "Run this script from the nmc-ai-accelerator repository root. Missing:\n"
            + "\n".join(missing)
        )
    if not Path(args.image).is_file():
        raise SystemExit(f"Image not found: {args.image}")

    for tool in ("iverilog", "vvp"):
        if shutil.which(tool) is None:
            raise SystemExit(f"{tool} is not installed or not on PATH.")

    os.makedirs("sim/data", exist_ok=True)
    os.makedirs("sim/build", exist_ok=True)
    os.makedirs("sim/waveform", exist_ok=True)

    # Load the actual pretrained VGG16 CONV1-1 weights and standard ImageNet input.
    if args.weights_file:
        model = vgg16(weights=None)
        try:
            state = torch.load(args.weights_file, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(args.weights_file, map_location="cpu")
        model.load_state_dict(state)
        weights_enum = VGG16_Weights.IMAGENET1K_V1
    else:
        weights_enum = VGG16_Weights.IMAGENET1K_V1
        model = vgg16(weights=weights_enum)

    model.eval()
    preprocess = weights_enum.transforms()
    image = Image.open(args.image).convert("RGB")
    x = preprocess(image).unsqueeze(0)  # [1,3,224,224], normalized as VGG16 expects

    conv = model.features[0]
    w_float = conv.weight.detach().cpu().numpy().astype(np.float32)
    bias_float = conv.bias.detach().cpu().numpy().astype(np.float32)
    x_float = x[0].detach().cpu().numpy().astype(np.float32)

    x_q, x_scale = quantize_symmetric_per_tensor(x_float)
    w_q, w_scales = quantize_weights_per_output_channel(w_float)

    # NMC stores one signed INT8 value in the low byte of each 32-bit SRAM word.
    layer = tp.layer_by_name("1-1")
    tiling = tp.Tiling(n_if=4, tc=1, tk=16)
    policy = "spatial"
    plan = tp.build_plan(layer, tiling, policy)

    ifm_pad = np.pad(x_q, ((0, 0), (layer.pad, layer.pad), (layer.pad, layer.pad)))
    golden_acc = tiled.conv_golden(ifm_pad, w_q, layer.S)

    print("\nVGG16 CONV1-1 actual-model run")
    print("--------------------------------")
    print(f"Input image       : {args.image}")
    print(f"Input tensor      : {x_float.shape}")
    print(f"Layer             : {layer.H}x{layer.W}x{layer.Cin} -> {layer.Cout}")
    print(f"Quantization      : INT8 symmetric, input scale={x_scale:.8g}")
    print(f"Weight scales     : per output channel ({len(w_scales)} scales)")
    print(f"Tiling            : n_if={tiling.n_if}, tc={tiling.tc}, tk={tiling.tk}, {policy}")
    print(f"Spatial tiles     : {len(plan.spatial)} (4x4 grid, 56x56 outputs/tile)")
    print(f"Output-channel tiles: {len(plan.ktiles)} (4 output channels/tile)")
    print(f"Concurrent banks  : {tp.NUM_BANKS}")
    print(f"Total jobs        : {plan.stats['jobs']}")
    print(f"Total rounds      : {len(plan.rounds)}")
    print(f"Channel passes    : {len(plan.passes)}")

    max_rounds = args.rounds if args.rounds > 0 else 0
    if max_rounds > len(plan.rounds):
        raise SystemExit(f"--rounds cannot exceed {len(plan.rounds)}")

    # Remove stale files BEFORE build_program writes this run's IFMap/weight tiles.
    for filename in glob.glob("sim/data/r*_b*.hex"):
        os.remove(filename)
    for filename in glob.glob("sim/data/r*_p*_b*.hex"):
        os.remove(filename)

    rounds, npass, loads = tiled.build_program(plan, ifm_pad, w_q, max_rounds)
    print(f"Rounds to execute : {len(rounds)}")
    print(f"Expected coverage : {100.0 * len(rounds) / len(plan.rounds):.2f}%")

    # Compile the same cluster testbench used by tiled_layer_golden.py.
    vvp = "sim/build/vgg16_conv1_1.vvp"
    rtl = sorted(glob.glob("rtl/**/*.v", recursive=True))
    compile_cmd = ["iverilog", "-g2012", "-o", vvp, "tb/integration/cluster_tb.v"] + rtl
    comp = subprocess.run(compile_cmd, capture_output=True, text=True)
    if comp.returncode:
        print(comp.stdout)
        print(comp.stderr)
        raise SystemExit("Icarus compilation failed.")

    # Run every selected round. In the default mode this is all 64 rounds.
    sim_cmd = ["vvp", vvp, f"+NROUNDS={len(rounds)}", f"+NPASS={npass}", "+NOISE=0"]
    print("\nStarting RTL simulation. Full CONV1-1 may take several minutes...")
    sim = subprocess.run(sim_cmd, capture_output=True, text=True, timeout=3600)
    sim_out = sim.stdout
    if sim.returncode != 0 or "CLUSTER" not in sim_out or "TIMEOUT" in sim_out:
        print(sim_out[-5000:])
        print(sim.stderr[-2000:])
        raise SystemExit("RTL simulation failed or timed out.")

    # Bit-exact INT32 comparison against NumPy using the exact quantized tensors.
    tag = f"VGG16 CONV1-1 224x224x3->64 n_if=4 tc=1 tk=16 rounds={len(rounds)}"
    ok = tiled.check(plan, rounds, npass, golden_acc, tag)
    got_acc, coverage = read_stitched_output(plan, rounds, npass)

    # Save the tensors so they can be inspected/reused by a later host-runtime stage.
    np.save("sim/data/vgg16_conv1_1_acc_int32.npy", got_acc)
    np.save("sim/data/vgg16_conv1_1_input_int8.npy", x_q)
    np.save("sim/data/vgg16_conv1_1_weights_int8.npy", w_q)

    # Dequantize the RTL accumulator. Bias + ReLU are included in the software
    # activation reference, but are not yet driven through the RTL postproc unit.
    dequant = got_acc.astype(np.float32) * x_scale * w_scales[:, None, None]
    activation = np.maximum(dequant + bias_float[:, None, None], 0.0)
    np.save("sim/data/vgg16_conv1_1_dequant_float.npy", dequant)
    np.save("sim/data/vgg16_conv1_1_relu_float.npy", activation)

    # Compare quantized NMC convolution with the original FP32 pretrained convolution.
    with torch.no_grad():
        fp32_conv = F.conv2d(x, conv.weight.detach(), bias=None, stride=1, padding=1)
    fp32_np = fp32_conv[0].cpu().numpy()
    if coverage == 1.0:
        diff = dequant - fp32_np
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff * diff)))
        maxerr = float(np.max(np.abs(diff)))
        print("\nNumerical comparison against pretrained FP32 CONV1-1 (before bias/ReLU)")
        print(f"  Mean absolute error : {mae:.6f}")
        print(f"  RMSE                : {rmse:.6f}")
        print(f"  Maximum absolute error: {maxerr:.6f}")
    else:
        print("\nPartial run: FP32 error metrics are omitted because output coverage is incomplete.")

    # Print cluster performance counters from the Verilog testbench.
    m = re.search(r"CLUSTER rounds=(\d+) passes=(\d+) max_busy=(\d+) overlap_cycles=(\d+) wall_nmc=(\d+)", sim_out)
    q = re.search(r"PERF crit=(\d+) busy_sum=(\d+) mac=(\d+) ops=(\d+) stall=(\d+) blocked=(\d+)", sim_out)
    if m and q:
        _, _, max_busy, overlap, wall = map(int, m.groups())
        crit, busy_sum, mac, ops, stalls, blocked = map(int, q.groups())
        print("\nRTL performance")
        print(f"  Max NMC units busy simultaneously : {max_busy}/{tp.NUM_BANKS}")
        print(f"  Overlap cycles                    : {overlap}/{wall}")
        print(f"  Critical-path cycles              : {crit}")
        print(f"  Sum of per-unit busy cycles       : {busy_sum}")
        print(f"  PE activity                       : {100.0 * ops / (16 * mac) if mac else 0.0:.2f}%")
        print(f"  Cluster PE efficiency             : {100.0 * ops / (16 * tp.NUM_BANKS * crit) if crit else 0.0:.2f}%")
        print(f"  Stalls / blocked cycles            : {stalls} / {blocked}")
    print(f"  Host-loaded IFMap words            : {loads['ifw']}")
    print(f"  Host-loaded weight words           : {loads['ww']}")

    metadata = {
        "model": "VGG16",
        "layer": "features.0 (CONV1-1)",
        "input_shape": [3, 224, 224],
        "output_shape": [64, 224, 224],
        "tiling": {"n_if": 4, "tc": 1, "tk": 16, "policy": policy},
        "rounds_run": len(rounds),
        "rounds_total": len(plan.rounds),
        "coverage": coverage,
        "input_scale": x_scale,
        "weight_scales": w_scales.tolist(),
        "rtl_bit_exact_match": bool(ok),
        "note": "RTL computes INT8 convolution to INT32. Bias and ReLU are software-side only in this script."
    }
    with open("sim/data/vgg16_conv1_1_run.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved outputs:")
    print("  sim/data/vgg16_conv1_1_acc_int32.npy")
    print("  sim/data/vgg16_conv1_1_dequant_float.npy")
    print("  sim/data/vgg16_conv1_1_relu_float.npy")
    print("  sim/data/vgg16_conv1_1_run.json")

    if ok and coverage == 1.0:
        print("\nSUCCESS: complete VGG16 CONV1-1 convolution executed and matched the INT32 NumPy golden.")
    elif ok:
        print("\nPARTIAL SUCCESS: executed tiles matched, but the complete layer was not run.")
    else:
        raise SystemExit("FAIL: one or more NMC output values differ from the NumPy golden.")


if __name__ == "__main__":
    main()
