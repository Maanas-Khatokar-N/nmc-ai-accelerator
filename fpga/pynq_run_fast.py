#!/usr/bin/env python3
"""
Whole-network VGG-16 inference on the PYNQ-Z2 NMC cluster - fast runtime (see nmc_fast.py for what changed).

On the board (as root), with nmc.bit/.hwh of the DMA build (axi_cdma_0 in the block design):

  sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --image test_img/ball.jpg
  sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --image a.jpg b.jpg c.jpg   # model loaded once
  sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --ref models/vgg16/ref.npz  # bit-exact check
       --mhz 66.67     set FCLK0 before running (only up to what the timing report allows, see FPGA_SPEEDUP.md)

  old bitstream (no CDMA):  add --mmio   (keeps on-chip post-proc, sparse FC, model in RAM; loads stay slow)
  post-proc on the ARM:     add --host-pp
  no hardware at all:       --sim       (functional model of the cluster; checks the host flow against ref.npz)

The first run writes transposed FC weights next to the model (L13_fc6_wT.npy ...), one time only.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nmc_fast as nf

def main():
    ap = argparse.ArgumentParser(description="Fast whole-network inference on the PYNQ-Z2 NMC cluster")
    ap.add_argument("--bit"); ap.add_argument("--sim", action="store_true", help="no hardware: functional model")
    ap.add_argument("--mmio", action="store_true", help="no CDMA in the overlay: ARM copies (old bitstream)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tilings", help="default <model>/tilings.json")
    ap.add_argument("--image", nargs="+"); ap.add_argument("--classes")
    ap.add_argument("--input", help="quantised int8 input [C,H,W] .npy (default <model>/input_q.npy)")
    ap.add_argument("--ref", help="integer reference npz (single input) -> per-layer bit-exact check")
    ap.add_argument("--cfg-base", type=lambda x: int(x, 0), default=0x40000000)
    ap.add_argument("--mem-base", type=lambda x: int(x, 0), default=0x40080000)
    ap.add_argument("--cdma-base", type=lambda x: int(x, 0), help="default: from the .hwh (axi_cdma_0)")
    ap.add_argument("--dma-mem-base", type=lambda x: int(x, 0),
                    help="s_axi_mem address in the CDMA's address map (default = --mem-base)")
    ap.add_argument("--banks", type=int, default=4); ap.add_argument("--bank-words", type=int, default=32768)
    ap.add_argument("--mhz", type=float, help="set FCLK0 to this (MHz) after loading the overlay")
    ap.add_argument("--host-pp", action="store_true", help="requant/ReLU/pool on the ARM instead of on chip")
    ap.add_argument("--no-perf", action="store_true", help="do not read the cycle counters (saves a little time)")
    ap.add_argument("--json", help="write predictions + timings of the last image to this file (web console)")
    ap.add_argument("--prep-only", action="store_true", help="only build the transposed FC weight files, then exit")
    a = ap.parse_args()

    model = nf.FastModel(a.model)
    if a.prep_only: print("model files prepared"); return
    til = json.load(open(a.tilings or os.path.join(a.model, "tilings.json")))["layers"]
    mhz = 60.0
    if a.sim:
        bd = nf.SimBoard(a.banks, a.bank_words); print("SIMULATION (functional model, no hardware)")
    else:
        if not a.bit: ap.error("--bit is required (or --sim)")
        from pynq import Overlay
        ol = Overlay(a.bit)
        try:
            from pynq import Clocks
            if a.mhz:
                Clocks.fclk0_mhz = a.mhz
            mhz = float(Clocks.fclk0_mhz); print(f"PL clock FCLK0 = {mhz:.2f} MHz")
        except Exception as e: print("(could not read/set FCLK0:", e, ")")
        cdma = None
        if not a.mmio:
            cdma = a.cdma_base
            if cdma is None:
                ips = {k: v for k, v in ol.ip_dict.items() if "cdma" in k.lower()}
                if not ips: sys.exit("no AXI CDMA in this overlay: rebuild with FPGA_SPEEDUP.md step 2, or run with --mmio")
                cdma = list(ips.values())[0]["phys_addr"]
            print(f"AXI CDMA at 0x{cdma:08x}, s_axi_mem at 0x{(a.dma_mem_base or a.mem_base):08x} in its map")
        bd = nf.HwBoard(a.cfg_base, a.mem_base, a.banks, a.bank_words, cdma_base=cdma, dma_mem_base=a.dma_mem_base)
        bd.cfg_w(nf.G["IRQEN"], 0)

    inputs = []
    if a.image:
        from imagenet_pre import preprocess, quantize_input
        crop, xs = model.input["shape"][1], model.input["x_scale"]
        for f in a.image:
            t0 = time.perf_counter(); x = quantize_input(preprocess(f, crop), xs)
            inputs.append((os.path.basename(f), x, time.perf_counter() - t0))
    else:
        f = a.input or os.path.join(a.model, "input_q.npy")
        inputs.append((os.path.basename(f), np.load(f).astype(np.int8), 0.0))
    ref = np.load(a.ref) if a.ref else None
    classes = None
    try:
        from imagenet_pre import load_classes, print_top5
        classes = load_classes(a.classes or os.path.join(a.model, "classes.txt"))
    except ImportError:
        print_top5 = None

    ok_all = True
    for name, x, t_pre in inputs:
        print(f"\n=== {name} ===")
        logits, acts, ok, tot, wall = nf.run_network(bd, model, x, til, host_pp=a.host_pp, read_perf=not a.no_perf,
                                                     ref=ref, mhz=mhz)
        ok_all &= ok
        if logits is not None and np.ndim(logits) == 1 and logits.size > 10:
            print("top-5:")
            if print_top5: print_top5(logits, classes)
            else:
                for i in np.argsort(-logits)[:5]: print(f"   {i:4d}  {logits[i]:.3f}")
            if ref is not None:
                same = np.allclose(logits, ref["logits"], rtol=0, atol=1e-9)
                print(f"logits identical to the integer reference: {same}"); ok_all &= bool(same)
        if tot["crit"]:
            print(f"NMC compute {tot['crit']} cycles = {tot['crit'] / (mhz * 1e6):.3f} s at {mhz:.1f} MHz "
                  f"({tot['macs']} MACs, {2 * tot['macs'] / tot['crit'] * mhz / 1e3:.2f} GOPS compute-only)")
        print(f"host: load+desc {tot['load']:.2f} s  run {tot['run']:.2f} s  post-proc {tot['pp']:.2f} s  "
              f"drain {tot['drain']:.2f} s  FC {tot['fc']:.2f} s  host-pp {tot['hostpp']:.2f} s;  "
              f"DMA words {tot['words'] / 1e6:.1f} M;  image preprocessing {t_pre:.2f} s")
        print(f"inference time (input tensor -> logits): {wall:.2f} s"
              f"  = {(wall + t_pre):.2f} s including JPEG decode/resize")
        if a.json and logits is not None:
            top = np.argsort(-logits)[:5]
            p = np.exp(logits - logits.max()); p /= p.sum()
            lab = lambda i: classes[i] if classes is not None else str(int(i))
            json.dump(dict(predictions=[dict(label=lab(i), score=round(100 * float(p[i]), 2)) for i in top],
                           metrics={"inference s": round(wall, 2), "NMC compute s": round(tot['crit'] / (mhz * 1e6), 2),
                                    "GOPS": round(2 * tot['macs'] / max(wall, 1e-9) / 1e9, 2)}), open(a.json, "w"))
    print("\nNETWORK RUN PASSED" if ok_all else "\nNETWORK RUN FAILED")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
