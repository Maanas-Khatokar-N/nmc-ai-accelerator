#!/usr/bin/env python3
"""
Run a whole quantised VGG-style network on the PYNQ-Z2 NMC cluster (hybrid execution, like the paper):
  conv layers   -> NMC cluster (tiled over the 2 banks, rounds of 2 jobs)           [pynq_run_layer.run_layer]
  requant/ReLU/pool, flatten, FC layers, softmax -> ARM (numpy, bit-exact integer model, vgg_int.py)

On the board (as root; copy scripts/*.py, nmc.bit/.hwh and the model folder next to each other):

  # classify your own picture(s): resize/crop/normalise/quantise happen here (PIL + numpy, no torch)
  sudo -E python3 pynq_run_network.py --bit nmc.bit --model models/vgg16 --tilings models/vgg16/tilings.json \
       --image cat.jpg --classes models/vgg16/classes.txt

  # check against the PC (see classify_pc.py --save-ref):  --input input_q_cat.npy --ref ref_cat.npz
  # the model's own test input:                           (no --image)  uses <model>/input_q.npy and --ref ref.npz
  # first few layers only (debug):                         --max-layers 3

With --ref every layer's int8 activation is compared with the integer reference and the run stops at the first
difference. Offline self-test: --emit ops.txt (no hardware) + tb/integration/nmc_axi_replay_tb.v.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiling_plan as tp
import vgg_int as vi
from imagenet_pre import preprocess, quantize_input, load_classes, print_top5
from pynq_run_layer import run_layer
from pynq_bringup import HwBackend, RecordBackend, G

def main():
    ap = argparse.ArgumentParser(description="Whole-network inference on the PYNQ-Z2 NMC cluster")
    ap.add_argument("--bit"); ap.add_argument("--emit")
    ap.add_argument("--model", required=True); ap.add_argument("--tilings")
    ap.add_argument("--image", nargs="+", help="one or more image files to classify")
    ap.add_argument("--classes", help="class names, one per line (default <model>/classes.txt if present)")
    ap.add_argument("--input", help="already quantised int8 input [C,H,W] .npy (default <model>/input_q.npy)")
    ap.add_argument("--ref", help="integer reference npz (one input only)")
    ap.add_argument("--cfg-base", type=lambda x: int(x, 0), default=0x40000000)
    ap.add_argument("--mem-base", type=lambda x: int(x, 0), default=0x40080000)
    ap.add_argument("--banks", type=int, default=2); ap.add_argument("--bank-words", type=int, default=32768)
    ap.add_argument("--mhz", type=float, default=60.0); ap.add_argument("--no-burst", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="stop after this many layers (debug)")
    ap.add_argument("--save", help="write the int8 activations and logits of the (last) input to this npz")
    a = ap.parse_args()

    model = vi.load_model(a.model)
    crop, xs = model["input"]["shape"][1], model["input"]["x_scale"]
    classes = load_classes(a.classes or os.path.join(a.model, "classes.txt"))
    if classes is None and a.image:
        print("(no class names file: showing class indices; make one with scripts/export_imagenet_classes.py)")
    inputs = []
    if a.image:
        if a.ref and len(a.image) > 1: ap.error("--ref works with a single input")
        for f in a.image:
            inputs.append((os.path.basename(f), quantize_input(preprocess(f, crop), xs)))
    else:
        inputs.append((a.input or "input_q.npy", np.load(a.input or os.path.join(a.model, "input_q.npy")).astype(np.int8)))
    ref = np.load(a.ref) if a.ref else None
    til = json.load(open(a.tilings))["layers"] if a.tilings else {}
    tp.NUM_BANKS = a.banks

    if a.emit:
        be = RecordBackend(a.banks, a.bank_words)
    else:
        if not a.bit: ap.error("--bit is required (or use --emit)")
        from pynq import Overlay
        Overlay(a.bit)
        try:
            from pynq import Clocks
            a.mhz = float(Clocks.fclk0_mhz); print(f"PL clock FCLK0 = {a.mhz:.2f} MHz")
        except Exception: pass
        be = HwBackend(a.cfg_base, a.mem_base, a.banks, a.bank_words)
        if a.no_burst: be.burst = False
        be.cfg_w(G["IRQEN"], 0)
    ra = argparse.Namespace(banks=a.banks, bank_words=a.bank_words, policy="spatial", rounds=0, golden=False,
                            mhz=a.mhz, quiet=True)

    ok_all, last = True, None
    for name, x in inputs:
        print(f"\n=== {name} ===")
        t_all = time.perf_counter()
        rows, tot = [], dict(crit=0, macs=0, load=0.0, run=0.0, drain=0.0, host=0.0)
        acts, logits, ok = [], None, True
        H, W = x.shape[1:]
        print(f"{'layer':9} {'shape':>16} {'NMC cycles':>11} {'load s':>7} {'run s':>7} {'drain s':>8} {'host s':>7}  check")
        for i, L in enumerate(model["layers"]):
            if a.max_layers and i >= a.max_layers: break
            if L["type"] == "conv":
                layer = tp.Layer(L["name"], H, W, L["cin"], L["cout"])
                t = til.get(L["name"])
                if t is None:
                    print(f"   no tiling for {L['name']} in --tilings, running the auto search (slow on the ARM)...")
                    p = tp.auto_tiling(layer); t = dict(n_if=p.tiling.n_if, tc=p.tiling.tc, tk=p.tiling.tk, policy=p.policy)
                ra.policy = t["policy"]
                out = run_layer(be, ra, layer, tp.Tiling(t["n_if"], t["tc"], t["tk"]), x, np.asarray(L["w"]))
                if out is None: ok = False; break
                acc, have, _ = out
                st = ra.last_stats
                th = time.perf_counter()
                y = vi.requant(acc, L["bias"], L["mult"], L["shift"], L["relu"])
                if L.get("pool", 1) > 1: y = vi.maxpool(y, L["pool"])
                x = y.astype(np.int8); H, W = x.shape[1:]
                del acc, y
                th = time.perf_counter() - th
                tot["crit"] += st["crit"]; tot["macs"] += st["macs"]
                tot["load"] += st["T"]["load"] + st["T"]["cfg"]; tot["run"] += st["T"]["run"]; tot["drain"] += st["T"]["drain"]; tot["host"] += th
                shape = f"{L['cin']}x{layer.H}x{layer.W}->{L['cout']}"
                line = (f"{L['name']:9} {shape:>16} {st['crit']:>11} {st['T']['load'] + st['T']['cfg']:>7.2f} {st['T']['run']:>7.3f} "
                        f"{st['T']['drain']:>8.2f} {th:>7.2f}")
                acts.append(x)
            else:
                th = time.perf_counter()
                acc = vi.fc_acc(x, np.asarray(L["w"])) + np.asarray(L["bias"], dtype=np.int64)
                if L.get("final"):
                    logits = acc.astype(np.float64) * np.asarray(L["scale"], dtype=np.float64)
                    x = logits
                else:
                    x = vi.requant(acc, np.zeros_like(acc), L["mult"], L["shift"], L["relu"]).astype(np.int8)
                    acts.append(x)
                th = time.perf_counter() - th; tot["host"] += th
                line = f"{L['name']:9} {str(L['nin']) + '->' + str(L['nout']):>16} {'(ARM)':>11} {'':>7} {'':>7} {'':>8} {th:>7.2f}"
            chk = ""
            if ref is not None and not L.get("final"):
                same = np.array_equal(acts[-1], ref[f"act{len(acts) - 1}"])
                chk = "OK" if same else "MISMATCH"; ok &= same
            print(f"{line}  {chk}", flush=True)
            if chk == "MISMATCH": break

        wall = time.perf_counter() - t_all
        if logits is not None:
            print("top-5:"); print_top5(logits, classes)
            if ref is not None:
                same = np.allclose(logits, ref["logits"], rtol=0, atol=1e-9)
                print(f"logits identical to the integer reference: {same}"); ok &= bool(same)
        if tot["crit"]:
            print(f"NMC compute {tot['crit']} cycles = {tot['crit'] / (a.mhz * 1e6):.3f} s at {a.mhz:.1f} MHz ({tot['macs']} MACs, "
                  f"{2 * tot['macs'] / tot['crit'] * a.mhz / 1e3:.2f} GOPS compute-only)")
        print(f"host: load {tot['load']:.1f} s  run {tot['run']:.2f} s  drain {tot['drain']:.1f} s  requant/pool/fc {tot['host']:.1f} s;  wall {wall:.1f} s")
        last = (logits, acts); ok_all &= ok
    if a.emit:
        be.dump(a.emit); print(f"wrote {len(be.ops)} operations to {a.emit}")
    elif a.save and last is not None:
        np.savez(a.save, logits=last[0] if last[0] is not None else np.zeros(1), **{f"act{i}": t for i, t in enumerate(last[1])})
    ok_all &= be.errors == 0
    print("\nNETWORK RUN PASSED" if ok_all else "\nNETWORK RUN FAILED")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()