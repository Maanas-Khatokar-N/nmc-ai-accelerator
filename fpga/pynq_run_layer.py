#!/usr/bin/env python3
"""
Run a tiled VGG-16 convolution layer on the NMC cluster of the PYNQ-Z2 (2 banks), driven from the ARM.

On the board (as root, scripts/ folder + nmc.bit/.hwh copied over, tiling_plan.py and pynq_bringup.py next to this file):

  # quick sanity: CONV1-1 shrunk 8x (28x28x3 -> 64), random int8 data, checked against NumPy
  sudo -E python3 pynq_run_layer.py --bit nmc.bit --layer 1-1 --scale 8 --n-if 2 --tk 16

  # the real CONV1-1, paper tiling (4x4 spatial tiles, 16 kernel tiles, 128 rounds on 2 banks)
  sudo -E python3 pynq_run_layer.py --bit nmc.bit --layer 1-1 --n-if 4 --tc 1 --tk 16 --npz vgg16_conv1_1_quant.npz --save conv1_1_out.npz

  # first rounds only (cheap):         add  --rounds 4
  # skip the (slow on ARM) NumPy check: add  --no-golden
  # no burst transfers (debug):         add  --no-burst

Flow per round (same as tb/integration/cluster_tb.v): the host loads IFMap tile / weights into each bank
(skipped when the bank already holds them), writes the descriptors, starts all banks of the round together
(UNIT_MASK + CTRL), waits for the round to finish, and drains the OFMap tiles after the last Cin pass.
Output is the raw int32 accumulator [Cout, OH, OW] (INT8 x INT8 convolution, bit exact vs NumPy).

Offline self-test: --emit ops.txt writes the access sequence, tb/integration/nmc_axi_replay_tb.v replays it on the RTL.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiling_plan as tp
from pynq_bringup import HwBackend, RecordBackend, G, BA, CONV_F, PERF_F, conv_golden

if not (hasattr(HwBackend, "mem_read_many") and hasattr(RecordBackend, "mem_read_many")):
    sys.exit("The pynq_bringup.py next to this script is an OLD version (no bulk memory read/write).\n"
             "Copy the current pynq_bringup.py to the board (same folder as pynq_run_layer.py) and run again.")

def load_data(a, layer):
    if a.npz:
        d = np.load(a.npz)
        x_q, w_q = d["x_q"].astype(np.int8), d["w_q"].astype(np.int8)
        extra = {k: d[k] for k in d.files if k not in ("x_q", "w_q")}
        if x_q.shape != (layer.Cin, layer.H, layer.W) or w_q.shape != (layer.Cout, layer.Cin, layer.K, layer.K):
            sys.exit(f"npz shapes {x_q.shape} / {w_q.shape} do not match layer {layer}")
        return x_q, w_q, extra
    rng = np.random.default_rng(a.seed)
    return (rng.integers(-128, 128, (layer.Cin, layer.H, layer.W), dtype=np.int8),
            rng.integers(-128, 128, (layer.Cout, layer.Cin, layer.K, layer.K), dtype=np.int8), {})

def run_layer(be, a, layer, tiling, ifm, wgt):
    nb = a.banks
    tp.NUM_BANKS = nb
    verbose = not getattr(a, "quiet", False)
    plan = tp.build_plan(layer, tiling, a.policy, bank_words=a.bank_words, nb=nb)
    rounds = plan.rounds[:a.rounds] if a.rounds else plan.rounds
    passes = plan.passes
    S, K = layer.S, layer.K
    ifm_pad = np.pad(ifm, ((0, 0), (layer.pad, layer.pad), (layer.pad, layer.pad)))
    need_gold = a.golden or isinstance(be, RecordBackend)
    t = time.perf_counter()
    gold = conv_golden(ifm_pad, wgt, S) if need_gold else None
    if need_gold and verbose: print(f"   NumPy golden: {time.perf_counter() - t:.1f} s")

    if verbose: print(f"   layer {layer.name}: {layer.H}x{layer.W}x{layer.Cin} -> {layer.Cout}, K{K} S{S} pad{layer.pad}")
    if verbose: print(f"   tiling n_if={tiling.n_if} tc={tiling.tc} tk={tiling.tk} ({a.policy}): {plan.stats['jobs']} jobs, "
          f"{len(plan.rounds)} rounds on {nb} banks, {len(passes)} Cin pass(es), {plan.words_used} words/bank "
          f"of {a.bank_words}; running {len(rounds)} rounds")

    res_if, res_w = [None] * nb, [None] * nb
    got = np.zeros((layer.Cout, layer.OH, layer.OW), dtype=np.int32)
    have = np.zeros(got.shape, dtype=bool)
    T = dict(load=0.0, cfg=0.0, run=0.0, drain=0.0)
    crit = busy = mac = ops = stall = blocked = macs_done = words_in = words_out = 0

    for r, rnd in enumerate(rounds):
        for p, (c0, c1) in enumerate(passes):
            cg, last, mask = c1 - c0, p == len(passes) - 1, 0
            for b, j in enumerate(rnd):
                if j is None: continue
                mask |= 1 << b
                ih, iw = tp.in_extent(j.oh, layer), tp.in_extent(j.ow, layer)
                base = b * a.bank_words
                t0 = time.perf_counter()
                if res_if[b] != (j.sid, c0, c1):
                    res_if[b] = (j.sid, c0, c1)
                    tile = ifm_pad[c0:c1, j.oy0 * S:j.oy0 * S + ih, j.ox0 * S:j.ox0 * S + iw]
                    be.mem_write_many(base + plan.if_base, tile.astype(np.int64).flatten() & 0xFF)
                    words_in += tile.size
                if res_w[b] != (j.kt, c0, c1):
                    res_w[b] = (j.kt, c0, c1)
                    wk = wgt[j.k0:j.k1, c0:c1]
                    be.mem_write_many(base + plan.w_base, wk.astype(np.int64).flatten() & 0xFF)
                    words_in += wk.size
                t1 = time.perf_counter(); T["load"] += t1 - t0
                patch_c = 3 * S + K
                d = dict(ifmap_cols=iw, kernel_rows=K, kernel_cols=K, input_channels=cg, output_channels=j.nk,
                         stride=S, ofmap_rows=j.oh, ofmap_cols=j.ow, weights_per_oc=cg * K * K,
                         ifmap_ch_size=ih * iw, ofmap_ch_size=j.oh * j.ow, ifmap_row_step=S * iw,
                         ifmap_base=plan.if_base, weight_base=plan.w_base, ofmap_base=plan.of_base,
                         rf_row_step=S * patch_c, rf_ch_stride=patch_c * patch_c, accumulate=0 if p == 0 else 1)
                for k, v in d.items(): be.cfg_w(BA(b, CONV_F[k]), v)
                T["cfg"] += time.perf_counter() - t1
                macs_done += j.oh * j.ow * j.nk * cg * K * K
            t0 = time.perf_counter()
            be.cfg_w(G["UMASK"], mask); be.cfg_w(G["CTRL"], 1)
            if not be.wait_idle(f"round {r} pass {p}"): return None
            be.cfg_w(G["IRQST"], 0xF)
            T["run"] += time.perf_counter() - t0
            tot = []
            for b, j in enumerate(rnd):
                if j is None: continue
                tot.append(be.cfg_r(BA(b, PERF_F["total"])))
                busy += tot[-1]; mac += be.cfg_r(BA(b, PERF_F["mac"])); ops += be.cfg_r(BA(b, PERF_F["ops"]))
                stall += be.cfg_r(BA(b, PERF_F["stall"])); blocked += be.cfg_r(BA(b, PERF_F["blocked"]))
            crit += max(tot) if tot else 0
            if last:
                t0 = time.perf_counter()
                for b, j in enumerate(rnd):
                    if j is None: continue
                    n = j.nk * j.oh * j.ow
                    exp = gold[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] if gold is not None else None
                    words = be.mem_read_many(b * a.bank_words + plan.of_base, n, expect=exp)
                    got[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] = words.astype(np.uint32).view(np.int32).reshape(j.nk, j.oh, j.ow)
                    have[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] = True
                    words_out += n
                T["drain"] += time.perf_counter() - t0
        if verbose and ((r + 1) % 16 == 0 or r + 1 == len(rounds)):
            print(f"   round {r + 1}/{len(rounds)}  elapsed {sum(T.values()):.1f} s", flush=True)

    cover = 100.0 * have.mean()
    ok = True
    if gold is not None:
        bad = int(np.count_nonzero((got != gold) & have))
        ok = bad == 0
        if verbose: print(f"   {'PASS' if ok else 'FAIL'}: {int(have.sum())} OFMap words checked ({cover:.0f}% of the layer), {bad} mismatches")
        if not ok:
            idx = np.argwhere((got != gold) & have)[:5]
            for c, y, x in idx: print(f"      [{c},{y},{x}] got {got[c, y, x]} expected {gold[c, y, x]}")
            be.fail(f"{bad} OFMap mismatches")
    else:
        if verbose: print(f"   {int(have.sum())} OFMap words produced ({cover:.0f}% of the layer), no golden check")

    mhz = a.mhz
    a.last_stats = dict(crit=crit, macs=macs_done, T=dict(T), words_in=words_in, words_out=words_out,
                        rounds=len(rounds), passes=len(passes), jobs=plan.stats["jobs"])
    if crit and verbose:
        sec = crit / (mhz * 1e6)
        print(f"   NMC critical path: {crit} cycles = {sec * 1e3:.2f} ms at {mhz:.1f} MHz;  {macs_done} MACs -> "
              f"{2 * macs_done / sec / 1e9:.2f} GOPS (compute only)")
        print(f"   PE activity {100.0 * ops / (16 * mac) if mac else 0:.1f}%   cluster PE efficiency "
              f"{100.0 * ops / (16 * nb * crit):.1f}%   stalls {stall}   blocked {blocked}")
    if verbose: print(f"   host time: load {T['load']:.2f} s ({words_in} words)  descriptors {T['cfg']:.2f} s  "
          f"compute+wait {T['run']:.2f} s  drain {T['drain']:.2f} s ({words_out} words)")
    return got, have, ok

def main():
    ap = argparse.ArgumentParser(description="Tiled VGG-16 conv layer on the PYNQ-Z2 NMC cluster")
    ap.add_argument("--bit"); ap.add_argument("--emit")
    ap.add_argument("--cfg-base", type=lambda x: int(x, 0), default=0x40000000)
    ap.add_argument("--mem-base", type=lambda x: int(x, 0), default=0x40040000)
    ap.add_argument("--banks", type=int, default=2); ap.add_argument("--bank-words", type=int, default=32768)
    ap.add_argument("--layer", default="1-1"); ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--n-if", type=int, default=4); ap.add_argument("--tc", type=int, default=1)
    ap.add_argument("--tk", type=int, default=16); ap.add_argument("--policy", default="spatial", choices=["spatial", "kernel"])
    ap.add_argument("--rounds", type=int, default=0, help="run only the first N rounds (0 = all)")
    ap.add_argument("--npz", help="quantised input/weights: x_q [Cin,H,W] int8, w_q [Cout,Cin,K,K] int8 (+ optional bias, x_scale, w_scales)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mhz", type=float, default=60.0, help="PL clock, for the throughput numbers")
    ap.add_argument("--no-golden", dest="golden", action="store_false")
    ap.add_argument("--no-burst", action="store_true")
    ap.add_argument("--save", help="write acc (int32) and, if the npz has scales, the dequantised ReLU output")
    a = ap.parse_args()

    layer = tp.layer_by_name(a.layer, a.scale)
    tiling = tp.Tiling(a.n_if, a.tc, a.tk)
    ifm, wgt, extra = load_data(a, layer)

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

    out = run_layer(be, a, layer, tiling, ifm, wgt)
    if a.emit:
        be.dump(a.emit); print(f"wrote {len(be.ops)} operations to {a.emit}")
    if out is not None and a.save and not a.emit:
        got, have, _ = out
        d = dict(acc=got, have=have)
        if "x_scale" in extra and "w_scales" in extra:
            deq = got.astype(np.float32) * float(extra["x_scale"]) * extra["w_scales"][:, None, None]
            if "bias" in extra: deq = deq + extra["bias"][:, None, None]
            d["relu_float"] = np.maximum(deq, 0.0)
        np.savez(a.save, **d); print(f"   saved {a.save}")
    ok = out is not None and be.errors == 0
    print("LAYER RUN PASSED" if ok else "LAYER RUN FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
