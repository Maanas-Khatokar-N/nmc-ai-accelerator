#!/usr/bin/env python3
"""
Cluster-level regression for the post-processing units (4 banks, concurrent, different shapes per bank).

  python3 scripts/cluster_pp_golden.py

Per case: every enabled bank gets its own random int32 OFMap, requant parameters and geometry.
The host fills the banks, starts one post-proc round (pp_start), waits for pp_irq, drains each bank's
int8 output and compares bit-exactly with the NumPy golden of postproc_golden.py.
Also checks: units overlap in time, pp_irq waits for the slowest bank, a conv `start` issued during the
round is ignored (unit_busy stays 0), and guard words after each output are untouched.
"""
import argparse, glob, os, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from postproc_golden import golden, wh

DATA = "sim/data"; REC = 16; NB = 4
SRC_BASE, DST_BASE = 0, 16384

# per bank: (rows, cols, ch, pool, relu, zp, mag)  or None (bank idle this round)
CASES = [
    [(8, 8, 2, 2, 1, 0, 12), (16, 16, 4, 2, 0, 5, 14), (12, 12, 3, 1, 1, -3, 12), (6, 10, 5, 2, 1, 0, 30)],
    [None, (14, 14, 4, 2, 1, -7, 16), None, (4, 4, 1, 2, 1, 0, 8)],
    [None, None, (24, 24, 8, 2, 1, -128, 20), None],
    [(10, 10, 3, 2, 1, 0, 16)] * 4,
]

def mk(rng, spec):
    rows, cols, ch, pool, relu, zp, mag = spec
    acc = np.clip(rng.integers(-(1 << mag), 1 << mag, size=(ch, rows, cols), dtype=np.int64), -(1 << 31), (1 << 31) - 1)
    bias = rng.integers(-(1 << 10), 1 << 10, size=ch, dtype=np.int64)
    mult = rng.integers(1, 1 << 16, size=ch, dtype=np.int64)
    shift = np.zeros(ch, dtype=np.int64)
    for c in range(ch):
        need = max(0, int(np.ceil(np.log2(max(1, (abs(acc[c]).max() + 1024)) * int(mult[c]) / 100.0))))
        shift[c] = min(31, need) if c % 3 != 0 else 0
    return acc, bias, mult, shift

def run_case(ci, specs, noise, rng, vvp):
    prog = np.zeros((NB, REC), dtype=np.int64); gold = {}
    for b, s in enumerate(specs):
        if s is None: continue
        rows, cols, ch, pool, relu, zp, mag = s
        acc, bias, mult, shift = mk(rng, s)
        gold[b] = golden(acc, bias, mult, shift, zp, relu, pool)
        orows, ocols = rows // pool, cols // pool
        wh(f"{DATA}/ppc_b{b}_src.hex", acc.flatten())
        qp = []
        for c in range(ch): qp += [bias[c], mult[c], shift[c]]
        wh(f"{DATA}/ppc_b{b}_qp.hex", qp)
        prog[b] = [1, cols, orows, ocols, ch, pool, rows*cols, pool*cols, ch*orows*ocols,
                   zp & 0xFF, relu, SRC_BASE, DST_BASE, acc.size, 0, 0]
    with open(f"{DATA}/pp_prog.hex", "w") as f:
        for w in prog.flatten(): f.write("%08x\n" % (int(w) & 0xFFFFFFFF))
    for b in range(NB):
        p = f"{DATA}/ppc_b{b}_dst.hex"
        if os.path.exists(p): os.remove(p)
    res = subprocess.run(["vvp", vvp, f"+NOISE={noise}"], capture_output=True, text=True)
    out = res.stdout
    tag = f"case{ci} banks={[b for b in gold]} noise={noise}"
    if "TIMEOUT" in out or "CLUSTERPP" not in out or "ERROR" in out:
        print(f"FAIL  {tag}: sim problem\n{out[-400:]}{res.stderr[-300:]}"); return False
    ok = True
    for b, g in gold.items():
        lines = [l.strip() for l in open(f"{DATA}/ppc_b{b}_dst.hex") if l.strip()]
        n = g.size
        try: w = np.array([int(x, 16) for x in lines[:n]], dtype=np.uint64)
        except ValueError: print(f"FAIL  {tag} bank{b}: X/Z"); ok = False; continue
        got = (w & 0xFF).astype(np.uint8).view(np.int8).astype(np.int64).reshape(g.shape)
        bad = int(np.count_nonzero(got != g)); up = bool(np.all(w >> 8 == 0))
        guard = len(lines) >= n + 16 and all(l.lower() == "deadbeef" for l in lines[n:n+16])
        if bad or not up or not guard:
            ok = False
            print(f"FAIL  {tag} bank{b}: {bad}/{n} differ, upper_ok={up}, guard_ok={guard}")
    m = out.split("CLUSTERPP")[1].split("\n")[0]
    vals = dict(kv.split("=") for kv in m.split())
    if int(vals["conv_busy"]) != 0: print(f"FAIL  {tag}: conv start was NOT ignored during post-proc"); ok = False
    if len(gold) > 1 and int(vals["max_busy"]) < len(gold): print(f"FAIL  {tag}: units did not overlap"); ok = False
    print(f"{'PASS' if ok else 'FAIL'}  {tag}: max units busy={vals['max_busy']} overlap={vals['overlap_cycles']}/{vals['wall']} cycles, irq at {vals['irq_cycles']}")
    return ok

def main():
    argparse.ArgumentParser().parse_args()
    os.makedirs(DATA, exist_ok=True); os.makedirs("sim/waveform", exist_ok=True); os.makedirs("sim/build", exist_ok=True)
    vvp = "sim/build/cluster_pp.vvp"
    rtl = sorted(glob.glob("rtl/**/*.v", recursive=True))
    r = subprocess.run(["iverilog", "-g2012", "-o", vvp, "tb/integration/cluster_pp_tb.v"] + rtl, capture_output=True, text=True)
    if r.returncode: print(r.stderr); sys.exit(2)
    rng = np.random.default_rng(1)
    res = [run_case(i, c, nz, rng, vvp) for i, c in enumerate(CASES) for nz in (0, 1)]
    print(f"\n{sum(res)}/{len(res)} cluster post-proc cases match the NumPy golden model")
    sys.exit(0 if all(res) else 1)

if __name__ == "__main__":
    main()