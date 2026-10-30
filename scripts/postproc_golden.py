#!/usr/bin/env python3
"""
NumPy golden-model regression for the post-processing unit (max-pool -> requant -> ReLU).

  python3 scripts/postproc_golden.py            # all cases, quiet + core-noise
  python3 scripts/postproc_golden.py --seed 7

Golden (bit-exact, int64):
  p = maxpool_k(acc)                       (k = 1: identity; rows/cols beyond out*k dropped)
  v = (p + bias[c]) * mult[c]
  q = (v + (1 << (shift-1))) >> shift      (shift = 0: no rounding)      arithmetic shift
  y = clip(q + zp, lo, 127),  lo = zp if relu else -128

The golden pools AFTER requant to prove the hardware's pool-before-requant ordering is exact.
"""
import argparse, glob, os, subprocess, sys
import numpy as np

DATA = "sim/data"

# rows cols ch pool relu zp  mag(log2 of |acc|)
CASES = [
    (8, 8, 2, 2, 1, 0, 12),
    (8, 8, 2, 1, 1, 0, 12),      # no pooling
    (16, 16, 3, 2, 0, 5, 14),    # no ReLU, non-zero zero-point
    (14, 14, 4, 2, 1, -7, 16),   # 7x7 pooled output (odd out dims)
    (13, 17, 2, 2, 1, 3, 12),    # odd src dims: last row/col dropped
    (6, 10, 5, 2, 1, 0, 30),     # large accumulators -> saturation
    (4, 4, 1, 2, 1, 0, 8),       # single output row
    (24, 24, 8, 2, 1, -128, 20), # zp at the int8 limit
]

def golden(acc, bias, mult, shift, zp, relu, pool):
    ch, rows, cols = acc.shape
    orows, ocols = rows // pool, cols // pool
    a = acc[:, :orows*pool, :ocols*pool].astype(np.int64)
    v = (a + bias[:, None, None].astype(np.int64)) * mult[:, None, None].astype(np.int64)
    sh = shift[:, None, None].astype(np.int64)
    rnd = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    q = (v + rnd) >> sh
    lo = zp if relu else -128
    y = np.clip(q + zp, lo, 127)
    y = y.reshape(ch, orows, pool, ocols, pool).max(axis=(2, 4))     # pool after requant
    return y.astype(np.int64)

def wh(path, words):
    with open(path, "w") as f:
        for w in words: f.write("%08x\n" % (int(w) & 0xFFFFFFFF))

def run_case(case, noise, rng, vvp):
    rows, cols, ch, pool, relu, zp, mag = case
    acc = rng.integers(-(1 << mag), 1 << mag, size=(ch, rows, cols), dtype=np.int64)
    acc = np.clip(acc, -(1 << 31), (1 << 31) - 1)
    bias = rng.integers(-(1 << 10), 1 << 10, size=ch, dtype=np.int64)
    mult = rng.integers(1, 1 << 16, size=ch, dtype=np.int64)
    shift = rng.integers(0, 32, size=ch, dtype=np.int64)
    shift[0] = 0 if ch > 1 else shift[0]                              # exercise shift = 0
    # pick shifts so results are not all saturated: aim near the int8 range
    for c in range(ch):
        if c % 2 == 1 or mag < 20:
            need = max(0, int(np.ceil(np.log2(max(1, (abs(acc[c]).max() + 1024)) * int(mult[c]) / 100.0))))
            shift[c] = min(31, need)
    gold = golden(acc, bias, mult, shift, zp, relu, pool)

    wh(f"{DATA}/pp_src.hex", acc.flatten())
    qp = []
    for c in range(ch): qp += [bias[c], mult[c], shift[c]]
    wh(f"{DATA}/pp_qp.hex", qp)
    if os.path.exists(f"{DATA}/pp_rtl.hex"): os.remove(f"{DATA}/pp_rtl.hex")

    args = ["vvp", vvp, f"+ROWS={rows}", f"+COLS={cols}", f"+CH={ch}", f"+POOL={pool}",
            f"+RELU={relu}", f"+ZP={zp}", f"+NOISE={noise}"]
    res = subprocess.run(args, capture_output=True, text=True)
    tag = f"{ch}x{rows}x{cols} pool{pool} relu={relu} zp={zp} mag=2^{mag} noise={noise}"
    if "TIMEOUT" in res.stdout or not os.path.exists(f"{DATA}/pp_rtl.hex"):
        print(f"FAIL  {tag}: timeout / no output\n{res.stdout[-300:]}{res.stderr[-300:]}"); return False
    lines = [l.strip() for l in open(f"{DATA}/pp_rtl.hex") if l.strip()]
    n = gold.size
    try: words = np.array([int(x, 16) for x in lines[:n]], dtype=np.uint64)
    except ValueError: print(f"FAIL  {tag}: X/Z in read-back"); return False
    upper_ok = bool(np.all(words >> 8 == 0))                           # low byte only, upper 24 bits zero
    got = (words & 0xFF).astype(np.uint8).view(np.int8).astype(np.int64).reshape(gold.shape)
    guard_ok = all(l.lower() == "deadbeef" for l in lines[n:n+16]) and len(lines) >= n + 16
    bad = np.argwhere(got != gold)
    sat = 100.0 * np.mean((gold == 127) | (gold == (zp if relu else -128)))
    ok = len(bad) == 0 and upper_ok and guard_ok
    cyc = res.stdout.split("CYCLES=")[1].split()[0] if "CYCLES=" in res.stdout else "?"
    print(f"{'PASS' if ok else 'FAIL'}  {tag}  cycles={cyc} ({int(cyc)/n:.2f}/out)  at-clamp={sat:.0f}%")
    if len(bad):
        print(f"      {len(bad)}/{got.size} differ; first: " +
              ", ".join(f"[{c},{r},{x}] got {got[c,r,x]} gold {gold[c,r,x]}" for c, r, x in bad[:4]))
    if not upper_ok: print("      upper 24 bits of a dst word are non-zero")
    if not guard_ok: print("      guard words after the output were overwritten")
    return ok

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True); os.makedirs("sim/waveform", exist_ok=True); os.makedirs("sim/build", exist_ok=True)
    vvp = "sim/build/postproc.vvp"
    rtl = sorted(glob.glob("rtl/memory/*.v") + glob.glob("rtl/postproc/*.v"))
    r = subprocess.run(["iverilog", "-g2012", "-o", vvp, "tb/integration/postproc_tb.v"] + rtl, capture_output=True, text=True)
    if r.returncode: print(r.stderr); sys.exit(2)
    rng = np.random.default_rng(a.seed)
    res = [run_case(c, nz, rng, vvp) for c in CASES for nz in (0, 1)]
    print(f"\n{sum(res)}/{len(res)} cases match the NumPy golden model")
    sys.exit(0 if all(res) else 1)

if __name__ == "__main__":
    main()