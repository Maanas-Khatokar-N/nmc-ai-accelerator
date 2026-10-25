#!/usr/bin/env python3
"""
NumPy golden-model regression for the NMC unit.

  python3 scripts/layer_golden.py            # all cases, quiet + core-noise
  python3 scripts/layer_golden.py --quick    # first 3 cases, quiet only
  python3 scripts/layer_golden.py --seed 7

For each case: random int8 IFMap / weights -> NumPy int32 convolution (golden)
-> hex files -> iverilog/vvp -> OFMap dumped from SRAM -> compared bit-exactly.
Layouts (one element per 32-bit word, same as the testbenches):
  IFMap   [ic][r][c]          at word 0
  weights [oc][ic][kr][kc]    at word 8192
  OFMap   [oc][r][c]          at word 12288  (+16 guard words = 0xDEADBEEF)
"""
import argparse, glob, os, re, subprocess, sys
import numpy as np

DATA = "sim/data"
PE = 16

# ifr ifc kr kc cin cout stride
CASES = [
    (18, 18, 3, 3, 3, 8, 1),   # 16x16 OFMap: 16 full tiles, 8 output channels
    (15, 15, 3, 3, 3, 4, 1),   # 13x13 OFMap: partial edge tiles
    (20, 20, 3, 3, 2, 5, 2),   # stride 2, 9x9 OFMap
    (14, 14, 5, 5, 1, 3, 1),   # 5x5 kernel
    (13, 17, 2, 3, 3, 3, 1),   # rectangular IFMap, 2x3 kernel
    (10, 10, 1, 1, 7, 4, 1),   # 1x1 kernel, 7 input channels
    (34, 34, 3, 3, 3, 4, 1),   # 32x32 OFMap, 64 tiles (VGG CONV1-like, Cin=3)
]


def conv_golden(ifm, wgt, s):
    """ifm [Cin,H,W] int8, wgt [Cout,Cin,Kr,Kc] int8 -> [Cout,OH,OW] int32 (valid conv)."""
    cin, h, w = ifm.shape
    cout, _, kr, kc = wgt.shape
    oh, ow = (h - kr) // s + 1, (w - kc) // s + 1
    out = np.zeros((cout, oh, ow), dtype=np.int64)
    for ki in range(kr):
        for kj in range(kc):
            win = ifm[:, ki:ki + (oh - 1) * s + 1:s, kj:kj + (ow - 1) * s + 1:s].astype(np.int64)
            out += np.einsum("oc,cxy->oxy", wgt[:, :, ki, kj].astype(np.int64), win)
    return out.astype(np.int32)


def write_hex(path, words):
    with open(path, "w") as f:
        for w in words:
            f.write("%08x\n" % (int(w) & 0xFFFFFFFF))


def check_fits(ifr, ifc, kr, kc, cin, cout, s):
    pr, pc = 3 * s + kr, 3 * s + kc
    per_ch = pr * pc
    oh, ow = (ifr - kr) // s + 1, (ifc - kc) // s + 1
    assert per_ch <= 255, f"patch {pr}x{pc} per channel does not fit 8-bit rf_ch_stride"
    assert cin * per_ch <= 256, f"cin*patch={cin*per_ch} > 256 (IFMap RF) -> split Cin in software"
    assert cin * kr * kc <= 256, f"weights_per_oc={cin*kr*kc} > 256 (weight RF)"
    assert cin * ifr * ifc <= 8192 and cout * cin * kr * kc <= 4096
    assert cout * oh * ow + 16 <= 20480
    return oh, ow


def run_case(case, noise, rng, vvp):
    ifr, ifc, kr, kc, cin, cout, s = case
    oh, ow = check_fits(*case)
    ifm = rng.integers(-128, 128, size=(cin, ifr, ifc), dtype=np.int8)
    wgt = rng.integers(-128, 128, size=(cout, cin, kr, kc), dtype=np.int8)
    gold = conv_golden(ifm, wgt, s)

    write_hex(f"{DATA}/ifmap.hex",   ifm.astype(np.int64).flatten() & 0xFF)
    write_hex(f"{DATA}/weights.hex", wgt.astype(np.int64).flatten() & 0xFF)
    write_hex(f"{DATA}/golden.hex",  gold.astype(np.int64).flatten())
    if os.path.exists(f"{DATA}/ofmap_rtl.hex"):
        os.remove(f"{DATA}/ofmap_rtl.hex")

    args = ["vvp", vvp, f"+IFR={ifr}", f"+IFC={ifc}", f"+KR={kr}", f"+KC={kc}",
            f"+CIN={cin}", f"+COUT={cout}", f"+S={s}", f"+NOISE={noise}"]
    res = subprocess.run(args, capture_output=True, text=True)
    out = res.stdout
    tag = f"ifm {ifr}x{ifc}x{cin} K{kr}x{kc} Cout{cout} S{s} noise={noise}"

    if "TIMEOUT" in out or not os.path.exists(f"{DATA}/ofmap_rtl.hex"):
        print(f"FAIL  {tag}: timeout / no output\n{out[-400:]}{res.stderr[-400:]}")
        return False

    lines = [l.strip() for l in open(f"{DATA}/ofmap_rtl.hex") if l.strip()]
    n = cout * oh * ow
    try:
        words = np.array([int(x, 16) for x in lines[:n]], dtype=np.uint64)
    except ValueError:
        print(f"FAIL  {tag}: X/Z in OFMap read-back")
        return False
    got = words.astype(np.uint32).view(np.int32).reshape(cout, oh, ow)
    guard_ok = all(l.lower() == "deadbeef" for l in lines[n:n + 16]) and len(lines) >= n + 16

    bad = np.argwhere(got != gold)
    ok = (len(bad) == 0) and guard_ok

    m = re.search(r"PERF total=(\d+) mac=(\d+) ops=(\d+) stall=(\d+) blocked=(\d+)", out)
    perf = ""
    if m:
        total, mac, ops, stall, blk = map(int, m.groups())
        act = 100.0 * ops / (PE * mac) if mac else 0.0
        eff = 100.0 * ops / (PE * total) if total else 0.0
        perf = f"cycles={total} PE-activity={act:.1f}% layer-eff={eff:.1f}% stalls={stall} blocked={blk}"
    print(f"{'PASS' if ok else 'FAIL'}  {tag}  {perf}")
    if len(bad):
        print(f"      {len(bad)}/{got.size} words differ; first mismatches (oc,r,c: got vs gold):")
        for oc, r, c in bad[:5]:
            print(f"        {oc},{r},{c}: {got[oc, r, c]} vs {gold[oc, r, c]}")
    if not guard_ok:
        print("      guard words after the OFMap were overwritten")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    os.makedirs("sim/waveform", exist_ok=True)
    vvp = "sim/build/layer_golden.vvp"
    rtl = sorted(glob.glob("rtl/**/*.v", recursive=True))
    cmd = ["iverilog", "-g2012", "-o", vvp, "tb/integration/layer_golden_tb.v"] + rtl
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr); sys.exit(2)

    rng = np.random.default_rng(a.seed)
    cases = CASES[:3] if a.quick else CASES
    noises = [0] if a.quick else [0, 1]
    results = [run_case(c, nz, rng, vvp) for c in cases for nz in noises]
    print(f"\n{sum(results)}/{len(results)} cases match the NumPy golden model")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()