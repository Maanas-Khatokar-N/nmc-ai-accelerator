#!/usr/bin/env python3
"""
NumPy golden-model regression for the NMC unit.

  python3 scripts/layer_golden.py            # all cases, quiet + core-noise
  python3 scripts/layer_golden.py --quick    # first 3 cases, quiet only
  python3 scripts/layer_golden.py --seed 7
  python3 scripts/layer_golden.py --cg 2     # force Cin groups of at most 2 channels
                                             # (exercises the psum reload path on small cases)

For each case: random int8 IFMap / weights -> NumPy int32 convolution (golden)
-> hex files -> iverilog/vvp -> OFMap dumped from SRAM -> compared bit-exactly.

Cin splitting: the IFMap RF holds Cg * patch words and the weight RF Cg*Kr*Kc words
(256 each), so layers with more input channels are run as several passes ("Cin groups").
Group 0 stores its partial sums; groups 1.. run with accumulate=1 (the testbench sets it),
reload the OFMap and add on top. Each group is one nmc_unit run.

Layouts (one element per 32-bit word, same as the testbenches):
  IFMap   [ic][r][c]                       at word 0       (groups are contiguous channel ranges)
  weights [g][oc][ic_in_g][kr][kc]         at word 8192    (GROUP-MAJOR: each group repacked so
                                                            weights_per_oc stays contiguous)
  OFMap   [oc][r][c]                       at word 12288   (+16 guard words = 0xDEADBEEF)
"""
import argparse, glob, os, re, subprocess, sys
import numpy as np

DATA = "sim/data"
PE = 16
RF_DEPTH = 256          # IFMap RF and weight RF depth (words)

FORCE_CG = 0            # set by --cg: upper bound on the Cin group size (0 = use the RF limit)

# ifr ifc kr kc cin cout stride
CASES = [
    (18, 18, 3, 3, 3, 8, 1),    # 16x16 OFMap: 16 full tiles, 8 output channels
    (15, 15, 3, 3, 3, 4, 1),    # 13x13 OFMap: partial edge tiles
    (20, 20, 3, 3, 2, 5, 2),    # stride 2, 9x9 OFMap
    (14, 14, 5, 5, 1, 3, 1),    # 5x5 kernel
    (13, 17, 2, 3, 3, 3, 1),    # rectangular IFMap, 2x3 kernel
    (10, 10, 1, 1, 7, 4, 1),    # 1x1 kernel, 7 input channels
    (34, 34, 3, 3, 3, 4, 1),    # 32x32 OFMap, 64 tiles (VGG CONV1-like, Cin=3)
    # ---- Cin splitting (psum RF + reload path) ----
    (10, 10, 3, 3, 64, 4, 1),   # Cin=64: groups 7 x9 + 1, 8x8 OFMap  (VGG CONV1-2 channel count)
    (11, 11, 3, 3, 20, 3, 1),   # 9x9 OFMap (partial edge tiles) + split Cin: 7,7,6
    (21, 21, 3, 3, 10, 2, 2),   # stride 2 (9x9 patch) -> groups 3,3,3,1; 10x10 OFMap
    (9, 9, 5, 5, 12, 2, 1),     # 5x5 kernel (8x8 patch) -> groups 4,4,4
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


# ---------------------------------------------------------------------------
# Cin splitting (software side of the psum reload path)
# ---------------------------------------------------------------------------
def cin_group_max(kr, kc, s):
    """Largest Cin group that fits both RFs.
    IFMap RF : Cg * patch_rows * patch_cols <= 256   (patch = (3*s+kr) x (3*s+kc))
    weight RF: Cg * kr * kc                 <= 256
    rf_ch_stride is 8 bit, so one channel's patch must be <= 255 words."""
    per_ch = (3 * s + kr) * (3 * s + kc)
    assert per_ch <= 255, f"patch {3*s+kr}x{3*s+kc} per channel does not fit 8-bit rf_ch_stride"
    return max(1, min(RF_DEPTH // per_ch, RF_DEPTH // (kr * kc)))


def split_cin(cin, cg_max):
    """Greedy split: as many full groups as possible, remainder last."""
    sizes, rem = [], cin
    while rem > 0:
        sizes.append(min(cg_max, rem))
        rem -= sizes[-1]
    return sizes


def pack_weights_by_group(wgt, groups):
    """[oc][ic][kr][kc] -> group-major [g][oc][ic_in_g][kr][kc] (flat)."""
    chunks, c0 = [], 0
    for g in groups:
        chunks.append(wgt[:, c0:c0 + g].flatten())
        c0 += g
    return np.concatenate(chunks)


def check_fits(ifr, ifc, kr, kc, cin, cout, s):
    cg_max = cin_group_max(kr, kc, s)
    oh, ow = (ifr - kr) // s + 1, (ifc - kc) // s + 1
    assert cin * ifr * ifc <= 8192 and cout * cin * kr * kc <= 4096, "exceeds testbench SRAM regions"
    assert cout * oh * ow + 16 <= 20480
    return oh, ow, cg_max


def run_case(case, noise, rng, vvp):
    ifr, ifc, kr, kc, cin, cout, s = case
    oh, ow, cg_max = check_fits(*case)
    cg = min(cg_max, FORCE_CG) if FORCE_CG else cg_max
    cg = min(cg, cin)
    groups = split_cin(cin, cg)

    ifm = rng.integers(-128, 128, size=(cin, ifr, ifc), dtype=np.int8)
    wgt = rng.integers(-128, 128, size=(cout, cin, kr, kc), dtype=np.int8)
    gold = conv_golden(ifm, wgt, s)

    write_hex(f"{DATA}/ifmap.hex",   ifm.astype(np.int64).flatten() & 0xFF)
    write_hex(f"{DATA}/weights.hex", pack_weights_by_group(wgt, groups).astype(np.int64) & 0xFF)
    write_hex(f"{DATA}/golden.hex",  gold.astype(np.int64).flatten())
    if os.path.exists(f"{DATA}/ofmap_rtl.hex"):
        os.remove(f"{DATA}/ofmap_rtl.hex")

    args = ["vvp", vvp, f"+IFR={ifr}", f"+IFC={ifc}", f"+KR={kr}", f"+KC={kc}",
            f"+CIN={cin}", f"+COUT={cout}", f"+S={s}", f"+NOISE={noise}", f"+CG={cg}"]
    res = subprocess.run(args, capture_output=True, text=True)
    out = res.stdout
    gtxt = f" groups={len(groups)}x{cg}" if len(groups) > 1 else ""
    tag = f"ifm {ifr}x{ifc}x{cin} K{kr}x{kc} Cout{cout} S{s}{gtxt} noise={noise}"

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

    # perf is summed over all Cin groups by the testbench
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
    global FORCE_CG
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cg", type=int, default=0,
                    help="force Cin groups of at most this many channels (tests the psum reload path)")
    a = ap.parse_args()
    FORCE_CG = a.cg

    os.makedirs(DATA, exist_ok=True)
    os.makedirs("sim/waveform", exist_ok=True)
    os.makedirs("sim/build", exist_ok=True)
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