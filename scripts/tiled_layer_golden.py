#!/usr/bin/env python3
"""
Tiled whole-layer regression on the 4-bank cluster (4 NMC units running concurrently).

  python3 scripts/tiled_layer_golden.py --small                      # scaled-down CONV1-1-like layer, all policies
  python3 scripts/tiled_layer_golden.py --layer 1-1 --n-if 4 --tk 16 --rounds 2   # paper's CONV1-1 tiling, first 2 rounds
  python3 scripts/tiled_layer_golden.py --ifm 20 20 --cin 5 --cout 8 --n-if 2 --tc 2 --tk 4 --noise

Flow: random int8 IFMap/weights -> zero pad -> NumPy golden conv -> tiling_plan -> per-round,
per-pass hex images + descriptors (sim/data/prog.hex) -> iverilog cluster_tb -> OFMap tiles read
back from every bank -> stitched -> compared bit-exactly with the golden (only the tiles that ran).
"""
import argparse, os, re, subprocess, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import tiling_plan as tp
from layer_golden import conv_golden, write_hex

DATA = "sim/data"; REC = 24; NB = tp.NUM_BANKS

def build_program(plan, ifm_pad, wgt, max_rounds):
    L = plan.layer
    rounds = plan.rounds[:max_rounds] if max_rounds else plan.rounds
    npass = len(plan.passes)
    prog = np.zeros((len(rounds), npass, NB, REC), dtype=np.int64)
    res_if = [None]*NB; res_w = [None]*NB
    loads = dict(ifw=0, ww=0)
    for r, rnd in enumerate(rounds):
        for p, (c0, c1) in enumerate(plan.passes):
            for b, j in enumerate(rnd):
                if j is None: continue
                cg = c1 - c0
                ih, iw = tp.in_extent(j.oh, L), tp.in_extent(j.ow, L)
                y0, x0 = j.oy0*L.S, j.ox0*L.S
                tile = ifm_pad[c0:c1, y0:y0+ih, x0:x0+iw]
                wk = wgt[j.k0:j.k1, c0:c1]                      # [oc][ic][kr][kc]
                n_if = n_w = 0
                if res_if[b] != (j.sid, c0, c1):
                    res_if[b] = (j.sid, c0, c1)
                    write_hex(f"{DATA}/r{r}_p{p}_b{b}_if.hex", tile.astype(np.int64).flatten() & 0xFF)
                    n_if = tile.size
                if res_w[b] != (j.kt, c0, c1):
                    res_w[b] = (j.kt, c0, c1)
                    write_hex(f"{DATA}/r{r}_p{p}_b{b}_w.hex", wk.astype(np.int64).flatten() & 0xFF)
                    n_w = wk.size
                loads["ifw"] += n_if; loads["ww"] += n_w
                patch_c = 3*L.S + L.K; patch_r = patch_c
                last = (p == npass - 1)
                prog[r, p, b] = [1, iw, L.K, L.K, cg, j.nk, L.S, j.oh, j.ow, ih*iw, j.oh*j.ow,
                                 L.S*iw, L.S*patch_c, patch_r*patch_c, cg*L.K*L.K,
                                 plan.if_base, plan.w_base, plan.of_base,
                                 0 if p == 0 else 1, n_if, n_w, j.nk*j.oh*j.ow if last else 0, 0, 0]
    with open(f"{DATA}/prog.hex", "w") as f:
        for w in prog.flatten(): f.write("%08x\n" % (int(w) & 0xFFFFFFFF))
    return rounds, npass, loads

def check(plan, rounds, npass, gold, tag):
    L = plan.layer
    got = np.zeros_like(gold); have = np.zeros(gold.shape, dtype=bool); bad = 0
    for r, rnd in enumerate(rounds):
        for b, j in enumerate(rnd):
            if j is None: continue
            fn = f"{DATA}/r{r}_p{npass-1}_b{b}_of.hex"
            if not os.path.exists(fn): print(f"FAIL {tag}: missing {fn}"); return False
            lines = [l.strip() for l in open(fn) if l.strip()]
            try: w = np.array([int(x, 16) for x in lines], dtype=np.uint64)
            except ValueError: print(f"FAIL {tag}: X/Z in {fn}"); return False
            t = w.astype(np.uint32).view(np.int32).reshape(j.nk, j.oh, j.ow)
            got[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] = t
            have[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] = True
    n_bad = int(np.count_nonzero((got != gold) & have))
    cover = have.mean()*100
    print(f"{'PASS' if n_bad == 0 else 'FAIL'}  {tag}: {int(have.sum())} OFMap words checked ({cover:.0f}% of layer), {n_bad} mismatches")
    return n_bad == 0

def run(layer, tiling, policy, noise, seed, max_rounds, vvp):
    plan = tp.build_plan(layer, tiling, policy)
    rng = np.random.default_rng(seed)
    ifm = rng.integers(-128, 128, size=(layer.Cin, layer.H, layer.W), dtype=np.int8)
    wgt = rng.integers(-128, 128, size=(layer.Cout, layer.Cin, layer.K, layer.K), dtype=np.int8)
    pad = layer.pad
    ifm_pad = np.pad(ifm, ((0,0),(pad,pad),(pad,pad)))
    gold = conv_golden(ifm_pad, wgt, layer.S)
    for f in glob.glob(f"{DATA}/r*_b*.hex"): os.remove(f)
    rounds, npass, loads = build_program(plan, ifm_pad, wgt, max_rounds)
    tag = (f"{layer.name} {layer.H}x{layer.W}x{layer.Cin}->{layer.Cout} n_if={tiling.n_if} tc={tiling.tc} "
           f"tk={tiling.tk} {policy} noise={noise}")
    res = subprocess.run(["vvp", vvp, f"+NROUNDS={len(rounds)}", f"+NPASS={npass}", f"+NOISE={noise}"],
                         capture_output=True, text=True)
    out = res.stdout
    if "TIMEOUT" in out or "CLUSTER" not in out:
        print(f"FAIL {tag}: sim problem\n{out[-500:]}{res.stderr[-300:]}"); return False
    ok = check(plan, rounds, npass, gold, tag)
    m = re.search(r"max_busy=(\d+) overlap_cycles=(\d+) wall_nmc=(\d+)", out)
    q = re.search(r"PERF crit=(\d+) busy_sum=(\d+) mac=(\d+) ops=(\d+) stall=(\d+) blocked=(\d+)", out)
    if m and q:
        mb, ov, wall = map(int, m.groups()); crit, bs, mac, ops, st, bl = map(int, q.groups())
        units = sum(1 for r in rounds for j in r if j)
        print(f"      rounds={len(rounds)} passes/job={npass} jobs={units} | max units busy at once={mb} "
              f"overlap cycles={ov}/{wall} | NMC critical path={crit} cyc, sum over units={bs}, "
              f"speedup vs 1 unit={bs/crit:.2f}x | PE activity={100*ops/(16*mac) if mac else 0:.1f}% "
              f"cluster PE eff={100*ops/(16*NB*crit) if crit else 0:.1f}% | stalls={st} blocked={bl}")
        print(f"      host traffic: IFMap {loads['ifw']} + weights {loads['ww']} words (residency reuse applied)")
        if mb < min(NB, max(sum(1 for j in r if j) for r in rounds)):
            print("FAIL: units did not overlap"); ok = False
    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer"); ap.add_argument("--ifm", type=int, nargs=2)
    ap.add_argument("--cin", type=int, default=3); ap.add_argument("--cout", type=int, default=8)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--s", type=int, default=1)
    ap.add_argument("--pad", type=int, default=1)
    ap.add_argument("--n-if", type=int, default=2); ap.add_argument("--tc", type=int, default=1)
    ap.add_argument("--tk", type=int, default=2)
    ap.add_argument("--policy", choices=["spatial", "kernel"])
    ap.add_argument("--noise", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=0, help="only run the first N rounds (0 = all)")
    ap.add_argument("--small", action="store_true", help="regression on small layers, both policies, quiet+noise")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True); os.makedirs("sim/waveform", exist_ok=True); os.makedirs("sim/build", exist_ok=True)
    vvp = "sim/build/cluster.vvp"
    rtl = sorted(glob.glob("rtl/**/*.v", recursive=True))
    r = subprocess.run(["iverilog", "-g2012", "-o", vvp, "tb/integration/cluster_tb.v"] + rtl, capture_output=True, text=True)
    if r.returncode: print(r.stderr); sys.exit(2)

    ok = True
    if a.small:
        cases = [  # layer, n_if, tc, tk
            (tp.Layer("c11s", 24, 24, 3, 8), 2, 1, 4),       # 4 spatial tiles x 4 ktiles   (CONV1-1 shape, scaled)
            (tp.Layer("edge", 19, 19, 3, 6), 3, 1, 3),       # 9 spatial tiles (6,6,5 edges), 3 ktiles, 9 jobs -> idle bank in last round
            (tp.Layer("cin",  14, 14, 20, 4), 2, 2, 2),      # channel tiles + RF sub-split -> accumulate passes
            (tp.Layer("str2", 25, 25, 4, 4, S=2, pad=1), 2, 1, 2),
        ]
        for L, n, c, k in cases:
            for pol in ("spatial", "kernel"):
                for nz in (0, 1):
                    ok &= run(L, tp.Tiling(n, c, k), pol, nz, a.seed, 0, vvp)
    else:
        if a.layer: L = tp.layer_by_name(a.layer)
        elif a.ifm: L = tp.Layer("custom", a.ifm[0], a.ifm[1], a.cin, a.cout, a.k, a.s, a.pad)
        else: ap.error("give --layer, --ifm or --small")
        pols = [a.policy] if a.policy else ["spatial"]
        for pol in pols: ok &= run(L, tp.Tiling(a.n_if, a.tc, a.tk), pol, int(a.noise), a.seed, a.rounds, vvp)
    print("\nALL PASSED" if ok else "\nFAILURES")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()