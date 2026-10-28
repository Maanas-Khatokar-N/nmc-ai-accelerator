#!/usr/bin/env python3
"""
Software tiling plan: spatial tiling + bank partitioning for the 4-bank NMC cluster.

Vocabulary (paper Sec. IV-A, Eq. 3-5, Table 1):
  #IFMap tiles   : split of the OFMap/IFMap into n x n spatial tiles   (224/4 = 56 for CONV1-1)
  #Channel tiles : split of the input channels                          (3/1  = 3)
  #Kernel tiles  : split of the kernels (= OFMap channels)              (64/16 = 4)

Execution model
  job   = (spatial tile, kernel tile): lives in ONE bank, run by that bank's NMC unit
  pass  = one nmc_unit run of a job over a slice of input channels (pass 0 stores,
          later passes run accumulate=1 and add onto the OFMap already in the bank)
  round = one concurrent run of up to NUM_BANKS jobs, one per bank (irq when all done)

Bank memory map (one element per 32-bit word, same as the RTL testbenches)
  IF region | weight region | OFMap region        sized for the worst job of the plan

  python3 scripts/tiling_plan.py --layer 1-1 --n-if 4 --tc 1 --tk 16     # the paper's CONV1-1 example
  python3 scripts/tiling_plan.py --auto                                  # search all VGG-16 layers
  python3 scripts/tiling_plan.py --table table1.csv                      # layer,n_if,tc,tk from the paper
"""
import argparse, csv, itertools, math, sys
from dataclasses import dataclass, field

NUM_BANKS  = 4
BANK_WORDS = 8 * 4096          # 8 macros x 4096 words x 32 bit = 128 KB
RF_DEPTH   = 256
PE         = 4                 # 4x4 PE grid

# name, H(=W), Cin, Cout   (3x3, stride 1, pad 1)
VGG16 = [("1-1",224,3,64),("1-2",224,64,64),("2-1",112,64,128),("2-2",112,128,128),
         ("3-1",56,128,256),("3-2",56,256,256),("3-3",56,256,256),
         ("4-1",28,256,512),("4-2",28,512,512),("4-3",28,512,512),
         ("5-1",14,512,512),("5-2",14,512,512),("5-3",14,512,512)]

@dataclass
class Layer:
    name: str; H: int; W: int; Cin: int; Cout: int; K: int = 3; S: int = 1; pad: int = 1
    @property
    def OH(self): return (self.H + 2*self.pad - self.K)//self.S + 1
    @property
    def OW(self): return (self.W + 2*self.pad - self.K)//self.S + 1

@dataclass
class Tiling:
    n_if: int; tc: int; tk: int           # per-dimension spatial split, channel tiles, kernel tiles

@dataclass
class Job:
    sid: int; kt: int                     # spatial tile id, kernel tile id
    oy0: int; oy1: int; ox0: int; ox1: int
    k0: int; k1: int
    @property
    def oh(self): return self.oy1 - self.oy0
    @property
    def ow(self): return self.ox1 - self.ox0
    @property
    def nk(self): return self.k1 - self.k0

@dataclass
class Plan:
    layer: Layer; tiling: Tiling; policy: str
    spatial: list; ktiles: list; passes: list          # passes: [(c0,c1)]
    rounds: list                                       # list of [Job|None]*NUM_BANKS
    if_base: int; w_base: int; of_base: int; words_used: int
    stats: dict = field(default_factory=dict)

# ----------------------------------------------------------------------------
def cin_group_max(K, S):
    per_ch = (3*S + K) * (3*S + K)               # IFMap RF patch is (3S+K)^2 per channel
    if per_ch > 255: raise ValueError("patch does not fit 8-bit rf_ch_stride")
    return max(1, min(RF_DEPTH // per_ch, RF_DEPTH // (K*K)))

def split(n, parts):
    """tile size = ceil(n/parts) (paper Eq. 3); the last tile takes the remainder."""
    sz = -(-n // parts)
    return [(a, min(a+sz, n)) for a in range(0, n, sz)]

def channel_passes(Cin, tc, cg_max):
    """channel tiles first (Eq. 4), then balanced sub-split so each pass fits both RFs."""
    out = []
    for c0, c1 in split(Cin, tc):
        n = c1 - c0
        g = -(-n // cg_max)
        base, extra = divmod(n, g)
        c = c0
        for i in range(g):
            sz = base + (1 if i < extra else 0)
            out.append((c, c + sz)); c += sz
    return out

def in_extent(o, L): return (o - 1)*L.S + L.K

def make_jobs(L, T):
    rows = split(L.OH, T.n_if); cols = split(L.OW, T.n_if)
    spatial = [(r, c) for r in rows for c in cols]
    ktiles  = split(L.Cout, T.tk)
    return spatial, ktiles

def assign(spatial, ktiles, policy, nb=NUM_BANKS):
    """Return rounds = list of per-bank job lists."""
    jobs = {}
    def J(s, k):
        (r, c), (k0, k1) = spatial[s], ktiles[k]
        return Job(s, k, r[0], r[1], c[0], c[1], k0, k1)
    rounds = []
    if policy == "spatial":      # bank b owns spatial tiles b, b+nb, ... and walks all kernel tiles
        seqs = [[J(s, k) for s in range(b, len(spatial), nb) for k in range(len(ktiles))] for b in range(nb)]
        for i in range(max(len(s) for s in seqs)):
            rounds.append([sq[i] if i < len(sq) else None for sq in seqs])
    elif policy == "kernel":     # all banks share a spatial tile; kernel tiles are split over banks
        for s in range(len(spatial)):
            for q in range(0, len(ktiles), nb):
                rounds.append([J(s, k) if k < len(ktiles) else None for k in range(q, q+nb)])
    else: raise ValueError(policy)
    return rounds

def align(x, a=8): return (x + a - 1)//a*a

def build_plan(L, T, policy="spatial", bank_words=BANK_WORDS, nb=NUM_BANKS):
    spatial, ktiles = make_jobs(L, T)
    passes = channel_passes(L.Cin, T.tc, cin_group_max(L.K, L.S))
    max_if = max(in_extent(r[1]-r[0], L)*in_extent(c[1]-c[0], L) for r, c in spatial)
    max_cg = max(c1-c0 for c0, c1 in passes)
    max_nk = max(k1-k0 for k0, k1 in ktiles)
    if_w = max_cg*max_if
    w_w  = max_nk*max_cg*L.K*L.K
    of_w = max_nk*max((r[1]-r[0])*(c[1]-c[0]) for r, c in spatial)
    if_base = 0; w_base = align(if_w); of_base = align(w_base + w_w); used = of_base + of_w + 16
    if used > bank_words:
        raise ValueError(f"tile does not fit a bank: {used} > {bank_words} words")
    rounds = assign(spatial, ktiles, policy, nb)
    p = Plan(L, T, policy, spatial, ktiles, passes, rounds, if_base, w_base, of_base, used)
    p.stats = stats(p)
    return p

def pass_loads(plan):
    """Walk the rounds and count host->bank words, honouring IFMap/weight residency."""
    L = plan.layer; res = [None]*NUM_BANKS; resw = [None]*NUM_BANKS
    ifw = ww = ofw = 0
    for rnd in plan.rounds:
        for pi, (c0, c1) in enumerate(plan.passes):
            for b, j in enumerate(rnd):
                if j is None: continue
                r = (plan.spatial[j.sid]); ih = in_extent(j.oh, L); iw = in_extent(j.ow, L)
                if res[b] != (j.sid, c0, c1):
                    res[b] = (j.sid, c0, c1); ifw += (c1-c0)*ih*iw
                if resw[b] != (j.kt, c0, c1):
                    resw[b] = (j.kt, c0, c1); ww += j.nk*(c1-c0)*L.K*L.K
                if pi == len(plan.passes)-1: ofw += j.nk*j.oh*j.ow
    return ifw, ww, ofw

def job_cycles(j, L, passes):
    """compute-only estimate: one MAC cycle per kernel element per PE tile, per output channel."""
    tiles = -(-j.oh//PE) * -(-j.ow//PE)
    return sum(tiles * j.nk * (c1-c0) * L.K * L.K for c0, c1 in passes)

def stats(plan):
    L = plan.layer
    ifw, ww, ofw = pass_loads(plan)
    comp = sum(max(job_cycles(j, L, plan.passes) for j in r if j) for r in plan.rounds)
    macs = L.OH*L.OW*L.Cout*L.Cin*L.K*L.K
    jobs = sum(1 for r in plan.rounds for j in r if j)
    ideal = macs / (NUM_BANKS*PE*PE)
    return dict(jobs=jobs, rounds=len(plan.rounds), passes=len(plan.passes), if_words=ifw, w_words=ww,
                of_words=ofw, traffic=ifw+ww+ofw, est_cycles=comp, ideal_cycles=ideal,
                bank_util=plan.words_used/BANK_WORDS,
                bank_balance=jobs/(len(plan.rounds)*NUM_BANKS))

def cost(st): return st["est_cycles"] + st["traffic"]      # serial host port model

def auto_tiling(L, policies=("spatial", "kernel")):
    best = None
    pow2 = lambda n: [d for d in (1, 2, 4, 8, 16, 32, 64, 128) if d <= n]
    for n_if in [d for d in (1, 2, 3, 4, 5, 7, 8, 14, 16) if d <= L.OH]:
        for tc in pow2(L.Cin):
            for tk in pow2(L.Cout):
                if n_if*n_if*tk > 4096: continue          # keep the search cheap
                for pol in policies:
                    try: p = build_plan(L, Tiling(n_if, tc, tk), pol)
                    except ValueError: continue
                    c = cost(p.stats)
                    if best is None or c < best[0]: best = (c, p)
    if best is None: raise ValueError(f"no tiling fits layer {L.name}")
    return best[1]

HDR = f"{'layer':5} {'IFMap tile':>11} {'n_if':>4} {'ch/tile':>7} {'tc':>3} {'ker/tile':>8} {'tk':>3} {'pol':>7} " \
      f"{'jobs':>5} {'rnds':>5} {'pass':>4} {'bank KW':>7} {'util':>5} {'bal':>5} {'est cyc':>9} {'traffic':>9}"
def row(p):
    L, T, s = p.layer, p.tiling, p.stats
    tr, tc_ = p.spatial[0]
    tile = f"{in_extent(tr[1]-tr[0],L)}x{in_extent(tc_[1]-tc_[0],L)}"
    return (f"{L.name:5} {tile:>11} {T.n_if:>4} {-(-L.Cin//T.tc):>7} {T.tc:>3} {-(-L.Cout//T.tk):>8} {T.tk:>3} "
            f"{p.policy:>7} {s['jobs']:>5} {s['rounds']:>5} {s['passes']:>4} {p.words_used/1024:>7.1f} "
            f"{s['bank_util']*100:>4.0f}% {s['bank_balance']*100:>4.0f}% {s['est_cycles']:>9} {s['traffic']:>9}")

def layer_by_name(n, scale=1):
    for name, h, ci, co in VGG16:
        if name == n: return Layer(name, h//scale, h//scale, ci, co)
    raise KeyError(n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer"); ap.add_argument("--n-if", type=int, default=1)
    ap.add_argument("--tc", type=int, default=1); ap.add_argument("--tk", type=int, default=1)
    ap.add_argument("--policy", default="spatial", choices=["spatial", "kernel"])
    ap.add_argument("--auto", action="store_true"); ap.add_argument("--table")
    a = ap.parse_args()
    print(HDR)
    if a.table:
        for r in csv.DictReader(open(a.table)):
            print(row(build_plan(layer_by_name(r["layer"]), Tiling(int(r["n_if"]), int(r["tc"]), int(r["tk"])),
                                 r.get("policy", "spatial"))))
    elif a.auto:
        for name, *_ in VGG16: print(row(auto_tiling(layer_by_name(name))))
    elif a.layer:
        print(row(build_plan(layer_by_name(a.layer), Tiling(a.n_if, a.tc, a.tk), a.policy)))
    else: ap.error("give --layer, --auto or --table")

if __name__ == "__main__":
    main()