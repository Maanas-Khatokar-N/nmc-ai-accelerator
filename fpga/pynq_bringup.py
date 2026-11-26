#!/usr/bin/env python3
"""
NMC bring-up ladder for the PYNQ-Z2 (Zynq-7020) overlay.

Run ON THE BOARD (Jupyter terminal / ssh, as root):
    sudo -E python3 pynq_bringup.py --bit nmc.bit                 # whole ladder
    sudo -E python3 pynq_bringup.py --bit nmc.bit --stage ids     # one stage
    (nmc.hwh must sit next to nmc.bit, same base name)

Stages (in order, each prints PASS/FAIL, a failing stage stops 'all'):
    ids        NMC ID, wrapper ID, CONFIG register (bank count)
    regs       register write/read-back (UNIT_MASK, one descriptor word)
    sram       corner words of every bank, random blocks, alias check
    soft       wrapper soft reset (0xF04) clears the register block
    milestone  6x6 IFMap, 3x3 kernel -> 4x4 OFMap (9 MAC cycles), then 2x2 pool + requant + ReLU
    layer      real tiled conv (default 18x18x3 -> 8, 3x3) with Cin-group handling, then post-proc
    burst      numpy slice (burst) writes/reads through the AXI memory port (hardware only)

Everything is checked against a NumPy golden model (bit exact).
Do not touch unmapped / read-only addresses from here: an AXI SLVERR becomes a bus error
on the PS and can hang the Python process.

Offline self-test (no board, no pynq): --emit ops.txt writes the exact access sequence;
tb/integration/nmc_axi_replay_tb.v replays it through the RTL and compares with NumPy.
"""
import argparse, math, sys, time
import numpy as np

# ----------------------------------------------------------------------------- register map
ID_NMC, ID_WRAP = 0x4E4D0001, 0x41584901
G = dict(ID=0x00, CTRL=0x04, UMASK=0x08, PMASK=0x0C, STATUS=0x10, UBUSY=0x14, PBUSY=0x18,
         PVALID=0x1C, IRQST=0x20, IRQEN=0x24, CONFIG=0x28,
         QPMASK=0x40, QPADDR=0x44, QPBIAS=0x48, QPMULT=0x4C, QPSHIFT=0x50)
WRAP_ID, WRAP_CTRL = 0xF00, 0xF04
CONV_F = dict(ifmap_cols=0, kernel_rows=1, kernel_cols=2, input_channels=3, output_channels=4,
              stride=5, ofmap_rows=6, ofmap_cols=7, weights_per_oc=8, ifmap_ch_size=9,
              ofmap_ch_size=10, ifmap_row_step=11, ifmap_base=12, weight_base=13, ofmap_base=14,
              rf_row_step=15, rf_ch_stride=16, accumulate=17)
PP_F = dict(src_base=20, dst_base=21, src_cols=22, out_rows=23, out_cols=24, channels=25,
            pool_k=26, src_ch_size=27, row_step=28, total_out=29, out_zp=30, relu_en=31)
PERF_F = dict(total=32, mac=34, stall=36, blocked=38, ops=40)   # LO word; HI = +1

def BA(bank, f):                      # byte offset of word f in bank block 'bank'
    return 0x100 * (bank + 1) + 4 * f

# local (per-bank) word addresses used by the tests, same plan as layer_golden_tb.v
IFM_L, W_L, OFM_L, DST_L = 0, 8192, 12288, 28000
RF_DEPTH = 256

# ----------------------------------------------------------------------------- golden models
def conv_golden(ifm, wgt, s):
    cin, h, w = ifm.shape
    cout, _, kr, kc = wgt.shape
    oh, ow = (h - kr) // s + 1, (w - kc) // s + 1
    out = np.zeros((cout, oh, ow), dtype=np.int64)
    for ki in range(kr):
        for kj in range(kc):
            win = ifm[:, ki:ki + (oh - 1) * s + 1:s, kj:kj + (ow - 1) * s + 1:s].astype(np.int64)
            out += np.einsum("oc,cxy->oxy", wgt[:, :, ki, kj].astype(np.int64), win)
    return out.astype(np.int32)

def cin_group_max(kr, kc, s):
    per_ch = (3 * s + kr) * (3 * s + kc)
    assert per_ch <= 255, "patch does not fit the 8-bit rf_ch_stride"
    return max(1, min(RF_DEPTH // per_ch, RF_DEPTH // (kr * kc)))

def split_cin(cin, cg):
    sizes, rem = [], cin
    while rem > 0:
        sizes.append(min(cg, rem)); rem -= sizes[-1]
    return sizes

def pack_weights_by_group(wgt, groups):
    chunks, c0 = [], 0
    for g in groups:
        chunks.append(wgt[:, c0:c0 + g].flatten()); c0 += g
    return np.concatenate(chunks)

def pp_golden(acc, bias, mult, shift, zp, relu, pool):
    ch, rows, cols = acc.shape
    orows, ocols = rows // pool, cols // pool
    a = acc[:, :orows * pool, :ocols * pool].astype(np.int64)
    v = (a + bias[:, None, None]) * mult[:, None, None]
    sh = shift[:, None, None]
    rnd = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    q = (v + rnd) >> sh
    lo = zp if relu else -128
    y = np.clip(q + zp, lo, 127)
    return y.reshape(ch, orows, pool, ocols, pool).max(axis=(2, 4)).astype(np.int64)

# ----------------------------------------------------------------------------- backends
class Backend:
    def __init__(self, banks, bank_words):
        self.banks, self.bank_words, self.errors = banks, bank_words, 0
        self.shown = 0
    def fail(self, msg):
        self.errors += 1
        if self.shown < 12:
            print("   FAIL", msg); self.shown += 1
    def cfg_set(self, off, val): self.cfg_w(off, val)
    def mem_check_many(self, word0, expected, label):
        for i, e in enumerate(np.asarray(expected, dtype=np.uint64).flatten()):
            self.mem_check(word0 + i, int(e), f"{label}[{i}]")
    def mem_write_many(self, word0, values):
        for i, v in enumerate(np.asarray(values, dtype=np.uint64).flatten()):
            self.mem_w(word0 + i, int(v))
    def mem_read_many(self, word0, n, expect=None):
        return np.array([self.mem_r(word0 + i) for i in range(n)], dtype=np.uint32)

class HwBackend(Backend):
    def __init__(self, cfg_base, mem_base, banks, bank_words):
        super().__init__(banks, bank_words)
        from pynq import MMIO
        self.cfg = MMIO(cfg_base, 0x1000)
        self.mem = MMIO(mem_base, banks * bank_words * 4)
        self.burst = True                      # bulk transfers through MMIO.array (AXI bursts); --no-burst disables
    def cfg_w(self, off, v): self.cfg.write(off, int(v) & 0xFFFFFFFF)
    def cfg_r(self, off):    return self.cfg.read(off)
    def mem_w(self, w, v):   self.mem.write(w * 4, int(v) & 0xFFFFFFFF)
    def mem_r(self, w):      return self.mem.read(w * 4)
    def mem_write_many(self, word0, values):
        v = np.asarray(values).astype(np.uint32, copy=False).ravel()          # no int64 round trip (values are < 2^32)
        if self.burst and v.size >= 8:
            try:
                self.mem.array[word0:word0 + v.size] = v; return
            except Exception as e:
                print("   burst write failed, falling back to single writes:", e); self.burst = False
        for i, x in enumerate(v): self.mem.write((word0 + i) * 4, int(x))
    def mem_read_many(self, word0, n, expect=None):
        if self.burst and n >= 8:
            try:
                return np.array(self.mem.array[word0:word0 + n], dtype=np.uint32)
            except Exception as e:
                print("   burst read failed, falling back to single reads:", e); self.burst = False
        return np.array([self.mem.read((word0 + i) * 4) for i in range(n)], dtype=np.uint32)
    def cfg_check(self, off, exp, label, mask=0xFFFFFFFF):
        got = self.cfg_r(off)
        if (got & mask) != (exp & mask):
            self.fail(f"{label}: cfg[0x{off:x}] = 0x{got:08x}, expected 0x{exp:08x} (mask 0x{mask:x})")
    def mem_check(self, w, exp, label):
        got = self.mem_r(w)
        if got != (exp & 0xFFFFFFFF):
            self.fail(f"{label}: mem[word {w}] = 0x{got:08x}, expected 0x{exp & 0xFFFFFFFF:08x}")
    def wait_idle(self, what, timeout_s=20.0):
        t0 = time.time()
        while self.cfg_r(G["STATUS"]) != 0:
            if time.time() - t0 > timeout_s:
                self.fail(f"timeout waiting for {what} (STATUS=0x{self.cfg_r(G['STATUS']):x})"); return False
        return True
    def delay(self, s): time.sleep(s)

class RecordBackend(Backend):
    """no hardware: log the access sequence for the RTL replay testbench"""
    def __init__(self, banks, bank_words):
        super().__init__(banks, bank_words); self.ops = []
    def cfg_w(self, off, v): self.ops.append(("CW", off, int(v) & 0xFFFFFFFF, 0))
    def cfg_r(self, off):    return 0
    def mem_w(self, w, v):   self.ops.append(("MW", w, int(v) & 0xFFFFFFFF, 0))
    def mem_r(self, w):      return 0
    def cfg_check(self, off, exp, label, mask=0xFFFFFFFF):
        self.ops.append(("CC", off, exp & 0xFFFFFFFF, mask))
    def mem_check(self, w, exp, label):
        self.ops.append(("MC", w, exp & 0xFFFFFFFF, 0))
    def mem_read_many(self, word0, n, expect=None):
        exp = np.asarray(expect, dtype=np.int64).flatten() & 0xFFFFFFFF
        for i in range(n): self.ops.append(("MC", word0 + i, int(exp[i]), 0))
        return exp.astype(np.uint32)
    def wait_idle(self, what, timeout_s=0):
        self.ops.append(("WI", 3000000, 0, 0)); return True
    def delay(self, s): self.ops.append(("DL", 60, 0, 0))
    def dump(self, path):
        with open(path, "w") as f:
            for op, a, b, c in self.ops: f.write(f"{op} {a:x} {b:x} {c:x}\n")
            f.write("END 0 0 0\n")

# ----------------------------------------------------------------------------- stages
def stage_ids(be, a):
    be.cfg_check(G["ID"], ID_NMC, "NMC ID")
    be.cfg_check(WRAP_ID, ID_WRAP, "wrapper ID")
    be.cfg_check(G["CONFIG"], (512 << 16) | (48 << 8) | a.banks, "CONFIG (MAX_OC<<16 | CNT_W<<8 | NUM_BANKS)")

def stage_regs(be, a):
    nbm = (1 << a.banks) - 1
    for v in (0x5, 0x3, 0x0):
        be.cfg_w(G["UMASK"], v); be.cfg_check(G["UMASK"], v & nbm, f"UNIT_MASK write 0x{v:x}")
    be.cfg_w(BA(0, CONV_F["ifmap_cols"]), 0x1234); be.cfg_check(BA(0, CONV_F["ifmap_cols"]), 0x1234, "bank0 IFMAP_COLS")
    be.cfg_w(BA(0, CONV_F["ifmap_cols"]), 0)
    be.cfg_check(G["STATUS"], 0, "STATUS idle")

def stage_sram(be, a):
    n_words = a.banks * a.bank_words
    words = sorted({0, 1, 7, 8, 4095, 4096, a.bank_words - 1, a.bank_words, a.bank_words + 1, n_words - 1})
    words = [w for w in words if w < n_words]
    for w in words: be.mem_w(w, 0xA5000000 ^ (w * 2654435761 & 0xFFFFFF))     # unique per word
    for w in words: be.mem_check(w, 0xA5000000 ^ (w * 2654435761 & 0xFFFFFF), f"corner word {w}")
    rng = np.random.default_rng(7)
    for bank in range(a.banks):
        base = bank * a.bank_words + 1000
        blk = rng.integers(0, 2**32, 256, dtype=np.uint64)
        be.mem_write_many(base, blk); be.mem_check_many(base, blk, f"bank{bank} block")
    for w in words[:3]: pass
    # alias check: corner words must still hold their values after the block writes
    for w in words:
        if not any(b * a.bank_words + 1000 <= w < b * a.bank_words + 1256 for b in range(a.banks)):
            be.mem_check(w, 0xA5000000 ^ (w * 2654435761 & 0xFFFFFF), f"alias check word {w}")

def stage_soft(be, a):
    be.cfg_w(G["UMASK"], 0x3); be.cfg_check(G["UMASK"], (1 << a.banks) - 1 & 0x3, "UNIT_MASK before reset")
    be.cfg_w(WRAP_CTRL, 1)
    be.delay(0.01)
    be.cfg_check(G["UMASK"], 0, "UNIT_MASK cleared by soft reset")
    be.cfg_check(G["ID"], ID_NMC, "NMC ID after soft reset")
    be.cfg_check(WRAP_CTRL, 0, "soft reset finished")

def write_desc(be, bank, fields):
    for k, v in fields.items(): be.cfg_w(BA(bank, CONV_F[k]), v)

def run_conv(be, a, ifr, ifc, kr, kc, cin, cout, s, bank=0, seed=1, label="conv"):
    rng = np.random.default_rng(seed)
    oh, ow = (ifr - kr) // s + 1, (ifc - kc) // s + 1
    cg = cin_group_max(kr, kc, s); groups = split_cin(cin, cg)
    n_if, n_w, n_of = cin * ifr * ifc, cout * cin * kr * kc, cout * oh * ow
    assert n_if <= W_L - IFM_L and n_w <= OFM_L - W_L and OFM_L + n_of + 16 <= DST_L, "layer does not fit the test memory plan"
    ifm = rng.integers(-128, 128, (cin, ifr, ifc), dtype=np.int8)
    wgt = rng.integers(-128, 128, (cout, cin, kr, kc), dtype=np.int8)
    gold = conv_golden(ifm, wgt, s)
    base = bank * a.bank_words                                   # host (global) word offset of the bank
    be.mem_write_many(base + IFM_L, ifm.astype(np.int64).flatten() & 0xFF)
    be.mem_write_many(base + W_L, pack_weights_by_group(wgt, groups).astype(np.int64) & 0xFF)
    be.mem_write_many(base + OFM_L, np.full(n_of + 16, 0xDEADBEEF, dtype=np.uint64))
    patch_r, patch_c = 3 * s + kr, 3 * s + kc
    common = dict(ifmap_cols=ifc, kernel_rows=kr, kernel_cols=kc, output_channels=cout, stride=s,
                  ofmap_rows=oh, ofmap_cols=ow, ifmap_ch_size=ifr * ifc, ofmap_ch_size=oh * ow,
                  ifmap_row_step=s * ifc, weight_base=W_L, ofmap_base=OFM_L,
                  rf_row_step=s * patch_c, rf_ch_stride=patch_r * patch_c)
    be.cfg_w(G["UMASK"], 1 << bank); be.cfg_w(G["IRQEN"], 0)
    c0, w_off, t0 = 0, 0, time.time()
    for gi, cg_now in enumerate(groups):
        write_desc(be, bank, dict(common, input_channels=cg_now, weights_per_oc=cg_now * kr * kc,
                                  ifmap_base=IFM_L + c0 * ifr * ifc, weight_base=W_L + w_off,
                                  accumulate=1 if gi else 0))
        be.cfg_w(G["CTRL"], 1)                                   # CONV_START
        if not be.wait_idle(f"{label} group {gi}"): return None
        be.cfg_check(G["IRQST"], 0x1, f"{label} IRQ_STATUS conv done", mask=0x5)   # done set, start not rejected
        be.cfg_w(G["IRQST"], 0xF)
        w_off += cout * cg_now * kr * kc; c0 += cg_now
    dt = time.time() - t0
    be.mem_check_many(base + OFM_L, gold.astype(np.int64).flatten() & 0xFFFFFFFF, f"{label} OFMap")
    be.mem_check_many(base + OFM_L + n_of, np.full(16, 0xDEADBEEF), f"{label} guard")
    mac, ops, tot = (be.cfg_r(BA(bank, PERF_F[k])) for k in ("mac", "ops", "total"))
    macs = oh * ow * cout * cin * kr * kc
    print(f"   {label}: {ifr}x{ifc}x{cin} K{kr}x{kc} -> {cout}x{oh}x{ow}, stride {s}, {len(groups)} Cin group(s), "
          f"{macs} MACs, host-side time {dt*1e3:.1f} ms")
    if tot: print(f"   last group perf: cycles={tot} mac_cycles={mac} mac_ops={ops} PE activity={100.0*ops/(16*mac) if mac else 0:.1f}%")
    return gold

def run_postproc(be, a, gold, pool, bank=0, relu=1, zp=0, shifts=None, label="postproc"):
    ch, rows, cols = gold.shape
    orows, ocols = rows // pool, cols // pool
    bias = np.zeros(ch, dtype=np.int64); mult = np.ones(ch, dtype=np.int64)
    if shifts is None: shifts = np.zeros(ch, dtype=np.int64)
    shifts = np.asarray(shifts, dtype=np.int64)
    exp = pp_golden(gold, bias, mult, shifts, zp, relu, pool)
    base = bank * a.bank_words
    be.mem_write_many(base + DST_L, np.full(ch * orows * ocols + 16, 0xDEADBEEF, dtype=np.uint64))
    d = dict(src_base=OFM_L, dst_base=DST_L, src_cols=cols, out_rows=orows, out_cols=ocols, channels=ch,
             pool_k=pool, src_ch_size=rows * cols, row_step=pool * cols, total_out=ch * orows * ocols,
             out_zp=zp & 0xFF, relu_en=relu)
    for k, v in d.items(): be.cfg_w(BA(bank, PP_F[k]), v)
    be.cfg_w(G["QPMASK"], 1 << bank); be.cfg_w(G["QPADDR"], 0)
    for c in range(ch):                                          # QP_ADDR auto-increments on every commit
        be.cfg_w(G["QPBIAS"], int(bias[c])); be.cfg_w(G["QPMULT"], int(mult[c])); be.cfg_w(G["QPSHIFT"], int(shifts[c]))
    be.cfg_w(G["PMASK"], 1 << bank)
    be.cfg_w(G["CTRL"], 2)                                       # PP_START
    if not be.wait_idle(label): return
    be.cfg_check(G["IRQST"], 0x2, f"{label} IRQ_STATUS pp done", mask=0x6)
    be.cfg_w(G["IRQST"], 0xF)
    be.mem_check_many(base + DST_L, exp.flatten() & 0xFF, f"{label} output (int8 in low byte)")
    be.mem_check_many(base + DST_L + ch * orows * ocols, np.full(16, 0xDEADBEEF), f"{label} guard")
    print(f"   {label}: {ch}x{rows}x{cols} -> {ch}x{orows}x{ocols}, pool {pool}, relu {relu}, zp {zp}")

def stage_milestone(be, a):
    gold = run_conv(be, a, 6, 6, 3, 3, 1, 1, 1, bank=0, seed=3, label="milestone conv")
    if gold is not None:
        if not a.no_perf_check:
            be.cfg_check(BA(0, PERF_F["mac"]), 9, "milestone MAC cycles (9)")
            be.cfg_check(BA(0, PERF_F["ops"]), 144, "milestone MAC ops (9*16)")
            be.cfg_check(BA(0, PERF_F["stall"]), 0, "milestone stalls")
        if a.pp_mask & 1: run_postproc(be, a, gold, 2, bank=0, relu=1, zp=0, label="milestone postproc")

def stage_layer(be, a):
    banks = [b for b in range(a.banks)]
    for b in banks:
        gold = run_conv(be, a, a.ifm, a.ifm, 3, 3, a.cin, a.cout, 1, bank=b, seed=10 + b, label=f"layer bank{b}")
        if gold is None: return
        peak = np.abs(gold.reshape(a.cout, -1)).max(axis=1)
        shifts = np.clip(np.ceil(np.log2(np.maximum(peak, 1) / 100.0)), 0, 31).astype(np.int64)
        if (a.pp_mask >> b) & 1: run_postproc(be, a, gold, 2, bank=b, relu=1, zp=0, shifts=shifts, label=f"layer bank{b} postproc")
    # accumulate path: Cin = 64 does not fit the register files, so it runs as 10 Cin groups (psum reload)
    run_conv(be, a, 10, 10, 3, 3, 64, 2, 1, bank=0, seed=21, label="accumulate (Cin=64, 10 groups)")

def stage_burst(be, a):
    if not isinstance(be, HwBackend):
        print("   (burst stage is hardware only, skipped)"); return
    try:
        arr = be.mem.array
        rng = np.random.default_rng(5)
        for base in (2000, a.bank_words + 3000):
            data = rng.integers(0, 2**32, 1024, dtype=np.uint64).astype(np.uint32)
            arr[base:base + 1024] = data
            back = np.array(arr[base:base + 1024])
            bad = int(np.count_nonzero(back != data))
            if bad: be.fail(f"burst block at word {base}: {bad}/1024 words differ")
    except Exception as e:
        be.fail(f"burst stage raised {type(e).__name__}: {e}")

STAGES = [("ids", stage_ids), ("regs", stage_regs), ("sram", stage_sram), ("soft", stage_soft),
          ("milestone", stage_milestone), ("layer", stage_layer), ("burst", stage_burst)]

def main():
    ap = argparse.ArgumentParser(description="NMC bring-up ladder for PYNQ-Z2")
    ap.add_argument("--bit", help="overlay .bit (hardware run)")
    ap.add_argument("--cfg-base", type=lambda x: int(x, 0), default=0x40000000, help="s_axi_cfg base address")
    ap.add_argument("--mem-base", type=lambda x: int(x, 0), default=0x40080000, help="s_axi_mem base address")
    ap.add_argument("--banks", type=int, default=2)
    ap.add_argument("--bank-words", type=int, default=32768, help="32-bit words per bank (8 macros x DEPTH_PER_MACRO)")
    ap.add_argument("--stage", default="all", choices=["all"] + [s for s, _ in STAGES])
    ap.add_argument("--ifm", type=int, default=18, help="layer stage: IFMap height/width")
    ap.add_argument("--cin", type=int, default=3); ap.add_argument("--cout", type=int, default=8)
    ap.add_argument("--pp-mask", type=lambda x: int(x, 0), default=0xFFFF, help="banks that have a post-proc unit (the PP_BANKS build parameter)")
    ap.add_argument("--no-perf-check", action="store_true", help="build has PERF_FULL=0 (only the cycle counter exists)")
    ap.add_argument("--emit", help="no hardware: write the access sequence to this file (for the RTL replay TB)")
    a = ap.parse_args()

    if a.emit:
        be = RecordBackend(a.banks, a.bank_words)
    else:
        if not a.bit: ap.error("--bit is required (or use --emit)")
        from pynq import Overlay
        ol = Overlay(a.bit)
        try:
            from pynq import Clocks
            print(f"PL clock FCLK0 = {Clocks.fclk0_mhz:.2f} MHz")
        except Exception: pass
        be = HwBackend(a.cfg_base, a.mem_base, a.banks, a.bank_words)

    todo = STAGES if a.stage == "all" else [(n, f) for n, f in STAGES if n == a.stage]
    ok_all = True
    for name, fn in todo:
        before = be.errors; be.shown = 0
        print(f"[{name}]")
        try:
            fn(be, a)
        except Exception as e:
            be.fail(f"stage raised {type(e).__name__}: {e}")
        ok = be.errors == before
        print(f"[{name}] {'PASS' if ok else 'FAIL'}\n")
        ok_all &= ok
        if not ok and a.stage == "all": break
    if a.emit:
        be.dump(a.emit); print(f"wrote {len(be.ops)} operations to {a.emit}")
    print("ALL STAGES PASSED" if ok_all else f"FAILED ({be.errors} errors)")
    sys.exit(0 if ok_all else 1)

if __name__ == "__main__":
    main()
