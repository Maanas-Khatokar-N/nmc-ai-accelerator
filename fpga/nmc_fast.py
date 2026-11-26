#!/usr/bin/env python3
"""
Fast runtime for the PYNQ-Z2 NMC cluster (DMA build).

Why the first version took ~85 s per image (measured on the board, ball.jpg):
  load 53.1 s   every int8 was written by the ARM as ONE 32-bit MMIO store over M_AXI_GP0
                (~1.5 M words/s = 6 MB/s); ~84 M words per image
  drain 5.1 s   same path for reads, int32 psums
  host 12.9 s   requant/ReLU/pool of every conv output in numpy int64 (6.3 s) + FC layers whose weights
                were memory-mapped .npy files, i.e. streamed from the SD card during the run (6.5 s)
  run 10.2 s    the accelerator itself (557 M cycles at 58.8 MHz)

What this runtime does instead
  * bulk data moves with an AXI CDMA (DDR via S_AXI_HP0 -> s_axi_mem, 256-beat bursts); the reworked
    nmc_axi_wrapper takes one beat per clock -> ~60-65 M words/s instead of 1.5 M
  * requant + ReLU + max-pool run on the chip (postproc_unit in every bank, in place over the OFMap);
    only the final int8 (pooled) tile is drained
  * the ARM prepares the next pass's IF/weight tiles while the NMC computes the current one
  * descriptor registers are only written when they change
  * the model is loaded into RAM once; FC weights are kept transposed so the FC layers only touch the
    rows of non-zero activations (pool5 is ~86 % zeros after ReLU) - exact float64 sums, bit-exact
  * Board backends:  HwBoard (CDMA, or --mmio: no DMA, works with the old bitstream)
                     SimBoard (no hardware: emulates the bank memory, conv and post-proc engines, so the
                               whole host flow can be checked bit-exact against vgg_int / ref.npz on a PC)
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiling_plan as tp
import vgg_int as vi

# ----------------------------------------------------------------------------- register map (nmc_regs.v)
G = dict(ID=0x00, CTRL=0x04, UMASK=0x08, PMASK=0x0C, STATUS=0x10, IRQST=0x20, IRQEN=0x24,
         QPMASK=0x40, QPADDR=0x44, QPBIAS=0x48, QPMULT=0x4C, QPSHIFT=0x50)
CONV_F = dict(ifmap_cols=0, kernel_rows=1, kernel_cols=2, input_channels=3, output_channels=4,
              stride=5, ofmap_rows=6, ofmap_cols=7, weights_per_oc=8, ifmap_ch_size=9,
              ofmap_ch_size=10, ifmap_row_step=11, ifmap_base=12, weight_base=13, ofmap_base=14,
              rf_row_step=15, rf_ch_stride=16, accumulate=17)
PP_F = dict(src_base=20, dst_base=21, src_cols=22, out_rows=23, out_cols=24, channels=25,
            pool_k=26, src_ch_size=27, row_step=28, total_out=29, out_zp=30, relu_en=31)
PERF_TOTAL = 32
F_INV = {v: k for k, v in list(CONV_F.items()) + list(PP_F.items())}

def BA(bank, f): return 0x100 * (bank + 1) + 4 * f
OFFS = [[BA(b, f) for f in range(18)] for b in range(4)]

# AXI CDMA (PG034, simple mode)
CDMACR, CDMASR, CDMA_SA, CDMA_DA, CDMA_BTT = 0x00, 0x04, 0x18, 0x20, 0x28
SR_IDLE, SR_ERR = 0x2, 0x70

# ============================================================================= boards
class BoardBase:
    """common part: cached register writes, staging/drain buffers (uint32, one per bank, bank layout)"""
    def __init__(self, nb, bank_words):
        self.nb, self.bank_words = nb, bank_words
        self.shadow = {}                       # register offset -> last value written
        self.dma_words = 0; self.dma_time = 0.0
    def cfg_set(self, off, v):                 # write only if different from the last write
        v = int(v) & 0xFFFFFFFF
        if self.shadow.get(off) != v:
            self.cfg_w(off, v); self.shadow[off] = v
    def invalidate_shadow(self): self.shadow.clear()
    def wait_idle(self, what="", timeout_s=30.0):
        t0 = time.perf_counter()
        while self.cfg_r(G["STATUS"]) != 0:
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(f"timeout waiting for {what} (STATUS=0x{self.cfg_r(G['STATUS']):x})")

class HwBoard(BoardBase):
    def __init__(self, cfg_base, mem_base, nb, bank_words, cdma_base=None, dma_mem_base=None):
        super().__init__(nb, bank_words)
        from pynq import MMIO, allocate
        self.cfg = MMIO(cfg_base, 0x1000)
        self.mem = MMIO(mem_base, nb * bank_words * 4)
        self.cfg_a = self.cfg.array             # numpy uint32 view of the register block
        self.use_dma = cdma_base is not None
        if self.use_dma:
            self.dma = MMIO(cdma_base, 0x1000)
            self.dma_mem_base = mem_base if dma_mem_base is None else dma_mem_base
            self.stage = [allocate(shape=(bank_words,), dtype=np.uint32) for _ in range(nb)]
            try:                                # cacheable drain buffer + invalidate = fast ARM reads
                self.drainb = [allocate(shape=(bank_words,), dtype=np.uint32, cacheable=1) for _ in range(nb)]
                self.drainb[0].invalidate(); self.drain_cached = True
            except Exception:
                self.drainb = [allocate(shape=(bank_words,), dtype=np.uint32) for _ in range(nb)]
                self.drain_cached = False
            self.dma_a = self.dma.array
            self.stage_pa = [int(x.physical_address) for x in self.stage]
            self.dma.write(CDMACR, 0x4)         # soft reset
            t0 = time.time()
            while self.dma.read(CDMACR) & 0x4:
                if time.time() - t0 > 1: raise RuntimeError("CDMA does not come out of reset")
            if not (self.dma.read(CDMASR) & SR_IDLE): raise RuntimeError("CDMA not idle after reset")
        else:                                   # no DMA in the overlay: plain numpy buffers + MMIO copies
            self.stage = [np.zeros(bank_words, dtype=np.uint32) for _ in range(nb)]
            self.drainb = [np.zeros(bank_words, dtype=np.uint32) for _ in range(nb)]
    # register access through the raw uint32 view of the mapping (MMIO.write/read add a lot of Python per call)
    def cfg_w(self, off, v): self.cfg_a[off >> 2] = v
    def cfg_r(self, off):    return int(self.cfg_a[off >> 2])
    def _cdma_start(self, sa, da, nbytes):
        d = self.dma_a
        d[CDMA_SA >> 2] = sa; d[CDMA_DA >> 2] = da; d[CDMA_BTT >> 2] = nbytes      # BTT write starts it
        self._last = (sa, da, nbytes)
    def _cdma_wait(self):
        d = self.dma_a
        while True:
            sr = int(d[CDMASR >> 2])
            if sr & SR_ERR:
                d[CDMACR >> 2] = 0x4
                sa, da, n = self._last
                raise RuntimeError(f"CDMA error, CDMASR=0x{sr:08x} (SA 0x{sa:08x} DA 0x{da:08x} BTT {n}); "
                                   "check the CDMA address map (--dma-mem-base)")
            if sr & SR_IDLE: return
    def _cdma(self, sa, da, nbytes): self._cdma_start(sa, da, nbytes); self._cdma_wait()
    def load_start(self, b, off, n):            # staging[b][off:off+n] -> bank b words off..off+n-1
        if self.use_dma:
            self._cdma_start(self.stage_pa[b] + 4 * off, self.dma_mem_base + 4 * (b * self.bank_words + off), 4 * n)
        else:
            self.mem.array[b * self.bank_words + off: b * self.bank_words + off + n] = self.stage[b][off:off + n]
        self.dma_words += n
    def load_wait(self):
        if self.use_dma: self._cdma_wait()
    def load(self, b, off, n): self.load_start(b, off, n); self.load_wait()
    def drain(self, b, off, n):                 # bank b words -> drainb[b][off:off+n]
        t0 = time.perf_counter()
        if self.use_dma:
            self._cdma(self.dma_mem_base + 4 * (b * self.bank_words + off),
                       self.drainb[b].physical_address + 4 * off, 4 * n)
            if self.drain_cached: self.drainb[b].invalidate()
        else:
            self.drainb[b][off:off + n] = self.mem.array[b * self.bank_words + off: b * self.bank_words + off + n]
        self.dma_words += n; self.dma_time += time.perf_counter() - t0
        return self.drainb[b]

class SimBoard(BoardBase):
    """Functional model of nmc_top behind the wrapper (bank memory + conv + post-proc engines + regs).
    Implements the same data layouts the bring-up script checks on the real board."""
    def __init__(self, nb, bank_words):
        super().__init__(nb, bank_words)
        self.memw = np.zeros(nb * bank_words, dtype=np.uint32)
        self.regs = {}; self.qp = [dict() for _ in range(nb)]; self.qpaddr = 0
        self.stage = [np.zeros(bank_words, dtype=np.uint32) for _ in range(nb)]
        self.drainb = [np.zeros(bank_words, dtype=np.uint32) for _ in range(nb)]
        self.cycles = [0] * nb; self.cfg_writes = 0
    def cfg_r(self, off):
        if off == G["STATUS"]: return 0
        for b in range(self.nb):
            if off == BA(b, PERF_TOTAL): return self.cycles[b] & 0xFFFFFFFF
        return self.regs.get(off, 0)
    def f(self, b, name):
        idx = CONV_F[name] if name in CONV_F else PP_F[name]
        return self.regs.get(BA(b, idx), 0)
    def cfg_w(self, off, v):
        self.cfg_writes += 1
        self.regs[off] = v
        if off == G["QPADDR"]: self.qpaddr = v
        elif off == G["QPSHIFT"]:
            bias = np.int64(np.uint32(self.regs.get(G["QPBIAS"], 0)).view(np.int32))
            for b in range(self.nb):
                if (self.regs.get(G["QPMASK"], 0) >> b) & 1:
                    self.qp[b][self.qpaddr] = (int(bias), self.regs.get(G["QPMULT"], 0) & 0xFFFF, v & 31)
            self.qpaddr += 1
        elif off == G["CTRL"]:
            if v & 1:
                for b in range(self.nb):
                    if (self.regs.get(G["UMASK"], 0) >> b) & 1: self._conv(b)
            elif v & 2:
                for b in range(self.nb):
                    if (self.regs.get(G["PMASK"], 0) >> b) & 1: self._pp(b)
    def _words(self, b, base, n): return self.memw[b * self.bank_words + base: b * self.bank_words + base + n]
    def _conv(self, b):
        f = lambda k: self.f(b, k)
        cg, nk, kr, kc, s = f("input_channels"), f("output_channels"), f("kernel_rows"), f("kernel_cols"), f("stride")
        iw = f("ifmap_cols"); ih = f("ifmap_ch_size") // iw; oh, ow = f("ofmap_rows"), f("ofmap_cols")
        assert f("ifmap_row_step") == s * iw and f("weights_per_oc") == cg * kr * kc and f("ofmap_ch_size") == oh * ow
        x = (self._words(b, f("ifmap_base"), cg * ih * iw) & 0xFF).astype(np.uint8).view(np.int8).reshape(cg, ih, iw)
        w = (self._words(b, f("weight_base"), nk * cg * kr * kc) & 0xFF).astype(np.uint8).view(np.int8).reshape(nk, cg, kr, kc)
        acc = vi.conv_acc(x, w, s)[:, :oh, :ow]
        o = self._words(b, f("ofmap_base"), nk * oh * ow)
        if f("accumulate") & 1: acc = (o.view(np.int32).reshape(acc.shape).astype(np.int64) + acc).astype(np.int32)
        o[:] = acc.astype(np.int32).ravel().view(np.uint32)
        self.cycles[b] = -(-oh // 4) * -(-ow // 4) * nk * cg * kr * kc
    def _pp(self, b):
        f = lambda k: self.f(b, k)
        ch, p, orows, ocols, cols = f("channels"), f("pool_k"), f("out_rows"), f("out_cols"), f("src_cols")
        rows = f("src_ch_size") // cols
        assert f("row_step") == p * cols and f("total_out") == ch * orows * ocols
        a = self._words(b, f("src_base"), ch * rows * cols).view(np.int32).reshape(ch, rows, cols).astype(np.int64)
        a = a[:, :orows * p, :ocols * p].reshape(ch, orows, p, ocols, p).max(axis=(2, 4))
        bias = np.array([self.qp[b][c][0] for c in range(ch)]); mult = np.array([self.qp[b][c][1] for c in range(ch)])
        shift = np.array([self.qp[b][c][2] for c in range(ch)])
        zp = np.int64(np.int8(np.uint8(f("out_zp") & 0xFF)))
        y = vi.requant(a, bias, mult, shift, bool(f("relu_en") & 1), int(zp)).astype(np.int64)
        self._words(b, f("dst_base"), ch * orows * ocols)[:] = (y.ravel() & 0xFF).astype(np.uint32)  # in place is fine
    def load(self, b, off, n):
        self._words(b, off, n)[:] = self.stage[b][off:off + n]; self.dma_words += n
    def load_start(self, b, off, n): self.load(b, off, n)
    def load_wait(self): pass
    def drain(self, b, off, n):
        self.drainb[b][off:off + n] = self._words(b, off, n); self.dma_words += n
        return self.drainb[b]

# ============================================================================= model
class FastModel:
    """whole model in RAM; conv weights as uint8 (low byte = the int8 the bank stores); FC weights transposed"""
    def __init__(self, d, log=print):
        t0 = time.perf_counter()
        m = vi.load_model(d, mmap=True)
        self.input = m["input"]; self.layers = []
        for i, L in enumerate(m["layers"]):
            L = dict(L)
            if L["type"] == "conv":
                L["w_u8"] = np.ascontiguousarray(np.asarray(L["w"])).view(np.uint8)
                L["w"] = L["w_u8"].view(np.int8)
                for k in ("bias", "mult", "shift"): L[k] = np.asarray(L[k], dtype=np.int64)
            else:
                L["wT"] = self._fc_transposed(d, m, i, L, log)
                L.pop("w", None)
                L["bias"] = np.asarray(L["bias"], dtype=np.int64)
                for k in ("mult", "shift", "scale"):
                    if k in L: L[k] = np.asarray(L[k])
            self.layers.append(L)
        log(f"model loaded into RAM in {time.perf_counter() - t0:.1f} s")

    @staticmethod
    def _fc_transposed(d, m, i, L, log):
        import json
        man = json.load(open(os.path.join(d, "manifest.json")))
        fn = man["layers"][i]["w"]; tfn = os.path.join(d, fn.replace("_w.npy", "_wT.npy"))
        if os.path.exists(tfn):
            return np.load(tfn)                      # fully in RAM (no mmap)
        w = np.load(os.path.join(d, fn), mmap_mode="r")
        nout, nin = w.shape
        log(f"   first run: writing transposed FC weights {os.path.basename(tfn)} ({nout}x{nin}), one time only ...")
        wT = np.empty((nin, nout), dtype=np.int8)
        for r0 in range(0, nout, 256):
            wT[:, r0:r0 + 256] = np.asarray(w[r0:r0 + 256]).T
        try: np.save(tfn, wT)
        except OSError as e: log(f"   (could not save {tfn}: {e})")
        return wT

def fc_sparse(x, wT, rows=1024):
    """exact int FC over the NON-ZERO inputs only. Each chunk of <= 1024 rows is summed in float32, which is
    exact because every partial sum is an integer below 1024*127*128 < 2^24; chunks are added in float64."""
    x = np.asarray(x).reshape(-1)
    nz = np.flatnonzero(x)
    acc = np.zeros(wT.shape[1], dtype=np.float64)
    for i in range(0, nz.size, rows):
        idx = nz[i:i + rows]
        acc += x[idx].astype(np.float32) @ wT[idx].astype(np.float32)
    return np.rint(acc).astype(np.int64)

# ============================================================================= conv layer
_PLANS = {}
class ConvRunner:
    def __init__(self, bd, host_pp=False, read_perf=True):
        self.bd, self.host_pp, self.read_perf = bd, host_pp, read_perf
        self.qp_loaded = [None] * bd.nb          # (layer id, k0, k1) whose requant params are in the bank
        self.regs = {}                           # conv descriptor / UNIT_MASK values already written

    def run(self, lid, L, x, til):
        """x int8 [Cin,H,W] -> int8 [Cout,H/p,W/p] (requant+ReLU+pool on chip) ; returns (y, stats)"""
        bd, nb = self.bd, self.bd.nb
        cin, H, W = x.shape
        pool = int(L.get("pool", 1)); pad = int(L.get("pad", 1))
        layer = tp.Layer(L["name"], H, W, cin, L["cout"], pad=pad)
        T = tp.Tiling(til["n_if"], til["tc"], til["tk"])
        key = (L["name"], H, W, cin, T.n_if, T.tc, T.tk, til["policy"], bd.bank_words, nb, self.host_pp)
        plan = _PLANS.get(key)
        if plan is None:
            plan = _PLANS[key] = tp.build_plan(layer, T, til["policy"], bank_words=bd.bank_words, nb=nb,
                                               pp_pool=0 if self.host_pp else pool, with_stats=False)
        S, K = layer.S, layer.K
        xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad))).view(np.uint8)
        wu = L["w_u8"]
        passes, rounds = plan.passes, plan.rounds
        if len(passes) > 1:
            # with several Cin passes nothing stays resident between two jobs of a bank (every pass reloads its
            # IF/weight slice), so the job order is free: kernel tile outer -> the post-proc requant params
            # (one set per kernel tile) are loaded once per kernel tile instead of once per round
            seqs = [sorted((rnd[b] for rnd in rounds if rnd[b] is not None), key=lambda j: (j.kt, j.sid)) for b in range(nb)]
            rounds = [[sq[i] if i < len(sq) else None for sq in seqs] for i in range(max(len(sq) for sq in seqs))]
        steps = [(r, p) for r in range(len(rounds)) for p in range(len(passes))]
        if self.host_pp: acc = np.zeros((layer.Cout, layer.OH, layer.OW), dtype=np.int32)
        y = np.empty((layer.Cout, layer.OH // pool, layer.OW // pool), dtype=np.int8) if not self.host_pp else None
        res_if, res_w = [None] * nb, [None] * nb
        st = dict(crit=0, macs=0, words_in=0, words_out=0, steps=len(steps), t_load=0.0, t_run=0.0, t_pp=0.0,
                  t_drain=0.0, t_prep=0.0)
        patch_c = 3 * S + K

        regs = self.regs                         # register state as it WILL be after the steps prepared so far
        def prep(step):                          # host side of a step (runs while the previous step computes):
            t0 = time.perf_counter()             # residency, staging buffers, descriptor diff
            r, p = step; c0, c1 = passes[p]; cg = c1 - c0; loads, writes, mask = [], [], 0
            for b, j in enumerate(rounds[r]):
                if j is None: continue
                mask |= 1 << b
                ih, iw = tp.in_extent(j.oh, layer), tp.in_extent(j.ow, layer)
                lo = hi = None
                if res_if[b] != (j.sid, c0, c1):
                    res_if[b] = (j.sid, c0, c1)
                    n = cg * ih * iw
                    bd.stage[b][plan.if_base:plan.if_base + n].reshape(cg, ih, iw)[...] = \
                        xp[c0:c1, j.oy0 * S:j.oy0 * S + ih, j.ox0 * S:j.ox0 * S + iw]
                    lo, hi = plan.if_base, plan.if_base + n
                if res_w[b] != (j.kt, c0, c1):
                    res_w[b] = (j.kt, c0, c1)
                    n = j.nk * cg * K * K
                    bd.stage[b][plan.w_base:plan.w_base + n].reshape(j.nk, cg, K, K)[...] = wu[j.k0:j.k1, c0:c1]
                    lo = plan.w_base if lo is None else lo; hi = plan.w_base + n
                if lo is not None: loads.append((b, lo, hi - lo))    # one transfer (IF..W are adjacent)
                if p == 0:                       # new job: whole descriptor (only changed words are written)
                    vals = (iw, K, K, cg, j.nk, S, j.oh, j.ow, cg * K * K, ih * iw, j.oh * j.ow, S * iw,
                            plan.if_base, plan.w_base, plan.of_base, S * patch_c, patch_c * patch_c, 0)
                    fields = enumerate(vals)
                else:                            # later Cin pass of the same job: 3 words can change
                    fields = ((3, cg), (8, cg * K * K), (17, 1))
                offs = OFFS[b]
                for f, v in fields:
                    off = offs[f]
                    if regs.get(off) != v: regs[off] = v; writes.append((off, v))
                st["macs"] += j.oh * j.ow * j.nk * cg * K * K
            if regs.get(G["UMASK"]) != mask: regs[G["UMASK"]] = mask; writes.append((G["UMASK"], mask))
            st["t_prep"] += time.perf_counter() - t0
            return loads, writes

        pending = None                           # drained round waiting to be unpacked
        def unpack(pend):
            for b, j, buf in pend:
                if self.host_pp:
                    n = j.nk * j.oh * j.ow
                    acc[j.k0:j.k1, j.oy0:j.oy1, j.ox0:j.ox1] = buf[plan.of_base:plan.of_base + n].view(np.int32).reshape(j.nk, j.oh, j.ow)
                else:
                    ph, pw = j.oh // pool, j.ow // pool; n = j.nk * ph * pw
                    lowb = buf.view(np.uint8)[4 * plan.pp_base: 4 * (plan.pp_base + n): 4]
                    y[j.k0:j.k1, j.oy0 // pool:j.oy0 // pool + ph, j.ox0 // pool:j.ox0 // pool + pw] = lowb.view(np.int8).reshape(j.nk, ph, pw)

        nxt = prep(steps[0])
        for i, (r, p) in enumerate(steps):
            (loads, writes), last = nxt, p == len(passes) - 1
            t0 = time.perf_counter()
            for k, (b, off, n) in enumerate(loads):
                bd.load_start(b, off, n); st["words_in"] += n
                if k == 0:                                     # register writes while the first DMA runs
                    for off_, v in writes: bd.cfg_w(off_, v)
                bd.load_wait()
            if not loads:
                for off_, v in writes: bd.cfg_w(off_, v)
            mask = regs[G["UMASK"]]
            t1 = time.perf_counter(); st["t_load"] += t1 - t0
            bd.cfg_w(G["CTRL"], 1)                       # CONV_START
            # ---- overlap: host work while the NMC runs ----
            if i + 1 < len(steps): nxt = prep(steps[i + 1])
            if pending is not None: unpack(pending); pending = None
            bd.wait_idle(f"{L['name']} round {r} pass {p}")
            t2 = time.perf_counter(); st["t_run"] += t2 - t1
            if self.read_perf:
                st["crit"] += max(bd.cfg_r(BA(b, PERF_TOTAL)) for b, j in enumerate(rounds[r]) if j is not None)
            if not last: continue
            # ---- round finished: post-proc on chip, then drain ----
            jobs = [(b, j) for b, j in enumerate(rounds[r]) if j is not None]
            if not self.host_pp:
                self._load_qp(lid, L, jobs)
                for b, j in jobs:
                    ph, pw = j.oh // pool, j.ow // pool
                    d = dict(src_base=plan.of_base, dst_base=plan.pp_base, src_cols=j.ow, out_rows=ph, out_cols=pw,
                             channels=j.nk, pool_k=pool, src_ch_size=j.oh * j.ow, row_step=pool * j.ow,
                             total_out=j.nk * ph * pw, out_zp=0, relu_en=1 if L["relu"] else 0)
                    for k, v in d.items(): bd.cfg_set(BA(b, PP_F[k]), v)
                bd.cfg_set(G["PMASK"], mask)
                bd.cfg_w(G["CTRL"], 2)                   # PP_START
                bd.wait_idle(f"{L['name']} post-proc round {r}")
            t3 = time.perf_counter(); st["t_pp"] += t3 - t2
            pend = []
            for b, j in jobs:
                if self.host_pp: off, n = plan.of_base, j.nk * j.oh * j.ow
                else: off, n = plan.pp_base, j.nk * (j.oh // pool) * (j.ow // pool)
                pend.append((b, j, bd.drain(b, off, n))); st["words_out"] += n
            # the drain buffer of a bank is reused next round: copy out before the next drain (it is
            # unpacked during the next step's compute; one round always has >= 1 step in between)
            pending = pend
            st["t_drain"] += time.perf_counter() - t3
        if pending is not None: unpack(pending)
        bd.cfg_w(G["IRQST"], 0xF)                        # sticky done bits (IRQEN = 0, so no interrupt)
        if self.host_pp:
            t0 = time.perf_counter()
            a = vi.maxpool(acc, pool) if pool > 1 else acc        # pool first (requant is monotone): 4x less work
            y = vi.requant(a, L["bias"], L["mult"], L["shift"], L["relu"]).astype(np.int8)
            st["t_hostpp"] = time.perf_counter() - t0
        return y, st

    def _load_qp(self, lid, L, jobs):
        bd = self.bd
        groups = {}
        for b, j in jobs:
            key = (lid, j.k0, j.k1)
            if self.qp_loaded[b] != key: groups.setdefault(key, []).append(b)
        for (lid_, k0, k1), banks in groups.items():
            bd.cfg_set(G["QPMASK"], sum(1 << b for b in banks))
            bd.cfg_w(G["QPADDR"], 0)
            for c in range(k0, k1):
                bd.cfg_set(G["QPBIAS"], int(L["bias"][c]) & 0xFFFFFFFF)
                bd.cfg_set(G["QPMULT"], int(L["mult"][c]))
                bd.cfg_w(G["QPSHIFT"], int(L["shift"][c]))          # commit (always written)
            for b in banks: self.qp_loaded[b] = (lid_, k0, k1)

# ============================================================================= network
def run_network(bd, model, x, tilings, host_pp=False, read_perf=True, ref=None, log=print, mhz=60.0):
    cr = ConvRunner(bd, host_pp=host_pp, read_perf=read_perf)
    tot = dict(crit=0, macs=0, words=0, load=0.0, run=0.0, pp=0.0, drain=0.0, fc=0.0, hostpp=0.0)
    acts, ok = [], True
    t_all = time.perf_counter()
    log(f"{'layer':9} {'shape':>16} {'NMC cycles':>11} {'steps':>6} {'DMA Mw':>7} {'load s':>7} {'run s':>7} {'pp+drain':>8} {'layer s':>8}  check")
    for lid, L in enumerate(model.layers):
        t0 = time.perf_counter()
        if L["type"] == "conv":
            til = tilings[L["name"]]
            cin, H, W = x.shape
            x, st = cr.run(lid, L, x, til)
            dt = time.perf_counter() - t0
            tot["crit"] += st["crit"]; tot["macs"] += st["macs"]; tot["words"] += st["words_in"] + st["words_out"]
            tot["load"] += st["t_load"]; tot["run"] += st["t_run"]; tot["pp"] += st["t_pp"]; tot["drain"] += st["t_drain"]
            tot["hostpp"] += st.get("t_hostpp", 0.0)
            shape = f"{cin}x{H}x{W}->{L['cout']}"
            line = (f"{L['name']:9} {shape:>16} {st['crit']:>11} {st['steps']:>6} "
                    f"{(st['words_in'] + st['words_out']) / 1e6:>7.2f} {st['t_load']:>7.2f} {st['t_run']:>7.2f} "
                    f"{st['t_pp'] + st['t_drain']:>8.2f} {dt:>8.2f}")
            acts.append(x)
        else:
            acc = fc_sparse(x, L["wT"]) + L["bias"]
            if L.get("final"):
                x = acc.astype(np.float64) * np.asarray(L["scale"], dtype=np.float64)
            else:
                x = vi.requant(acc, np.zeros_like(acc), L["mult"], L["shift"], L["relu"]).astype(np.int8); acts.append(x)
            dt = time.perf_counter() - t0; tot["fc"] += dt
            line = f"{L['name']:9} {str(L['nin']) + '->' + str(L['nout']):>16} {'(ARM)':>11} {'':>6} {'':>7} {'':>7} {'':>7} {'':>8} {dt:>8.2f}"
        chk = ""
        if ref is not None and not L.get("final"):
            same = np.array_equal(acts[-1], ref[f"act{len(acts) - 1}"]); chk = "OK" if same else "MISMATCH"; ok &= same
        log(f"{line}  {chk}")
        if chk == "MISMATCH": break
    wall = time.perf_counter() - t_all
    return x, acts, ok, tot, wall
