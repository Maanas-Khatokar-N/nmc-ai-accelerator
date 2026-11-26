# Cutting VGG-16 time on the PYNQ-Z2 from 84.6 s

> **4 banks:** your bitstream has 4 banks, but `pynq_run_network.py` defaults to `--banks 2`, so the 84.6 s run used
> only half the PEs (557 M cycles matches the 2-bank estimate). `pynq_run_fast.py` defaults to `--banks 4`:
> 246 M ideal cycles, about 4.1 s at 60 MHz / 3.7 s at 66.67 MHz plus stalls. Check the bank count with
> `pynq_bringup.py --bit nmc.bit --banks 4 --stage ids` (the CONFIG register reports NUM_BANKS).

## 1. Where the 84.6 s went (ball.jpg run, 58.82 MHz)

| part | time | cause |
|---|---|---|
| load | **53.1 s** | every int8 of every IF/weight tile was written by the ARM as **one 32-bit MMIO store** over M_AXI_GP0 → SmartConnect → `s_axi_mem`. Measured **~1.5 M words/s (6 MB/s)**, ~650 ns per word, for ~84 M words per image. `MMIO.array[...] = v` does not make AXI bursts. |
| requant / pool (ARM) | 6.3 s | int64 numpy on every int32 conv output (64×224×224 …). The on-chip post-proc unit was never used. |
| FC (ARM) | 6.5 s | `load_model(mmap=True)`: the 123 MB of FC weights were **memory-mapped** `.npy` files, so fc6 read 103 MB **from the SD card** during the run. |
| drain | 5.1 s | int32 psums read back word by word (~10 MB/s) |
| run (NMC) | 10.2 s | the accelerator itself: 557 M cycles. **This is the only part that is the architecture.** |

So, on your three guesses: the **AXI path was the bottleneck** (not Vivado timing, not the ARM clock), and the
model was **not** sitting in DDR: it was mmap'd and paged in from the SD card. The 512 MB DDR is fine.
Re-loading the model on every script run does not count in the 84.6 s (that was measured after loading), but
the web console re-runs the script for every image, so it pays it each time (see section 5).

## 2. What changed

| change | where | effect |
|---|---|---|
| **AXI CDMA** (DDR → `s_axi_mem` via S_AXI_HP0, 256-beat bursts) | block design, `vivado/add_cdma.tcl` | load: ARM MMIO → DMA |
| `s_axi_mem` streams **1 beat / clock** (was ~4) for writes and reads | `rtl/fpga/nmc_axi_wrapper.v` | DMA ≈ 1 word/clock ≈ 65 M words/s at 66.7 MHz |
| **requant + ReLU + max-pool on chip**, in place over the OFMap | `nmc_fast.py` (RTL unchanged) | no ARM requant; drain only the int8 (pooled) result: 4.5 M words instead of 13.5 M |
| ARM prepares the next pass while the NMC computes; only changed registers are written; requant params loaded once per kernel tile | `nmc_fast.py` | host overhead hidden behind compute |
| model fully in RAM; FC weights stored transposed; FC uses only **non-zero** inputs (pool5 is 86 % zeros) | `nmc_fast.py` | FC 6.5 s → well under 1 s, still bit-exact |
| FCLK0 58.82 → **66.67 MHz** (17 → 15 ns; WNS was +2.63 ns) | `add_cdma.tcl` | compute 9.47 s → 8.36 s |

Verification done here, all bit-exact against `vgg_int` / `ref.npz` (every layer plus the logits):
* `tb/unit/nmc_axi_wrapper_tb.v`: all old tests pass, plus new ones for streaming throughput (256 beats in 260 clocks,
  both ways), read back-pressure, cfg accesses during a streamed write, and partial-strobe beats inside a burst.
* `scripts/postproc_golden.py --inplace`: post-proc with dst = src, 16/16 cases, with and without host noise.
* `pynq_run_fast.py --sim`: whole VGG-16 through a functional model of the cluster, both reference images,
  repeated images in one process, `--host-pp` too.
* `nmc_fast.HwBoard` against a mock `pynq` (fake MMIO/CDMA registers on top of the simulator), both CDMA and `--mmio`.

What I could **not** test: the real CDMA and SmartConnect on the board, the Vivado run, timing at 15 ns.

## 3. Tonight: build steps

1. **Copy the files** from this branch into your project: `rtl/fpga/nmc_axi_wrapper.v`, `fpga/*.py`, `fpga/vivado/add_cdma.tcl`.
2. **Refresh the wrapper** in the block design. If it is a module reference: Vivado shows "module references out of
   date" → *Refresh Changed Modules*. If you packaged it as an IP: re-package it, then *Upgrade IP*.
3. **Add the CDMA.** Tcl console:
   ```
   source K:/IIT Dh/RnD/Vivado/<repo>/fpga/vivado/add_cdma.tcl
   ```
   It prints the CDMA address map. Check that `s_axi_mem` is at **0x4008_0000** and HP0 DDR at **0x0000_0000**
   in `axi_cdma_0/Data`. If s_axi_mem ended up somewhere else, pass that address as `--dma-mem-base`.
4. Generate Block Design → Synthesis → Implementation → Bitstream. Check `report_timing_summary`:
   * WNS ≥ 0 at 15 ns → run at 66.67 MHz (PYNQ sets it from the .hwh automatically).
   * WNS slightly negative → still usable: run with `--mhz 62.5` (16 ns) or `--mhz 58.82` (17 ns).
     The design is safe at any period ≥ 15 ns − WNS.
   * Utilization was 73.6 % LUT / 89 % slices. CDMA + one SmartConnect add roughly 4–5 k LUTs, which should fit.
5. Copy `nmc.bit` + `nmc.hwh` (same base name) to the board.

## 4. On the board

```
# once (writes L13_fc6_wT.npy etc. next to the model, ~1 min, one time only)
sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --prep-only

# bit-exact check against the PC reference first
sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --ref models/vgg16/ref.npz

# your image(s): model and bitstream are loaded once, every image is timed separately
sudo -E python3 pynq_run_fast.py --bit nmc.bit --model models/vgg16 --image test_img/ball.jpg test_img/cat.jpg
```
Needs `imagenet_pre.py` next to it, as before.

**Before the rebuild** you can already run the new software on the old bitstream with `--mmio`. That gives on-chip
post-proc, the fast FC and the model in RAM, but loads stay at ~6 MB/s, so expect about 65 s (load ≈ 54 s + compute).

If something fails on the board:
* `no AXI CDMA in this overlay`: the .hwh next to the .bit is the old one.
* `CDMA error ... DMADecErr (0x40)`: the CDMA cannot reach that address. Check the address map (step 3), then `--dma-mem-base`.
* Mismatch with `--ref`: run the same command with `--host-pp`. If that passes, the on-chip post-proc is the issue;
  keep `--host-pp` for the demo (costs ~2–3 s).
* Hang in `wait_idle`: check that STATUS reads 0 after `pynq_bringup.py --stage ids`.

## 5. Expected time (estimate, not measured)

| part | before | after (66.67 MHz) |
|---|---|---|
| NMC compute (557 M cycles) | 9.5 s | **8.4 s** |
| IF/weight DMA (≈ 98 M words incl. drain, ~1 word/clock) | 58 s | ~1.5 s |
| per-pass host overhead (~6.7 k passes, ~14 k DMA starts) | n/a | ~0.5–1 s |
| post-proc on chip | 6.3 s (ARM) | ~0.1 s |
| FC 6/7/8 | 6.5 s | ~0.2–0.5 s |
| **total** | **84.6 s** | **≈ 11 s with 2 banks; ≈ 6–7 s with your 4 banks** |

The floor is the compute itself: 15.35 G MACs on 2 banks × 16 PEs = 32 MAC/cycle is 480 M cycles at best.
Measured 557 M (13 % psum store/reload stalls) / 66.67 MHz = 8.4 s. To go well below 10 s you need:
* **A faster clock.** Re-run `add_cdma.tcl` with `set FCLK_MHZ 71.428571` (14 ns) and an implementation strategy
  of Performance_ExplorePostRoutePhysOpt. The critical path is BRAM → LUT → LUTRAM in the IF load path, mostly routing (7.2 ns of 13.5 ns).
* **Overlapping DMA with compute** (ping-pong IF/weight regions in the bank). The memory map is full today (32 232 of 32 768 words).
* **Fewer Cin passes** (RF deeper than 256) cuts both the 13 % stall and the per-pass overhead.

For the **web console**, every run re-downloads the bitstream and re-reads ~140 MB of model from the SD card.
That startup is not part of the per-image time printed. Use `--json` with `result_json` in `config.json` to show the numbers.
