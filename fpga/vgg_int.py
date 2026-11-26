#!/usr/bin/env python3
"""
Integer (bit-exact) model of the network as the NMC hardware computes it, plus the model-package format.

  conv   : acc = sum int8*int8 (int32, exact)             -> PE array
  requant: y = clip(((acc + bias) * mult + 2^(shift-1)) >> shift + zp, lo, 127)   mult uint16, shift 0..31,
           lo = zp when ReLU else -128                      -> rtl/postproc/requant.v (zp = 0 here)
  pool   : max-pool k x k                                   -> pool before requant == pool after (requant is monotone)
  fc     : same arithmetic on the host (ARM)               -> the paper's hybrid execution

Model package = directory with manifest.json + one .npy per array (big weights can be memory-mapped).
"""
import json, os
import numpy as np

# ----------------------------------------------------------------------------- requant parameters
def fold_requant(M):
    """real multiplier M (>0) per channel -> (mult uint16, shift 0..31) with mult / 2^shift ~= M"""
    M = np.atleast_1d(np.asarray(M, dtype=np.float64))
    mult = np.zeros(M.shape, dtype=np.int64); shift = np.zeros(M.shape, dtype=np.int64)
    for i, m in np.ndenumerate(M):
        if m <= 0: continue
        sh = int(np.floor(np.log2(65535.0 / m)))              # largest shift that keeps mult <= 65535
        sh = max(0, min(31, sh))
        mult[i] = min(int(np.rint(m * 2.0 ** sh)), 65535); shift[i] = sh
    return mult, shift

def _bc(v, ndim):                                              # per-channel vector -> broadcastable
    v = np.asarray(v, dtype=np.int64)
    return v.reshape((-1,) + (1,) * (ndim - 1))

def _requant_block(acc, bias, mult, shift, relu, zp):
    v = (acc.astype(np.int64) + _bc(bias, acc.ndim)) * _bc(mult, acc.ndim)
    sh = _bc(shift, acc.ndim)
    rnd = np.where(sh > 0, np.int64(1) << np.maximum(sh - 1, 0), 0)
    q = (v + rnd) >> sh
    return np.clip(q + zp, zp if relu else -128, 127).astype(np.int16)

def requant(acc, bias, mult, shift, relu=True, zp=0):
    """returns int16 holding values in [-128,127]; channel-blocked so a 64x224x224 layer needs only a few MB of temporaries"""
    acc = np.asarray(acc)
    if acc.ndim < 3 or acc.shape[0] <= 16:
        return _requant_block(acc, bias, mult, shift, relu, zp)
    bias, mult, shift = (np.asarray(t, dtype=np.int64) for t in (bias, mult, shift))
    out = np.empty(acc.shape, dtype=np.int16)
    for c in range(0, acc.shape[0], 16):
        out[c:c + 16] = _requant_block(acc[c:c + 16], bias[c:c + 16], mult[c:c + 16], shift[c:c + 16], relu, zp)
    return out

def maxpool(x, k):
    c, h, w = x.shape
    x = x[:, :h // k * k, :w // k * k]
    return x.reshape(c, h // k, k, w // k, k).max(axis=(2, 4))

# ----------------------------------------------------------------------------- exact convolution / fc (float64 BLAS)
def _conv_mm(xp, w, s=1, max_elems=1 << 24):
    """valid convolution of a padded input; float64 matmul (exact for integers: sums stay far below 2^53)"""
    cin, hp, wp = xp.shape
    cout, _, kr, kc = w.shape
    oh, ow = (hp - kr) // s + 1, (wp - kc) // s + 1
    W2 = w.reshape(cout, -1).astype(np.float64)
    xf = xp.astype(np.float64)
    out = np.empty((cout, oh, ow), dtype=np.float64)
    rows = max(1, max_elems // (cin * kr * kc * ow))
    for r0 in range(0, oh, rows):
        r1 = min(oh, r0 + rows)
        patches = np.empty((cin, kr, kc, r1 - r0, ow), dtype=np.float64)
        for ki in range(kr):
            for kj in range(kc):
                patches[:, ki, kj] = xf[:, r0 * s + ki:(r1 - 1) * s + ki + 1:s, kj:(ow - 1) * s + kj + 1:s]
        out[:, r0:r1] = (W2 @ patches.reshape(cin * kr * kc, -1)).reshape(cout, r1 - r0, ow)
    return out

def conv_acc(xp, w, s=1):
    return np.rint(_conv_mm(xp, w, s)).astype(np.int32)

def fc_acc(x, w, rows=64):
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    out = np.empty(w.shape[0], dtype=np.int64)
    for i in range(0, w.shape[0], rows):
        out[i:i + rows] = np.rint(np.asarray(w[i:i + rows], dtype=np.float64) @ x64).astype(np.int64)
    return out

# ----------------------------------------------------------------------------- integer forward
def forward_int(model, x_q, collect=True):
    """x_q int8 [C,H,W] -> (logits_float, logits_acc, list of int8 activations after every layer)"""
    x, acts = x_q, []
    for L in model["layers"]:
        if L["type"] == "conv":
            pad = L.get("pad", 1)
            xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
            y = requant(conv_acc(xp, L["w"], 1), L["bias"], L["mult"], L["shift"], L["relu"])
            if L.get("pool", 1) > 1: y = maxpool(y, L["pool"])
            x = y.astype(np.int8)
            if collect: acts.append(x)
        else:
            acc = fc_acc(x, L["w"]) + np.asarray(L["bias"], dtype=np.int64)
            if L.get("final"):
                return acc.astype(np.float64) * np.asarray(L["scale"], dtype=np.float64), acc, acts
            x = requant(acc, np.zeros_like(acc), L["mult"], L["shift"], L["relu"]).astype(np.int8)
            if collect: acts.append(x)
    raise ValueError("model has no final fc layer")

# ----------------------------------------------------------------------------- model package IO
ARRAYS = ("w", "bias", "mult", "shift", "scale")

def save_model(d, model):
    os.makedirs(d, exist_ok=True)
    meta = dict(input=model["input"], layers=[])
    for i, L in enumerate(model["layers"]):
        m = {k: v for k, v in L.items() if k not in ARRAYS}
        for a in ARRAYS:
            if a in L:
                fn = f"L{i:02d}_{L['name']}_{a}.npy"; np.save(os.path.join(d, fn), np.asarray(L[a])); m[a] = fn
        meta["layers"].append(m)
    with open(os.path.join(d, "manifest.json"), "w") as f: json.dump(meta, f, indent=1)

def load_model(d, mmap=True):
    meta = json.load(open(os.path.join(d, "manifest.json")))
    layers = []
    for m in meta["layers"]:
        L = dict(m)
        for a in ARRAYS:
            if a in m: L[a] = np.load(os.path.join(d, m[a]), mmap_mode="r" if (mmap and a == "w") else None)
        layers.append(L)
    return dict(input=meta["input"], layers=layers)