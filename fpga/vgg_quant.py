#!/usr/bin/env python3
"""
Post-training quantisation of a VGG-style float network to the NMC integer format (numpy only, no torch).
  - weights: symmetric INT8, per output channel      - activations: symmetric INT8 (ReLU outputs use 0..127), per tensor
  - bias folded into the accumulator domain (int32)   - requant multiplier folded into (uint16 mult, shift) per channel
Layer spec (float):  conv: dict(type="conv", name, w[Cout,Cin,3,3], b[Cout], relu=True, pool=1|2)
                     fc:   dict(type="fc",   name, w[out,in],       b[out],  relu=bool, final=bool)
"""
import numpy as np
from vgg_int import fold_requant, _conv_mm, maxpool

def float_forward(layers, x, collect=False):
    acts = []
    for L in layers:
        if L["type"] == "conv":
            xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
            y = _conv_mm(xp, L["w"].astype(np.float64)) + L["b"].astype(np.float64)[:, None, None]
            if L["relu"]: y = np.maximum(y, 0.0)
            if collect: acts.append(y)                 # pre-pool, post-ReLU: what the requant stage sees
            x = maxpool(y, L["pool"]) if L.get("pool", 1) > 1 else y
        else:
            y = L["w"].astype(np.float64) @ x.reshape(-1) + L["b"].astype(np.float64)
            if L["relu"]: y = np.maximum(y, 0.0)
            if collect: acts.append(y)
            x = y
    return x, acts

def quantize_model(layers, calib, pct=99.99, peaks=None):
    """calib: list of float inputs [C,H,W]. peaks: optional precomputed activation range per layer (e.g. from torch,
    faster for the real VGG-16); otherwise they are measured here with the numpy float forward.
    Returns (model, x_scale)."""
    x_scale = float(max(np.abs(c).max() for c in calib)) / 127.0
    if peaks is None:                       # activation range of every layer output (percentile over the calibration set)
        for c in calib:
            _, acts = float_forward(layers, c, collect=True)
            p = [float(np.percentile(np.abs(a), pct)) if a.size > 1 else float(np.abs(a).max()) for a in acts]
            peaks = p if peaks is None else [max(u, v) for u, v in zip(peaks, p)]
    model = dict(input=dict(shape=list(calib[0].shape), x_scale=x_scale), layers=[])
    s_in = x_scale
    for L, peak in zip(layers, peaks):
        w = L["w"].astype(np.float32)
        axes = tuple(range(1, w.ndim))
        w_peak = np.max(np.abs(w), axis=axes)
        w_scale = np.where(w_peak > 0, w_peak / 127.0, 1.0).astype(np.float64)
        w_q = np.clip(np.rint(w / w_scale.reshape((-1,) + (1,) * (w.ndim - 1))), -127, 127).astype(np.int8)
        bias_q = np.rint(L["b"].astype(np.float64) / (s_in * w_scale)).astype(np.int64)
        assert np.abs(bias_q).max() < 2 ** 31, "bias does not fit int32"
        m = dict(type=L["type"], name=L["name"], relu=int(L["relu"]), w=w_q, bias=bias_q.astype(np.int32))
        if L["type"] == "conv": m.update(pad=1, pool=int(L.get("pool", 1)), cin=int(w.shape[1]), cout=int(w.shape[0]))
        else:                   m.update(nin=int(w.shape[1]), nout=int(w.shape[0]))
        if L.get("final"):
            m.update(final=1, scale=(s_in * w_scale).astype(np.float64))      # logits = (acc + bias) * scale
        else:
            s_out = max(peak, 1e-12) / 127.0
            mult, shift = fold_requant(s_in * w_scale / s_out)
            m.update(mult=mult.astype(np.int32), shift=shift.astype(np.int32))
            s_in = s_out
        model["layers"].append(m)
    return model, x_scale

def quantize_input(x, x_scale):
    return np.clip(np.rint(x / x_scale), -127, 127).astype(np.int8)
