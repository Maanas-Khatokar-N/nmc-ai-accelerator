# Scalable Near-Memory AI Accelerator

A synthesizable Verilog implementation of a scalable
all-digital Near-Memory Computing (NMC) architecture
for AI acceleration.

## Motivation

Conventional AI accelerators suffer from significant
data movement between memory and computation.

This project implements a digital NMC architecture where
processing elements are placed close to the memory subsystem,
reducing unnecessary data movement.

## Reference Paper

A Scalable All-Digital Near-Memory Computing Architecture
for Edge AIoT Applications

IEEE Access, 2025

## Architecture


RISC-V / Host
     |
     v
Shared Memory
     |
     +---- Bank 0 ---- NMC Unit ---- PE Array
     |
     +---- Bank 1 ---- NMC Unit ---- PE Array
     |
     +---- Bank 2 ---- NMC Unit ---- PE Array
     |
     +---- Bank 3 ---- NMC Unit ---- PE Array

## Main Modules

- Memory Banks
- IFMap / OFMap Memory
- Register File
- Address Generator
- Access Controller
- Arbitration Logic
- Processing Elements
- MAC Units
- NMC Controller

## Dataflow

Input Feature Map
        ↓
Memory
        ↓
Register File
        ↓
PE Array
        ↓
MAC
        ↓
Partial Sum
        ↓
OFMap Memory

## Supported Operations

- Multiply
- Accumulate
- Convolution
- Partial-sum accumulation

## Verification

Simulation using:
- Icarus Verilog
- GTKWave

## Status

[ ] Memory subsystem
[ ] PE
[ ] MAC
[ ] Address generator
[ ] Controller
[ ] Arbiter
[ ] AI datapath
[ ] Top-level integration
[ ] Verification
[ ] Synthesis