# IMC Challenge 2025

Solutions developed for the **IMC Challenge 2025**, a signal-processing competition. Our team placed **10th overall** and won **3rd prize**.

Both problems revolve around recovering a clean digital signal from a noisy, distorted analog channel — the two classic sides of a communications link: given what you *sent*, predict what the channel *produced* (Problem A), and given what the channel *produced*, recover what was *sent* (Problem B).

## Problem A — Analog Waveform Reconstruction (`Problem_A.cpp`)

Given a training pair of a known digital symbol sequence and the corresponding analog waveform it produced, predict the analog waveform that a *new* digital sequence would produce over the same channel.

**Approach:** an ensemble of several independent predictors, each modeling the channel differently, combined with a learned blending weight:

- **1801** — context-based symbol lookup (symbol, duplet, triplet, quintuplet, septuplet patterns).
- **5150 / 1620 (SignalModel)** — cached polynomial ridge regression blended with an internal residual LUT model.
- **1425** — memory-polynomial model (linear + log-magnitude terms) with delay-line taps.

The models are combined per test case via one of several ensembling strategies (fixed weights, inverse-RMSE, softmax-RMSE, or NNLS-on-simplex), then refined with per-phase affine calibration and a small learned FIR post-filter. Channel-case detection (via the training symbol count `M` and waveform heuristics) selects pre-tuned hyperparameters for each configuration, since the optimal model mix differs by test case.

**Input format** (stdin):
```
N
<N analog samples>              (training waveform)
M
<M digital symbols>             (training sequence that produced the waveform above)
K
<K digital symbols>             (new sequence to predict the waveform for)
```

**Output format** (stdout):
```
Lout
<Lout predicted analog samples>   (Lout = K * oversampling_ratio)
```

## Problem B — PAM4 Signal Equalization (`Problem_B.cpp`)

Given a noisy, inter-symbol-interference-distorted analog waveform sampled at 4x the symbol rate, recover the transmitted PAM4 digital symbols (levels `-3, -1, 1, 3`).

**Approach:** classic adaptive equalization —

1. **Downsample** 4 samples/symbol down to 1 sample/symbol using tunable combining weights.
2. **Normalize** amplitude using a percentile-based gain estimate (robust to outliers).
3. **Equalize** with a feedforward equalizer (FFE) combined with a decision-feedback equalizer (DFE), trained online via LMS with a decaying learning rate and a warmup period before DFE kicks in.
4. **Adapt slicer thresholds** dynamically from running per-level means (self-calibrating against DC/gain drift).
5. **Re-run** the converged filter over the full signal for final symbol decisions.

Equalizer parameters (FFE/DFE tap counts, learning rates, warmup length, downsampling weights) are selected based on input length `N`, since different test cases correspond to different channel conditions.

**Input format** (stdin):
```
N
<N analog samples>
```

**Output format** (stdout):
```
L
<L decoded PAM4 symbols, each in {-3, -1, 1, 3}>
```
(`L = N / 4`)

## Building

Both solutions are single-file, dependency-free C++17 programs (Problem A uses GCC's `<bits/stdc++.h>`, so a GCC-compatible compiler is required):

```bash
g++ -O2 -std=c++17 -o problem_a Problem_A.cpp
g++ -O2 -std=c++17 -o problem_b Problem_B.cpp
```

## Running

```bash
./problem_a < input_a.txt > output_a.txt
./problem_b < input_b.txt > output_b.txt
```
