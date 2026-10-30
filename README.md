# Echo State Network (Reservoir Computing) for Chaotic Time-Series Prediction

A fixed random recurrent tanh network (2 % connectivity, spectral radius ρ) is driven by the normalised Lorenz-63 state; only
a ridge-regression readout is trained for one-step-ahead prediction (small input noise during training for closed-loop
stability). The network is then run **autonomously** and scored by *valid prediction time*: Lyapunov times until the
normalised error exceeds 0.4 (8 start points × 3 reservoirs).

```
pip install numpy scipy matplotlib
python esn.py     # ~3 min; writes results.json and forecast.png
```

## Results (mean valid time in Lyapunov times, `results.json`)
| Spectral radius ρ (n=500) | 0.3 | 0.6 | 0.9 | 1.1 | 1.3 |
|---|---|---|---|---|---|
| valid time | 4.6 | 5.3 | 4.9 | 3.9 | 1.1 |

| Reservoir size n (ρ=0.9) | 100 | 300 | 500 | 1000 |
|---|---|---|---|---|
| valid time | 0.9 | 4.0 | 4.9 | 6.0 |

Larger reservoirs predict longer; performance collapses for ρ well above 1, while ρ ∈ [0.3, 0.9] are statistically similar.

## Caveats
Noise-free simulated Lorenz data, single noise level (1e-3) and tolerance (0.4) chosen in development on the same system;
no comparison against other forecasters; hyper-parameters were tuned on the same trajectory family used for testing.
