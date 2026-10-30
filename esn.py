"""Echo state network (reservoir computing) for chaotic time-series prediction.

A fixed random recurrent rate network (tanh units, spectral radius rho, leak a) is driven by the
Lorenz-63 x-coordinate; only a ridge-regression readout is trained (one-step-ahead prediction). The
trained network is then run autonomously (closed loop) and we measure how many Lyapunov times it
stays on the true trajectory (valid prediction time). Sweeps spectral radius and reservoir size.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

DT = 0.02; LYAP = 0.906           # largest Lyapunov exponent of Lorenz-63 (1/time units)

def lorenz_series(T=400, seed=0):
    rng = np.random.default_rng(seed)
    f = lambda t, s: [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]]
    sol = solve_ivp(f, [0, T], rng.uniform(-5, 5, 3) + [0, 0, 25], t_eval=np.arange(0, T, DT), rtol=1e-9, atol=1e-9)
    y = sol.y.T[int(10 / DT):]                            # drop transient
    return (y - y.mean(0)) / y.std(0)                      # normalise so tanh units are not saturated

class ESN:
    def __init__(self, n=500, rho=0.9, leak=1.0, in_scale=0.5, density=0.02, seed=0, ridge=1e-6):
        rng = np.random.default_rng(seed)
        W = rng.standard_normal((n, n)) * (rng.random((n, n)) < density)
        W *= rho / max(abs(np.linalg.eigvals(W)))
        self.W, self.Win = W, in_scale * rng.uniform(-1, 1, (n, 3)); self.n, self.leak, self.ridge = n, leak, ridge
    def step(self, r, u): return (1 - self.leak) * r + self.leak * np.tanh(self.W @ r + self.Win @ u)
    def fit(self, U, wash=200, noise=0.0, seed=0):
        rng = np.random.default_rng(seed); r = np.zeros(self.n); R = []
        for u in U[:-1]: r = self.step(r, u + noise * rng.standard_normal(3)); R.append(r)   # input noise regularises closed-loop stability
        R = np.array(R)[wash:]; Y = U[1:][wash:]
        self.Wout = np.linalg.solve(R.T @ R + self.ridge * np.eye(self.n), R.T @ Y).T; self.r = r
    def warm(self, U):
        r = np.zeros(self.n)
        for u in U: r = self.step(r, u)
        return r
    def forecast(self, r, steps):
        """r: state after consuming input up to time t-1 -> predicts x_t, x_{t+1}, ... in closed loop."""
        out = []; u = self.Wout @ r; out.append(u)
        for _ in range(steps - 1):
            r = self.step(r, u); u = self.Wout @ r; out.append(u)
        return np.array(out)

def valid_time(esn, series, starts, horizon=600, tol=0.4):
    sd = series.std(0); vts = []
    for s0 in starts:
        r = esn.warm(series[s0 - 200:s0]); pred = esn.forecast(r, horizon)
        err = np.linalg.norm((pred - series[s0:s0 + horizon]) / sd, axis=1) / np.sqrt(3)
        bad = np.flatnonzero(err > tol); k = bad[0] if len(bad) else horizon
        vts.append(k * DT * LYAP)                          # in Lyapunov times
    return float(np.mean(vts)), float(np.std(vts))

if __name__ == "__main__":
    S = lorenz_series(); ntr = 12000; train, test = S[:ntr], S[ntr:]
    starts = [300 + 700 * i for i in range(8)]; starts = [s for s in starts if s + 600 < len(test)]
    out = {"rho_sweep": {}, "size_sweep": {}}
    for rho in (0.3, 0.6, 0.9, 1.1, 1.3):
        vt = []
        for seed in range(3):
            e = ESN(n=500, rho=rho, seed=seed); e.fit(train, noise=1e-3); vt.append(valid_time(e, test, starts)[0])
        out["rho_sweep"][rho] = [float(np.mean(vt)), float(np.std(vt))]; print("rho", rho, out["rho_sweep"][rho], flush=True)
    for n in (100, 300, 500, 1000):
        vt = []
        for seed in range(3):
            e = ESN(n=n, rho=0.9, seed=seed); e.fit(train, noise=1e-3); vt.append(valid_time(e, test, starts)[0])
        out["size_sweep"][n] = [float(np.mean(vt)), float(np.std(vt))]; print("n", n, out["size_sweep"][n], flush=True)
    json.dump(out, open("results.json", "w"), indent=2)
    e = ESN(n=500, rho=0.9); e.fit(train, noise=1e-3); r = e.warm(test[100:300]); pred = e.forecast(r, 600)
    t = np.arange(600) * DT * LYAP; plt.figure(figsize=(7, 3))
    plt.plot(t, test[300:900, 0], label="true x"); plt.plot(t, pred[:, 0], "--", label="ESN closed-loop")
    plt.xlabel("Lyapunov times"); plt.legend(); plt.tight_layout(); plt.savefig("forecast.png", dpi=150)
