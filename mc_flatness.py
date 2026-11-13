"""Why do better volatility forecasts barely change inverse-vol portfolios?
Monte Carlo in a one-factor world: r_i = beta_i * f + e_i.  For a basket of k stocks we compare the
true portfolio volatility of
  EW, inverse-vol with forecast noise s (log sigma_hat = log sigma + N(0, s^2)), inverse-vol with the
  true sigma (s = 0), and long-only minimum variance with the true covariance (uses correlations).
Parameters are set to roughly match the 1,000-stock universe (market vol 17%, median total vol ~32%).
Run: ~/MPCS/envs/base/bin/python mc_flatness.py"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)
K, N, SF = 10, 4000, 0.17


def draw():
    beta = np.clip(rng.normal(1.0, 0.35, K), 0.2, 2.2)
    idio = np.exp(rng.normal(np.log(0.25), 0.45, K))
    cov = SF ** 2 * np.outer(beta, beta) + np.diag(idio ** 2)
    return cov, np.sqrt(np.diag(cov))


def iv(sig, cap=2.0 / K):
    w = 1 / sig; w /= w.sum()
    for _ in range(20):
        w = np.minimum(w, cap); w /= w.sum()
    return w


def minvar(cov, it=3000):
    w = np.full(K, 1 / K); lr = 0.5 / np.linalg.eigvalsh(cov).max()
    for _ in range(it):
        w = w - lr * 2 * cov @ w
        u = np.sort(w)[::-1]; c = np.cumsum(u) - 1; r = np.nonzero(u - c / np.arange(1, K + 1) > 0)[0][-1]
        w = np.maximum(w - c[r] / (r + 1), 0)          # projection onto the simplex
    return w


pv = lambda w, cov: float(np.sqrt(w @ cov @ w))
S = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
draws = [draw() for _ in range(N)]
var_logsig = np.mean([np.var(np.log(s)) for _, s in draws])
ew = np.mean([pv(np.full(K, 1 / K), c) for c, _ in draws])
mv = np.mean([pv(minvar(c), c) for c, _ in draws[:800]])
res = {}
for s in S:
    v = [pv(iv(sig * np.exp(rng.normal(0, s, K))), c) for c, sig in draws]
    r2 = var_logsig / (var_logsig + s ** 2)
    res[s] = (r2, float(np.mean(v)))
iv0 = res[0.0][1]
out = {"k": K, "ew_vol": ew, "minvar_vol": mv, "iv_true_vol": iv0, "xs_var_log_sigma": var_logsig,
       "curve": {str(s): {"xs_r2_vs_true": r2, "vol": v, "share_of_ew_to_iv_gain": (ew - v) / (ew - iv0)} for s, (r2, v) in res.items()}}
json.dump(out, open(os.path.join(HERE, "results", "mc_flatness.json"), "w"), indent=1)
print(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
x = [res[s][0] for s in S]; y = [res[s][1] * 100 for s in S]
ax.plot(x, y, "o-", color="#d62728", label="inverse vol, noisy forecast")
ax.axhline(ew * 100, color="#888", ls="--", label="equal weight")
ax.axhline(mv * 100, color="#9467bd", ls=":", label="min variance (true covariance)")
ax.set_xlabel("forecast accuracy: cross-sectional R² of log vol (vs true)"); ax.set_ylabel("portfolio volatility (%)")
ax.set_title(f"Portfolio risk is flat in forecast accuracy ({K}-stock basket, simulated)")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "7_flatness_simulation.png"), dpi=160)
