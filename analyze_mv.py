"""Part 3: does using correlations (minimum variance) beat inverse vol, and does forecast accuracy
matter once correlations are used?  Analyses run 'full3' (same baskets and seeds as full2).
Run: ~/MPCS/envs/base/bin/python analyze_mv.py [full3]"""
import json, os, re, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "full3"
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
PM = ["EW", "ROLL63", "LGBM", "ORACLE", "MV_SAMPLE", "MV_ROLL63", "MV_LGBM"]
KS = [5, 10, 20, 50]
PC = ["ann", "vol", "sh", "mdd", "to", "sh_med"]
DC = ["dvol", "pv", "dsh", "ps", "s5", "s95", "dmdd", "pm", "dann"]
FAC = ["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]

raw = json.load(open(os.path.join(RES, f"{TAG}.json")))["runtimeStatistics"]
TOK = {}
for k in sorted(x for x in raw if re.fullmatch(r"O\d\d", x)):
    for t in raw[k].split(";"):
        if "~" not in t:
            print("WARNING truncated token:", t[:40]); continue
        a, b = t.split("~", 1); TOK.setdefault(a, []).extend(b.split(","))
num = lambda a: [float(x) for x in TOK[a]]
out = {"info": dict(zip(["months", "oos_months", "portfolio_months", "lgbm_fits", "errors", "first", "last"], TOK.get("info", []))),
       "avg_pairwise_corr": TOK.get("rbar"), "errors": {k: v for k, v in raw.items() if k.startswith("_err")}}
FC = ["r2", "mse", "ql", "sp", "dmse_t", "dql_t", "win_ql", "win_mse"]
out["forecast_check"] = {m: dict(zip(FC, num(f"F|{m}"))) for m in ("ROLL63", "LGBM") if f"F|{m}" in TOK}
print(out)

P, D = {}, {}
for k in KS:
    P[k] = pd.DataFrame({m: dict(zip(PC, num(f"P{k}|{m}"))) for m in PM if f"P{k}|{m}" in TOK}).T
    D[k] = {a.split("|", 1)[1]: dict(zip(DC, num(a))) for a in TOK if a.startswith(f"D{k}|")}
    print(f"\nk={k}\n", P[k].round(4).to_string(), "\n", pd.DataFrame(D[k]).T.round(3).to_string())
out["portfolio"] = {k: P[k].round(4).to_dict(orient="index") for k in KS}
out["paired"] = D
yd = {m: {2000 + int(a): (float(b), float(c)) for a, b, c in (x.split("/") for x in TOK[f"Y10|{m}"] if x.count("/") == 2)}
      for m in ("ROLL63", "LGBM", "MV_ROLL63", "MV_LGBM") if f"Y10|{m}" in TOK}
out["paired_by_year_k10"] = yd

ch = json.load(open(os.path.join(RES, f"{TAG}_charts.json")))
def ser(c, s):
    v = ch.get(c, {}).get(s)
    if not v:
        return None
    x = pd.Series({pd.Timestamp(d): val for d, val in v}).sort_index()
    x = x.groupby(x.index.to_period("M")).last(); x.index = x.index - 1
    return x
M = {k: pd.DataFrame({m: ser(f"R{k}", m) for m in PM}).dropna() for k in KS}

def load_ff():
    D_ = os.path.join(HERE, "data")
    lines = open(os.path.join(D_, "F-F_Research_Data_5_Factors_2x3_daily.csv")).read().splitlines()
    rows = [[x.strip() for x in l.split(",")] for l in lines if l[:8].isdigit() and len(l.split(",")) == 7]
    f5 = pd.DataFrame([[float(x) / 100 for x in r[1:]] for r in rows], index=pd.to_datetime([r[0] for r in rows]), columns=FAC[:5] + ["RF"])
    lines = open(os.path.join(D_, "F-F_Momentum_Factor_daily.csv")).read().splitlines()
    rows = [[x.strip() for x in l.split(",")] for l in lines if l[:8].isdigit() and len(l.split(",")) == 2]
    mom = pd.Series([float(r[1]) / 100 for r in rows], index=pd.to_datetime([r[0] for r in rows]), name="MOM")
    d = f5.join(mom, how="inner")
    return (1 + d).groupby(d.index.to_period("M")).prod() - 1

def ols_nw(y, X, lags=3):
    X = np.column_stack([np.ones(len(X)), X]); XtX = np.linalg.inv(X.T @ X); b = XtX @ X.T @ y; e = y - X @ b
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None]); S += (1 - L / (lags + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtX @ S @ XtX))
    return b, b / se, 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))

def stats(r):
    v = (1 + r).cumprod()
    return {"ann": float(v.iloc[-1] ** (12 / len(r)) - 1), "vol": float(r.std() * np.sqrt(12)),
            "sharpe": float(r.mean() / r.std() * np.sqrt(12)), "mdd": float((1 - v / v.cummax()).max())}

def boot(a, b, n=5000, block=6, seed=0):
    """Block bootstrap of Sharpe difference and volatility ratio a/b."""
    rng = np.random.default_rng(seed); T = len(a); A, B = a.values, b.values
    sh = lambda x: x.mean() / x.std() * np.sqrt(12)
    obs = (sh(A) - sh(B), A.std() / B.std()); d = np.empty((n, 2))
    for i in range(n):
        st = rng.integers(0, T, int(np.ceil(T / block)))
        ix = (st[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
        d[i] = sh(A[ix]) - sh(B[ix]), A[ix].std() / B[ix].std()
    return {"dsharpe": float(obs[0]), "p_dsharpe_le0": float(np.mean(d[:, 0] <= 0)), "ci_dsharpe": [float(x) for x in np.percentile(d[:, 0], [2.5, 97.5])],
            "vol_ratio": float(obs[1]), "p_volratio_ge1": float(np.mean(d[:, 1] >= 1)), "ci_vol_ratio": [float(x) for x in np.percentile(d[:, 1], [2.5, 97.5])]}

ff = load_ff(); agg = {}
for k in KS:
    m = M[k]; j = m.join(ff, how="inner")
    agg[k] = {"months": len(m), "stats": {c: stats(m[c]) for c in PM},
              "boot": {f"{a}-{b}": boot(m[a], m[b]) for a, b in [("MV_ROLL63", "EW"), ("MV_ROLL63", "ROLL63"), ("MV_LGBM", "MV_ROLL63"), ("MV_SAMPLE", "MV_ROLL63")]}}
    regs = {}
    for c in ("MV_ROLL63", "MV_LGBM"):
        b, t, r2 = ols_nw((j[c] - j["EW"]).values, j[FAC].values)
        regs[f"{c}-EW"] = {"alpha_ann": float(b[0] * 12), "t": float(t[0]), "r2": float(r2),
                           "loadings": dict(zip(FAC, map(float, b[1:]))), "tstats": dict(zip(FAC, map(float, t[1:])))}
    agg[k]["ff"] = regs
out["avg_basket"] = agg
print(json.dumps({k: {"stats": {c: {a: round(b, 4) for a, b in v.items()} for c, v in agg[k]["stats"].items()}, "boot": agg[k]["boot"], "ff": agg[k]["ff"]} for k in KS}, indent=1)[:6000])
json.dump(out, open(os.path.join(RES, f"summary_{TAG}.json"), "w"), indent=1, default=str)

# ---------------- figures
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COL = {"EW": "#888888", "ROLL63": "#1f77b4", "LGBM": "#d62728", "ORACLE": "#9467bd", "MV_SAMPLE": "#bcbd22", "MV_ROLL63": "#2ca02c", "MV_LGBM": "#ff7f0e"}
LAB = {"ROLL63": "Inverse vol (ROLL63)", "LGBM": "Inverse vol (LGBM)", "MV_SAMPLE": "Min var, sample cov", "MV_ROLL63": "Min var, shrunk corr + ROLL63",
       "MV_LGBM": "Min var, shrunk corr + LGBM", "ORACLE": "Inverse vol, perfect foresight"}
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for m in ("ROLL63", "LGBM", "MV_SAMPLE", "MV_ROLL63", "MV_LGBM"):
    ax[0].plot(KS, [D[k][f"{m}-EW"]["dvol"] * 100 for k in KS], "o-", color=COL[m], label=LAB[m])
    ax[1].plot(KS, [D[k][f"{m}-EW"]["dmdd"] * 100 for k in KS], "o-", color=COL[m], label=LAB[m])
    ax[2].plot(KS, [D[k][f"{m}-EW"]["dsh"] for k in KS], "o-", color=COL[m], label=LAB[m])
ax[0].set_title("Change in volatility vs EW (pp)"); ax[1].set_title("Change in max drawdown vs EW (pp)"); ax[2].set_title("Change in Sharpe vs EW")
for a in ax: a.set_xscale("log"); a.set_xticks(KS); a.set_xticklabels(KS); a.axhline(0, color="k", lw=0.6); a.set_xlabel("stocks per basket")
ax[0].legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "8_minvar_vs_invvol.png"), dpi=160); plt.close(fig)
if "MV_ROLL63" in yd:
    fig, ax = plt.subplots(figsize=(10, 3.4)); yrs = sorted(yd["ROLL63"]); w = 0.38
    ax.bar(np.array(yrs) - w / 2, [yd["ROLL63"][y][0] * 100 for y in yrs], w, color=COL["ROLL63"], label="Inverse vol − EW")
    ax.bar(np.array(yrs) + w / 2, [yd["MV_ROLL63"][y][0] * 100 for y in yrs], w, color=COL["MV_ROLL63"], label="Min var (shrunk) − EW")
    ax.axhline(0, color="k", lw=0.6); ax.set_title("Volatility change vs equal weight, by year (10-stock baskets, pp)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "9_minvar_by_year.png"), dpi=160); plt.close(fig)
print(f"saved results/summary_{TAG}.json, figures 8-9")
