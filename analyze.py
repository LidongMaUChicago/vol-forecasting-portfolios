"""Offline analysis of the QuantConnect volatility study.
Inputs : results/<tag>.json (runtime statistics) and results/<tag>_charts.json (monthly series)
Outputs: results/summary.json, figures/*.png, tables printed to stdout.
Run    : ~/MPCS/envs/base/bin/python analyze.py full2   (forecasting + inverse-vol run; figures 1-6)
The minimum-variance run (full3) is analysed by analyze_mv.py."""
import json, os, sys, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[1] if len(sys.argv) > 1 else "full"
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
MODELS = ["NAIVE21", "ROLL63", "EWMA", "HAR", "LINX", "LGBM"]
PM = ["EW", "ROLL63", "EWMA", "HAR", "LGBM", "ORACLE"]
KS = [5, 10, 20, 50]

bt = json.load(open(os.path.join(RES, f"{TAG}.json")))
raw = bt["runtimeStatistics"]
TOK = {}
for k in sorted(x for x in raw if re.fullmatch(r"O\d\d", x)):
    for t in raw[k].split(";"):
        if "~" not in t:
            print("WARNING truncated token:", t[:40]); continue
        a, b = t.split("~", 1); TOK.setdefault(a, []).extend(b.split(","))
num = lambda a: [float(x) for x in TOK[a]]
info = TOK.get("info", [])
out = {"info": dict(zip(["months", "oos_months", "portfolio_months", "lgbm_fits", "errors", "first", "last"], info)),
       "N": TOK.get("N"), "errors": {k: v for k, v in raw.items() if k.startswith("_err")}, "nkeys": raw.get("_nkeys")}
print(out)

# ---------------- forecasting
FC = ["r2", "mse", "ql", "sp", "dmse_t", "dql_t", "win_ql", "win_mse"]
F = pd.DataFrame({m: dict(zip(FC, num(f"F|{m}"))) for m in MODELS if f"F|{m}" in TOK}).T
X = pd.DataFrame({m: dict(zip(["xs_r2", "sp_t"], num(f"X|{m}"))) for m in MODELS if f"X|{m}" in TOK}).T
F = F.join(X) if len(X) else F
print("\nForecast accuracy (out of sample; t-stats vs ROLL63)\n", F.round(4).to_string())
out["forecast"] = F.round(4).to_dict(orient="index")
imp = [(a, float(b)) for a, b in (x.split("/") for x in TOK.get("imp", []))]
out["importance"] = imp
fy = {m: {2000 + int(a): (float(b), float(c)) for a, b, c in (x.split("/") for x in TOK[f"FY|{m}"] if x.count("/") == 2)} for m in MODELS if f"FY|{m}" in TOK}
out["forecast_by_year"] = fy

# ---------------- portfolios (per-basket averages from QC)
PC = ["ann", "vol", "sh", "mdd", "to", "sh_med"]
DC = ["dvol", "pv", "dsh", "ps", "s5", "s95", "dmdd", "pm", "dann"]
P, D = {}, {}
for k in KS:
    P[k] = pd.DataFrame({m: dict(zip(PC, num(f"P{k}|{m}"))) for m in PM if f"P{k}|{m}" in TOK}).T
    D[k] = {a.split("|", 1)[1]: dict(zip(DC, num(a))) for a in TOK if a.startswith(f"D{k}|")}
    print(f"\nk={k}  per-basket averages\n", P[k].round(4).to_string())
    print(pd.DataFrame(D[k]).T.round(3).to_string())
out["portfolio"] = {k: P[k].round(4).to_dict(orient="index") for k in KS}
out["paired"] = D
yd = {}
for k in KS:
    for m in ("ROLL63", "LGBM"):
        if f"Y{k}|{m}" in TOK:
            yd[f"{k}|{m}"] = {2000 + int(a): (float(b), float(c)) for a, b, c in (x.split("/") for x in TOK[f"Y{k}|{m}"] if x.count("/") == 2)}
out["paired_by_year"] = yd

# ---------------- monthly average-basket series: factor regressions and block bootstrap
cp = os.path.join(RES, f"{TAG}_charts.json")
if os.path.exists(cp):
    ch = json.load(open(cp))
    def ser(c, s):
        v = ch.get(c, {}).get(s)
        if not v:
            return None
        x = pd.Series({pd.Timestamp(d): val for d, val in v}).sort_index()
        x = x.groupby(x.index.to_period("M")).last()
        x.index = x.index - 1        # a point plotted at the start of month t+1 is month t's outcome
        return x
    M = {k: pd.DataFrame({m: ser(f"R{k}", m) for m in PM}).dropna() for k in KS}
    spy = ser("MKT", "SPY")

    def load_ff():
        D_ = os.path.join(HERE, "data")
        lines = open(os.path.join(D_, "F-F_Research_Data_5_Factors_2x3_daily.csv")).read().splitlines()
        rows = [[x.strip() for x in l.split(",")] for l in lines if l[:8].isdigit() and len(l.split(",")) == 7]
        f5 = pd.DataFrame([[float(x) / 100 for x in r[1:]] for r in rows], index=pd.to_datetime([r[0] for r in rows]),
                          columns=["MKT", "SMB", "HML", "RMW", "CMA", "RF"])
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

    def boot_sharpe_diff(a, b, n=5000, block=6, seed=0):
        rng = np.random.default_rng(seed); T = len(a); A, B = a.values, b.values
        sd = lambda x: x.mean() / x.std() * np.sqrt(12)
        obs = sd(A) - sd(B); d = np.empty(n)
        for i in range(n):
            starts = rng.integers(0, T, int(np.ceil(T / block)))
            ix = (starts[:, None] + np.arange(block)[None, :]).ravel()[:T] % T
            d[i] = sd(A[ix]) - sd(B[ix])
        return float(obs), float(np.mean(d <= 0)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]

    ff = load_ff()
    agg = {}
    for k in KS:
        m = M[k]
        agg[k] = {"months": len(m), "first": str(m.index[0]), "last": str(m.index[-1]),
                  "stats": {c: stats(m[c]) for c in PM}}
        agg[k]["boot"] = {f"{a}-{b}": boot_sharpe_diff(m[a], m[b]) for a, b in
                          [("ROLL63", "EW"), ("LGBM", "EW"), ("LGBM", "ROLL63"), ("ORACLE", "ROLL63")]}
        j = m.join(ff, how="inner")
        regs = {}
        for c in ("EW", "ROLL63", "LGBM"):
            b, t, r2 = ols_nw((j[c] - j["RF"]).values, j[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]].values)
            regs[c] = {"alpha_ann": float(b[0] * 12), "t": float(t[0]), "beta": float(b[1]), "r2": float(r2),
                       "loadings": dict(zip(["MKT", "SMB", "HML", "RMW", "CMA", "MOM"], map(float, b[1:])))}
        b, t, r2 = ols_nw((j["ROLL63"] - j["EW"]).values, j[["MKT", "SMB", "HML", "RMW", "CMA", "MOM"]].values)
        regs["IV-EW"] = {"alpha_ann": float(b[0] * 12), "t": float(t[0]), "r2": float(r2),
                         "loadings": dict(zip(["MKT", "SMB", "HML", "RMW", "CMA", "MOM"], map(float, b[1:]))),
                         "tstats": dict(zip(["MKT", "SMB", "HML", "RMW", "CMA", "MOM"], map(float, t[1:])))}
        agg[k]["ff"] = regs
    out["avg_basket"] = agg
    if spy is not None:
        out["spy"] = stats(spy.dropna())
    print("\nAverage-basket monthly series\n", json.dumps({k: {"boot": agg[k]["boot"], "IV-EW": agg[k]["ff"]["IV-EW"]} for k in KS}, indent=1)[:3000])

    # ---------------- figures
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    COL = {"EW": "#888888", "ROLL63": "#1f77b4", "EWMA": "#17becf", "HAR": "#2ca02c", "LGBM": "#d62728", "ORACLE": "#9467bd"}
    # 1 forecast accuracy
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    F["r2"].plot.bar(ax=ax[0], color=["#bbb", "#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"])
    ax[0].set_title("Out-of-sample R² (log volatility)"); ax[0].set_ylim(F["r2"].min() - 0.05, F["r2"].max() + 0.03)
    F["sp"].plot.bar(ax=ax[1], color=["#bbb", "#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#d62728"])
    ax[1].set_title("Cross-sectional rank correlation"); ax[1].set_ylim(F["sp"].min() - 0.05, F["sp"].max() + 0.03)
    for a in ax: a.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_forecast_accuracy.png"), dpi=160); plt.close(fig)
    # 2 risk reduction vs basket size (per-basket averages)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    for m in PM[1:]:
        ax[0].plot(KS, [D[k][f"{m}-EW"]["dvol"] * 100 for k in KS], "o-", color=COL[m], label=m)
        ax[1].plot(KS, [D[k][f"{m}-EW"]["dmdd"] * 100 for k in KS], "o-", color=COL[m], label=m)
        ax[2].plot(KS, [D[k][f"{m}-EW"]["dsh"] for k in KS], "o-", color=COL[m], label=m)
    ax[0].set_title("Change in volatility vs EW (pp)"); ax[1].set_title("Change in max drawdown vs EW (pp)"); ax[2].set_title("Change in Sharpe vs EW")
    for a in ax: a.set_xscale("log"); a.set_xticks(KS); a.set_xticklabels(KS); a.axhline(0, color="k", lw=0.6); a.set_xlabel("stocks per basket")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_risk_vs_basket_size.png"), dpi=160); plt.close(fig)
    # 3 per-year vol reduction (k=10)
    if "10|LGBM" in yd:
        fig, ax = plt.subplots(figsize=(10, 3.4)); yrs = sorted(yd["10|LGBM"])
        w = 0.38
        ax.bar(np.array(yrs) - w / 2, [yd["10|ROLL63"][y][0] * 100 for y in yrs], w, color=COL["ROLL63"], label="ROLL63 − EW")
        ax.bar(np.array(yrs) + w / 2, [yd["10|LGBM"][y][0] * 100 for y in yrs], w, color=COL["LGBM"], label="LGBM − EW")
        ax.axhline(0, color="k", lw=0.6); ax.set_title("Volatility change vs equal weight, by year (10-stock baskets, pp)"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_vol_change_by_year.png"), dpi=160); plt.close(fig)
    # 4 rank correlation over time
    if "FCsp" in ch:
        fig, ax = plt.subplots(figsize=(10, 3.2))
        for m in ("ROLL63", "HAR", "LGBM"):
            s = ser("FCsp", m)
            if s is not None: ax.plot(s.index.to_timestamp(), s.values, color=COL[m], lw=1.2, label=m)
        ax.set_title("Monthly cross-sectional rank correlation: forecast vs realized volatility"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_rank_corr_over_time.png"), dpi=160); plt.close(fig)
    # 5 average-basket wealth (k=10)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for m in PM:
        ax.plot(M[10].index.to_timestamp(), (1 + M[10][m]).cumprod(), color=COL[m], lw=1.3, label=m)
    if spy is not None:
        s = spy.reindex(M[10].index).fillna(0); ax.plot(M[10].index.to_timestamp(), (1 + s).cumprod(), "k--", lw=1, label="SPY")
    ax.set_yscale("log"); ax.set_title("Growth of $1, average of 500 random 10-stock baskets"); ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_wealth_k10.png"), dpi=160); plt.close(fig)
    # 6 feature importance
    if imp:
        fig, ax = plt.subplots(figsize=(6, 3.4)); names = [a for a, _ in imp][::-1]; vals = [b for _, b in imp][::-1]
        ax.barh(names, vals, color="#d62728"); ax.set_title("LightGBM feature importance (gain share)")
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "6_feature_importance.png"), dpi=160); plt.close(fig)

json.dump(out, open(os.path.join(RES, f"summary_{TAG}.json"), "w"), indent=1, default=str)
print(f"\nsaved results/summary_{TAG}.json and figures/")
