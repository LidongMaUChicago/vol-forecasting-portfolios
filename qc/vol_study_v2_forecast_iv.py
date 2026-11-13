# region imports
from AlgorithmImports import *
import numpy as np
import pandas as pd
# endregion

# Volatility forecasting for risk-based portfolio construction.
# Research question: stock returns are hard to predict, but volatility is not. Do better
# volatility forecasts translate into better inverse-volatility portfolios, and by how much?
#
# Universe (point-in-time, survivorship-bias-free): each month, the 1,000 most liquid US stocks
# by prior-day dollar volume, price > $5, with fundamental data and >= 64 days of prices.
#
# Part 1 - forecasting. At the first trading day of each month t, forecast each stock's realized
# volatility over month t (daily log returns, annualized) using only data before t:
#   NAIVE21 (last 21d), ROLL63 (last 63d, our midterm rule), EWMA (RiskMetrics 0.94),
#   HAR (pooled OLS on log 5d/21d/126d vol), LINX (ridge on all features), LGBM (gradient boosting).
#   HAR/LINX refit monthly, LGBM refit every January, all on an expanding window of past months only.
#   Hyperparameters fixed before any out-of-sample result was seen.
#   Metrics (out of sample): pooled R^2 of log vol, MSE(log), QLIKE, cross-sectional Spearman;
#   Diebold-Mariano style Newey-West t-stats on monthly loss differences.
#
# Part 2 - portfolios. NB random baskets of each size k in {5,10,20,50}, drawn at OOS start from
# the universe; a name that leaves the universe is replaced by a random current member.
# Each basket is held under EW and inverse-forecast-vol weights (cap 2/k), plus an ORACLE that
# uses the realized vol of the coming month (upper bound). Monthly rebalance, 10 bps per unit
# traded, daily marking within the month. Paired comparison per basket.

START = (2010, 1, 1)
END = (2026, 8, 12)
OOS_START = "2013-01"
NB = 500
LOOK = 300
COST = 0.0010
KS = [5, 10, 20, 50]
MODELS = ["NAIVE21", "ROLL63", "EWMA", "HAR", "LINX", "LGBM"]
PM = ["EW", "ROLL63", "EWMA", "HAR", "LGBM", "ORACLE"]
FEATS = ["l_rv5", "l_rv21", "l_rv63", "l_rv126", "l_rv252", "l_ewma", "l_pk21", "l_semi63", "l_idio63",
         "beta252", "ret21", "ret252", "maxabs21", "l_dv", "l_mcap", "l_px", "l_mkt5", "l_mkt21", "mkt_ret21", "sector"]
SQ = np.sqrt(252.0)


def nw_t(x, L=6):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 10:
        return float("nan")
    e = x - x.mean(); s = e @ e / n
    for l in range(1, L + 1):
        s += 2 * (1 - l / (L + 1)) * (e[l:] @ e[:-l]) / n
    return float(x.mean() / np.sqrt(s / n)) if s > 0 else float("nan")


def cap_norm(w, cap):
    w = w / w.sum(axis=1, keepdims=True)
    for _ in range(20):
        w = np.minimum(w, cap); w = w / w.sum(axis=1, keepdims=True)
    return w


class VolStudy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(*START)
        self.set_end_date(*END)
        self.set_cash(100000)
        self.universe_settings.resolution = Resolution.DAILY
        self.spy = self.add_equity("SPY", Resolution.DAILY).symbol
        self.add_universe(self.select)
        self.month = -1
        self.rng = np.random.RandomState(7)
        # panel
        self.months = []            # 'YYYY-MM' per rebalance
        self.Xm, self.Ym, self.Sm = [], [], []   # per month: features (n,F), targets (n,), symbols
        self.Pm = []                # per month: dict model -> forecast vol (n,)
        self.prev_cut = None
        self.lgbm = None; self.lgbm_imp = None; self.lgbm_fits = 0
        # forecast metrics
        self.fm = {m: {"mse": [], "ql": [], "sp": [], "se": 0.0} for m in MODELS}
        self.pool_y = []            # OOS log targets (for pooled R^2 denominators)
        # portfolios
        self.slots = None           # k -> (NB,k) array of symbols (object)
        self.W = None               # k -> dict method -> (NB,k) target weights set at last rebalance
        self.repl = None            # k -> (NB,k) bool: slot replaced at last rebalance
        self.Wd = None              # k -> dict method -> drifted weights at end of last month
        self.st = None              # k -> dict of arrays (NB, M)
        self.year_ix = {}
        self.yst = {}               # (k) -> arrays (Y, NB, M, 3)
        self.turn = {k: np.zeros((NB, len(PM))) for k in KS}
        self.pmonths = 0
        self.errs = 0

    # ------------------------------------------------------------------ helpers
    def features(self, C, H, L, info, spy_lr):
        lr = np.log(C).diff().iloc[1:]
        def vol(n, minn):
            x = lr.iloc[-n:]
            v = x.std() * SQ
            return v.where(x.count() >= minn)
        rv5, rv21, rv63, rv126, rv252 = vol(5, 4), vol(21, 15), vol(63, 45), vol(126, 90), vol(252, 180)
        ew = np.sqrt((lr ** 2).ewm(alpha=0.06, ignore_na=True).mean().iloc[-1]) * SQ
        hl = np.log(H / L).iloc[-21:] ** 2
        pk = np.sqrt(hl.mean() / (4 * np.log(2))) * SQ
        x63 = lr.iloc[-63:]
        semi = np.sqrt((np.minimum(x63, 0) ** 2).mean()) * SQ
        Cf = C.ffill()
        ret21 = Cf.iloc[-1] / Cf.iloc[-22] - 1
        ret252 = Cf.iloc[-1] / Cf.iloc[-253] - 1 if len(Cf) > 253 else Cf.iloc[-1] * np.nan
        maxabs = lr.iloc[-21:].abs().max()
        m = spy_lr.reindex(lr.index)
        x = lr.iloc[-252:]; mm = m.iloc[-252:]
        xd = x - x.mean(); md = mm - mm.mean()
        beta = (xd.mul(md, axis=0)).mean() / (md ** 2).mean()
        res = x63 - np.outer(m.iloc[-63:].fillna(0).values, beta.fillna(1).values)
        idio = res.std() * SQ
        mk5 = m.iloc[-5:].std() * SQ; mk21 = m.iloc[-21:].std() * SQ
        mkr = float(np.exp(m.iloc[-21:].sum()) - 1)
        lg = lambda s: np.log(np.maximum(s.astype(float), 0.01))
        F = pd.DataFrame({
            "l_rv5": lg(rv5), "l_rv21": lg(rv21), "l_rv63": lg(rv63), "l_rv126": lg(rv126), "l_rv252": lg(rv252),
            "l_ewma": lg(ew), "l_pk21": lg(pk), "l_semi63": lg(semi), "l_idio63": lg(idio),
            "beta252": beta, "ret21": ret21, "ret252": ret252, "maxabs21": maxabs})
        F["l_dv"] = [np.log(max(info[s].dollar_volume, 1.0)) for s in F.index]
        F["l_mcap"] = [np.log(max(float(info[s].market_cap or 0), 1.0)) if info[s].market_cap else np.nan for s in F.index]
        F["l_px"] = np.log(np.maximum(Cf.iloc[-1].astype(float), 0.01))
        F["l_mkt5"] = np.log(max(mk5, 0.01)); F["l_mkt21"] = np.log(max(mk21, 0.01)); F["mkt_ret21"] = mkr
        F["sector"] = [float(info[s].asset_classification.morningstar_sector_code or 0) for s in F.index]
        raw = {"NAIVE21": rv21, "ROLL63": rv63, "EWMA": ew}
        ok = rv63.notna() & rv21.notna() & ew.notna()
        return F[FEATS], raw, ok

    def train_mats(self):
        X = [x for x, y in zip(self.Xm, self.Ym) if y is not None]
        Y = [y for y in self.Ym if y is not None]
        if not X:
            return None, None
        X = np.vstack(X); Y = np.concatenate(Y)
        g = np.isfinite(Y)
        return X[g], np.log(np.maximum(Y[g], 0.01))

    def fit_linear(self, X, y, cols):
        A = X[:, cols]
        lo = np.nanpercentile(A, 1, axis=0); hi = np.nanpercentile(A, 99, axis=0)
        A = np.clip(A, lo, hi); mu = np.nanmean(A, axis=0); sd = np.nanstd(A, axis=0) + 1e-9
        Z = np.nan_to_num((A - mu) / sd)
        Z1 = np.column_stack([np.ones(len(Z)), Z])
        lam = 1.0 * np.eye(Z1.shape[1]); lam[0, 0] = 0
        b = np.linalg.solve(Z1.T @ Z1 + lam, Z1.T @ y)
        s2 = float(np.var(y - Z1 @ b))
        return (cols, lo, hi, mu, sd, b, s2)

    def pred_linear(self, mdl, X):
        cols, lo, hi, mu, sd, b, s2 = mdl
        Z = np.nan_to_num((np.clip(X[:, cols], lo, hi) - mu) / sd)
        return np.column_stack([np.ones(len(Z)), Z]) @ b, s2

    # ------------------------------------------------------------------ main monthly step
    def select(self, fundamentals):
        if self.time.month == self.month:
            return Universe.UNCHANGED
        self.month = self.time.month
        try:
            return self.step(fundamentals)
        except Exception as e:
            self.errs += 1
            if self.errs <= 3:
                self.set_runtime_statistic(f"_err{self.errs}", (self.time.strftime('%Y-%m') + " " + str(e))[:240])
            return Universe.UNCHANGED

    def step(self, fundamentals):
        liquid = [f for f in fundamentals if f.has_fundamental_data and f.price > 5 and f.dollar_volume > 0]
        liquid = sorted(liquid, key=lambda f: f.dollar_volume, reverse=True)[:1000]
        if not liquid:
            return Universe.UNCHANGED
        info = {f.symbol: f for f in liquid}
        need = set(info) | {self.spy}
        if self.Sm:
            need |= set(self.Sm[-1])
        if self.slots:
            for k in KS:
                need |= set(self.slots[k].ravel())
        hist = self.history(list(need), LOOK, Resolution.DAILY)
        if hist.empty:
            return Universe.UNCHANGED
        C = hist["close"].unstack(level=0); H = hist["high"].unstack(level=0); L = hist["low"].unstack(level=0)
        keep = C.index < self.time
        C, H, L = C[keep], H[keep], L[keep]
        cut = C.index[-1]
        lrall = np.log(C).diff()
        spy_lr = lrall[self.spy] if self.spy in lrall.columns else None
        if spy_lr is None:
            return Universe.UNCHANGED

        # ---------- 1) realize last month: targets, forecast metrics, portfolio paths
        if self.prev_cut is not None and self.Sm:
            path = C.index > self.prev_cut
            seg = lrall.loc[path]
            prevS = self.Sm[-1]
            cols = [s for s in prevS if s in seg.columns]
            sd = seg[cols].std() * SQ
            cnt = seg[cols].count()
            y = np.array([sd[s] if (s in sd.index and cnt[s] >= 15) else np.nan for s in prevS], float)
            self.Ym[-1] = y
            if self.months[-1] >= OOS_START and self.Pm[-1]:
                self.evaluate(y, self.Pm[-1])
            if self.slots is not None and self.W is not None:
                self.portfolio_month(C, path, y, prevS)

        # ---------- 2) features for the current universe
        cols = [s for s in info if s in C.columns]
        F, raw, ok = self.features(C[cols], H[cols], L[cols], info, spy_lr)
        F = F[ok.reindex(F.index).fillna(False).values]
        S = list(F.index)
        X = F.values.astype(float)
        mkey = self.time.strftime("%Y-%m")
        self.months.append(mkey); self.Xm.append(X); self.Ym.append(None); self.Sm.append(S)
        self.prev_cut = cut

        # ---------- 3) forecasts (only from months with known targets)
        preds = {}
        if mkey >= OOS_START:
            Xt, yt = self.train_mats()
            if Xt is not None and len(yt) > 5000:
                for m in ("NAIVE21", "ROLL63", "EWMA"):
                    preds[m] = (raw[m].reindex(S).values.astype(float), 0.0)
                har = self.fit_linear(Xt, yt, [0, 1, 3])
                p, s2 = self.pred_linear(har, X); preds["HAR"] = (np.exp(p), s2)
                lin = self.fit_linear(Xt, yt, list(range(len(FEATS) - 1)))
                p, s2 = self.pred_linear(lin, X); preds["LINX"] = (np.exp(p), s2)
                if self.lgbm is None or self.time.month == 1:
                    import lightgbm as lgb
                    mdl = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                                            n_jobs=2, verbose=-1, random_state=1)
                    mdl.fit(Xt, yt)
                    s2l = float(np.var(yt - mdl.predict(Xt)))
                    self.lgbm = (mdl, s2l); self.lgbm_fits += 1
                    self.lgbm_imp = mdl.booster_.feature_importance(importance_type="gain")
                p = self.lgbm[0].predict(X); preds["LGBM"] = (np.exp(p), self.lgbm[1])
        self.Pm.append(preds)

        # ---------- 4) baskets and weights for the coming month
        if preds:
            self.set_baskets(S, C, preds)
        return [self.spy]

    def evaluate(self, y, preds):
        g = np.isfinite(y) & (y > 0.01)
        ly = np.log(np.maximum(y, 0.01))
        for m in MODELS:
            if m not in preds:
                return
            f, s2 = preds[m]
            gg = g & np.isfinite(f) & (f > 0)
            if gg.sum() < 50:
                return
        allg = g.copy()
        for m in MODELS:
            allg &= np.isfinite(preds[m][0]) & (preds[m][0] > 0)
        self.pool_y.append(ly[allg])
        lyc = ly[allg] - ly[allg].mean()
        self.xs_sst = getattr(self, "xs_sst", 0.0) + float(lyc @ lyc)
        ry = pd.Series(y[allg]).rank().values
        for m in MODELS:
            f, s2 = preds[m]
            lf = np.log(np.maximum(f[allg], 0.01))
            e = ly[allg] - lf
            h = np.exp(2 * lf + 2 * s2)              # variance forecast (lognormal bias correction)
            r = y[allg] ** 2 / h
            self.fm[m]["mse"].append(float(np.mean(e ** 2)))
            self.fm[m]["ql"].append(float(np.mean(r - np.log(r) - 1)))
            self.fm[m]["sp"].append(float(np.corrcoef(pd.Series(f[allg]).rank().values, ry)[0, 1]))
            self.fm[m]["se"] += float(e @ e)
            ec = e - e.mean()
            self.fm[m]["xse"] = self.fm[m].get("xse", 0.0) + float(ec @ ec)
        for m in ("ROLL63", "HAR", "LGBM"):
            self.plot("FCsp", m, self.fm[m]["sp"][-1])

    # ------------------------------------------------------------------ portfolios
    def set_baskets(self, S, C, preds):
        Sset = set(S); idx = {s: i for i, s in enumerate(S)}
        last = C.ffill().iloc[-1]
        valid = [s for s in S if last.get(s, np.nan) == last.get(s, np.nan)]
        if self.slots is None:
            self.slots = {k: np.array([self.rng.choice(len(valid), k, replace=False) for _ in range(NB)]) for k in KS}
            self.slots = {k: np.vectorize(lambda i: valid[i], otypes=[object])(v) for k, v in self.slots.items()}
            self.repl = {k: np.ones((NB, k), bool) for k in KS}
            self.st = {k: {"n": np.zeros((NB, len(PM))), "s": np.zeros((NB, len(PM))), "s2": np.zeros((NB, len(PM))),
                           "lw": np.zeros((NB, len(PM))), "pk": np.zeros((NB, len(PM))), "mdd": np.zeros((NB, len(PM)))} for k in KS}
            self.Wd = {k: {m: np.zeros((NB, k)) for m in PM} for k in KS}
        else:
            vset = set(valid)
            for k in KS:
                sl = self.slots[k]; rp = np.zeros((NB, k), bool)
                for b in range(NB):
                    cur = set(sl[b])
                    for j in range(k):
                        if sl[b, j] not in vset:
                            while True:
                                c = valid[self.rng.randint(len(valid))]
                                if c not in cur:
                                    break
                            cur.discard(sl[b, j]); cur.add(c); sl[b, j] = c; rp[b, j] = True
                self.repl[k] = rp
        self.W = {}
        self.fc_now = {m: preds[m][0] for m in ("ROLL63", "EWMA", "HAR", "LGBM")}
        self.S_now = idx
        for k in KS:
            ii = np.vectorize(lambda s: idx[s])(self.slots[k])
            Wk = {"EW": np.full((NB, k), 1.0 / k)}
            for m in ("ROLL63", "EWMA", "HAR", "LGBM"):
                v = preds[m][0][ii]
                med = np.nanmedian(v)
                v = np.where(np.isfinite(v) & (v > 0), v, med)
                Wk[m] = cap_norm(1.0 / v, 2.0 / k)
            Wk["ORACLE"] = None
            self.W[k] = Wk
        self.slot_ix = {k: np.vectorize(lambda s: idx[s])(self.slots[k]) for k in KS}

    def portfolio_month(self, C, path, y, prevS):
        Cf = C.ffill()
        base_row = Cf.loc[C.index <= self.prev_cut].iloc[-1]
        P = Cf.loc[path]
        if len(P) == 0:
            return
        ymap = {s: y[i] for i, s in enumerate(prevS)}
        days = P.index
        yrs = np.array([d.year for d in days])
        for k in KS:
            sl = self.slots[k]
            flat = list(sl.ravel())
            colset = list(dict.fromkeys(flat))
            p0 = base_row.reindex(colset).values.astype(float)
            rel = P.reindex(columns=colset).values.astype(float) / p0
            rel = np.where(np.isfinite(rel), rel, 1.0)
            cix = {s: i for i, s in enumerate(colset)}
            ii = np.vectorize(lambda s: cix[s])(sl)
            R = rel[:, ii]                                     # (D, NB, k)
            # oracle weights from the realized vol of this month
            yv = np.array([[ymap.get(s, np.nan) for s in row] for row in sl], float)
            fb = self.W[k]["ROLL63"]
            Wo = np.where(np.isfinite(yv) & (yv > 0), 1.0 / np.where(yv > 0, yv, 1), np.nan)
            Wo = np.where(np.isfinite(Wo), Wo, np.nanmedian(Wo))
            self.W[k]["ORACLE"] = cap_norm(Wo, 2.0 / k)
            st = self.st[k]
            Dret = np.zeros((len(days), NB, len(PM)))
            for mi, m in enumerate(PM):
                w = self.W[k][m]
                wd = self.Wd[k][m]
                tr = np.where(self.repl[k], w + wd, np.abs(w - wd)).sum(axis=1)
                if self.pmonths == 0:
                    tr = w.sum(axis=1)
                self.turn[k][:, mi] += tr
                V = np.einsum("dbk,bk->db", R, w) * (1 - COST * tr)[None, :]
                Vprev = np.vstack([np.ones((1, NB)), V[:-1]])
                Dret[:, :, mi] = V / Vprev - 1
                end = w * R[-1]
                self.Wd[k][m] = end / end.sum(axis=1, keepdims=True)
            for d in range(len(days)):
                r = Dret[d]
                st["n"] += 1; st["s"] += r; st["s2"] += r * r
                st["lw"] += np.log1p(r)
                st["pk"] = np.maximum(st["pk"], st["lw"])
                st["mdd"] = np.maximum(st["mdd"], 1 - np.exp(st["lw"] - st["pk"]))
                yi = self.year_ix.setdefault(int(yrs[d]), len(self.year_ix))
                if (k, yi) not in self.yst:
                    self.yst[(k, yi)] = np.zeros((3, NB, len(PM)))
                a = self.yst[(k, yi)]; a[0] += 1; a[1] += r; a[2] += r * r
            mret = np.prod(1 + Dret, axis=0) - 1                 # (NB, M) monthly return per basket
            avg = mret.mean(axis=0)
            for mi, m in enumerate(PM):
                self.plot(f"R{k}", m, float(avg[mi]))
        sp = C[self.spy].ffill()
        self.plot("MKT", "SPY", float(sp.loc[path].iloc[-1] / sp.loc[C.index <= self.prev_cut].iloc[-1] - 1))
        self.pmonths += 1

    # ------------------------------------------------------------------ report
    def on_end_of_algorithm(self):
        # QC keeps ~50 runtime statistics of <=200 chars, so results are packed as tokens
        # "name~v1,v2,..." joined by ";" into keys O00, O01, ...  (decoded by analyze.py)
        T = []
        g = lambda x: f"{x:.4g}"
        T.append(f"info~{len(self.months)},{len(self.fm['ROLL63']['mse'])},{self.pmonths},{self.lgbm_fits},{self.errs},{self.months[0] if self.months else '-'},{self.months[-1] if self.months else '-'}")
        if self.pool_y:
            ly = np.concatenate(self.pool_y); sst = float(((ly - ly.mean()) ** 2).sum())
            T.append(f"N~{len(ly)}")
            base = self.fm["ROLL63"]
            for m in MODELS:
                f = self.fm[m]
                if not f["mse"]:
                    continue
                d_mse = np.array(f["mse"]) - np.array(base["mse"]); d_ql = np.array(f["ql"]) - np.array(base["ql"])
                T.append(f"F|{m}~" + ",".join(g(x) for x in (1 - f['se'] / sst, np.mean(f['mse']), np.mean(f['ql']), np.mean(f['sp']),
                                                            nw_t(d_mse), nw_t(d_ql), np.mean(d_ql < 0), np.mean(d_mse < 0))))
                T.append(f"X|{m}~" + ",".join(g(x) for x in (1 - f.get('xse', 0) / self.xs_sst, nw_t(np.array(f['sp']) - np.array(base['sp'])))))
            mm = [m for m in self.months if m >= OOS_START][:len(base["sp"])]
            yrs = sorted(set(x[:4] for x in mm))
            for m in ("ROLL63", "HAR", "LGBM"):
                f = self.fm[m]; out = []
                for yy in yrs:
                    ix = [i for i, x in enumerate(mm) if x[:4] == yy]
                    out.append(f"{yy[2:]}/{np.mean([f['mse'][i] for i in ix]):.3f}/{np.mean([f['sp'][i] for i in ix]):.3f}")
                T.append(f"FY|{m}~" + ",".join(out[:7])); T.append(f"FY|{m}~" + ",".join(out[7:]))
        if self.lgbm_imp is not None:
            imp = self.lgbm_imp / self.lgbm_imp.sum(); o = np.argsort(-imp)
            T.append("imp~" + ",".join(f"{FEATS[i]}/{imp[i]:.3f}" for i in o[:12]))
        if self.st:
            iy = sorted(self.year_ix.items())
            for k in KS:
                st = self.st[k]; n = st["n"]
                mu = st["s"] / n; var = st["s2"] / n - mu ** 2; sdv = np.sqrt(np.maximum(var, 1e-12))
                vol = sdv * SQ; sh = mu / sdv * SQ
                ann = np.exp(st["lw"] * 252 / n) - 1; mdd = st["mdd"]
                yrs_n = n[0, 0] / 252.0
                for mi, m in enumerate(PM):
                    T.append(f"P{k}|{m}~" + ",".join(g(x) for x in (ann[:, mi].mean(), vol[:, mi].mean(), sh[:, mi].mean(), mdd[:, mi].mean(),
                                                                     self.turn[k][:, mi].mean() / yrs_n, np.median(sh[:, mi]))))
                for ref in ("EW", "ROLL63"):
                    r0 = PM.index(ref)
                    for mi, m in enumerate(PM):
                        if m == ref or (ref == "ROLL63" and m in ("EW", "EWMA", "HAR")):
                            continue
                        dv = vol[:, mi] - vol[:, r0]; ds = sh[:, mi] - sh[:, r0]; dd = mdd[:, mi] - mdd[:, r0]; da = ann[:, mi] - ann[:, r0]
                        T.append(f"D{k}|{m}-{ref}~" + ",".join(g(x) for x in (dv.mean(), np.mean(dv < 0), ds.mean(), np.mean(ds > 0),
                                                                              np.percentile(ds, 5), np.percentile(ds, 95), dd.mean(), np.mean(dd < 0), da.mean())))
                for m in (("ROLL63", "LGBM") if k in (10, 50) else ()):
                    mi = PM.index(m); out = []
                    for yy, yi in iy:
                        a = self.yst.get((k, yi))
                        if a is None:
                            continue
                        mu_ = a[1] / a[0]; sd_ = np.sqrt(np.maximum(a[2] / a[0] - mu_ ** 2, 1e-12)); s_ = mu_ / sd_ * SQ
                        out.append(f"{yy % 100}/{(sd_[:, mi] - sd_[:, 0]).mean() * SQ:.3f}/{(s_[:, mi] - s_[:, 0]).mean():.2f}")
                    T.append(f"Y{k}|{m}~" + ",".join(out[:7])); T.append(f"Y{k}|{m}~" + ",".join(out[7:]))
        keys, cur = [], ""
        for t in T:
            t = t[:190]
            if cur and len(cur) + 1 + len(t) > 195:
                keys.append(cur); cur = t
            else:
                cur = t if not cur else cur + ";" + t
        if cur:
            keys.append(cur)
        for i, v in enumerate(keys[:47]):
            self.set_runtime_statistic(f"O{i:02d}", v)
        self.set_runtime_statistic("_nkeys", f"{len(keys)}")
