"""주봉 기반 매매전략 비교 백테스트

모든 전략: 롱온리, 금요일 종가 신호 -> 다음 주부터 포지션 반영 (pos.shift(1)),
거래비용은 포지션 변경 시 편도 부과. 지표는 analyze.py와 동일 계산.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import fetch_daily, to_weekly, compute_indicators

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
START = "2010-01-01"
WEEKS_PER_YEAR = 52

ASSETS = [
    {"ticker": "^GSPC", "name": "S&P500", "market": "US", "source": "yf"},
    {"ticker": "^IXIC", "name": "NASDAQ", "market": "US", "source": "yf"},
    {"ticker": "KS11", "name": "KOSPI", "market": "KR", "source": "fdr"},
    {"ticker": "KQ11", "name": "KOSDAQ", "market": "KR", "source": "fdr"},
    {"ticker": "AAPL", "name": "Apple", "market": "US", "source": "yf"},
    {"ticker": "MSFT", "name": "Microsoft", "market": "US", "source": "yf"},
    {"ticker": "NVDA", "name": "NVIDIA", "market": "US", "source": "yf"},
    {"ticker": "TSLA", "name": "Tesla", "market": "US", "source": "yf"},
    {"ticker": "005930", "name": "SamsungElec", "market": "KR", "source": "fdr"},
    {"ticker": "000660", "name": "SKhynix", "market": "KR", "source": "fdr"},
    {"ticker": "005380", "name": "HyundaiMotor", "market": "KR", "source": "fdr"},
    {"ticker": "035420", "name": "NAVER", "market": "KR", "source": "fdr"},
]

COST = {"US": 0.0010, "KR": 0.0025}  # 편도 (수수료+세금+슬리피지)


# ---------------------------------------------------------------- score series

def score_series(w):
    """analyze.py의 종합점수를 시계열로 (각 주 금요일 종가 기준, 인과적)."""
    c, s20, s60 = w.Close, w.SMA20, w.SMA60
    s = pd.DataFrame(index=w.index)

    s["ma"] = np.where((c > s20) & (s20 > s60), 1,
              np.where((c < s20) & (s20 < s60), -1, 0))

    diff = s20 - s60
    gc = (diff > 0) & (diff.shift(8) < 0)
    dc = (diff < 0) & (diff.shift(8) > 0)
    s["cross"] = np.where(gc, 1, np.where(dc, -1, 0))

    s["macd"] = np.where((w.MACD > w.MACD_sig) & (w.MACD_hist > w.MACD_hist.shift(1)), 1,
                np.where((w.MACD < w.MACD_sig) & (w.MACD_hist < w.MACD_hist.shift(1)), -1, 0))

    s["rsi"] = np.where(w.RSI >= 70, -1,
               np.where(w.RSI <= 30, 1,
               np.where(w.RSI > 50, 1, 0)))

    s["bb"] = np.where(c < w.BB_lo, -1, 0)

    up_week = c > c.shift(1)
    burst = w.Volume > 1.5 * w.VOL20
    s["vol"] = np.where(burst & up_week, 1, np.where(burst & ~up_week, -1, 0))

    hi52 = w.High.rolling(52).max()
    lo52 = w.Low.rolling(52).min()
    pos = (c - lo52) / (hi52 - lo52)
    s["pos52"] = np.where(pos >= 0.8, 1, np.where(pos <= 0.2, -1, 0))

    return s.sum(axis=1)


# ---------------------------------------------------------------- strategies
# 각 전략은 0/1 포지션 시리즈를 반환 (해당 주 종가 시점에 결정)

def strat_buyhold(w):
    return pd.Series(1.0, index=w.index)


def strat_sma_cross(fast, slow):
    def f(w):
        sf = w.Close.rolling(fast).mean()
        sl = w.Close.rolling(slow).mean()
        return (sf > sl).astype(float)
    return f


def strat_macd(w):
    return (w.MACD > w.MACD_sig).astype(float)


def strat_tsmom_52(w):
    return (w.Close > w.Close.shift(52)).astype(float)


def strat_donchian(entry_n=26, exit_n=13):
    def f(w):
        hi = w.Close.rolling(entry_n).max().shift(1)
        lo = w.Close.rolling(exit_n).min().shift(1)
        pos, cur = [], 0.0
        for c, h, l in zip(w.Close, hi, lo):
            if cur == 0 and not np.isnan(h) and c > h:
                cur = 1.0
            elif cur == 1 and not np.isnan(l) and c < l:
                cur = 0.0
            pos.append(cur)
        return pd.Series(pos, index=w.index)
    return f


def rsi_n(close, n):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)


def strat_rsi_mr(w):
    """RSI(2) 역추세 (코너스 스타일 주봉 적용): 40주선 위 + RSI2<20 매수, RSI2>70 또는 추세이탈 청산."""
    sma40 = w.Close.rolling(40).mean()
    r2 = rsi_n(w.Close, 2)
    pos, cur = [], 0.0
    for c, r, m in zip(w.Close, r2, sma40):
        if np.isnan(m) or np.isnan(r):
            pos.append(0.0); continue
        if cur == 0 and c > m and r < 20:
            cur = 1.0
        elif cur == 1 and (r > 70 or c < m):
            cur = 0.0
        pos.append(cur)
    return pd.Series(pos, index=w.index)


def strat_score(entry=3, exit_=0):
    def f(w):
        sc = score_series(w)
        pos, cur = [], 0.0
        for v in sc:
            if cur == 0 and v >= entry:
                cur = 1.0
            elif cur == 1 and v <= exit_:
                cur = 0.0
            pos.append(cur)
        return pd.Series(pos, index=w.index)
    return f


def strat_score_trail(entry=3, exit_=0, atr_mult=3.0):
    """종합점수 진입 + 3ATR 트레일링스탑 (점수 청산과 스탑 중 먼저 걸리는 쪽)."""
    def f(w):
        sc = score_series(w)
        tr = pd.concat([w.High - w.Low,
                        (w.High - w.Close.shift(1)).abs(),
                        (w.Low - w.Close.shift(1)).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        pos, cur, stop = [], 0.0, np.nan
        for c, v, a in zip(w.Close, sc, atr):
            if cur == 0:
                if v >= entry and not np.isnan(a):
                    cur, stop = 1.0, c - atr_mult * a
            else:
                stop = max(stop, c - atr_mult * a) if not np.isnan(a) else stop
                if c < stop or v <= exit_:
                    cur, stop = 0.0, np.nan
            pos.append(cur)
        return pd.Series(pos, index=w.index)
    return f


def atr_series(w, n=10):
    tr = pd.concat([w.High - w.Low,
                    (w.High - w.Close.shift(1)).abs(),
                    (w.Low - w.Close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def strat_supertrend(n=10, mult=3.0):
    """SuperTrend: ATR 밴드 추적 추세 지표 (GitHub 리테일 저장소 최다 구현)."""
    def f(w):
        atr = atr_series(w, n)
        mid = (w.High + w.Low) / 2
        up_base, dn_base = mid + mult * atr, mid - mult * atr
        pos, trend, up, dn = [], 1, np.nan, np.nan
        for i in range(len(w)):
            ub, db, c = up_base.iloc[i], dn_base.iloc[i], w.Close.iloc[i]
            pc = w.Close.iloc[i - 1] if i else c
            up = ub if (np.isnan(up) or ub < up or pc > up) else up
            dn = db if (np.isnan(dn) or db > dn or pc < dn) else dn
            if np.isnan(atr.iloc[i]):
                pos.append(0.0); continue
            if trend == 1 and c < dn:
                trend = -1
            elif trend == -1 and c > up:
                trend = 1
            pos.append(1.0 if trend == 1 else 0.0)
        return pd.Series(pos, index=w.index)
    return f


def strat_ichimoku(w):
    """일목균형표 주봉(9/26/52): 구름 위 + 전환선>기준선 매수, 이탈 시 청산."""
    tenkan = (w.High.rolling(9).max() + w.Low.rolling(9).min()) / 2
    kijun = (w.High.rolling(26).max() + w.Low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((w.High.rolling(52).max() + w.Low.rolling(52).min()) / 2).shift(26)
    cloud_top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    return ((w.Close > cloud_top) & (tenkan > kijun)).astype(float)


def strat_weinstein(w):
    """Weinstein 스테이지: 30주선 위 + 30주선 상승 = 스테이지2 매수, 30주선 이탈 청산."""
    sma30 = w.Close.rolling(30).mean()
    rising = sma30 > sma30.shift(4)
    pos, cur = [], 0.0
    for c, m, r in zip(w.Close, sma30, rising):
        if np.isnan(m):
            pos.append(0.0); continue
        if cur == 0 and c > m and r:
            cur = 1.0
        elif cur == 1 and c < m:
            cur = 0.0
        pos.append(cur)
    return pd.Series(pos, index=w.index)


def strat_psar(af_step=0.02, af_max=0.2):
    """Parabolic SAR: 가속 추적 지표. 종가가 SAR 위면 보유."""
    def f(w):
        hi, lo = w.High.values, w.Low.values
        n = len(w)
        pos = np.zeros(n)
        if n < 3:
            return pd.Series(pos, index=w.index)
        bull, sar, ep, af = True, lo[0], hi[0], af_step
        for i in range(1, n):
            sar = sar + af * (ep - sar)
            if bull:
                sar = min(sar, lo[i - 1], lo[i - 2] if i >= 2 else lo[i - 1])
                if lo[i] < sar:
                    bull, sar, ep, af = False, ep, lo[i], af_step
                elif hi[i] > ep:
                    ep, af = hi[i], min(af + af_step, af_max)
            else:
                sar = max(sar, hi[i - 1], hi[i - 2] if i >= 2 else hi[i - 1])
                if hi[i] > sar:
                    bull, sar, ep, af = True, ep, hi[i], af_step
                elif lo[i] < ep:
                    ep, af = lo[i], min(af + af_step, af_max)
            pos[i] = 1.0 if bull else 0.0
        return pd.Series(pos, index=w.index)
    return f


def strat_minervini(w):
    """Minervini 트렌드 템플릿 필터 + 26주 돌파 진입 / 13주 저가 청산."""
    sma30 = w.Close.rolling(30).mean()
    rising = sma30 > sma30.shift(4)
    lo52 = w.Low.rolling(52).min()
    hi52 = w.High.rolling(52).max()
    template = (w.Close > sma30) & rising & (w.Close >= 1.25 * lo52) & (w.Close >= 0.75 * hi52)
    hi26 = w.Close.rolling(26).max().shift(1)
    lo13 = w.Close.rolling(13).min().shift(1)
    pos, cur = [], 0.0
    for c, h, l, t in zip(w.Close, hi26, lo13, template):
        if cur == 0 and not np.isnan(h) and c > h and t:
            cur = 1.0
        elif cur == 1 and not np.isnan(l) and c < l:
            cur = 0.0
        pos.append(cur)
    return pd.Series(pos, index=w.index)


def strat_combo(entry_n=26, exit_n=13, score_min=2):
    """돈치안 돌파 진입 + 종합점수 필터: 26주 신고가 돌파 '이면서' score>=2일 때만 진입, 13주 저가 이탈 청산."""
    def f(w):
        sc = score_series(w)
        hi = w.Close.rolling(entry_n).max().shift(1)
        lo = w.Close.rolling(exit_n).min().shift(1)
        pos, cur = [], 0.0
        for c, h, l, v in zip(w.Close, hi, lo, sc):
            if cur == 0 and not np.isnan(h) and c > h and v >= score_min:
                cur = 1.0
            elif cur == 1 and not np.isnan(l) and c < l:
                cur = 0.0
            pos.append(cur)
        return pd.Series(pos, index=w.index)
    return f


STRATEGIES = {
    "BuyHold": strat_buyhold,
    "GoldenCross 10/40w(≈50/200d)": strat_sma_cross(10, 40),
    "SMA 20/60w cross": strat_sma_cross(20, 60),
    "MACD cross": strat_macd,
    "Donchian 26w in/13w out": strat_donchian(26, 13),
    "TS-Momentum 52w": strat_tsmom_52,
    "RSI MeanRev(40w filter)": strat_rsi_mr,
    "Score>=3 / <=0": strat_score(3, 0),
    "Score>=4 / <=-2": strat_score(4, -2),
    "Score>=2 / <=-2": strat_score(2, -2),
    "Score>=3 + 3ATR trail": strat_score_trail(3, 0, 3.0),
    "Donchian 52w in/26w out": strat_donchian(52, 26),
    "Combo: Donchian26+Score>=2": strat_combo(26, 13, 2),
    "SuperTrend 10w/3.0": strat_supertrend(10, 3.0),
    "Ichimoku 9/26/52w": strat_ichimoku,
    "Weinstein Stage2 30w": strat_weinstein,
    "Parabolic SAR": strat_psar(),
    "Minervini+Donchian26": strat_minervini,
}


# ---------------------------------------------------------------- engine

def backtest(w, pos, cost_side):
    rets = w.Close.pct_change().fillna(0.0)
    held = pos.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    strat = held * rets - turnover.shift(1).fillna(0.0) * cost_side
    eq = (1 + strat).cumprod()

    n_years = len(w) / WEEKS_PER_YEAR
    cagr = eq.iloc[-1] ** (1 / n_years) - 1 if n_years > 0 else np.nan
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    vol = strat.std() * np.sqrt(WEEKS_PER_YEAR)
    sharpe = (strat.mean() * WEEKS_PER_YEAR) / vol if vol > 0 else np.nan

    # 트레이드 단위 통계
    entries = w.index[(pos == 1) & (pos.shift(1) != 1)]
    exits = w.index[(pos == 0) & (pos.shift(1) == 1)]
    trades = []
    for i, ein in enumerate(entries):
        eout = next((x for x in exits if x > ein), None)
        px_in = w.Close.loc[ein]
        px_out = w.Close.loc[eout] if eout is not None else w.Close.iloc[-1]
        trades.append(px_out / px_in - 1 - 2 * cost_side)
        if eout is None:
            break
    trades = pd.Series(trades, dtype=float)
    return {
        "CAGR": cagr, "MDD": mdd, "Sharpe": sharpe,
        "Exposure": held.mean(),
        "Trades": int(len(trades)),
        "WinRate": float((trades > 0).mean()) if len(trades) else np.nan,
        "AvgTrade": float(trades.mean()) if len(trades) else np.nan,
    }


def main():
    data = {}
    for a in ASSETS:
        print(f"fetch {a['name']} ...", flush=True)
        try:
            d = fetch_daily({**a}, start=START)
            d = d[d.index >= START]
            w = compute_indicators(to_weekly(d))
            if len(w) < 120:
                print(f"  skip (short history: {len(w)}w)")
                continue
            data[a["name"]] = (w, a)
        except Exception as e:
            print(f"  ERROR {e}")

    rows = []
    for sname, sfn in STRATEGIES.items():
        for name, (w, a) in data.items():
            pos = sfn(w).fillna(0.0)
            m = backtest(w, pos, COST[a["market"]])
            rows.append({"strategy": sname, "asset": name, "market": a["market"],
                         "weeks": len(w), **m})
    df = pd.DataFrame(rows)
    df.to_csv(BASE / "reports" / "backtest_detail.csv", index=False)

    agg = df.groupby("strategy").agg(
        CAGR_mean=("CAGR", "mean"), CAGR_med=("CAGR", "median"),
        MDD_mean=("MDD", "mean"), MDD_worst=("MDD", "min"),
        Sharpe_mean=("Sharpe", "mean"),
        Win=("WinRate", "mean"), Trades=("Trades", "mean"),
        Expo=("Exposure", "mean"),
    ).round(3)

    # 자산별 Buy&Hold 대비 샤프 우위 개수
    bh = df[df.strategy == "BuyHold"].set_index("asset")["Sharpe"]
    beats = {}
    for sname in STRATEGIES:
        sub = df[df.strategy == sname].set_index("asset")["Sharpe"]
        beats[sname] = int((sub > bh.reindex(sub.index)).sum())
    agg["BeatsBH"] = pd.Series(beats)
    agg["N"] = df.groupby("strategy")["asset"].count()

    order = agg.sort_values("Sharpe_mean", ascending=False)
    print("\n=== 전략 비교 (12개 자산 평균, 2010~, 주봉, 비용 반영) ===")
    print(order.to_string())
    order.to_csv(BASE / "reports" / "backtest_summary.csv")

    # 시장별 분해 (지수 vs 개별주, US vs KR)
    df["group"] = np.where(df.asset.isin(["S&P500", "NASDAQ", "KOSPI", "KOSDAQ"]),
                           df.market + "-Index", df.market + "-Stock")
    piv = df.pivot_table(index="strategy", columns="group", values="Sharpe", aggfunc="mean").round(2)
    print("\n=== 그룹별 평균 Sharpe ===")
    print(piv.reindex(order.index).to_string())

    # 기간 분할 견고성 (상위 전략만): 2010-2017 vs 2018-현재
    checks = ["BuyHold", "Donchian 26w in/13w out", "GoldenCross 10/40w(≈50/200d)",
              "TS-Momentum 52w", "Combo: Donchian26+Score>=2",
              "SuperTrend 10w/3.0", "Ichimoku 9/26/52w", "Weinstein Stage2 30w",
              "Parabolic SAR", "Minervini+Donchian26"]
    print("\n=== 기간 분할 견고성 (평균 Sharpe / CAGR / MDD) ===")
    for lo_, hi_ in [("2010", "2018"), ("2018", "2027")]:
        print(f"\n[{lo_} ~ {hi_}]")
        for sname in checks:
            sfn = STRATEGIES[sname]
            ms = []
            for name, (w, a) in data.items():
                ws = w[(w.index >= lo_) & (w.index < hi_)]
                if len(ws) < 120:
                    continue
                pos = sfn(ws).fillna(0.0)
                ms.append(backtest(ws, pos, COST[a["market"]]))
            m = pd.DataFrame(ms).mean()
            print(f"  {sname:32s} Sharpe {m.Sharpe:5.2f}  CAGR {m.CAGR*100:6.1f}%  MDD {m.MDD*100:6.1f}%")
    print("\nsaved -> reports/backtest_summary.csv, backtest_detail.csv")


if __name__ == "__main__":
    main()
