"""주간 주식 차트 분석 파이프라인 (샘플 버전)

일봉 수집 -> 주봉 변환 -> 지표 계산 -> 패턴 후보 검출 -> 차트 이미지 -> JSON 요약
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import FinanceDataReader as fdr
import mplfinance as mpf
from scipy.signal import argrelextrema
from scipy.stats import linregress
import ta

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
CHARTS = BASE / "charts"
REPORTS = BASE / "reports"
CHARTS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

INDICES = [
    {"ticker": "^GSPC", "name": "S&P 500", "market": "US", "source": "yf"},
    {"ticker": "^IXIC", "name": "NASDAQ", "market": "US", "source": "yf"},
    {"ticker": "KS11", "name": "KOSPI", "market": "KR", "source": "fdr"},
    {"ticker": "KQ11", "name": "KOSDAQ", "market": "KR", "source": "fdr"},
]

# 미국 시총 상위 15 (분기마다 수동 점검)
US_TOP15 = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"),
    ("GOOGL", "Alphabet"), ("AMZN", "Amazon"), ("META", "Meta"),
    ("AVGO", "Broadcom"), ("TSLA", "Tesla"), ("BRK-B", "Berkshire"),
    ("LLY", "EliLilly"), ("JPM", "JPMorgan"), ("V", "Visa"),
    ("WMT", "Walmart"), ("XOM", "ExxonMobil"), ("UNH", "UnitedHealth"),
]

# 한국 시총 상위 15 조회 실패 시 폴백
KR_TOP15_FALLBACK = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("373220", "LG에너지솔루션"),
    ("207940", "삼성바이오로직스"), ("005380", "현대차"), ("051910", "LG화학"),
    ("035420", "NAVER"), ("000270", "기아"), ("068270", "셀트리온"),
    ("105560", "KB금융"), ("055550", "신한지주"), ("012450", "한화에어로스페이스"),
    ("035720", "카카오"), ("006400", "삼성SDI"), ("086790", "하나금융지주"),
]


def kr_top15():
    """KRX 시가총액 상위 15 보통주 (우선주 제외) — 매주 자동 갱신."""
    try:
        lst = fdr.StockListing("KRX")
        lst = lst[lst["Code"].str.endswith("0")]  # 보통주만
        lst = lst.sort_values("Marcap", ascending=False).head(15)
        return [(r.Code, r.Name) for r in lst.itertuples()]
    except Exception:
        return KR_TOP15_FALLBACK


def build_stocks():
    stocks = [{"ticker": t, "name": n, "market": "US", "source": "yf"} for t, n in US_TOP15]
    stocks += [{"ticker": t, "name": n, "market": "KR", "source": "fdr"} for t, n in kr_top15()]
    return stocks

START = "2020-06-01"  # 120주선 계산 여유분 포함 (~6년)


# ---------------------------------------------------------------- data
# 다중 소스 폴백: 실행 환경(로컬/클라우드)의 네트워크 정책이 달라 소스별 도달성이 다름.
# 클라우드에서 fc.yahoo.com(yfinance 인증 호스트)이 차단된 사례가 있어 직접 API 호출 경로 포함.

YAHOO_HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _src_yfinance(item, start):
    df = yf.download(item["ticker"], start=start, interval="1d",
                     auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def _yahoo_symbols(item):
    t = item["ticker"]
    if item["source"] == "yf":
        return [t]
    if t == "KS11":
        return ["^KS11"]
    if t == "KQ11":
        return ["^KQ11"]
    return [f"{t}.KS", f"{t}.KQ"]


def _src_yahoo_direct(item, start):
    """Yahoo v8 chart API 직접 호출 — 크럼 인증 호스트(fc.yahoo.com) 불필요."""
    last_err = None
    for sym in _yahoo_symbols(item):
        for host in ("query1", "query2"):
            try:
                r = requests.get(
                    f"https://{host}.finance.yahoo.com/v8/finance/chart/{sym}",
                    params={"period1": int(pd.Timestamp(start).timestamp()),
                            "period2": int(pd.Timestamp.now().timestamp()) + 86400,
                            "interval": "1d", "events": "splits"},
                    headers=YAHOO_HDR, timeout=30)
                r.raise_for_status()
                res = r.json()["chart"]["result"][0]
                q = res["indicators"]["quote"][0]
                tz = res.get("meta", {}).get("exchangeTimezoneName", "UTC")
                idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
                       .tz_convert(tz).tz_localize(None).normalize())
                df = pd.DataFrame({"Open": q["open"], "High": q["high"], "Low": q["low"],
                                   "Close": q["close"], "Volume": q["volume"]}, index=idx)
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
                if adj:
                    ratio = pd.Series(adj, index=idx) / df["Close"]
                    for col in ("Open", "High", "Low", "Close"):
                        df[col] *= ratio
                df = df.dropna(subset=["Close"])
                if len(df) >= 10:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"yahoo direct 실패: {last_err}")


def _src_fdr(item, start):
    return fdr.DataReader(item["ticker"], start)


def fetch_daily(item, start=None):
    start = start or START
    chain = ([_src_yfinance, _src_yahoo_direct, _src_fdr]
             if item["source"] == "yf" else
             [_src_fdr, _src_yahoo_direct])
    last_err = None
    for fn in chain:
        try:
            df = fn(item, start)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
            if len(df) < 10:
                raise RuntimeError("결과 비어있음")
            df.index = pd.to_datetime(df.index)
            if fn is not chain[0]:
                print(f"  [{item['ticker']}] 대체 소스 사용: {fn.__name__}", flush=True)
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"{item['ticker']} 모든 데이터 소스 실패: {last_err}")


def to_weekly(daily):
    w = daily.resample("W-FRI").agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    ).dropna(subset=["Close"])
    # 미완성 주봉 제거: 주봉 라벨(그 주 금요일)이 아직 오지 않았으면 진행 중인 주
    # (마지막 일봉 날짜 기준으로 자르면 금요일 휴장 주를 미완성으로 오판함)
    today = pd.Timestamp.now().normalize()
    if len(w) and w.index[-1] > today:
        w = w.iloc[:-1]
    return w


# ---------------------------------------------------------------- indicators

def compute_indicators(w):
    c = w["Close"]
    w["SMA20"] = c.rolling(20).mean()
    w["SMA60"] = c.rolling(60).mean()
    w["SMA120"] = c.rolling(120).mean()
    w["RSI"] = ta.momentum.RSIIndicator(c, window=14).rsi()
    macd = ta.trend.MACD(c)
    w["MACD"] = macd.macd()
    w["MACD_sig"] = macd.macd_signal()
    w["MACD_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    w["BB_hi"] = bb.bollinger_hband()
    w["BB_lo"] = bb.bollinger_lband()
    w["BB_width"] = (w["BB_hi"] - w["BB_lo"]) / w["SMA20"]
    w["VOL20"] = w["Volume"].rolling(20).mean()
    return w


def score_indicators(w):
    """지표별 +1/0/-1 점수와 근거."""
    last, prev = w.iloc[-1], w.iloc[-2]
    sig = {}

    # 이동평균 배열
    if last.Close > last.SMA20 > last.SMA60:
        sig["ma"] = (1, "주가>20주선>60주선 정배열")
    elif last.Close < last.SMA20 < last.SMA60:
        sig["ma"] = (-1, "주가<20주선<60주선 역배열")
    else:
        sig["ma"] = (0, "이평선 혼조")

    # 골든/데드크로스 (최근 8주 내 20/60주선 교차)
    cross = 0
    recent = w.tail(8)
    diff = recent.SMA20 - recent.SMA60
    if (diff.iloc[0] < 0) and (diff.iloc[-1] > 0):
        cross = 1
        sig["cross"] = (1, "최근 8주 내 20/60주선 골든크로스")
    elif (diff.iloc[0] > 0) and (diff.iloc[-1] < 0):
        cross = -1
        sig["cross"] = (-1, "최근 8주 내 20/60주선 데드크로스")
    else:
        sig["cross"] = (0, "최근 이평선 교차 없음")

    # MACD
    if last.MACD > last.MACD_sig and last.MACD_hist > prev.MACD_hist:
        sig["macd"] = (1, "MACD 시그널 상회 + 히스토그램 증가")
    elif last.MACD < last.MACD_sig and last.MACD_hist < prev.MACD_hist:
        sig["macd"] = (-1, "MACD 시그널 하회 + 히스토그램 감소")
    else:
        sig["macd"] = (0, "MACD 중립/전환 구간")

    # RSI
    r = last.RSI
    if r >= 70:
        sig["rsi"] = (-1, f"RSI {r:.0f} 과매수 구간")
    elif r <= 30:
        sig["rsi"] = (1, f"RSI {r:.0f} 과매도 구간")
    elif r > 50:
        sig["rsi"] = (1, f"RSI {r:.0f} 강세 영역(50 상회)")
    else:
        sig["rsi"] = (0, f"RSI {r:.0f} 중립~약세 영역")

    # 볼린저밴드
    if last.Close > last.BB_hi:
        sig["bb"] = (0, "볼린저 상단 돌파(강한 추세 or 단기과열)")
    elif last.Close < last.BB_lo:
        sig["bb"] = (-1, "볼린저 하단 이탈")
    elif last.BB_width < w.BB_width.tail(52).quantile(0.2):
        sig["bb"] = (0, "밴드 스퀴즈(변동성 응축) — 방향 분출 대기")
    else:
        sig["bb"] = (0, "밴드 내 정상 범위")

    # 거래량
    if last.Volume > 1.5 * last.VOL20 and last.Close > prev.Close:
        sig["vol"] = (1, "상승 주에 거래량 20주 평균 1.5배 이상")
    elif last.Volume > 1.5 * last.VOL20 and last.Close < prev.Close:
        sig["vol"] = (-1, "하락 주에 거래량 급증")
    else:
        sig["vol"] = (0, "거래량 평이")

    # 52주 위치
    hi52 = w.High.tail(52).max()
    lo52 = w.Low.tail(52).min()
    pos = (last.Close - lo52) / (hi52 - lo52) if hi52 > lo52 else 0.5
    if pos >= 0.8:
        sig["pos52"] = (1, f"52주 밴드 상단 {pos*100:.0f}% 위치 (신고가권)")
    elif pos <= 0.2:
        sig["pos52"] = (-1, f"52주 밴드 하단 {pos*100:.0f}% 위치 (신저가권)")
    else:
        sig["pos52"] = (0, f"52주 밴드 {pos*100:.0f}% 위치")

    total = sum(v[0] for v in sig.values())
    return sig, total


# ---------------------------------------------------------------- patterns

def find_pivots(series, order=3):
    lows = argrelextrema(series.values, np.less_equal, order=order)[0]
    highs = argrelextrema(series.values, np.greater_equal, order=order)[0]
    return sorted(set(lows)), sorted(set(highs))


def detect_double_triple_bottom(w, lookback=80):
    """쌍바닥/삼중바닥: 근접 저점 2~3개(5% 이내) + 넥라인 돌파 여부."""
    seg = w.tail(lookback)
    lows_idx, _ = find_pivots(seg.Low, order=3)
    lows_idx = [i for i in lows_idx if i < len(seg) - 2]
    if len(lows_idx) < 2:
        return None
    # 최근 저점부터 역방향으로 5% 이내 저점 군집 탐색
    cands = []
    for i in range(len(lows_idx) - 1, 0, -1):
        a, b = lows_idx[i - 1], lows_idx[i]
        la, lb = seg.Low.iloc[a], seg.Low.iloc[b]
        if b - a >= 4 and abs(la - lb) / min(la, lb) <= 0.05:
            cands.append((a, b))
    if not cands:
        return None
    a, b = cands[0]
    neckline = seg.High.iloc[a:b + 1].max()
    last_close = seg.Close.iloc[-1]
    broke = bool(last_close > neckline)
    # 돌파 시점: 두 번째 저점 이후 종가가 처음 넥라인을 넘은 주
    breakout_ago = None
    if broke:
        after = seg.Close.iloc[b:]
        crossed = after[after > neckline]
        if len(crossed):
            breakout_ago = len(seg) - 1 - seg.index.get_loc(crossed.index[0])
    # 신호 유효성: 미돌파면 넥라인 15% 이내 접근 중일 때만, 돌파면 최근 8주 이내일 때만
    if broke and (breakout_ago is None or breakout_ago > 8):
        return None
    if not broke and last_close < neckline * 0.85:
        return None
    n_bottoms = 2
    # 삼중바닥 확인
    for j in lows_idx:
        if j < a and abs(seg.Low.iloc[j] - seg.Low.iloc[a]) / seg.Low.iloc[a] <= 0.05:
            n_bottoms = 3
            break
    return {
        "pattern": "삼중바닥" if n_bottoms == 3 else "쌍바닥",
        "bottoms": [str(seg.index[a].date()), str(seg.index[b].date())],
        "neckline": round(float(neckline), 2),
        "neckline_broken": broke,
        "status": f"{breakout_ago}주 전 넥라인 돌파" if broke else "형성 중(넥라인 미돌파, 접근 중)",
    }


def detect_cup_handle(w, lookback=65):
    """컵위드핸들(단순화한 오닐 기준): 12~33% 깊이 둥근 바닥 + 핸들 + 피벗 접근."""
    seg = w.tail(lookback)
    if len(seg) < 30:
        return None
    left_rim_i = int(seg.High.iloc[:15].idxmax() == seg.index[:15]) if False else seg.High.iloc[:15].argmax()
    left_rim = seg.High.iloc[:15].max()
    bottom_i = seg.Low.iloc[left_rim_i:].argmin() + left_rim_i
    bottom = seg.Low.iloc[bottom_i]
    depth = (left_rim - bottom) / left_rim
    if not (0.12 <= depth <= 0.35):
        return None
    after = seg.iloc[bottom_i:]
    if len(after) < 8:
        return None
    recover = after.High.max()
    if recover < left_rim * 0.95:
        return None
    # 핸들: 우측 고점 이후 5~15% 조정
    rim2_i = after.High.argmax()
    handle = after.iloc[rim2_i:]
    if len(handle) < 3:
        return None
    handle_pullback = (handle.High.max() - handle.Low.min()) / handle.High.max()
    if not (0.03 <= handle_pullback <= 0.15):
        return None
    pivot = handle.High.max()
    last_close = seg.Close.iloc[-1]
    return {
        "pattern": "컵위드핸들",
        "depth_pct": round(depth * 100, 1),
        "pivot": round(float(pivot), 2),
        "status": "피벗 돌파" if last_close > pivot else f"피벗 대비 {last_close/pivot*100-100:.1f}%",
    }


def detect_wedge_rsi(w, lookback=26):
    """쐐기형: 고점열·저점열 회귀선 수렴 + RSI 다이버전스."""
    seg = w.tail(lookback).copy()
    x = np.arange(len(seg))
    hi = linregress(x, seg.High.values)
    lo = linregress(x, seg.Low.values)
    price_scale = seg.Close.mean()
    hi_slope, lo_slope = hi.slope / price_scale, lo.slope / price_scale
    converging = abs(hi_slope - lo_slope) > 1e-4 and (hi.slope < lo.slope)
    result = None
    if converging and hi.slope < 0 and lo.slope < 0:
        # 하락쐐기 — RSI 상승 다이버전스 확인
        lows_idx, _ = find_pivots(seg.Low, order=2)
        if len(lows_idx) >= 2:
            a, b = lows_idx[-2], lows_idx[-1]
            if seg.Low.iloc[b] < seg.Low.iloc[a] and seg.RSI.iloc[b] > seg.RSI.iloc[a]:
                result = {"pattern": "하락쐐기 + RSI 상승 다이버전스", "signal": "반등 가능 신호"}
            else:
                result = {"pattern": "하락쐐기", "signal": "다이버전스 미확인"}
    elif converging and hi.slope > 0 and lo.slope > 0:
        result = {"pattern": "상승쐐기", "signal": "추세 피로 주의(약세 반전형)"}
    return result


def detect_candle_seq(w):
    """양양음양양양: 최근 6주 캔들 (양,양,음,양,양,양) — 상승추세 필터 결합."""
    seg = w.tail(6)
    if len(seg) < 6:
        return None
    body = (seg.Close > seg.Open).tolist()
    if body == [True, True, False, True, True, True]:
        uptrend = seg.Close.iloc[-1] > w.SMA20.iloc[-1]
        return {
            "pattern": "양양음양양양 (6주 캔들 배열)",
            "signal": "상승추세 지속형" if uptrend else "추세 필터 미충족(참고용)",
        }
    return None


def detect_patterns(w):
    out = []
    for fn in (detect_double_triple_bottom, detect_cup_handle, detect_wedge_rsi, detect_candle_seq):
        try:
            r = fn(w)
            if r:
                out.append(r)
        except Exception as e:
            out.append({"pattern": f"{fn.__name__} 오류", "error": str(e)})
    return out


# ---------------------------------------------------------------- chart

def make_chart(item, w, kind="stock"):
    seg = w.tail(104).copy()  # 최근 2년 주봉
    aps = [
        mpf.make_addplot(seg["SMA20"], color="#e8862d", width=1.2),
        mpf.make_addplot(seg["SMA60"], color="#2d7fe8", width=1.2),
        mpf.make_addplot(seg["SMA120"], color="#8e44ad", width=1.2),
        mpf.make_addplot(seg["BB_hi"], color="#aaaaaa", width=0.7, linestyle="--"),
        mpf.make_addplot(seg["BB_lo"], color="#aaaaaa", width=0.7, linestyle="--"),
        mpf.make_addplot(seg["RSI"], panel=2, color="#444444", width=1.0, ylabel="RSI", secondary_y=False),
        mpf.make_addplot(pd.Series(70.0, index=seg.index), panel=2, color="#cc4444", width=0.6, linestyle=":", secondary_y=False),
        mpf.make_addplot(pd.Series(30.0, index=seg.index), panel=2, color="#44aa44", width=0.6, linestyle=":", secondary_y=False),
        mpf.make_addplot(seg["MACD"], panel=3, color="#2d7fe8", width=1.0, ylabel="MACD", secondary_y=False),
        mpf.make_addplot(seg["MACD_sig"], panel=3, color="#e8862d", width=1.0, secondary_y=False),
        mpf.make_addplot(seg["MACD_hist"], panel=3, type="bar", color="#999999", alpha=0.5, secondary_y=False),
    ]
    style = mpf.make_mpf_style(base_mpf_style="yahoo", rc={"font.size": 9})
    fname = CHARTS / f"{item['ticker'].replace('^','')}_weekly.png"
    # 실행 환경에 한글 폰트가 없을 수 있어 차트 제목은 ASCII만 사용
    label = item["name"] if item["name"].isascii() else item["ticker"]
    mpf.plot(
        seg, type="candle", volume=True, addplot=aps, style=style,
        title=f"{label} ({item['ticker']}) Weekly  SMA20/60/120wk",
        panel_ratios=(6, 1.5, 1.5, 1.5), figsize=(14, 10),
        savefig=dict(fname=str(fname), dpi=110, bbox_inches="tight"),
    )
    return str(fname)


# ---------------------------------------------------------------- trade signals
# 백테스트 검증 규칙(2010~2026, 12자산): 시장 필터 = 지수 10/40주 골든크로스,
# 진입 = 26주 신고가 돌파, 청산 = 13주 저가 이탈 또는 10/40주 데드크로스

def market_filter(idx_w):
    s10 = idx_w.Close.rolling(10).mean()
    s40 = idx_w.Close.rolling(40).mean()
    return bool(s10.iloc[-1] > s40.iloc[-1])


def weinstein_stage(w):
    """Weinstein 스테이지 분류 (30주선 기준). 백테스트: 타이밍 전략 중 샤프 1위(0.67)."""
    sma30 = w.Close.rolling(30).mean()
    rising = bool(sma30.iloc[-1] > sma30.iloc[-5])
    above = bool(w.Close.iloc[-1] > sma30.iloc[-1])
    if above and rising:
        return 2, "스테이지2 상승"
    if above and not rising:
        return 3, "스테이지3 천장권"
    if not above and not rising:
        return 4, "스테이지4 하락"
    return 1, "스테이지1 바닥권"


def minervini_ok(w):
    """Minervini 트렌드 템플릿(간이): 30주선 위+상승, 52주 저가 +25% 이상, 고가 -25% 이내.
    백테스트: 돈치안 진입에 이 필터를 더하면 샤프 유지, 평균 MDD -39%→-33%."""
    sma30 = w.Close.rolling(30).mean()
    if np.isnan(sma30.iloc[-1]):
        return False
    rising = bool(sma30.iloc[-1] > sma30.iloc[-5])
    c = w.Close.iloc[-1]
    lo52 = w.Low.tail(52).min()
    hi52 = w.High.tail(52).max()
    return bool(c > sma30.iloc[-1] and rising and c >= 1.25 * lo52 and c >= 0.75 * hi52)


def trade_signal(w, market_ok):
    c = w.Close
    hi26 = c.rolling(26).max().shift(1)
    lo13 = c.rolling(13).min().shift(1)
    s10, s40 = c.rolling(10).mean(), c.rolling(40).mean()

    above_break = bool(c.iloc[-1] > hi26.iloc[-1])
    fresh_break = above_break and not bool(c.iloc[-2] > hi26.iloc[-2])
    below_stop = bool(c.iloc[-1] < lo13.iloc[-1])
    fresh_stop = below_stop and not bool(c.iloc[-2] < lo13.iloc[-2])
    dead_now = bool(s10.iloc[-1] < s40.iloc[-1])
    fresh_dead = dead_now and bool(s10.iloc[-2] >= s40.iloc[-2])

    stop = float(lo13.iloc[-1])
    risk_pct = (c.iloc[-1] - stop) / c.iloc[-1] * 100
    template = minervini_ok(w)
    stage_n, stage_label = weinstein_stage(w)

    if fresh_stop or fresh_dead:
        status, note = "EXIT", "청산 신호: " + ("13주 저가 이탈" if fresh_stop else "10/40주 데드크로스")
    elif fresh_break and market_ok and template:
        status, note = "ENTRY", "신규 진입 신호: 26주 신고가 돌파 + 시장 필터 + 트렌드 템플릿 통과"
    elif fresh_break:
        blockers = []
        if not market_ok:
            blockers.append("시장 필터 미통과")
        if not template:
            blockers.append("트렌드 템플릿 미충족(30주선/52주 위치)")
        status, note = "WATCH", "26주 신고가 돌파했으나 " + "·".join(blockers) + " — 진입 보류"
    elif above_break and not dead_now and not below_stop:
        status, note = "HOLD", "추세 지속 — 보유 적합 구간"
    else:
        status, note = "NEUTRAL", "신호 없음 — 관망"

    return {
        "status": status, "note": note,
        "stop_13w": round(stop, 2),
        "risk_to_stop_pct": round(float(risk_pct), 2),
        "trend_10_40": "골든" if not dead_now else "데드",
        "market_filter": market_ok,
        "template": template,
        "stage": stage_n,
        "stage_label": stage_label,
    }


# ---------------------------------------------------------------- main

def analyze(item, kind, market_ok=True, with_chart=True):
    daily = fetch_daily(item)
    w = compute_indicators(to_weekly(daily))
    if len(w) < 60:
        raise ValueError(f"insufficient history: {len(w)}w")
    sig, total = score_indicators(w)
    patterns = detect_patterns(w)
    trade = trade_signal(w, market_ok) if kind == "stock" else None
    chart = make_chart(item, w, kind) if with_chart else None
    last = w.iloc[-1]
    wk_ret = (w.Close.iloc[-1] / w.Close.iloc[-2] - 1) * 100
    verdict = "강세" if total >= 3 else ("약세" if total <= -3 else "중립")
    return {
        "ticker": item["ticker"], "name": item["name"], "market": item["market"],
        "kind": kind,
        "last_week": str(w.index[-1].date()),
        "close": round(float(last.Close), 2),
        "week_return_pct": round(float(wk_ret), 2),
        "score_total": total, "verdict": verdict,
        "signals": {k: {"score": v[0], "note": v[1]} for k, v in sig.items()},
        "patterns": patterns,
        "trade": trade,
        "chart": chart,
        "_weekly": w,
        "levels": {
            "SMA20": round(float(last.SMA20), 2),
            "SMA60": round(float(last.SMA60), 2),
            "SMA120": round(float(last.SMA120), 2) if not np.isnan(last.SMA120) else None,
            "RSI": round(float(last.RSI), 1),
            "52w_high": round(float(w.High.tail(52).max()), 2),
            "52w_low": round(float(w.Low.tail(52).min()), 2),
        },
    }


def main():
    results = {"indices": [], "stocks": []}

    # 1) 지수 분석 + 시장 필터 산출
    mkt_ok = {}
    for item in INDICES:
        print(f"[index] {item['name']} ...", flush=True)
        r = analyze(item, "index", with_chart=True)
        results["indices"].append(r)
        if item["ticker"] in ("^GSPC", "KS11"):
            mkt_ok[item["market"]] = market_filter(r["_weekly"])
    print(f"시장 필터: US={'통과' if mkt_ok.get('US') else '차단'} / KR={'통과' if mkt_ok.get('KR') else '차단'}")

    # 2) 종목 분석 (차트는 신호/주목 종목만 나중에)
    for item in build_stocks():
        print(f"[stock] {item['name']} ...", flush=True)
        try:
            results["stocks"].append(
                analyze(item, "stock", market_ok=mkt_ok.get(item["market"], True), with_chart=False))
        except Exception as e:
            print(f"  skip {item['ticker']}: {e}")

    # 3) 차트 대상 선정: ENTRY/EXIT/WATCH 전부 + 점수 상위 3 + 하위 1
    stocks = results["stocks"]
    chart_set = {s["ticker"] for s in stocks if s["trade"]["status"] in ("ENTRY", "EXIT", "WATCH")}
    by_score = sorted(stocks, key=lambda s: s["score_total"], reverse=True)
    chart_set |= {s["ticker"] for s in by_score[:3]} | {by_score[-1]["ticker"]}
    for s in stocks:
        if s["ticker"] in chart_set:
            item = {"ticker": s["ticker"], "name": s["name"]}
            s["chart"] = make_chart(item, s["_weekly"], "stock")

    # 4) 저장 (내부용 주봉 데이터 제거)
    for r in results["indices"] + stocks:
        r.pop("_weekly", None)
    out = REPORTS / "weekly_analysis.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    n_e = sum(1 for s in stocks if s["trade"]["status"] == "ENTRY")
    n_x = sum(1 for s in stocks if s["trade"]["status"] == "EXIT")
    n_h = sum(1 for s in stocks if s["trade"]["status"] == "HOLD")
    print(f"\n종목 {len(stocks)}개 | 진입 {n_e} / 청산 {n_x} / 보유적합 {n_h} | 차트 {len(chart_set)+len(INDICES)}장")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
