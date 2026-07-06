"""Notion 주간 리포트 빌더 (정식 File Upload API 사용)

입력: reports/weekly_analysis.json (analyze.py 산출물)
      reports/commentary.json     (AI 차트 판독: {"market_overview": str, "per_ticker": {ticker: str}})
      config.json                 ({"notion_token": "...", "parent_page_id": "..."})
출력: 상위 페이지 아래 새 주간 리포트 페이지, 표준출력으로 페이지 URL
"""
import json
import sys
from pathlib import Path

import requests

BASE = Path(__file__).parent
API = "https://api.notion.com/v1"

cfg = json.loads((BASE / "config.json").read_text())
HDR = {
    "Authorization": f"Bearer {cfg['notion_token']}",
    "Notion-Version": "2022-06-28",
}
JHDR = {**HDR, "Content-Type": "application/json"}

STATUS_LABEL = {
    "ENTRY": "🟢 신규 진입",
    "EXIT": "🔴 청산",
    "WATCH": "🟡 돌파·조건 대기",
    "HOLD": "🔵 보유 적합",
    "NEUTRAL": "⚪ 관망",
}


# ---------------------------------------------------------------- notion helpers

def upload_png(path):
    r = requests.post(f"{API}/file_uploads", headers=JHDR,
                      json={"mode": "single_part", "filename": Path(path).name})
    r.raise_for_status()
    fid = r.json()["id"]
    with open(path, "rb") as f:
        r2 = requests.post(f"{API}/file_uploads/{fid}/send", headers=HDR,
                           files={"file": (Path(path).name, f, "image/png")})
    r2.raise_for_status()
    return fid


def rt(text, bold=False):
    return [{"type": "text", "text": {"content": str(text)[:2000]},
             "annotations": {"bold": bold}}]


def h2(t):
    return {"type": "heading_2", "heading_2": {"rich_text": rt(t)}}


def h3(t):
    return {"type": "heading_3", "heading_3": {"rich_text": rt(t)}}


def para(t, bold=False):
    return {"type": "paragraph", "paragraph": {"rich_text": rt(t, bold)}}


def quote(t):
    return {"type": "quote", "quote": {"rich_text": rt(t)}}


def divider():
    return {"type": "divider", "divider": {}}


def image(fid):
    return {"type": "image",
            "image": {"type": "file_upload", "file_upload": {"id": fid}}}


def table(rows, header=True):
    width = len(rows[0])
    return {"type": "table",
            "table": {"table_width": width, "has_column_header": header,
                      "children": [
                          {"type": "table_row",
                           "table_row": {"cells": [rt(c) for c in row]}}
                          for row in rows]}}


def create_page(title, blocks):
    first, rest = blocks[:90], blocks[90:]
    r = requests.post(f"{API}/pages", headers=JHDR, json={
        "parent": {"page_id": cfg["parent_page_id"]},
        "icon": {"type": "emoji", "emoji": "📈"},
        "properties": {"title": {"title": rt(title)}},
        "children": first,
    })
    r.raise_for_status()
    page = r.json()
    while rest:
        chunk, rest = rest[:90], rest[90:]
        r = requests.patch(f"{API}/blocks/{page['id']}/children",
                           headers=JHDR, json={"children": chunk})
        r.raise_for_status()
    return page


# ---------------------------------------------------------------- report

def fmt_close(v, market):
    return f"₩{v:,.0f}" if market == "KR" else f"${v:,.2f}"


def build():
    data = json.loads((BASE / "reports" / "weekly_analysis.json").read_text())
    comm = json.loads((BASE / "reports" / "commentary.json").read_text())
    per = comm.get("per_ticker", {})
    stocks = data["stocks"]
    indices = data["indices"]
    week = indices[0]["last_week"]

    blocks = [quote(
        f"분석 기준: {week} 마감 주봉 · 대상 지수 4 + 종목 {len(stocks)} · "
        "전략: 시장필터(지수 10/40주 골든크로스) + 진입(26주 신고가 돌파 & Minervini 트렌드 템플릿) + "
        "청산(13주 저가 이탈 또는 데드크로스) · 스테이지 = Weinstein 30주선 분류 · "
        "본 리포트는 투자 참고 자료입니다.")]

    # --- 매매 신호
    blocks.append(h2("📋 매매 신호"))
    idx_by = {i["ticker"]: i for i in indices}
    us_ok = next(s["trade"]["market_filter"] for s in stocks if s["market"] == "US")
    kr_ok = next(s["trade"]["market_filter"] for s in stocks if s["market"] == "KR")
    blocks.append(para(f"시장 필터 — 미국: {'✅ 통과' if us_ok else '⛔ 차단(신규 진입 금지)'} · "
                       f"한국: {'✅ 통과' if kr_ok else '⛔ 차단(신규 진입 금지)'}"))

    actionable = [s for s in stocks if s["trade"]["status"] in ("ENTRY", "EXIT", "WATCH")]
    holds = [s for s in stocks if s["trade"]["status"] == "HOLD"]
    if actionable:
        rows = [["신호", "종목", "종가", "청산선(13주 저가)", "스탑까지 거리", "종합점수"]]
        for s in actionable:
            t = s["trade"]
            rows.append([STATUS_LABEL[t["status"]], f"{s['name']} ({s['ticker']})",
                         fmt_close(s["close"], s["market"]),
                         fmt_close(t["stop_13w"], s["market"]),
                         f"{t['risk_to_stop_pct']}%", f"{s['score_total']:+d}"])
        blocks.append(table(rows))
        blocks.append(para("포지션 사이징: 계좌 리스크 1% ÷ 스탑까지 거리 = 투입 비중. "
                           "예) 스탑 거리 15%면 계좌의 6.7%만 투입.", bold=False))
    else:
        blocks.append(para("이번 주 신규 진입/청산 신호 없음."))
    if holds:
        blocks.append(para("보유 적합(추세 지속): " +
                           ", ".join(f"{s['name']}({s['ticker']})" for s in holds)))

    # --- 시황 총평
    if comm.get("market_overview"):
        blocks.append(h2("🌎 주간 시황 총평"))
        blocks.append(para(comm["market_overview"]))

    # --- 지수
    blocks.append(h2("📉 지수 동향"))
    for i in indices:
        blocks.append(h3(f"{i['name']} — {i['verdict']} ({i['score_total']:+d}) · 주간 {i['week_return_pct']:+.2f}%"))
        if i.get("chart") and Path(i["chart"]).exists():
            blocks.append(image(upload_png(i["chart"])))
        lv = i["levels"]
        blocks.append(table([
            ["종가", "RSI", "20주선", "60주선", "52주 고가", "52주 저가"],
            [f"{i['close']:,}", lv["RSI"], f"{lv['SMA20']:,}", f"{lv['SMA60']:,}",
             f"{lv['52w_high']:,}", f"{lv['52w_low']:,}"]]))
        if per.get(i["ticker"]):
            blocks.append(para(f"💬 {per[i['ticker']]}"))

    # --- 주목 종목 카드 (차트 생성된 종목)
    featured = [s for s in stocks if s.get("chart")]
    if featured:
        blocks.append(divider())
        blocks.append(h2("🔍 주목 종목"))
        for s in featured:
            t = s["trade"]
            blocks.append(h3(f"{s['name']} ({s['ticker']}) — {STATUS_LABEL[t['status']]} · "
                             f"점수 {s['score_total']:+d} · 주간 {s['week_return_pct']:+.2f}%"))
            if Path(s["chart"]).exists():
                blocks.append(image(upload_png(s["chart"])))
            notes = " · ".join(v["note"] for v in s["signals"].values() if v["score"] != 0) or "특이 신호 없음"
            blocks.append(para(f"지표: {notes}"))
            for p in s["patterns"]:
                blocks.append(para(f"패턴: {p.get('pattern')} — {p.get('status', p.get('signal', ''))}"))
            blocks.append(para(f"신호: {t['note']} · {t.get('stage_label', '')} · "
                               f"청산선 {fmt_close(t['stop_13w'], s['market'])} "
                               f"(현재가 대비 -{t['risk_to_stop_pct']}%)"))
            if per.get(s["ticker"]):
                blocks.append(para(f"💬 {per[s['ticker']]}"))

    # --- 전 종목 점수표
    blocks.append(divider())
    blocks.append(h2("📊 전 종목 요약"))
    rows = [["종목", "종가", "주간", "점수", "판정", "신호", "10/40주", "스테이지"]]
    for s in sorted(stocks, key=lambda x: -x["score_total"]):
        t = s["trade"]
        rows.append([f"{s['name']} ({s['ticker']})", fmt_close(s["close"], s["market"]),
                     f"{s['week_return_pct']:+.2f}%", f"{s['score_total']:+d}",
                     s["verdict"], STATUS_LABEL[t["status"]], t["trend_10_40"],
                     t.get("stage_label", "-")])
    blocks.append(table(rows))

    page = create_page(f"📈 주간 차트분석 리포트 — {week}", blocks)
    print(page["url"])


if __name__ == "__main__":
    try:
        build()
    except requests.HTTPError as e:
        print(f"NOTION API ERROR: {e.response.status_code} {e.response.text[:500]}", file=sys.stderr)
        sys.exit(1)
