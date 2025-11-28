# app.py — Growlio SaaS-style multi-page starter
import os
import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sqlite3
import feedparser
import requests
from openai import OpenAI
from google.oauth2.service_account import Credentials
import gspread
from datetime import datetime as dt, timedelta

# --------------------
# Minimal config
# --------------------
st.set_page_config(page_title="Growlio Super-App", layout="wide")
APP_TITLE = "Growlio Super-App"

# --------------------
# API keys & clients
# --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
SHEET_ID = os.getenv("SHEET_ID") or st.secrets.get("SHEET_ID", None)
GCP_SERVICE_ACCOUNT = os.getenv("GCP_SERVICE_ACCOUNT") or st.secrets.get("gcp_service_account", None)

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
else:
    openai_client = None

HAS_GCP = bool(GCP_SERVICE_ACCOUNT or "gcp_service_account" in st.secrets)

# --------------------
# Simple auth placeholder
# --------------------
def simple_login_ui():
    if "auth" not in st.session_state:
        st.session_state.auth = {"logged_in": False, "user": None}
    if st.session_state.auth["logged_in"]:
        st.sidebar.success(f"Signed in as {st.session_state.auth['user']}")
        if st.sidebar.button("Sign out"):
            st.session_state.auth = {"logged_in": False, "user": None}
            st.experimental_rerun()
    else:
        st.sidebar.info("Sign in (placeholder)")
        username = st.sidebar.text_input("Email")
        pw = st.sidebar.text_input("Password", type="password")
        if st.sidebar.button("Sign in"):
            # TODO: replace with Firebase/Auth
            if username and pw:
                st.session_state.auth = {"logged_in": True, "user": username}
                st.experimental_rerun()
            else:
                st.sidebar.error("Enter email + password (placeholder)")

# --------------------------------
# Utilities: data, news, sheet push
# --------------------------------
@st.cache_data
def load_data(tickers, start, end):
    try:
        df = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True, progress=False)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def fetch_news_rss(ticker, limit=6):
    """Use Google News RSS (robust for hosts)"""
    try:
        q = f"{ticker}+stock"
        url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        out = []
        for entry in feed.entries[:limit]:
            out.append({
                "title": entry.title,
                "url": entry.link,
                "date": entry.get("published", "")
            })
        return out
    except Exception as e:
        st.warning(f"RSS error: {e}")
        return []

def update_sheet(sheet_key, rows):
    if not HAS_GCP:
        st.warning("GCP creds missing; cannot update sheet.")
        return False
    try:
        # credential object either from st.secrets or env var JSON string
        info = st.secrets.get("gcp_service_account") if "gcp_service_account" in st.secrets else None
        if not info:
            import json
            info = json.loads(os.environ.get("GCP_SERVICE_ACCOUNT") or "{}")
        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_key)
        ws = sh.sheet1
        df = pd.DataFrame(rows)
        ws.clear()
        ws.update([df.columns.tolist()] + df.values.tolist())
        return True
    except Exception as e:
        st.error(f"Sheet update failed: {e}")
        return False

def openai_summary(headlines, ticker):
    if not openai_client:
        return "OpenAI API key missing — cannot generate summary."
    try:
        combined = " | ".join(headlines[:10])
        prompt = f"Explain in plain English why {ticker} might have moved based on these headlines: {combined}. Keep under 2 sentences and finish with 'Lesson:' one sentence."
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You explain finance simply."},{"role":"user","content":prompt}],
            temperature=0.7
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI error: {e}"

# --------------------
# Pages — modular functions
# --------------------
def growlio_page():
    st.title("📊 Growlio — Investing Learning")
    st.sidebar.header("Growlio: Stock Settings")
    tickers = st.sidebar.text_input("Tickers (comma separated)", "AAPL, MSFT, TSLA").upper()
    start = st.sidebar.date_input("Start Date", datetime.date(2023,1,1))
    end = st.sidebar.date_input("End Date", datetime.date.today())
    tickers = [t.strip() for t in tickers.split(",") if t.strip()]
    if not tickers:
        st.info("Enter tickers")
        return
    data = load_data(tickers, start, end)
    if data is None or data.empty:
        st.warning("No price data.")
        return

    # metrics
    st.subheader("Stock Metrics")
    cols = st.columns(len(tickers))
    for i,t in enumerate(tickers):
        try:
            last = data[t]["Close"].iloc[-1]
            first = data[t]["Close"].iloc[0]
            change = (last-first)/first*100
            cols[i].metric(t, f"${last:.2f}", f"{change:.2f}%")
        except Exception:
            cols[i].warning("No data")

    # comparison chart
    fig = go.Figure()
    for t in tickers:
        try:
            fig.add_trace(go.Scatter(x=data[t].index, y=data[t]["Close"], mode="lines", name=t))
        except Exception:
            pass
    fig.update_layout(title="Prices", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig, use_container_width=True)

    # news & sheet
    st.subheader("News & Insights")
    sheet_key = st.text_input("Google Sheet ID (optional)", value=SHEET_ID or "")
    if st.button("Fetch News & (optionally) update sheet"):
        rows = []
        for t in tickers:
            news = fetch_news_rss(t, limit=6)
            if not news:
                st.warning(f"No news for {t}")
            for n in news:
                rows.append({"ticker":t, "title":n["title"], "url":n["url"], "date":n["date"]})
        if rows:
            st.write("Found headlines:")
            for r in rows:
                st.markdown(f"- [{r['ticker']}] [{r['title']}]({r['url']})")
            if sheet_key:
                ok = update_sheet(sheet_key, rows)
                if ok:
                    st.success("Sheet updated.")
        else:
            st.info("No headlines found.")

    # quick AI summary
    st.subheader("AI Summary (optional)")
    if st.button("Create quick summary from fetched headlines"):
        headlines = []
        for t in tickers:
            news = fetch_news_rss(t, limit=4)
            headlines += [n["title"] for n in news]
        if headlines:
            s = openai_summary(headlines, ", ".join(tickers))
            st.info(s)
        else:
            st.warning("No headlines to summarize.")

def portfolio_page():
    st.title("💼 Portfolio Risk Dashboard")
    st.sidebar.header("Portfolio Inputs")
    mode = st.sidebar.radio("Input Method", ["Manual", "Upload Excel"])
    start = st.sidebar.date_input("Start Date", datetime.date(2022,1,1), key="p_start")
    end = st.sidebar.date_input("End Date", datetime.date.today(), key="p_end")
    rf = st.sidebar.number_input("Risk-free %", 0.0, 10.0, 0.0)/100.0
    if mode=="Manual":
        tickers_raw = st.sidebar.text_input("Tickers", "AAPL, MSFT, TSLA")
        weights_raw = st.sidebar.text_input("Weights", "0.4,0.4,0.2")
        try:
            tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            weights = np.array([float(w) for w in weights_raw.split(",") if w.strip()])
            weights = weights/weights.sum()
        except Exception as e:
            st.error(f"Input error: {e}")
            return
    else:
        uploaded = st.sidebar.file_uploader("Upload xlsx with sheet 'weights'", type=["xlsx"])
        if not uploaded:
            st.info("Upload file or switch to Manual")
            return
        dfw = pd.read_excel(uploaded, sheet_name="weights")
        tickers = [str(x).upper() for x in dfw["Ticker"].tolist()]
        weights = np.array(dfw["Weight"].astype(float).tolist())
        weights = weights/weights.sum()

    prices = load_data(tickers, start, end)
    if prices is None or prices.empty:
        st.warning("No prices")
        return
    # normalize to closings
    if isinstance(prices.columns, pd.MultiIndex):
        close = prices["Close"].copy()
    else:
        close = prices[["Close"]].copy()
        close.columns = [tickers[0]]
    close = close.dropna(how="all")
    returns = close.pct_change().dropna()
    port_norm = (close / close.iloc[0] * weights).sum(axis=1)

    st.subheader("Portfolio value (normalized)")
    st.line_chart(port_norm)

    # metrics
    ann_return = returns.mean() * 252
    ann_vol = returns.std()*np.sqrt(252)
    st.dataframe(pd.DataFrame({"Ann Return":ann_return, "Ann Vol":ann_vol}))

def tradeflow_page():
    st.title("📈 TradeFlow Analyzer (synthetic)")
    n = st.sidebar.slider("Rows", 100, 10000, 2000)
    np.random.seed(42)
    ts = [dt.now() - timedelta(minutes=i) for i in range(n)]
    prices = np.random.normal(100,2,n).round(2)
    vols = np.random.randint(1,1000,n)
    df = pd.DataFrame({"timestamp":ts,"price":prices,"volume":vols})
    if st.checkbox("Show sample"):
        st.dataframe(df.head(20))
    # liquidity
    df["minute"] = df["timestamp"].dt.floor("T")
    liquidity = df.groupby("minute")["volume"].sum().reset_index()
    st.line_chart(liquidity.set_index("minute"))

# -----------------------
# App navigation
# -----------------------
simple_login_ui()  # show auth placeholder

st.sidebar.title(APP_TITLE)
page = st.sidebar.radio("Go to", ["Growlio", "Portfolio Risk Dashboard", "TradeFlow Analyzer", "About"])

if page == "Growlio":
    growlio_page()
elif page == "Portfolio Risk Dashboard":
    portfolio_page()
elif page == "TradeFlow Analyzer":
    tradeflow_page()
else:
    st.title("About Growlio")
    st.markdown("""
    - Multi-tool edition: Growlio + Portfolio risk + TradeFlow
    - Add real auth by replacing the placeholder with Firebase or streamlit-authenticator.
    - Add scheduled updates using Render Cron Jobs or GitHub Actions calling a webhook to update Sheets.
    """)


