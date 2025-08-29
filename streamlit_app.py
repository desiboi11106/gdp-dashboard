import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import requests
import plotly.graph_objects as go

# --------------------
# App Title
# --------------------
st.set_page_config(page_title="Growlio 📈", layout="wide")
st.title("📊 Growlio - Investment Learning App")

# --------------------
# Sidebar Inputs
# --------------------
st.sidebar.header("Stock Settings")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma separated)", "AAPL, MSFT, TSLA")
start = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end = st.sidebar.date_input("End Date", datetime.date.today())

# Split and clean tickers
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

# --------------------
# Download Data
# --------------------
@st.cache_data
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data(tickers, start, end)

if data is None or data.empty:
    st.warning("⚠️ No data found for the selected tickers and date range.")
    st.stop()

# --------------------
# Metrics
# --------------------
st.subheader("📈 Stock Metrics")
cols = st.columns(len(tickers))

for i, ticker in enumerate(tickers):
    try:
        last_close = data[ticker]["Close"].iloc[-1]
        first_close = data[ticker]["Close"].iloc[0]
        change = ((last_close - first_close) / first_close) * 100
        cols[i].metric(ticker, f"${last_close:.2f}", f"{change:.2f}%")
    except Exception:
        cols[i].warning(f"No close price data for {ticker}")

# --------------------
# Stock Comparison Chart
# --------------------
st.subheader("📉 Stock Price Comparison")

fig = go.Figure()

for ticker in tickers:
    try:
        fig.add_trace(go.Scatter(
            x=data[ticker].index,
            y=data[ticker]["Close"],
            mode="lines",
            name=ticker
        ))
    except Exception:
        pass

fig.update_layout(title="Stock Prices", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(fig, use_container_width=True)

# --------------------
# Individual Stock Analysis
# --------------------
st.subheader("🔍 Detailed Analysis per Stock")

for ticker in tickers:
    st.markdown(f"### {ticker}")

    try:
        df = data[ticker].copy()
        df["50MA"] = df["Close"].rolling(window=50).mean()
        df["200MA"] = df["Close"].rolling(window=200).mean()
        df["Volatility"] = df["Close"].rolling(window=20).std()

        # Buy signals (when MA50 crosses above MA200)
        df["Signal"] = (df["50MA"] > df["200MA"]) & (df["50MA"].shift(1) <= df["200MA"].shift(1))
        buy_signals = df[df["Signal"]]

        # Candlestick + Moving Averages + Buy Signals
        fig2 = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Candlestick"
        )])
        fig2.add_trace(go.Scatter(x=df.index, y=df["50MA"], mode="lines", name="50MA"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["200MA"], mode="lines", name="200MA"))
        fig2.add_trace(go.Scatter(
            x=buy_signals.index,
            y=buy_signals["Close"],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="Buy Signal"
        ))
        fig2.update_layout(title=f"{ticker} Price with MAs & Buy Signals")
        st.plotly_chart(fig2, use_container_width=True)

        # Volatility
        st.subheader(f"📉 {ticker} Volatility")
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(x=df.index, y=df["Volatility"], mode="lines", name="Volatility"))
        vol_fig.update_layout(title=f"{ticker} 20-Day Rolling Volatility")
        st.plotly_chart(vol_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not process {ticker}: {e}")

# --------------------
# News Section
# --------------------
st.subheader("📰 Stock News & Articles")

def fetch_news(ticker):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={ticker}"
        response = requests.get(url, timeout=5).json()
        news_items = response.get("news", [])
        return news_items[:5]
    except Exception:
        return []

for ticker in tickers:
    st.markdown(f"#### {ticker} News")
    news_list = fetch_news(ticker)
    if not news_list:
        st.write("No news found.")
    else:
        for item in news_list:
            title = item.get("title", "No title")
            link = item.get("link", "#")
            st.markdown(f"- [{title}]({link})")
