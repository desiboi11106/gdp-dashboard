import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import os
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials

# --------------------
# App Setup
# --------------------
st.set_page_config(page_title="Growlio 📈", layout="wide")
st.title("📊 Growlio - Investment Learning App")

# --------------------
# API Keys
# --------------------
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
st.write("🔑 OpenAI key loaded:", "✅ Yes" if openai_key else "❌ No")

if not openai_key:
    st.error("❌ Missing OpenAI API key. Add it in Streamlit Secrets as OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=openai_key)

# --------------------
# Sidebar Inputs
# --------------------
st.sidebar.header("Stock Settings")
tickers = st.sidebar.text_input("Enter Stock Tickers (comma separated)", "AAPL, MSFT, TSLA")
start = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1))
end = st.sidebar.date_input("End Date", datetime.date.today())
tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]

# --------------------
# Data Loader
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
# Stock Metrics
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
# Comparison Chart
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
# Google Sheets Connection
# --------------------
def fetch_news_from_sheet(ticker):
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        client = gspread.authorize(creds)
        sheet = client.open_by_key("10zj6tfkdwxNH9lPDeAx5QdM-vcx3G_FsICpC6Us8dx8")
        worksheet = sheet.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df = df[df['ticker'].str.upper() == ticker.upper()]
        return df.to_dict('records')
    except Exception as e:
        st.warning(f"Error fetching sheet data: {e}")
        return []

# --------------------
# Detailed Analysis per Stock
# --------------------
st.subheader("🔍 Detailed Analysis per Stock")

for ticker in tickers:
    st.markdown(f"## {ticker}")
    try:
        df = data[ticker].copy()
        df["50MA"] = df["Close"].rolling(window=50).mean()
        df["200MA"] = df["Close"].rolling(window=200).mean()
        df["Volatility"] = df["Close"].rolling(window=20).std()

        # Buy signal logic
        df["Signal"] = (df["50MA"] > df["200MA"]) & (df["50MA"].shift(1) <= df["200MA"].shift(1))
        buy_signals = df[df["Signal"]]

        # Chart: Candlestick + Moving Averages + Buy Signals
        fig2 = go.Figure(data=[go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Candlestick"
        )])
        fig2.add_trace(go.Scatter(x=df.index, y=df["50MA"], mode="lines", name="50MA"))
        fig2.add_trace(go.Scatter(x=df.index, y=df["200MA"], mode="lines", name="200MA"))
        fig2.add_trace(go.Scatter(
            x=buy_signals.index, y=buy_signals["Close"],
            mode="markers", marker=dict(symbol="triangle-up", color="green", size=10), name="Buy Signal"
        ))
        fig2.update_layout(title=f"{ticker} Price with MAs & Buy Signals")
        st.plotly_chart(fig2, use_container_width=True)

        # Volatility chart
        st.subheader(f"📉 {ticker} Volatility")
        vol_fig = go.Figure()
        vol_fig.add_trace(go.Scatter(x=df.index, y=df["Volatility"], mode="lines", name="Volatility"))
        vol_fig.update_layout(title=f"{ticker} 20-Day Rolling Volatility")
        st.plotly_chart(vol_fig, use_container_width=True)

        # 📰 News & Learning Insights
        st.subheader(f"📰 {ticker} News & Insights (From Google Sheet)")
        articles = fetch_news_from_sheet(ticker)
        if not articles:
            st.write("No articles found for this stock yet.")
            continue

        headlines = []
        for a in articles:
            st.markdown(f"- [{a['title']}]({a['url']}) ({a['date']})")
            headlines.append(a['title'])

        # 🤖 AI Summary
        st.subheader(f"🤖 Why Did {ticker} Move?")
        combined = " | ".join(headlines)
        prompt = f"""
        You are a finance educator explaining to new investors.
        Based on these news headlines for {ticker}, explain:
        1. Why this stock likely moved.
        2. What concept it illustrates (earnings, inflation, sentiment, etc.).
        Then give a 'Lesson:' line in one sentence.
        Headlines: {combined}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain finance clearly and simply."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        st.info(response.choices[0].message.content.strip())

    except Exception as e:
        st.warning(f"Could not process {ticker}: {e}")
