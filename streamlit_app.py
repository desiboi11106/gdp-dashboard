# app.py — Unified single-file Growlio + Portfolio Risk + TradeFlow app
import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import plotly.graph_objects as go
import os
import time
from openai import OpenAI
import gspread
from google.oauth2.service_account import Credentials
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import sqlite3
from datetime import datetime as dt, timedelta
import io

# --------------------
# App Setup
# --------------------
st.set_page_config(page_title="Growlio 📈", layout="wide")
# We'll show the chosen page's title within each page function

# --------------------
# Shared: API Keys + clients
# --------------------
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
st.sidebar.write("🔑 OpenAI key loaded:", "✅ Yes" if openai_key else "❌ No")

if openai_key:
    client = OpenAI(api_key=openai_key)
else:
    client = None

has_gcp = "gcp_service_account" in st.secrets
st.sidebar.write("🔒 Google Sheets loaded:", "✅" if has_gcp else "❌")

# --------------------
# Shared helpers
# --------------------
@st.cache_data
def load_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end, group_by="ticker", auto_adjust=True)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def fetch_news_from_sheet_by_key(sheet_key, ticker):
    try:
        creds = Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
        )
        gclient = gspread.authorize(creds)
        sheet = gclient.open_by_key(sheet_key)
        worksheet = sheet.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        if df.empty or 'ticker' not in df.columns:
            return []
        df = df[df['ticker'].astype(str).str.upper() == ticker.upper()]
        return df.to_dict('records')
    except Exception as e:
        st.warning(f"Error fetching sheet data: {e}")
        return []

def openai_summary_from_headlines(ticker, headlines, model="gpt-4o-mini"):
    if client is None:
        return "OpenAI key missing — cannot generate summary."
    try:
        combined = " | ".join(headlines[:12])
        prompt = (
            f"Summarize in plain English why the stock {ticker} might have moved today, "
            f"based on these news headlines: {combined}. "
            f"Keep it under 2 sentences and make it understandable for a beginner investor. "
            "Finish with a one-sentence 'Lesson:'."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst who explains simply."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM error: {e}"

# --------------------
# Page 1: Growlio (exactly preserve your original code)
# --------------------
def growlio_page():
    st.title("📊 Growlio - Investment Learning App")

    # --------------------
    # Sidebar Inputs (local to this page)
    # --------------------
    st.sidebar.header("Stock Settings (Growlio)")
    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma separated)", "AAPL, MSFT, TSLA", key="growlio_tickers")
    start = st.sidebar.date_input("Start Date", datetime.date(2023, 1, 1), key="growlio_start")
    end = st.sidebar.date_input("End Date", datetime.date.today(), key="growlio_end")
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    # --------------------
    # Data Loader
    # --------------------
    data = load_data(tickers, start, end)

    if data is None or data.empty:
        st.warning("⚠️ No data found for the selected tickers and date range.")
        return

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
    # Google Sheets Connection (sheet key input)
    # --------------------
    st.subheader("🔗 Google Sheet Settings")
    sheet_key = st.text_input("Google Sheet ID (Sheet key) — leave blank to use default", value="10zj6tfkdwxNH9lPDeAx5QdM-vcx3G_FsICpC6Us8dx8", key="growlio_sheet_key")

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

            # 📰 News & Learning Insights (From Google Sheet)
            st.subheader(f"📰 {ticker} News & Insights (From Google Sheet)")
            articles = []
            if sheet_key and has_gcp:
                articles = fetch_news_from_sheet_by_key(sheet_key, ticker)
            elif sheet_key and not has_gcp:
                st.info("Google credentials missing in secrets; cannot fetch private sheet.")
            else:
                st.info("No sheet key provided — skipping sheet fetch.")

            if not articles:
                st.write("No articles found for this stock yet.")
                continue

            headlines = []
            for a in articles:
                st.markdown(f"- [{a.get('title','No title')}]({a.get('url','#')}) ({a.get('date','')})")
                headlines.append(a.get('title','No title'))

            # 🤖 AI Summary
            st.subheader(f"🤖 Why Did {ticker} Move?")
            combined = " | ".join(headlines)
            # Use OpenAI only if client available
            if client:
                summary = openai_summary_from_headlines(ticker, headlines)
                st.info(summary)
            else:
                st.info("OpenAI key missing — cannot generate AI summary.")

        except Exception as e:
            st.warning(f"Could not process {ticker}: {e}")

# --------------------
# Page 2: Portfolio Risk Dashboard (kept largely as-is, minimal namespaced edits)
# --------------------
def portfolio_risk_page():
    st.title("💼 Portfolio Risk Dashboard")
    st.caption("Sharpe, beta, diversification metrics, and risk/return visualizations (Python • Excel • Matplotlib)")

    # ------------------------------
    # Helpers (local copies / small local functions)
    # ------------------------------
    @st.cache_data
    def fetch_prices(tickers, start, end, auto_adjust=True):
        df = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"].copy()
        else:
            close = df[["Close"]].copy()
            close.columns = [tickers[0]]
        close = close.dropna(how="all")
        return close

    def clean_weights(raw_tickers, raw_weights):
        tickers = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        weights = [w.strip() for w in raw_weights.split(",") if w.strip()]
        weights = np.array([float(w) for w in weights], dtype=float)
        if len(tickers) != len(weights):
            raise ValueError("Tickers and weights must have the same count.")
        if (weights < 0).any():
            raise ValueError("Weights must be non-negative.")
        if weights.sum() == 0:
            raise ValueError("Weights sum to 0. Provide non-zero weights.")
        weights = weights / weights.sum()
        return tickers, weights

    def ann_metrics(returns, rf=0.00):
        mu_d = returns.mean()
        sd_d = returns.std()
        mu = mu_d * 252
        vol = sd_d * np.sqrt(252)
        sharpe = (mu - rf) / vol.replace(0, np.nan) if isinstance(vol, pd.Series) else (mu - rf) / (vol if vol != 0 else np.nan)
        return mu, vol, sharpe

    def portfolio_series(prices: pd.DataFrame, weights: np.ndarray):
        norm = prices / prices.iloc[0]
        port = (norm * weights).sum(axis=1)
        return port

    def covariance_matrix(returns: pd.DataFrame):
        return returns.cov() * 252  # annualized

    def beta_vs_benchmark(asset_returns: pd.DataFrame, bench_returns: pd.Series):
        betas = {}
        var_b = bench_returns.var()
        if var_b == 0 or np.isnan(var_b):
            return pd.Series({c: np.nan for c in asset_returns.columns})
        for c in asset_returns.columns:
            cov = np.cov(asset_returns[c].dropna(), bench_returns.dropna())[0, 1]
            betas[c] = cov / var_b
        return pd.Series(betas)

    def risk_contributions(weights: np.ndarray, cov: pd.DataFrame):
        w = np.asarray(weights).reshape(-1, 1)
        port_var = float(w.T @ cov.values @ w)
        if port_var <= 0:
            mcr = np.zeros_like(w.flatten())
            pcr = np.zeros_like(w.flatten())
        else:
            mcr = (cov.values @ w).flatten() / np.sqrt(port_var)
            pcr = (w.flatten() * (cov.values @ w).flatten()) / port_var
        return mcr, pcr, np.sqrt(port_var)

    def diversification_stats(weights: np.ndarray, corr: pd.DataFrame):
        hhi = float(np.sum(np.square(weights)))
        div_score = 1 - hhi
        C = corr.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        n = C.shape[0]
        if n > 1:
            upper = C[np.triu_indices(n, 1)]
            avg_corr = float(np.mean(upper))
        else:
            avg_corr = np.nan
        return hhi, div_score, avg_corr

    def random_frontier(returns: pd.DataFrame, n_port=3000, rf=0.00):
        mu, Sigma = returns.mean().values * 252, returns.cov().values * 252
        n = len(mu)
        rr, vv, sh, WW = [], [], [], []
        for _ in range(n_port):
            w = np.random.rand(n)
            w = w / w.sum()
            mu_p = float(w @ mu)
            vol_p = float(np.sqrt(w @ Sigma @ w))
            rr.append(mu_p)
            vv.append(vol_p)
            sh.append((mu_p - rf) / (vol_p if vol_p != 0 else np.nan))
            WW.append(w)
        df = pd.DataFrame({"Return": rr, "Volatility": vv, "Sharpe": sh})
        return df, np.array(WW)

    def template_excel():
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT", "TSLA"], "Weight": [0.4, 0.4, 0.2]})
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="weights")
        buf.seek(0)
        return buf

    # ------------------------------
    # Sidebar inputs
    # ------------------------------
    st.sidebar.header("Portfolio Inputs")
    mode = st.sidebar.radio("Input Method", ["Manual (tickers & weights)", "Upload Excel (weights)"], key="pr_mode")
    default_start = datetime.date(2022, 1, 1)
    start = st.sidebar.date_input("Start Date", default_start, key="pr_start")
    end = st.sidebar.date_input("End Date", datetime.date.today(), key="pr_end")
    rf_pct = st.sidebar.number_input("Risk-free (annual, %)", min_value=0.0, max_value=10.0, value=0.0, step=0.10, key="pr_rf")
    rf = rf_pct / 100.0
    benchmark = st.sidebar.text_input("Benchmark ticker (for beta)", "^GSPC", key="pr_bench")

    if mode == "Manual (tickers & weights)":
        tickers_raw = st.sidebar.text_input("Tickers (comma-separated)", "AAPL, MSFT, TSLA", key="pr_tickers_raw")
        weights_raw = st.sidebar.text_input("Weights (comma-separated)", "0.4, 0.4, 0.2", key="pr_weights_raw")
        try:
            tickers, weights = clean_weights(tickers_raw, weights_raw)
        except Exception as e:
            st.error(f"Weight error: {e}")
            return
    else:
        uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx) with sheet 'weights' (Ticker, Weight)", type=["xlsx"], key="pr_uploaded")
        st.sidebar.download_button("Download template.xlsx", template_excel(), file_name="portfolio_template.xlsx")
        if uploaded is None:
            st.info("Upload an Excel file or switch to Manual mode.")
            return
        else:
            try:
                wdf = pd.read_excel(uploaded, sheet_name="weights")
                if not {"Ticker", "Weight"}.issubset(set(wdf.columns)):
                    st.error("Excel must contain columns: Ticker, Weight (sheet name: weights).")
                    return
                tickers = [str(t).upper().strip() for t in wdf["Ticker"].tolist()]
                weights = np.array(wdf["Weight"].astype(float).tolist())
                if len(tickers) == 0:
                    st.error("No tickers found in Excel.")
                    return
                if weights.sum() == 0:
                    st.error("Weights sum to 0. Please provide positive weights.")
                    return
                weights = weights / weights.sum()
            except Exception as e:
                st.error(f"Excel parsing error: {e}")
                return

    # ------------------------------
    # Data fetch
    # ------------------------------
    prices = fetch_prices(tickers, start, end)
    if prices.empty:
        st.warning("No price data found for your selection.")
        return

    # Align benchmark
    try:
        bench_px = fetch_prices([benchmark], start, end).iloc[:, 0]
    except Exception:
        bench_px = None

    rets = prices.pct_change().dropna()
    if bench_px is not None and not bench_px.empty:
        bench_rets = bench_px.pct_change().dropna()
    else:
        bench_rets = None

    port_series = portfolio_series(prices, weights)
    port_rets = port_series.pct_change().dropna()

    # ------------------------------
    # Top KPIs
    # ------------------------------
    colA, colB, colC, colD = st.columns(4)
    p_mu, p_vol, p_sharpe = ann_metrics(port_rets, rf=rf)
    colA.metric("Portfolio Annual Return", f"{p_mu*100:.2f}%")
    colB.metric("Portfolio Annual Volatility", f"{p_vol*100:.2f}%")
    colC.metric("Portfolio Sharpe", f"{p_sharpe:.2f}")
    colD.metric("Holdings / Sum of Weights", f"{len(tickers)} / {weights.sum():.2f}")

    # ------------------------------
    # Charts: Price & Rolling Vol
    # ------------------------------
    st.subheader("📈 Portfolio Value & Rolling Volatility")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(port_series.index, port_series.values, label="Portfolio (normalized)")
        ax.set_title("Portfolio Value (Start = 1.0)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, use_container_width=True)
    with c2:
        roll = port_rets.rolling(21).std() * np.sqrt(252)
        fig2, ax2 = plt.subplots(figsize=(6.5, 4))
        ax2.plot(roll.index, roll.values)
        ax2.set_title("Rolling 1M Volatility (Annualized)")
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

    # ------------------------------
    # Asset metrics (Return / Vol / Sharpe) & Beta
    # ------------------------------
    st.subheader("📊 Asset Metrics")
    asset_mu, asset_vol, asset_sh = ann_metrics(rets, rf=rf)
    metrics_df = pd.DataFrame({
        "Ann Return": asset_mu,
        "Ann Volatility": asset_vol,
        "Sharpe": asset_sh
    })
    if bench_rets is not None and not bench_rets.empty:
        betas = beta_vs_benchmark(rets, bench_rets)
        metrics_df["Beta vs " + benchmark] = betas
    st.dataframe(metrics_df.style.format({
        "Ann Return": "{:.2%}", "Ann Volatility": "{:.2%}", "Sharpe": "{:.2f}",
        **({f"Beta vs {benchmark}": "{:.2f}"} if bench_rets is not None else {})
    }), use_container_width=True)

    # ------------------------------
    # Risk contributions & Diversification
    # ------------------------------
    st.subheader("🧩 Risk Decomposition & Diversification")
    cov = covariance_matrix(rets)
    corr = rets.corr()
    mcr, pcr, port_vol_ann = risk_contributions(weights, cov)
    hhi, div_score, avg_corr = diversification_stats(weights, corr)
    r1, r2, r3 = st.columns(3)
    r1.metric("Portfolio Volatility", f"{port_vol_ann*100:.2f}%")
    r2.metric("Diversification Score (1-HHI)", f"{div_score:.3f}")
    r3.metric("Avg Pairwise Correlation", f"{avg_corr:.2f}" if not np.isnan(avg_corr) else "—")

    # Bar: Percent risk contribution
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    ax3.bar(tickers, pcr * 100.0)
    ax3.set_ylabel("% of Portfolio Variance")
    ax3.set_title("Percent Contribution to Risk")
    ax3.yaxis.set_major_formatter(PercentFormatter(100))
    ax3.grid(True, axis="y", alpha=0.3)
    st.pyplot(fig3, use_container_width=True)

    # Heatmap: Correlation
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    im = ax4.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax4.set_xticks(range(len(tickers))); ax4.set_xticklabels(tickers, rotation=45, ha="right")
    ax4.set_yticks(range(len(tickers))); ax4.set_yticklabels(tickers)
    ax4.set_title("Correlation Heatmap")
    cbar = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    st.pyplot(fig4, use_container_width=True)

    # ------------------------------
    # Efficient frontier (Monte Carlo)
    # ------------------------------
    st.subheader("🌈 Efficient Frontier (Monte Carlo)")
    n_sims = st.slider("Number of random portfolios", 1000, 10000, 3000, step=1000)
    frontier_df, _ = random_frontier(rets, n_port=n_sims, rf=rf)
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    ax5.scatter(frontier_df["Volatility"], frontier_df["Return"], s=8, alpha=0.3)
    ax5.scatter([p_vol], [p_mu], c="red", s=80, label="Your Portfolio")
    ax5.set_xlabel("Volatility"); ax5.set_ylabel("Return"); ax5.set_title("Risk/Return Cloud")
    ax5.grid(True, alpha=0.3); ax5.legend()
    st.pyplot(fig5, use_container_width=True)

    # ------------------------------
    # Downloads (metrics + prices)
    # ------------------------------
    st.subheader("⬇️ Export")
    exp1 = metrics_df.copy()
    exp1.index.name = "Ticker"
    csv_metrics = exp1.to_csv().encode("utf-8")
    st.download_button("Download Asset Metrics (CSV)", csv_metrics, "asset_metrics.csv", "text/csv")
    csv_prices = prices.to_csv().encode("utf-8")
    st.download_button("Download Prices (CSV)", csv_prices, "prices.csv", "text/csv")

    # ------------------------------
    # Notes / Resume bullets
    # ------------------------------
    with st.expander("📝 What this dashboard demonstrates (resume-ready)"):
        st.markdown("""
    - **Modeled portfolio Sharpe ratio, beta, and diversification** (HHI, correlations, risk contributions).
    - **Visualized risk/return trade-offs**: efficient frontier via Monte Carlo, portfolio vs. cloud.
    - **Supports Excel inputs** for practical portfolio allocations (sheet: `weights`).
    - Built with **Python, Excel I/O, Matplotlib**, and **Streamlit** for interactive UX.
    """)

# --------------------
# Page 3: TradeFlow Analyzer (kept as-is)
# --------------------
def tradeflow_page():
    st.title("📈 TradeFlow Analyzer")
    st.markdown("Synthetic trade data with SQL + anomaly detection")

    # Generate trades
    def generate_fake_trades(n=1000):
        np.random.seed(42)
        timestamps = [dt.now() - timedelta(minutes=i) for i in range(n)]
        prices = np.random.normal(100, 2, n).round(2)
        volumes = np.random.randint(10, 1000, n)
        trade_ids = range(1, n+1)
        df = pd.DataFrame({
            "trade_id": trade_ids,
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes
        })
        return df

    df = generate_fake_trades(5000)

    # Show raw data
    if st.checkbox("Show sample trades", key="tf_sample"):
        st.dataframe(df.head(20))

    # Run custom SQL query
    st.subheader("Run SQL on Trades")
    default_query = "SELECT AVG(price) as avg_price, SUM(volume) as total_volume FROM trades"
    query = st.text_area("Enter SQL query:", value=default_query, height=100, key="tf_sql")
    if st.button("Run Query", key="tf_run"):
        try:
            conn = sqlite3.connect(":memory:")
            df.to_sql("trades", conn, index=False, if_exists="replace")
            result = pd.read_sql_query(query, conn)
            st.dataframe(result)
            conn.close()
        except Exception as e:
            st.error(f"SQL error: {e}")

    # Liquidity patterns
    st.subheader("Liquidity Patterns")
    df["minute"] = df["timestamp"].dt.floor("T")
    liquidity = df.groupby("minute")["volume"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(liquidity["minute"], liquidity["volume"], label="Total Volume per Minute")
    ax.set_title("Liquidity Over Time")
    ax.set_ylabel("Volume")
    ax.set_xlabel("Time")
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # Anomaly detection: Price & Volume outliers
    st.subheader("Anomaly Detection")
    price_mean, price_std = df["price"].mean(), df["price"].std()
    outliers = df[(df["price"] > price_mean + 3*price_std) | (df["price"] < price_mean - 3*price_std)]

    if not outliers.empty:
        st.warning(f"Detected {len(outliers)} abnormal trades (3σ rule)")
        st.dataframe(outliers.head(10))
    else:
        st.success("No price anomalies detected.")

    # Heatmap of price-volume correlation
    st.subheader("Price-Volume Relationship")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="price", y="volume", alpha=0.3, ax=ax)
    ax.set_title("Price vs Volume")
    st.pyplot(fig)

# -----------------------
# Navigation (Growlio default)
# -----------------------
st.sidebar.title("Growlio Super-App")
page = st.sidebar.radio("Select page", ["Growlio (default)", "Portfolio Risk Dashboard", "TradeFlow Analyzer"], index=0)

if page == "Growlio (default)":
    growlio_page()
elif page == "Portfolio Risk Dashboard":
    portfolio_risk_page()
else:
    tradeflow_page()

