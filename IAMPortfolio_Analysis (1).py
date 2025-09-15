# app.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Portfolio Analysis (Indian equities)", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_data(show_spinner=False)
def download_prices(tickers, start_date, end_date):
    # yfinance can accept list of tickers; keep Adj Close
    try:
        df = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        if ('Adj Close' in df.columns):
            prices = df['Adj Close'].copy()
        else:
            # If single ticker, df might already be 'Adj Close' series
            prices = df.copy()
        prices = prices.dropna(how='all')
        return prices
    except Exception as e:
        st.error(f"Download error: {e}")
        return pd.DataFrame()

def compute_returns(prices):
    # daily simple returns and monthly/yearly aggregated returns
    daily = prices.pct_change().dropna()
    monthly = prices.resample('M').last().pct_change().dropna()
    yearly = prices.resample('Y').last().pct_change().dropna()
    return {"daily": daily, "monthly": monthly, "yearly": yearly}

def annualize_return(returns, periods_per_year=252):
    # arithmetic annualization of mean daily simple returns
    return (1 + returns.mean())**periods_per_year - 1

def annualize_vol(returns, periods_per_year=252):
    return returns.std() * np.sqrt(periods_per_year)

def compute_cagr(prices):
    try:
        start = prices.index[0]
        end = prices.index[-1]
        years = (end - start).days / 365.25
        return (prices.iloc[-1] / prices.iloc[0]) ** (1/years) - 1
    except Exception:
        return np.nan

def compute_beta_ols(series, benchmark):
    # linear regression: series ~ benchmark
    try:
        X = benchmark.values
        Y = series.values
        X = np.vstack([np.ones_like(X), X]).T
        params = np.linalg.lstsq(X, Y, rcond=None)[0]
        alpha, beta = params[0], params[1]
        # r2
        ss_res = ((Y - (alpha + beta*benchmark.values))**2).sum()
        ss_tot = ((Y - Y.mean())**2).sum()
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else np.nan
        return beta, alpha, r2
    except Exception:
        return np.nan, np.nan, np.nan

def compute_summary_stats(prices, returns_dict, benchmark_symbol=None, rf=0.06):
    daily = returns_dict['daily']
    cols = list(daily.columns)
    stats = []
    bench_ret = daily[benchmark_symbol] if (benchmark_symbol in daily.columns) else None

    for col in cols:
        r = daily[col]
        ann_ret = annualize_return(r)
        ann_vol = annualize_vol(r)
        cagr = compute_cagr(prices[col])
        sharpe = (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan
        # sortino: use downside std
        downside = r[r < 0].std() * np.sqrt(252)
        sortino = (ann_ret - rf) / downside if downside and downside != 0 else np.nan

        if benchmark_symbol and bench_ret is not None and col != benchmark_symbol:
            beta_cov = np.cov(r.values, bench_ret.values)[0,1] / np.var(bench_ret.values)
            beta_ols, alpha, r2 = compute_beta_ols(r, bench_ret)
        else:
            beta_cov, beta_ols, alpha, r2 = (1.0, 1.0, 0.0, 1.0)

        stats.append({
            "ticker": col,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "beta_cov": beta_cov,
            "beta_ols": beta_ols,
            "alpha": alpha,
            "r2": r2,
            "cagr": cagr,
            "sharpe": sharpe,
            "sortino": sortino
        })

    df = pd.DataFrame(stats).set_index('ticker')
    return df

# Portfolio funcs
def port_return(weights, mean_returns):
    return np.dot(weights, mean_returns) * 252

def port_vol(weights, cov):
    return np.sqrt(np.dot(weights.T, np.dot(cov, weights)))

def minimize_vol_for_ret(target, mean_returns, cov, no_shorting=True):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    bounds = [(0.0,1.0)]*n if no_shorting else [(None,None)]*n
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type':'eq', 'fun': lambda w: port_return(w, mean_returns) - target})
    res = minimize(lambda w: port_vol(w, cov), x0=x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res

def efficient_frontier(mean_returns, cov, no_shorting=True, points=50):
    mean_returns = np.array(mean_returns)
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    targets = np.linspace(min_ret, max_ret, points)
    vols = []
    weights = []
    for t in targets:
        res = minimize_vol_for_ret(t, mean_returns, cov, no_shorting=no_shorting)
        if res.success:
            vols.append(res.fun)
            weights.append(res.x)
        else:
            vols.append(np.nan)
            weights.append([np.nan]*len(mean_returns))
    return targets*252, vols, weights  # return annualized target returns

def max_sharpe(mean_returns, cov, rf=0.06, no_shorting=True):
    n = len(mean_returns)
    x0 = np.repeat(1/n, n)
    bounds = [(0.0,1.0)]*n if no_shorting else [(None,None)]*n
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1},)
    def neg_sharpe(w):
        r = port_return(w, mean_returns)
        v = port_vol(w, cov)
        return - (r - rf) / v if v != 0 else 1e9
    res = minimize(neg_sharpe, x0=x0, bounds=bounds, constraints=cons, method='SLSQP')
    return res

# ---------------------------
# Sidebar - controls
# ---------------------------
st.sidebar.title("Controls")
default_tickers = ['SHRIRAMFIN.NS', 'INDIGOPNTS.NS', 'IRB.NS', 'UNITDSPR.NS', 'VOLTAS.NS',
                   'PIIND.NS', 'ASTRAZEN.NS', 'BALKRISIND.NS', 'MANAPPURAM.NS', 'BLUESTARCO.NS']
tickers_text = st.sidebar.text_area("Tickers (comma separated)", value=",".join(default_tickers))
tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

benchmark = st.sidebar.text_input("Benchmark ticker", value="^NSEI")
start_date = st.sidebar.date_input("Start date", value=date.today() - relativedelta(years=5))
end_date = st.sidebar.date_input("End date", value=date.today())
rf_input = st.sidebar.number_input("Risk-free rate (annual, e.g. 0.06)", value=0.06, format="%.4f")
no_short = st.sidebar.checkbox("No shorting (weights >= 0)", value=True)
run_button = st.sidebar.button("Run analysis")

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: adjust tickers, date range, and risk-free rate. Press *Run analysis* to (re)compute.")

# Main layout
st.title("Portfolio Analysis — Interactive Dashboard")
st.markdown("A compact, interactive portfolio analysis for Indian equities — prices from Yahoo Finance.")

# If run or first load, execute
if run_button:
    with st.spinner("Downloading prices and computing analytics..."):
        all_tickers = tickers.copy()
        if benchmark not in all_tickers:
            all_tickers.append(benchmark)

        prices = download_prices(all_tickers, start_date, end_date)
        if prices.empty:
            st.error("No price data was downloaded. Check tickers or your internet connection.")
            st.stop()

        # Ensure all tickers exist as columns (yfinance can return single-column)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        # Some tickers may be missing — warn
        missing = [t for t in all_tickers if t not in prices.columns]
        if missing:
            st.warning(f"No data for: {missing}. They will be ignored in analysis.")

        # separate asset tickers (exclude benchmark if present)
        asset_tickers = [t for t in tickers if t in prices.columns]
        bench_symbol = benchmark if benchmark in prices.columns else None

        returns = compute_returns(prices)
        daily = returns['daily']

        # summary stats
        summary = compute_summary_stats(prices, returns, benchmark_symbol=bench_symbol, rf=rf_input)

        # Cov and corr for assets only
        if len(asset_tickers) < 2:
            st.warning("Need at least 2 asset tickers for covariance/correlation and optimization.")
        asset_daily = daily[asset_tickers]
        cov = asset_daily.cov() * 252
        corr = asset_daily.corr()

        # rolling metrics
        rolling_window = 252
        rolling_vol = asset_daily.rolling(window=rolling_window).std() * np.sqrt(252)
        rolling_beta = pd.DataFrame(index=daily.index)
        if bench_symbol:
            for t in asset_tickers:
                rolling_cov = asset_daily[t].rolling(window=rolling_window).cov(daily[bench_symbol])
                rolling_var = daily[bench_symbol].rolling(window=rolling_window).var()
                rolling_beta[t] = rolling_cov / rolling_var

        # efficient frontier & optimal portfolios
        mean_returns = asset_daily.mean()
        if len(asset_tickers) >= 2:
            ef_rets, ef_vols, ef_weights = efficient_frontier(mean_returns.values, cov.values, no_shorting=no_short)
            msp = max_sharpe(mean_returns.values, cov.values, rf=rf_input, no_shorting=no_short)
            msp_weights = msp.x if msp.success else np.repeat(1/len(asset_tickers), len(asset_tickers))
            mvp_res = minimize(lambda w: port_vol(w, cov.values), x0=np.repeat(1/len(asset_tickers), len(asset_tickers)),
                               bounds=[(0,1)]*len(asset_tickers) if no_short else None,
                               constraints=({'type':'eq','fun':lambda w: np.sum(w)-1},), method='SLSQP')
            mvp_weights = mvp_res.x if mvp_res.success else np.repeat(1/len(asset_tickers), len(asset_tickers))

        # portfolio simulated returns for MSP
        if len(asset_tickers) >= 1:
            port_msp_returns = asset_daily.dot(msp_weights)
            cum = (1 + port_msp_returns).cumprod()
            peak = cum.cummax()
            drawdown = (cum - peak) / peak
            max_dd = drawdown.min()

        # ---------------------------
        # Render panels
        # ---------------------------
        # Top metrics (CAGR, Max Sharpe, Min Var)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Assets (loaded)", value=len(asset_tickers))
            if bench_symbol:
                st.caption(f"Benchmark: {bench_symbol}")
        with col2:
            if len(asset_tickers) >= 1:
                msp_ret = port_return(msp_weights, mean_returns.values)
                msp_vol = port_vol(msp_weights, cov.values)
                st.metric("MSP Expected Return (ann.)", f"{msp_ret:.2%}", delta=None)
                st.metric("MSP Vol (ann.)", f"{msp_vol:.2%}")
        with col3:
            if len(asset_tickers) >= 1:
                mvp_vol_val = port_vol(mvp_weights, cov.values)
                mvp_ret_val = port_return(mvp_weights, mean_returns.values)
                st.metric("MVP Vol (ann.)", f"{mvp_vol_val:.2%}")
                st.metric("MVP Return (ann.)", f"{mvp_ret_val:.2%}")
        with col4:
            if len(asset_tickers) >= 1:
                st.metric("MSP Max Drawdown (hist)", f"{max_dd:.2%}")

        st.markdown("---")

        # Price chart (interactive)
        st.subheader("Price chart")
        price_fig = px.line(prices[asset_tickers + ([bench_symbol] if bench_symbol else [])].dropna(how='all'),
                            labels={"value":"Price", "index":"Date"}, title="Adjusted Close Prices")
        st.plotly_chart(price_fig, use_container_width=True)

        # Summary stats table
        st.subheader("Summary statistics")
        st.dataframe(summary.style.format({
            "annual_return":"{:.2%}",
            "annual_vol":"{:.2%}",
            "beta_cov":"{:.3f}",
            "beta_ols":"{:.3f}",
            "alpha":"{:.4f}",
            "r2":"{:.3f}",
            "cagr":"{:.2%}",
            "sharpe":"{:.2f}",
            "sortino":"{:.2f}"
        }), height=300)

        # Correlation heatmap
        st.subheader("Correlation matrix")
        if not corr.empty:
            fig_corr = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

        # Efficient frontier plot
        st.subheader("Efficient Frontier")
        if len(asset_tickers) >= 2:
            ef_df = pd.DataFrame({"Return": ef_rets, "Vol": ef_vols})
            ef_fig = go.Figure()
            # assets scatter
            ef_fig.add_trace(go.Scatter(x=summary.loc[asset_tickers, 'annual_vol'],
                                       y=summary.loc[asset_tickers, 'annual_return'],
                                       mode='markers+text', text=asset_tickers, name='Assets',
                                       marker=dict(size=8)))
            # frontier
            ef_fig.add_trace(go.Scatter(x=ef_vols, y=ef_rets, mode='lines', name='Efficient Frontier'))
            # MVP & MSP
            ef_fig.add_trace(go.Scatter(x=[mvp_vol_val], y=[mvp_ret_val], mode='markers', name='MVP',
                                       marker=dict(symbol='diamond', size=12)))
            ef_fig.add_trace(go.Scatter(x=[msp_vol], y=[msp_ret], mode='markers', name='MSP',
                                       marker=dict(symbol='star', size=14)))
            ef_fig.update_layout(xaxis_title="Annual Volatility", yaxis_title="Annual Return")
            st.plotly_chart(ef_fig, use_container_width=True)

            # display weights as pie charts
            wcol1, wcol2 = st.columns(2)
            with wcol1:
                st.markdown("*MVP Weights*")
                w_mvp = pd.Series(mvp_weights, index=asset_tickers).clip(lower=0)
                fig_mvp = px.pie(names=w_mvp.index, values=w_mvp.values, title="Min Variance Portfolio")
                st.plotly_chart(fig_mvp, use_container_width=True)
            with wcol2:
                st.markdown("*MSP Weights*")
                w_msp = pd.Series(msp_weights, index=asset_tickers).clip(lower=0)
                fig_msp = px.pie(names=w_msp.index, values=w_msp.values, title="Max Sharpe Portfolio")
                st.plotly_chart(fig_msp, use_container_width=True)

        # Rolling vol / beta
        st.subheader("Rolling 1-year volatility & beta")
        rv_fig = go.Figure()
        for t in asset_tickers[:6]:  # limit lines for readability
            rv_fig.add_trace(go.Scatter(x=rolling_vol.index, y=rolling_vol[t], mode='lines', name=f"{t} vol"))
        rv_fig.update_layout(yaxis_title="Annual Volatility")
        st.plotly_chart(rv_fig, use_container_width=True)

        if bench_symbol:
            rb_fig = go.Figure()
            for t in asset_tickers[:6]:
                rb_fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta[t], mode='lines', name=f"{t} beta"))
            rb_fig.update_layout(yaxis_title="Rolling Beta")
            st.plotly_chart(rb_fig, use_container_width=True)

        # Drawdown
        st.subheader("MSP Cumulative return & historical drawdown")
        if len(asset_tickers) >= 1:
            dd_fig = go.Figure()
            dd_fig.add_trace(go.Scatter(x=cum.index, y=cum.values, mode='lines', name='Cumulative (MSP)'))
            dd_fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, mode='lines', name='Drawdown'))
            dd_fig.update_layout(yaxis_title="Cumulative / Drawdown")
            st.plotly_chart(dd_fig, use_container_width=True)

        # Download results
        st.subheader("Export results")
        with io.BytesIO() as buffer:
            # write Excel
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                prices.to_excel(writer, sheet_name='Prices')
                daily.to_excel(writer, sheet_name='Daily_Returns')
                pd.DataFrame(cov).to_excel(writer, sheet_name='Covariance')
                corr.to_excel(writer, sheet_name='Correlation')
                summary.to_excel(writer, sheet_name='Summary_Stats')
                # efficient frontier
                if len(asset_tickers) >= 2:
                    pd.DataFrame({"Return": ef_rets, "Vol": ef_vols}).to_excel(writer, sheet_name='Efficient_Frontier')
                    pd.DataFrame(index=asset_tickers, data={"MVP": mvp_weights, "MSP": msp_weights}).to_excel(writer, sheet_name='Optimal_Weights')
            buffer.seek(0)
            st.download_button("Download Excel report", data=buffer.read(), file_name="portfolio_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # CSV export (prices & summary)
        csv_buf = io.StringIO()
        prices.to_csv(csv_buf)
        st.download_button("Download Prices CSV", data=csv_buf.getvalue(), file_name="prices.csv", mime="text/csv")

        st.success("Analysis complete.")
else:
    st.info("Set parameters in sidebar and click *Run analysis* to start. Default tickers are prefilled.")
    