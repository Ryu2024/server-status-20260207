import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="Data Dashboard", layout="wide")

# --- 2. 状态初始化 ---
if'ticker' not in st.session_state:
    st.session_state.ticker = "BTC-USD"

# --- 3. 样式设置 ---
st.markdown("""
<style>
    /* ============================================================ */
    /* 1. 全局基础与布局优化 */
    /* ============================================================ */
    .stApp {
        background-color: #ffffff;
        font-family: 'Times New Roman', Times, serif;
        color: #000000;
    }
    
    /* 限制页面主容器的最大宽度，保持紧凑 */
    div[data-testid="block-container"] {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }

    /* [核心修改]：图表容器强制拉宽 */
    /* 宽度设为 106% 并向左偏移 3%，让图表比上面的文字栏更宽，视觉上对齐边框 */
    div[data-testid="stPlotlyChart"] {
        width: 106% !important;
        margin-left: -3% !important;
        max-width: 106% !important;
    }

    h1, h2, h3, label, p, div, span, li {
        font-family: 'Times New Roman', Times, serif !important;
        color: #000000 !important;
    }
    .modebar { display: none !important; }

    /* ============================================================ */
    /* 2. 颜色定义 */
    /* ============================================================ */
    .retro-color-green, .retro-color-green * { color: #28a745 !important; }
    .retro-color-blue, .retro-color-blue * { color: #007bff !important; }
    .retro-color-orange, .retro-color-orange * { color: #fd7e14 !important; }
    .retro-color-red, .retro-color-red * { color: #dc3545 !important; }
    .retro-color-gray, .retro-color-gray * { color: #666666 !important; }

    /* ============================================================ */
    /* 3. 容器与边框 */
    /* ============================================================ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        background-color: #ffffff !important;
        padding: 0px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > * {
        border-radius: 0px !important;
    }

    /* ============================================================ */
    /* 4. 按钮样式 */
    /* ============================================================ */
    div.stButton {
        margin-top: 0px !important;
        width: 100%;
        padding-bottom: 0px !important;
    }

    .stButton > button {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #ffffff !important;
        color: #000 !important;
        font-weight: bold !important;
        box-shadow: 1px 1px 0px #888 !important;
        height: 32px !important;
        min-height: 32px !important;
        padding: 0px !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    .stButton > button:hover {
        background-color: #f0f0f0 !important;
        border-color: #000 !important;
        color: #000 !important;
    }
    
    /* ============================================================ */
    /* 5. 辅助样式 */
    /* ============================================================ */
    div[data-testid="stSliderTickBar"], div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p { display: none !important; }
    .stSlider > div > div > div > div { height: 6px !important; background-color: #c0c0c0 !important; border: 1px solid #808080 !important; border-radius: 0px !important; }
    [data-testid="stSliderThumb"] { height: 18px !important; width: 18px !important; border-radius: 0px !important; background-color: #000000 !important; border: 1px solid #ffffff !important; top: -6px !important; }
    
    .metric-container { display: flex; justify-content: space-between; background-color: #f9f9f9; padding: 15px; border: 1px solid #000000; margin-bottom: 20px; }
    .metric-item { text-align: center; width: 33%; border-right: 1px solid #ccc; }
    .metric-item:last-child { border-right: none; }
    .metric-value { font-size: 1.6em; font-weight: bold; }
    .legend-text { font-size: 0.85rem !important; margin-right: 15px !important; }
</style>
""", unsafe_allow_html=True)

# --- 4. 动态 CSS 注入 (用于按钮高亮) ---
if st.session_state.ticker == "BTC-USD":
    st.markdown("""<style>div[data-testid="column"]:nth-of-type(1) div.stButton > button { background-color: #000000 !important; color: #ffffff !important; box-shadow: none !important; }</style>""", unsafe_allow_html=True)
elif st.session_state.ticker == "ETH-USD":
    st.markdown("""<style>div[data-testid="column"]:nth-of-type(2) div.stButton > button { background-color: #000000 !important; color: #ffffff !important; box-shadow: none !important; }</style>""", unsafe_allow_html=True)


# --- 5. 数据逻辑 ---
@st.cache_data(ttl=3600)
def get_data_and_calc(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: return pd.DataFrame(), "Error: No data returned."
        if isinstance(df.columns, pd.MultiIndex):
            if'Close' in df.columns.get_level_values(0): df = df.xs('Close', axis=1, level=0, drop_level=True)
            else: df.columns = df.columns.droplevel(1)
        if'Close' not in df.columns:
            if len(df.columns) == 1: df.columns = ['Close']
            else:
                close_cols = [c for c in df.columns if'Close' in str(c)]
                if close_cols: df = df[[close_cols[0]]].copy(); df.columns = ['Close']
        if'Close' not in df.columns: return pd.DataFrame(), "Error: Could not find Close price."
        df = df[['Close']].copy().sort_index()
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = df[~df.index.duplicated(keep='last')].dropna()
        df = df[df['Close'] > 0]
        df['Log_Price'] = np.log(df['Close'])
        df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
        genesis_date = pd.Timestamp("2009-01-03")
        df['Days'] = (df.index - genesis_date).days
        df = df[df['Days'] > 0]
        
        if ticker == "BTC-USD":
            slope = 5.84; intercept = -17.01
            log_days = np.log10(df['Days'])
            df['Predicted'] = 10 ** (slope * log_days + intercept)
            note = "Method: Power Law (Fixed)"
        else:
            valid_data = df.dropna()
            if len(valid_data) > 0:
                x = np.log10(valid_data['Days'].values); y = np.log10(valid_data['Close'].values)
                slope, intercept, _, _, _ = linregress(x, y)
                df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))
                note = f"Method: Dynamic Reg (Beta {slope:.4f})"
            else: df['Predicted'] = np.nan; note = "Insufficient Data"
        df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
        return df, note
    except Exception as e: return pd.DataFrame(), f"System Error: {str(e)}"

# --- 6. 页面布局 ---

st.markdown("""
<div style="text-align: center; margin-bottom: 20px; border-bottom: 2px solid #000; padding-bottom: 10px;">
    <h1 style="font-family: 'Courier New', Courier, monospace; text-transform: uppercase; letter-spacing: 2px; font-size: 2.2rem; margin: 0;">
        Statistical Deviation Monitor
    </h1>
    <div style="font-family: 'Times New Roman'; font-size: 0.9rem; margin-top: 5px;">
        SYSTEM STATUS: ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    c_btc, c_eth, c_reload, c_spacer, c_legend = st.columns([0.8, 0.8, 1, 0.5, 4.5])
    with c_btc:
        if st.button("BTC", use_container_width=True): st.session_state.ticker = "BTC-USD"; st.rerun()
    with c_eth:
        if st.button("ETH", use_container_width=True): st.session_state.ticker = "ETH-USD"; st.rerun()
    with c_reload:
        if st.button("RELOAD", use_container_width=True): st.cache_data.clear(); st.rerun()
    with c_legend:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-end; height: 32px;">
            <div style="display: flex; align-items: center; margin-left: 15px;">
                <span style="width:10px; height:10px; background-color:#28a745; border:1px solid black; margin-right:6px;"></span>
                <span class="legend-text"><b>L:</b> Buy</span>
            </div>
            <div style="display: flex; align-items: center; margin-left: 15px;">
                <span style="width:10px; height:10px; background-color:#007bff; border:1px solid black; margin-right:6px;"></span>
                <span class="legend-text"><b>M:</b> Accum</span>
            </div>
            <div style="display: flex; align-items: center; margin-left: 15px;">
                <span style="width:10px; height:10px; background-color:#dc3545; border:1px solid black; margin-right:6px;"></span>
                <span class="legend-text"><b>H:</b> Sell</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

min_date = datetime(2009, 1, 3).date()
max_date = datetime.today().date()
slider_key = f"slider_{st.session_state.ticker}"
if slider_key not in st.session_state: st.session_state[slider_key] = (max_date - timedelta(days=365), max_date)

st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

with st.container(border=True):
    c_start, c_end = st.columns([1, 1])
    current_val = st.session_state[slider_key]
    with c_start: st.markdown(f"<div style='padding-left:10px;'><b>{current_val[0].strftime('%Y/%m/%d')}</b></div>", unsafe_allow_html=True)
    with c_end: st.markdown(f"<div style='text-align: right; padding-right:10px;'><b>{current_val[1].strftime('%Y/%m/%d')}</b></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='padding: 0px 10px;'>", unsafe_allow_html=True)
    start_date, end_date = st.slider("Time Range", min_value=min_date, max_value=max_date, value=st.session_state[slider_key], format="YYYY/MM/DD", label_visibility="collapsed", key=slider_key)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

# --- 7. 核心分析展示 ---
with st.spinner("Processing data..."):
    df, note = get_data_and_calc(st.session_state.ticker)
    df_full = df.copy()
    if not df_full.empty:
        mask = (df_full.index >= pd.to_datetime(start_date)) & (df_full.index <= pd.to_datetime(end_date))
        df_display = df_full.loc[mask]
        
        if len(df_display) > 0:
            last = df_full.iloc[-1]
            ahr = last['AHR999'] if'AHR999' in last else 0
            price = last['Close']
            
            if ahr < 0.45: state = "ZONE L (Undershoot)"; css_class = "retro-color-green"
            elif 0.45 <= ahr <= 1.2: state = "ZONE M (Accumulation)"; css_class = "retro-color-blue"
            elif 1.2 < ahr <= 4.0: state = "ZONE N (Neutral)"; css_class = "retro-color-orange"
            else: state = "ZONE H (Overshoot)"; css_class = "retro-color-red"

            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item"><div class="retro-color-gray" style="font-size:0.9em;">CURRENT VALUE</div><div class="metric-value">${price:,.2f}</div></div>
                <div class="metric-item"><div class="retro-color-gray" style="font-size:0.9em;">DEVIATION INDEX</div><div class="metric-value {css_class}">{ahr:.4f}</div></div>
                <div class="metric-item"><div class="retro-color-gray" style="font-size:0.9em;">STATUS</div><div class="metric-value {css_class}">{state}</div></div>
            </div>
            """, unsafe_allow_html=True)

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.6, 0.4], subplot_titles=("Asset Value & Regression", "Deviation Index (DI)"))
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Close'], name="Value", line=dict(color="#000080", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['GeoMean'], name="Geo-Mean", line=dict(color="#555555", width=1.5, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Predicted'], name="Model", line=dict(color="#800080", width=1.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['AHR999'], name="DI Value", line=dict(color="#d35400", width=1.5)), row=2, col=1)
            fig.add_hline(y=0.45, line_color="green", line_dash="dash", row=2, col=1)
            fig.add_hline(y=1.2, line_color="blue", line_dash="dot", row=2, col=1)
            fig.add_hline(y=4.0, line_color="red", line_dash="dash", row=2, col=1)

            fig.update_layout(
                height=900,  
                template="plotly_white",
                font=dict(family="Times New Roman", size=14, color="#000"),
                
                # [核心修改]：减小左右 Margin (原 l=60, r=40)，让图表内容更宽
                margin=dict(l=20, r=20, t=140, b=100),
                
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(
                    orientation="h",
                    y=-0.1,
                    x=0.5,
                    xanchor="center",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="black",
                    borderwidth=1
                )
            )

            fig.update_annotations(yshift=15)  
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#eee', linecolor='black', mirror=True, rangeslider=dict(visible=False))
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', linecolor='black', mirror=True, type="log")

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown(f"""<div style="background-color: #f0f0f0; padding: 8px; border: 1px solid #000; margin-top: 15px; font-size: 0.9em;"><b>SYSTEM STATUS:</b> Ready | <b>DATA POINTS:</b> {len(df_display)} | <b>MODE:</b> {note}</div>""", unsafe_allow_html=True)
        else: st.warning("No data in selected range.")
    else: st.error(f"Unable to fetch data. Error details: {note}")
