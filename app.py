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

# --- 2. 样式设置 ---
st.markdown("""
<style>
    /* ============================================================ */
    /* 1. 全局字体与基础 */
    /* ============================================================ */
    .stApp {
        background-color: #ffffff;
        font-family: 'Times New Roman', Times, serif;
        color: #000000;
    }
    
    /* 全局强制黑色 */
    h1, h2, h3, label, p, div, span, li {
        font-family: 'Times New Roman', Times, serif !important;
        color: #000000 !important;
    }
    
    .modebar { display: none !important; }

    /* ============================================================ */
    /* 2. [核心修复] 专用颜色类 (覆盖全局黑色) */
    /* ============================================================ */
    .retro-color-green, .retro-color-green * { color: #28a745 !important; }
    .retro-color-blue, .retro-color-blue * { color: #007bff !important; }
    .retro-color-orange, .retro-color-orange * { color: #fd7e14 !important; }
    .retro-color-red, .retro-color-red * { color: #dc3545 !important; }
    .retro-color-gray, .retro-color-gray * { color: #666666 !important; }

    /* ============================================================ */
    /* 3. 容器与边框 (统一控制) + [Flexbox 自动高度对齐] */
    /* ============================================================ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        background-color: #ffffff !important;
        padding: 0px !important;
        overflow: visible !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > * {
        border-radius: 0px !important;
    }

    /* [CSS 魔法]：强制左右两列的容器高度自动拉伸对齐 */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch !important;
    }
    div[data-testid="column"] {
        display: flex !important;
        flex-direction: column !important;
    }
    /* 让 Column 内部的黑框容器自动填充剩余高度 */
    div[data-testid="column"] > div > div > div > div[data-testid="stVerticalBlockBorderWrapper"] {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
    }
    div[data-testid="column"] div[data-testid="stVerticalBlock"] {
        flex-grow: 1 !important;
    }

    /* ============================================================ */
    /* 4. [调整点] 原生容器内的标题样式 */
    /* ============================================================ */
    .retro-header-native {
        background-color: #e0e0e0;
        color: #000000 !important;
        font-weight: bold;
        text-transform: uppercase;
        
        margin-top: 0px !important;
        margin-left: 0px !important;
        margin-right: 0px !important;
        
        /* [修改点]：将标题下方的间距由 10px 缩减为 5px，与下方对齐 */
        margin-bottom: 5px !important;
        
        width: 100% !important;
        padding: 8px 0px;
        border-bottom: 1px solid #000000;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-align: center;
        line-height: 1.2;
    }
    
    /* 容器内边距 */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] {
        padding-left: 10px; padding-right: 10px;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] .stElementContainer {
        padding-left: 10px; padding-right: 10px;
    }

    /* ============================================================ */
    /* 5. [调整点] Selectbox 间距优化 */
    /* ============================================================ */
    div[data-baseweb="select"] > div {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #ffffff !important;
        box-shadow: none !important;
        min-height: 32px !important;
    }
    div[data-baseweb="popover"] > div, div[data-baseweb="menu"] {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
    }
    
    /* [修改点]：针对 Selectbox 容器的间距控制 */
    div[data-testid="stSelectbox"] { 
        /* 强制去除顶部可能存在的间距 */
        margin-top: 0px !important;
        /* 下方间距设为 5px，与 Header 下方间距保持完全一致 */
        margin-bottom: 5px !important;
    }
    
    /* [修改点]：彻底隐藏 Label 元素，防止即使 collapsed 也占位 */
    div[data-testid="stSelectbox"] label {
        display: none !important;
    }

    /* ============================================================ */
    /* 6. [调整点] 按钮样式 */
    /* ============================================================ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton {
        /* 确保按钮顶部紧贴上方的 Selectbox (由 Selectbox 的 margin-bottom: 5px 控制间距) */
        margin-top: 0px !important;
        width: 100%;
        padding-bottom: 10px !important;
    }

    .stButton>button {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #e0e0e0 !important;
        color: #000 !important;
        font-weight: bold !important;
        box-shadow: 1px 1px 0px #888 !important;
        height: auto !important;
        min-height: 22px !important;
        padding-top: 3px !important;
        padding-bottom: 3px !important;
        font-size: 0.75rem !important; 
        line-height: 1.1 !important;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:active {
        box-shadow: none !important;
        transform: translate(1px, 1px);
    }
    .stButton>button:hover {
        background-color: #eaeaea !important;
        border-color: #000 !important;
        color: #000 !important;
    }

    /* 其他组件样式 (Slider, Metric) */
    div[data-testid="stSliderTickBar"],
    div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        display: none !important;
    }
    .stSlider > div > div > div > div {
        height: 6px !important;
        background-color: #c0c0c0 !important;
        border: 1px solid #808080 !important;
        border-radius: 0px !important;
    }
    [data-testid="stSliderThumb"] {
        height: 18px !important;
        width: 18px !important;
        border-radius: 0px !important;
        background-color: #000000 !important;
        border: 1px solid #ffffff !important;
        top: -6px !important;
    }
    
    .metric-container {
        display: flex;
        justify-content: space-between;
        background-color: #f9f9f9;
        padding: 15px;
        border: 1px solid #000000;
        margin-bottom: 20px;
    }
    .metric-item {
        text-align: center;
        width: 33%;
        border-right: 1px solid #ccc;
    }
    .metric-item:last-child { border-right: none; }
    .metric-value { 
        font-size: 1.6em; 
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 数据逻辑 ---
@st.cache_data(ttl=3600)
def get_data_and_calc(ticker):
    try:
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        
        if df.empty:
            return pd.DataFrame(), "Error: No data returned."

        if isinstance(df.columns, pd.MultiIndex):
            if'Close' in df.columns.get_level_values(0):
                df = df.xs('Close', axis=1, level=0, drop_level=True)
            else:
                 df.columns = df.columns.droplevel(1)
        
        if'Close' not in df.columns:
            if len(df.columns) == 1:
                df.columns = ['Close']
            else:
                close_cols = [c for c in df.columns if'Close' in str(c)]
                if close_cols:
                    df = df[[close_cols[0]]].copy()
                    df.columns = ['Close']
        
        if'Close' not in df.columns:
             return pd.DataFrame(), "Error: Could not find Close price."
             
        df = df[['Close']].copy().sort_index()
        
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        df = df[~df.index.duplicated(keep='last')]
        df = df.dropna()
        df = df[df['Close'] > 0]
        
        df['Log_Price'] = np.log(df['Close'])
        df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
        
        genesis_date = pd.Timestamp("2009-01-03")
        df['Days'] = (df.index - genesis_date).days
        df = df[df['Days'] > 0]
        
        if ticker == "BTC-USD":
            slope = 5.84
            intercept = -17.01
            log_days = np.log10(df['Days'])
            df['Predicted'] = 10 ** (slope * log_days + intercept)
            note = "Method: Power Law (Fixed)"
        else:
            valid_data = df.dropna()
            if len(valid_data) > 0:
                x = np.log10(valid_data['Days'].values)
                y = np.log10(valid_data['Close'].values)
                slope, intercept, _, _, _ = linregress(x, y)
                df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))
                note = f"Method: Dynamic Reg (Beta {slope:.4f})"
            else:
                df['Predicted'] = np.nan
                note = "Insufficient Data"

        df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
        return df, note
    except Exception as e:
        return pd.DataFrame(), f"System Error: {str(e)}"

# --- 4. 页面布局 ---

st.markdown("""
<div style="
    text-align: center;
    margin-bottom: 30px;
    border-bottom: 2px solid #000; 
    padding-bottom: 10px;">
    <h1 style="
        font-family: 'Courier New', Courier, monospace;
        text-transform: uppercase; 
        letter-spacing: 2px; 
        font-size: 2.2rem; 
        margin: 0;">
        Statistical Deviation Monitor
    </h1>
    <div style="font-family: 'Times New Roman';
        font-size: 0.9rem; margin-top: 5px;">
        SYSTEM STATUS: ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns([1, 2], gap="large")

# 左侧：配置 (高度将自动拉伸以对齐右侧)
with col_l:
    with st.container(border=True):
        st.markdown('<div class="retro-header-native">CONFIGURATION</div>', unsafe_allow_html=True)
        
        # Selectbox: 文字隐藏，样式紧凑
        ticker = st.selectbox(
            "Target Asset", 
            options=["BTC-USD", "ETH-USD"],
            index=0,
            label_visibility="collapsed"
        )
        
        # Button: 样式紧凑
        if st.button("RELOAD DATASET", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# 右侧：指南
with col_r:
    with st.container(border=True):
        st.markdown('<div class="retro-header-native">REFERENCE GUIDE</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="padding-bottom: 10px;">
            <div style="margin-bottom: 10px; display: flex; align-items: center;">
                <span style="display:inline-block; width:12px; height:12px; background-color:#28a745; border:1px solid black; margin-right:10px;"></span>
                <span><b>L-Line (0.45):</b> Lower statistical bound. Historical buy zone.</span>
            </div>
            <div style="margin-bottom: 10px; display: flex; align-items: center;">
                <span style="display:inline-block; width:12px; height:12px; background-color:#007bff; border:1px solid black; margin-right:10px;"></span>
                <span><b>M-Line (1.20):</b> Mean accumulation threshold.</span>
            </div>
            <div style="margin-bottom: 0px; display: flex; align-items: center;">
                <span style="display:inline-block; width:12px; height:12px; background-color:#dc3545; border:1px solid black; margin-right:10px;"></span>
                <span><b>H-Line (4.00):</b> Upper statistical bound. Variance warning.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# 时间选择器
min_date = datetime(2009, 1, 3).date()
max_date = datetime.today().date()
one_year_ago = max_date - timedelta(days=365)
default_start = one_year_ago
default_end = max_date

slider_key = f"slider_{ticker}"
if slider_key not in st.session_state:
    st.session_state[slider_key] = (default_start, default_end)

with st.container(border=True):
    st.markdown('<div class="retro-header-native">TIME RANGE SLICER</div>', unsafe_allow_html=True)
    
    c_start, c_end = st.columns([1, 1])
    current_val = st.session_state[slider_key]
    
    with c_start:
        st.markdown(f"<div style='padding-left:10px;'><b>{current_val[0].strftime('%Y/%m/%d')}</b></div>", unsafe_allow_html=True)
    with c_end:
        st.markdown(f"<div style='text-align: right; padding-right:10px;'><b>{current_val[1].strftime('%Y/%m/%d')}</b></div>", unsafe_allow_html=True)
    
    st.markdown("<div style='padding: 0px 10px;'>", unsafe_allow_html=True)
    start_date, end_date = st.slider(
        "Time Range",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state[slider_key],
        format="YYYY/MM/DD",
        label_visibility="collapsed",
        key=slider_key
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

# --- 5. 核心分析展示 ---

with st.spinner("Processing data..."):
    df, note = get_data_and_calc(ticker)
    
    df_full = df.copy()
    
    if not df_full.empty:
        mask = (df_full.index >= pd.to_datetime(start_date)) & (df_full.index <= pd.to_datetime(end_date))
        df_display = df_full.loc[mask]
        
        if len(df_display) > 0:
            last = df_full.iloc[-1]
            ahr = last['AHR999'] if'AHR999' in last else 0
            price = last['Close']
            
            # 状态逻辑
            if ahr < 0.45:
                state = "ZONE L (Undershoot)"
                css_class = "retro-color-green"
                color_hex = "#28a745"
            elif 0.45 <= ahr <= 1.2:
                state = "ZONE M (Accumulation)"
                css_class = "retro-color-blue"
                color_hex = "#007bff"
            elif 1.2 < ahr <= 4.0:
                state = "ZONE N (Neutral)"
                css_class = "retro-color-orange"
                color_hex = "#fd7e14"
            else:
                state = "ZONE H (Overshoot)"
                css_class = "retro-color-red"
                color_hex = "#dc3545"

            # 指标卡片
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div class="retro-color-gray" style="font-size:0.9em;">CURRENT VALUE</div>
                    <div class="metric-value">${price:,.2f}</div>
                </div>
                <div class="metric-item">
                    <div class="retro-color-gray" style="font-size:0.9em;">DEVIATION INDEX</div>
                    <div class="metric-value {css_class}">{ahr:.4f}</div>
                </div>
                <div class="metric-item">
                    <div class="retro-color-gray" style="font-size:0.9em;">STATUS</div>
                    <div class="metric-value {css_class}">{state}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fig = make_subplots(
                rows=2, cols=1, 
                shared_xaxes=True, 
                vertical_spacing=0.1, 
                row_heights=[0.6, 0.4],
                subplot_titles=("Asset Value & Regression", "Deviation Index (DI)"),
            )

            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Close'], name="Value", line=dict(color="#000080", width=1.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['GeoMean'], name="Geo-Mean", line=dict(color="#555555", width=1.5, dash='dot')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['Predicted'], name="Model", line=dict(color="#800080", width=1.5, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_display.index, y=df_display['AHR999'], name="DI Value", line=dict(color="#d35400", width=1.5)), row=2, col=1)

            fig.add_hline(y=0.45, line_color="green", line_dash="dash", row=2, col=1)
            fig.add_hline(y=1.2, line_color="blue", line_dash="dot", row=2, col=1)
            fig.add_hline(y=4.0, line_color="red", line_dash="dash", row=2, col=1)

            fig.update_layout(
                height=700,
                template="plotly_white",
                font=dict(family="Times New Roman", size=14, color="#000"),
                margin=dict(l=40, r=40, t=50, b=80), 
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(
                    orientation="h", 
                    y=-0.15, 
                    x=0.5, 
                    xanchor="center",
                    bgcolor="rgba(255,255,255,0.8)", 
                    bordercolor="black", 
                    borderwidth=1
                )
            )
            
            fig.update_xaxes(
                showgrid=True, gridwidth=1, gridcolor='#eee', linecolor='black', mirror=True,
                rangeslider=dict(visible=False) 
            )
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#eee', linecolor='black', mirror=True, type="log")

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown(f"""
            <div style="background-color: #f0f0f0;
                padding: 8px; border: 1px solid #000; margin-top: 15px; font-size: 0.9em;">
                <b>SYSTEM STATUS:</b> Ready | <b>DATA POINTS:</b> {len(df_display)} | <b>MODE:</b> {note}
            </div>
            """, unsafe_allow_html=True)
            
        else:
             st.warning("No data in selected range.")
    
    else:
        st.error(f"Unable to fetch data. Error details: {note}")
