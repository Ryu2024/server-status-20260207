import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="Data Dashboard", layout="wide")

# --- 2. 样式设置  ---
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
    
    h1, h2, h3, label {
        font-family: 'Times New Roman', Times, serif !important;
        color: #000000 !important;
    }
    
    p, div, span, li {
        font-family: 'Times New Roman', Times, serif !important;
    }
    
    /* 隐藏 Plotly 工具栏 */
    .modebar { display: none !important; }

    /* ============================================================ */
    /* 2. 容器样式 
    /* ============================================================ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #ffffff;
        /* 保留默认 padding，用下方 header 的负边距来贴边 */
    }

    /* ============================================================ */
    /* 3. 标题栏样式
    /* ============================================================ */
    .retro-header {
        background-color: #e0e0e0;
        color: #000000 !important;
        font-weight: bold;
        text-transform: uppercase;
        
        /* [核心魔法]：抵消 Streamlit 容器默认的内边距，让标题贴边 */
        margin-top: -16px !important;
        margin-left: -16px !important;
        margin-right: -16px !important;
        margin-bottom: 15px !important;
        
        /* 重新计算宽度，确保盖住左右 */
        width: calc(100% + 32px) !important;
        padding: 8px 0px;
        border-bottom: 1px solid #000000;
        font-size: 1rem;
        letter-spacing: 0.05em;
        text-align: center;
        line-height: 1.2;
    }
    
    .retro-content-pad {
        padding: 0px 10px;
    }

    /* ============================================================ */
    /* 4. 输入控件改造 (1px 细黑边) */
    /* ============================================================ */
    div[data-baseweb="select"] > div {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
        background-color: #ffffff !important;
        box-shadow: none !important;
        min-height: 38px;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        cursor: pointer !important;
    }
    div[data-baseweb="select"] svg {
        fill: #000000 !important;
    }
    div[data-baseweb="popover"] > div, div[data-baseweb="menu"] {
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
    }

    /* ============================================================ */
    /* 5. 时间滑块改造 */
    /* ============================================================ */
    /* 隐藏滑块自带数值 */
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
    
    /* 方形手柄 */
    [data-testid="stSliderThumb"] {
        height: 18px !important;
        width: 18px !important;
        border-radius: 0px !important;
        background-color: #000000 !important;
        border: 1px solid #ffffff !important;
        top: -6px !important;
    }
    [data-testid="stSliderThumb"]:focus { 
        box-shadow: none !important;
    }
    .stSlider > div > div > div > div > div {
         background-color: #666666 !important;
    }

    /* ============================================================ */
    /* 6. 指标卡片 (1px 细黑框) */
    /* ============================================================ */
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

    /* ============================================================ */
    /* 7. 按钮与杂项 */
    /* ============================================================ */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stButton>button {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #e0e0e0 !important;
        color: #000 !important;
        font-weight: bold !important;
        box-shadow: 1px 1px 0px #888 !important;
        min-height: 38px !important; 
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    .stButton>button:active {
        box-shadow: none !important;
        transform: translate(1px, 1px);
    }
    
    div[data-testid="stWidgetLabel"] p {
        font-size: 0.9rem !important;
        font-weight: bold !important;
        color: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 数据逻辑 (完全修复版) ---
@st.cache_data(ttl=3600)
def get_data_and_calc(ticker):
    try:
        # 下载数据
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        
        if df.empty:
            return pd.DataFrame(), "Error: No data returned from Yahoo Finance."

        # 清洗 MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            if'Close' in df.columns.get_level_values(0):
                 df = df.xs('Close', axis=1, level=0, drop_level=True)
            else:
                 df.columns = df.columns.droplevel(1)
        
        # 确保只有 Close 列
        if'Close' not in df.columns:
            if len(df.columns) == 1:
                df.columns = ['Close']
            else:
                # 尝试模糊匹配 Close
                close_cols = [c for c in df.columns if'Close' in str(c)]
                if close_cols:
                    df = df[[close_cols[0]]].copy()
                    df.columns = ['Close']
        
        # 最终检查 (这里是之前报错的地方，现在已修复)
        if'Close' not in df.columns:
             return pd.DataFrame(), f"Error: Could not find Close price. Columns: {df.columns.tolist()}"
             
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

st.title("Statistical Deviation Monitor")
st.markdown("---")

# 4.1 顶部配置
col_l, col_r = st.columns([1, 2], gap="large")

# 左侧：配置 (使用 border=True + 负边距 CSS)
with col_l:
    with st.container(border=True):
        st.markdown('<div class="retro-header">CONFIGURATION</div>', unsafe_allow_html=True)
        
        ticker = st.selectbox(
            "Target Asset", 
            options=["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD"],
            index=0
        )
        
        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)
        
        if st.button("RELOAD DATASET", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# 右侧：指南
with col_r:
    with st.container(border=True):
        st.markdown('<div class="retro-header">REFERENCE GUIDE</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="retro-content-pad" style="font-size: 0.95rem;">
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

# 4.2 时间选择器
min_date = datetime(2009, 1, 3).date()
max_date = datetime.today().date()
default_start = datetime(2020, 1, 1).date()
default_end = max_date

slider_key = f"slider_{ticker}"
if slider_key not in st.session_state:
    st.session_state[slider_key] = (default_start, default_end)

with st.container(border=True):
    st.markdown('<div class="retro-header">TIME RANGE SLICER</div>', unsafe_allow_html=True)
    
    # 左右日期显示
    c_start, c_end = st.columns([1, 1])
    current_val = st.session_state[slider_key]
    start_str = current_val[0].strftime("%Y/%m/%d")
    end_str = current_val[1].strftime("%Y/%m/%d")
    
    with c_start:
        st.markdown(f"<div style='padding-left:0px;'><b>{start_str}</b></div>", unsafe_allow_html=True)
    with c_end:
        st.markdown(f"<div style='text-align: right; padding-right:0px;'><b>{end_str}</b></div>", unsafe_allow_html=True)
    
    start_date, end_date = st.slider(
        "Time Range",
        min_value=min_date,
        max_value=max_date,
        value=st.session_state[slider_key],
        format="YYYY/MM/DD",
        label_visibility="collapsed",
        key=slider_key
    )

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
            
            # 状态颜色逻辑
            if ahr < 0.45:
                state = "ZONE L (Undershoot)"
                color = "#28a745"
            elif 0.45 <= ahr <= 1.2:
                state = "ZONE M (Accumulation)"
                color = "#007bff"
            elif 1.2 < ahr <= 4.0:
                state = "ZONE N (Neutral)"
                color = "#fd7e14"
            else:
                state = "ZONE H (Overshoot)"
                color = "#dc3545"

            # 指标卡片
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-item">
                    <div style="font-size:0.9em; color:#666;">CURRENT VALUE</div>
                    <div class="metric-value">${price:,.2f}</div>
                </div>
                <div class="metric-item">
                    <div style="font-size:0.9em; color:#666;">DEVIATION INDEX</div>
                    <div class="metric-value" style="color: {color}">{ahr:.4f}</div>
                </div>
                <div class="metric-item">
                    <div style="font-size:0.9em; color:#666;">STATUS</div>
                    <div class="metric-value" style="color: {color}">{state}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 图表
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

            # --- 关键修改：调整边距和图例位置 ---
            fig.update_layout(
                height=700,
                template="plotly_white",
                font=dict(family="Times New Roman", size=14, color="#000"),
                # t=50: 防止标题顶到边框; b=80: 给底部图例留出空间
                margin=dict(l=40, r=40, t=50, b=80), 
                plot_bgcolor="white",
                paper_bgcolor="white",
                # 图例移至最下方，避免覆盖图表或标题
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

            # 纯净图表 (已移除外框 div)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            
            st.markdown(f"""
            <div style="background-color: #f0f0f0; padding: 8px; border: 1px solid #000; margin-top: 15px; font-size: 0.9em;">
                <b>SYSTEM STATUS:</b> Ready | <b>DATA POINTS:</b> {len(df_display)} | <b>MODE:</b> {note}
            </div>
            """, unsafe_allow_html=True)
            
        else:
             st.warning("No data in selected range.")
    else:
        st.error(f"Unable to fetch data. Error details: {note}")
