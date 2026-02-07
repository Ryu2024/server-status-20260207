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

# --- 2. 样式设置 (保留你的 Retro 风格) ---
st.markdown("""
<style>
    .stApp { background-color: #ffffff; font-family: 'Times New Roman', serif; color: #000; }
    div[data-testid="block-container"] { max-width: 1200px; padding-top: 2rem; padding-bottom: 2rem; margin: 0 auto; }
    /* 隐藏 Streamlit 默认的图表工具栏，让界面更像原生 App */
    .modebar { display: none !important; }
    h1, div { font-family: 'Times New Roman', serif !important; color: #000 !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. 数据处理函数 (同时获取并计算 BTC 和 ETH) ---
@st.cache_data(ttl=3600)
def get_all_data():
    tickers = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
    data_store = {}
    
    for ticker, name in tickers.items():
        try:
            df = yf.download(ticker, period="max", interval="1d", progress=False)
            if df.empty: continue
            
            # 清洗数据
            if isinstance(df.columns, pd.MultiIndex):
                if'Close' in df.columns.get_level_values(0): df = df.xs('Close', axis=1, level=0, drop_level=True)
                else: df.columns = df.columns.droplevel(1)
            if'Close' not in df.columns:
                if len(df.columns) == 1: df.columns = ['Close']
                else: continue
            
            df = df[['Close']].copy().sort_index().dropna()
            df = df[df['Close'] > 0]
            
            # 计算指标
            df['Log_Price'] = np.log(df['Close'])
            df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
            
            genesis_date = pd.Timestamp("2009-01-03")
            df['Days'] = (df.index - genesis_date).days
            df = df[df['Days'] > 0]
            
            # 回归模型
            if ticker == "BTC-USD":
                slope, intercept = 5.84, -17.01 # 固定参数
            else:
                # 动态回归
                valid = df.dropna()
                x = np.log10(valid['Days'].values)
                y = np.log10(valid['Close'].values)
                slope, intercept, _, _, _ = linregress(x, y)
                
            log_days = np.log10(df['Days'])
            df['Predicted'] = 10 ** (intercept + slope * log_days)
            df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
            
            data_store[name] = df
        except:
            continue
            
    return data_store

# --- 4. 页面标题 ---
st.markdown("""
<div style="text-align: center; margin-bottom: 10px; border-bottom: 2px solid #000; padding-bottom: 10px;">
    <h1 style="font-family: 'Courier New', Courier, monospace; text-transform: uppercase; letter-spacing: 2px; font-size: 2.2rem; margin: 0;">
        Statistical Deviation Monitor
    </h1>
    <div style="font-family: 'Times New Roman'; font-size: 0.9rem; margin-top: 5px;">
        INTERACTIVE MODE (PLOTLY ENGINE)
    </div>
</div>
""", unsafe_allow_html=True)

# --- 5. 构建全交互图表 ---
with st.spinner("Initializing Plotly Engine..."):
    data = get_all_data()
    
    if "BTC" in data and "ETH" in data:
        # 创建子图结构
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.7, 0.3],
            subplot_titles=("Asset Price Model", "Deviation Index")
        )

        # ==========================================================
        # 1. 添加所有 Trace (先把所有线条都画上去)
        # ==========================================================
        
        # --- BTC Traces (默认显示) ---
        btc = data['BTC']
        # Trace 0: BTC Price
        fig.add_trace(go.Scatter(x=btc.index, y=btc['Close'], name="BTC Price", visible=True, line=dict(color="#000000", width=1.5)), row=1, col=1)
        # Trace 1: BTC Model
        fig.add_trace(go.Scatter(x=btc.index, y=btc['Predicted'], name="BTC Model", visible=True, line=dict(color="#800080", width=1.5, dash='dash')), row=1, col=1)
        # Trace 2: BTC AHR999
        fig.add_trace(go.Scatter(x=btc.index, y=btc['AHR999'], name="BTC Dev", visible=True, line=dict(color="#d35400", width=1.5)), row=2, col=1)

        # --- ETH Traces (默认隐藏 visible=False) ---
        eth = data['ETH']
        # Trace 3: ETH Price
        fig.add_trace(go.Scatter(x=eth.index, y=eth['Close'], name="ETH Price", visible=False, line=dict(color="#000080", width=1.5)), row=1, col=1)
        # Trace 4: ETH Model
        fig.add_trace(go.Scatter(x=eth.index, y=eth['Predicted'], name="ETH Model", visible=False, line=dict(color="#800080", width=1.5, dash='dash')), row=1, col=1)
        # Trace 5: ETH AHR999
        fig.add_trace(go.Scatter(x=eth.index, y=eth['AHR999'], name="ETH Dev", visible=False, line=dict(color="#2980b9", width=1.5)), row=2, col=1)

        # 添加参考线 (永远显示)
        fig.add_hline(y=0.45, line_color="green", line_dash="dash", row=2, col=1)
        fig.add_hline(y=1.2, line_color="blue", line_dash="dot", row=2, col=1)
        fig.add_hline(y=4.0, line_color="red", line_dash="dash", row=2, col=1)

        # ==========================================================
        # 2. 定义 Buttons (Update Menus)
        # ==========================================================
        # 逻辑：点击 BTC，设置 visible=[True, True, True, False, False, False]
        # 逻辑：点击 ETH，设置 visible=[False, False, False, True, True, True]
        
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                x=0.0, y=1.16, # 按钮位置 (图表左上角上方)
                showactive=True,
                bgcolor="#ffffff",
                bordercolor="#000000",
                borderwidth=1,
                font=dict(family="Courier New", size=12, color="#000"),
                buttons=list([
                    dict(
                        label="BTC-USD",
                        method="update",
                        args=[{"visible": [True, True, True, False, False, False]}, # 控制 Trace 可见性
                              {"title": "Bitcoin Logarithmic Regression"}]          # 同时更新标题
                    ),
                    dict(
                        label="ETH-USD",
                        method="update",
                        args=[{"visible": [False, False, False, True, True, True]},
                              {"title": "Ethereum Logarithmic Regression"}]
                    ),
                ]),
            )
        ]

        # ==========================================================
        # 3. 布局设置 (包含 Range Slider)
        # ==========================================================
        fig.update_layout(
            height=850,
            template="plotly_white",
            updatemenus=updatemenus,
            title_text="Bitcoin Logarithmic Regression", # 默认标题
            title_x=0.5,
            title_y=0.98,
            title_font=dict(family="Times New Roman", size=20),
            font=dict(family="Times New Roman", size=14, color="#000"),
            margin=dict(l=40, r=40, t=100, b=80), # 给按钮留出顶部空间
            
            # --- 核心：原生时间选择器 ---
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ]),
                    bgcolor="#f0f0f0",
                    bordercolor="#000",
                    borderwidth=1,
                    font=dict(family="Courier New", size=11)
                ),
                rangeslider=dict(
                    visible=True, # 开启底部滑块
                    thickness=0.08,
                    bgcolor="#f9f9f9",
                    bordercolor="#ccc"
                ),
                type="date"
            ),
            
            legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
        )

        fig.update_yaxes(type="log", row=1, col=1, gridcolor='#eee', zeroline=False)
        fig.update_yaxes(gridcolor='#eee', zeroline=False, row=2, col=1)
        fig.update_xaxes(gridcolor='#eee', showgrid=True)

        # 渲染图表
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # 底部状态栏 (静态)
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 10px; border-top: 2px solid #000; font-size: 0.9em; display: flex; justify-content: space-between;">
            <span><b>SYSTEM:</b> Plotly Native Mode</span>
            <span><b>NOTE:</b> Use chart buttons to switch assets. Use bottom slider to zoom.</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Failed to load market data.")
