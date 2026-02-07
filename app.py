import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置与基础样式 ---
st.set_page_config(page_title="Pro Quant Monitor", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    div[data-testid="block-container"] { max-width: 1250px; padding: 1.5rem 2rem; margin: 0 auto; }
    .modebar { display: none !important; }
    /* 强制隐藏 Streamlit 的默认交互元素 */
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 2. 高级数据引擎 ---
@st.cache_data(ttl=3600)
def get_quant_data():
    assets = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
    processed = {}
    
    for ticker, name in assets.items():
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if df.empty: continue
        
        # 统一清洗
        if isinstance(df.columns, pd.MultiIndex): df = df.xs('Close', axis=1, level=0)
        df = df[['Close']].copy().dropna().sort_index()
        df = df[df['Close'] > 0]
        
        # 指标计算
        df['Log_P'] = np.log(df['Close'])
        df['Geo'] = np.exp(df['Log_P'].rolling(200).mean())
        df['Days'] = (df.index - pd.Timestamp("2009-01-03")).days
        df = df[df['Days'] > 0]
        
        # 回归模型
        if name == "BTC": slope, intercept = 5.84, -17.01
        else:
            v = df.dropna(); x = np.log10(v['Days'].values); y = np.log10(v['Close'].values)
            slope, intercept, _, _, _ = linregress(x, y)
        
        df['Model'] = 10 ** (intercept + slope * np.log10(df['Days']))
        df['DI'] = (df['Close'] / df['Geo']) * (df['Close'] / df['Model'])
        processed[name] = df
    return processed

# --- 3. 构建专业级图表 ---
with st.spinner("QUANT ENGINE INITIALIZING..."):
    data = get_quant_data()
    if data:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
            row_heights=[0.72, 0.28]
        )

        # 预定义颜色
        colors = {'BTC': '#1e1e1e', 'ETH': '#3c3cbb', 'Model': '#a905b6', 'DI': '#e67e22'}

        # --- Trace 注入逻辑 ---
        for i, (name, df) in enumerate(data.items()):
            is_visible = (name == "BTC")
            # 价格线
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Close'], name=f"{name} Price",
                visible=is_visible, line=dict(color=colors[name], width=1.8)
            ), row=1, col=1)
            # 模型线
            fig.add_trace(go.Scatter(
                x=df.index, y=df['Model'], name=f"{name} Model",
                visible=is_visible, line=dict(color=colors['Model'], width=1.2, dash='dot')
            ), row=1, col=1)
            # DI 偏离值
            fig.add_trace(go.Scatter(
                x=df.index, y=df['DI'], name=f"{name} DI",
                visible=is_visible, line=dict(color=colors['DI'], width=1.5)
            ), row=2, col=1)

        # --- 背景带 (Rectangles) ---
        # 为下方的 DI 图表添加背景颜色带，直观识别区间
        fig.add_hrect(y0=0, y1=0.45, fillcolor="rgba(40, 167, 69, 0.1)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0.45, y1=1.2, fillcolor="rgba(0, 123, 255, 0.05)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=4.0, y1=10.0, fillcolor="rgba(220, 53, 69, 0.1)", line_width=0, row=2, col=1)

        # --- 交互 UI 控件 ---
        updatemenus = [
            # 币种切换按钮
            dict(
                type="buttons", direction="right", x=0, y=1.12,
                showactive=True, active=0,
                bgcolor="white", bordercolor="#444", borderwidth=1,
                font=dict(size=12, family="Arial", color="#000"),
                buttons=[
                    dict(label="   BITCOIN   ", method="update", args=[{"visible": [True, True, True, False, False, False]}]),
                    dict(label="   ETHEREUM   ", method="update", args=[{"visible": [False, False, False, True, True, True]}])
                ]
            )
        ]

        # --- 全局布局调整 ---
        fig.update_layout(
            height=900,
            template="plotly_white",
            updatemenus=updatemenus,
            margin=dict(l=50, r=20, t=130, b=50),
            hovermode="x unified",
            # 图表标题与辅助信息
            annotations=[
                dict(text="STATISTICAL DEVIATION MONITOR", font_size=24, xref="paper", yref="paper", x=0.5, y=1.18, showarrow=False, font_family="Courier New", font_color="black"),
                dict(text="L: Value Zone | M: Accumulation | H: Distribution", font_size=11, xref="paper", yref="paper", x=1, y=1.06, showarrow=False)
            ],
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            # 时间选择器
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ]),
                    y=1.02, x=0, bgcolor="rgba(255,255,255,0.8)"
                ),
                rangeslider=dict(visible=True, thickness=0.06),
                type="date",
                gridcolor="#f0f0f0"
            )
        )

        fig.update_yaxes(type="log", gridcolor="#f0f0f0", row=1, col=1, title="Price (USD)")
        fig.update_yaxes(gridcolor="#f0f0f0", row=2, col=1, title="DI Index", range=[0, 6])

        # --- 渲染 ---
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # 自定义状态页脚
        st.markdown("""
        <div style="border-top: 1px solid #000; padding: 10px; font-size: 0.8rem; font-family: 'Courier New'; display: flex; justify-content: space-between; opacity: 0.6;">
            <span>QUANT_SYSTEM_V2 // NO_REFRESH_ACTIVE</span>
            <span>ENGINE: PLOTLY_JS_6.4</span>
        </div>
        """, unsafe_allow_html=True)
