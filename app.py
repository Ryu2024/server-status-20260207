import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. 页面配置 ---
st.set_page_config(page_title="Pro Quant Monitor", layout="wide")
st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    div[data-testid="block-container"] { max-width: 1250px; padding: 1rem 2rem; margin: 0 auto; }
    .modebar { display: none !important; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --- 2. 数据处理与状态计算 ---
@st.cache_data(ttl=3600)
def get_quant_data():
    assets = {"BTC-USD": "BTC", "ETH-USD": "ETH"}
    data_store = {}
    
    for ticker, name in assets.items():
        try:
            df = yf.download(ticker, period="max", interval="1d", progress=False)
            if df.empty: continue
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
            
            # --- 提取最新状态 (用于 HUD 显示) ---
            last = df.iloc[-1]
            price = last['Close']
            di = last['DI']
            
            if di < 0.45: status = "BUY ZONE (Undervalued)"; color = "#28a745" # Green
            elif 0.45 <= di <= 1.2: status = "ACCUMULATION"; color = "#007bff" # Blue
            elif 1.2 < di <= 4.0: status = "NEUTRAL / HOLD"; color = "#fd7e14" # Orange
            else: status = "SELL ZONE (Overheated)"; color = "#dc3545" # Red
            
            data_store[name] = {
                "df": df,
                "info": {"price": price, "di": di, "status": status, "color": color}
            }
        except: continue
    return data_store

# --- 3. 辅助函数：生成 HUD 注解 ---
def create_hud_annotations(info, asset_name):
    """
    生成一套注解列表，包含：标题、价格、DI值、状态
    """
    price_str = f"${info['price']:,.2f}"
    di_str = f"{info['di']:.4f}"
    status_str = info['status']
    status_color = info['color']
    
    return [
        # 1. 左上角：资产名称
        dict(text=f"{asset_name} / USD", font=dict(size=14, color="#666"), x=0, y=1.25, xref="paper", yref="paper", showarrow=False, xanchor="left"),
        # 2. 左上角：大号价格
        dict(text=price_str, font=dict(size=32, color="#000", family="Arial Black"), x=0, y=1.15, xref="paper", yref="paper", showarrow=False, xanchor="left"),
        
        # 3. 中间偏右：DI 指标
        dict(text="DEVIATION INDEX", font=dict(size=10, color="#666"), x=0.45, y=1.25, xref="paper", yref="paper", showarrow=False),
        dict(text=di_str, font=dict(size=28, color=status_color, family="Arial"), x=0.45, y=1.15, xref="paper", yref="paper", showarrow=False),
        
        # 4. 右侧：状态
        dict(text="CURRENT STATUS", font=dict(size=10, color="#666"), x=1, y=1.25, xref="paper", yref="paper", showarrow=False, xanchor="right"),
        dict(text=status_str, font=dict(size=20, color="white"), bgcolor=status_color, borderpad=4, bordercolor=status_color, borderwidth=1, x=1, y=1.16, xref="paper", yref="paper", showarrow=False, xanchor="right"),
        
        # 5. 底部图例子标题 (固定)
        dict(text="Price Action & Regression", x=0, y=1.02, xref="paper", yref="paper", showarrow=False, xanchor="left", font=dict(size=12, color="#888")),
        dict(text="Deviation Oscillator", x=0, y=0.22, xref="paper", yref="paper", showarrow=False, xanchor="left", font=dict(size=12, color="#888")),
    ]

# --- 4. 构建可视化 ---
with st.spinner("CALCULATING MARKET METRICS..."):
    data = get_quant_data()
    
    if "BTC" in data and "ETH" in data:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15, row_heights=[0.7, 0.3])

        # --- A. 添加 Traces ---
        # 1. BTC (Visible by default)
        btc_df = data['BTC']['df']
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['Close'], visible=True, line=dict(color="#111", width=1.5), name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['Model'], visible=True, line=dict(color="#d63031", dash='dot', width=1), name="Reg Model"), row=1, col=1)
        fig.add_trace(go.Scatter(x=btc_df.index, y=btc_df['DI'], visible=True, line=dict(color="#e17055", width=1.5), name="DI"), row=2, col=1)

        # 2. ETH (Hidden by default)
        eth_df = data['ETH']['df']
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['Close'], visible=False, line=dict(color="#0984e3", width=1.5), name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['Model'], visible=False, line=dict(color="#6c5ce7", dash='dot', width=1), name="Reg Model"), row=1, col=1)
        fig.add_trace(go.Scatter(x=eth_df.index, y=eth_df['DI'], visible=False, line=dict(color="#00b894", width=1.5), name="DI"), row=2, col=1)

        # --- B. 背景色块 (区间) ---
        fig.add_hrect(y0=0, y1=0.45, fillcolor="rgba(46, 204, 113, 0.15)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=0.45, y1=1.2, fillcolor="rgba(52, 152, 219, 0.1)", line_width=0, row=2, col=1)
        fig.add_hrect(y0=4.0, y1=10.0, fillcolor="rgba(231, 76, 60, 0.15)", line_width=0, row=2, col=1)
        
        # --- C. 按钮交互 (核心魔法) ---
        # 预先生成两套 Annotations
        btc_ann = create_hud_annotations(data['BTC']['info'], "BITCOIN")
        eth_ann = create_hud_annotations(data['ETH']['info'], "ETHEREUM")
        
        updatemenus = [dict(
            type="buttons",
            direction="right",
            x=0.5, y=1.08, xanchor="center", # 按钮放在标题和图表之间
            bgcolor="white", bordercolor="#ddd", borderwidth=1,
            buttons=[
                dict(
                    label=" BTC ",
                    method="update",
                    args=[
                        {"visible": [True, True, True, False, False, False]}, # Traces
                        {"annotations": btc_ann} # Layout: 替换所有注解
                    ]
                ),
                dict(
                    label=" ETH ",
                    method="update",
                    args=[
                        {"visible": [False, False, False, True, True, True]}, # Traces
                        {"annotations": eth_ann} # Layout: 替换所有注解
                    ]
                )
            ]
        )]

        # --- D. 全局 Layout ---
        fig.update_layout(
            height=850,
            template="plotly_white",
            updatemenus=updatemenus,
            margin=dict(t=160, l=50, r=50, b=50), # 顶部留出大量空间给 HUD
            annotations=btc_ann, # 默认显示 BTC 注解
            hovermode="x unified",
            xaxis=dict(
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=4, label="4Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ],
                    x=0.0, y=-0.15 # 放到最下面
                ),
                rangeslider=dict(visible=False), # 关闭滑块，用按钮即可
                type="date"
            ),
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center")
        )

        fig.update_yaxes(type="log", gridcolor="#f4f4f4", row=1, col=1)
        fig.update_yaxes(gridcolor="#f4f4f4", range=[0, 5], row=2, col=1) # 固定 DI 范围以便视觉统一

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    else:
        st.error("Data processing failed.")
