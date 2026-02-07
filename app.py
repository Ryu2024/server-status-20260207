import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import linregress
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. Page Configuration (Generic Title to avoid SEO) ---
st.set_page_config(page_title="Data View v2.0", layout="wide")

# --- 2. CSS Styling (Web 1.0 Serif - Clean & Boring) ---
st.markdown("""
<style>
    /* Global Settings: Times New Roman */
    .stApp {
        background-color: #ffffff;
        font-family: 'Times New Roman', Times, serif;
        color: #000000;
    }

    /* Header Styles - Generic Academic Look */
    h1, h2, h3 {
        color: #333333 !important;
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: bold;
        padding-bottom: 5px;
        border-bottom: 1px solid #ccc;
    }

    /* Metric Dashboard Styles */
    .metric-container {
        display: flex;
        justify-content: space-between;
        background-color: #f9f9f9;
        padding: 15px;
        border: 1px solid #000000;
        margin-bottom: 20px;
        margin-top: 10px;
    }
    .metric-item {
        text-align: center;
        width: 33%;
        border-right: 1px solid #ccc;
    }
    .metric-item:last-child {
        border-right: none;
    }
    .metric-label {
        font-size: 1.0em;
        color: #444;
        text-transform: uppercase;
        font-family: 'Times New Roman', Times, serif;
        margin-bottom: 5px;
        font-weight: normal;
    }
    .metric-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #000000;
        font-family: 'Times New Roman', Times, serif;
    }

    /* Legend Box Style */
    .legend-box {
        font-size: 1.0em;
        color: #000;
        background-color: #fff;
        padding: 10px;
        border: 1px solid #000;
        font-family: 'Times New Roman', Times, serif;
    }
    .legend-item {
        margin-bottom: 4px;
    }
    .color-box {
        display: inline-block;
        width: 12px;
        height: 12px;
        margin-right: 8px;
        border: 1px solid #000;
    }

    /* Chart Container */
    .stPlotlyChart {
        background-color: white;
        border: 1px solid #000000;
        padding: 5px;
    }
    
    /* Button Override */
    .stButton>button {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        border-radius: 0px;
        border: 1px solid #000;
        background-color: #e0e0e0;
        color: #000;
    }

    /* Hide Streamlit Menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- 3. Core Logic (Power Law & Dynamic Fit) ---
@st.cache_data(ttl=3600)
def get_data_and_calc(ticker):
    try:
        # Fetch Data (Real Ticker)
        df = yf.download(ticker, period="max", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.index = df.index.tz_localize(None)
        
        df = df[['Close']].copy().sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = df[df['Close'] > 0]
        
        # Basic Metrics
        df['Log_Price'] = np.log(df['Close'])
        df['GeoMean'] = np.exp(df['Log_Price'].rolling(window=200).mean())
        
        genesis_date = pd.Timestamp("2009-01-03")
        df['Days'] = (df.index - genesis_date).days
        df = df[df['Days'] > 0]
        
        # Fitting Logic
        if ticker == "BTC-USD":
            # Bitcoin Power Law
            slope = 5.84
            intercept = -17.01
            log_days = np.log10(df['Days'])
            df['Predicted'] = 10 ** (slope * log_days + intercept)
            note = "Method: Power Law (Fixed)"
        else:
            # Dynamic Fit
            valid_data = df.dropna()
            x = np.log10(valid_data['Days'].values)
            y = np.log10(valid_data['Close'].values)
            slope, intercept, _, _, _ = linregress(x, y)
            df['Predicted'] = 10 ** (intercept + slope * np.log10(df['Days']))
            note = f"Method: Dynamic Reg (Beta {slope:.4f})"

        # Calculate Index (Hidden Name: Deviation Index)
        df['AHR999'] = (df['Close'] / df['GeoMean']) * (df['Close'] / df['Predicted'])
        return df, note
    except Exception as e:
        return pd.DataFrame(), str(e)

# --- 4. Top Layout ---

st.title("Statistical Deviation Monitor")

# Control & Legend
with st.container():
    col_controls, col_legend = st.columns([1, 2])
    
    with col_controls:
        st.markdown("### Configuration")
        # Direct Asset Names (As requested)
        ticker = st.selectbox("Target Asset", ["BTC-USD", "ETH-USD"])
        if st.button("Reload Dataset"):
            st.cache_data.clear()
            
    with col_legend:
        st.markdown("### Reference Guide")
        st.markdown("""
        <div class="legend-box">
            <div class="legend-item">
                <span class="color-box" style="background-color: #28a745;"></span>
                <b>L-Line (0.45)</b>: Lower statistical bound. Historical buy zone.
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: #007bff;"></span>
                <b>M-Line (1.20)</b>: Mean accumulation threshold.
            </div>
            <div class="legend-item">
                <span class="color-box" style="background-color: #dc3545;"></span>
                <b>H-Line (4.00)</b>: Upper statistical bound. Variance warning.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# --- 5. Main Analysis ---

with st.spinner("Calculating metrics..."):
    df, note = get_data_and_calc(ticker)
    
    if not df.empty and len(df) > 200:
        last = df.iloc[-1]
        ahr = last['AHR999']
        price = last['Close']
        
        # Status Logic (Generic Codes)
        if ahr < 0.45:
            state = "ZONE L (Undershoot)"
            color = "#28a745" # Green
        elif 0.45 <= ahr <= 1.2:
            state = "ZONE M (Accumulation)"
            color = "#007bff" # Blue
        elif 1.2 < ahr <= 4.0:
            state = "ZONE N (Neutral)"
            color = "#fd7e14" # Orange
        else:
            state = "ZONE H (Overshoot)"
            color = "#dc3545" # Red

        # Dashboard
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-item">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">{price:,.2f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Deviation Index</div>
                <div class="metric-value" style="color: {color}">{ahr:.4f}</div>
            </div>
            <div class="metric-item">
                <div class="metric-label">Zone Status</div>
                <div class="metric-value" style="color: {color}">{state}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- Charting ---
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.08, 
            row_heights=[0.6, 0.4],
            subplot_titles=("Asset Value & Regression", "Deviation Index (DI)")
        )

        # Top Chart
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], 
            name="Value", 
            line=dict(color="#000080", width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['GeoMean'], 
            name="Geo-Mean (200)", 
            line=dict(color="#555555", width=1.5, dash='dot')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Predicted'], 
            name="Reg. Model", 
            line=dict(color="#800080", width=1.5, dash='dash')
        ), row=1, col=1)

        # Bottom Chart
        fig.add_trace(go.Scatter(
            x=df.index, y=df['AHR999'], 
            name="DI Value", 
            line=dict(color="#d35400", width=1.5)
        ), row=2, col=1)

        # Backgrounds
        fig.add_hrect(y0=0, y1=0.45, fillcolor="green", opacity=0.1, layer="below", line_width=0, row=2, col=1)
        fig.add_hrect(y0=4.0, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0, row=2, col=1)

        # Lines
        fig.add_hline(y=0.45, line_color="green", line_dash="dash", annotation_text="L-Bound (0.45)", row=2, col=1)
        fig.add_hline(y=1.2, line_color="blue", line_dash="dot", annotation_text="M-Bound (1.20)", row=2, col=1)
        fig.add_hline(y=4.0, line_color="red", line_dash="dash", annotation_text="H-Bound (4.00)", row=2, col=1)

        # Layout
        fig.update_layout(
            height=750,
            template="plotly_white",
            hovermode="x unified",
            font=dict(family="Times New Roman", size=14, color="#000"),
            margin=dict(l=40, r=40, t=60, b=40),
            
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(step="all", label="MAX")
                    ]),
                    bgcolor="#f0f0f0",
                    activecolor="#d0d0d0",
                    font=dict(color="#000", family="Times New Roman")
                ),
                type="date",
                rangeslider=dict(visible=False)
            ),
            
            yaxis=dict(title="Value (Log)", type="log", gridcolor="#f0f0f0"),
            yaxis2=dict(title="Index (Log)", type="log", gridcolor="#f0f0f0", range=[np.log10(0.1), np.log10(50)]),
            
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Times New Roman"))
        )
        
        fig.update_xaxes(rangeslider_visible=True, rangeselector_font_family="Times New Roman")

        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={'scrollZoom': True, 'displayModeBar': False}
        )
        
        st.caption(f"Calculation: {note} | Source: Public Data Repository")

    else:
        st.error("Data unavailable.")
