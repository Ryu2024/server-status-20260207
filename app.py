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
        background-color: #ffffff;
        padding: 10px;
        border: 1px solid #000;
        font-family: 'Times New Roman', Times, serif;
    }
    .legend-item {
        margin-bottom: 8px;
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
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #e0e0e0 !important;
        color: #000 !important;
    }

    /* Input & Date Picker Override */
    input, input[type="date"], input[type="text"], .stDateInput>div>div>input {
        font-family: 'Times New Roman', Times, serif !important;
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #ffffff !important;
        color: #000 !important;
    }

    /* Select Box Override - Comprehensive */
    .stSelectbox, .stSelectbox > div, .stSelectbox > div > div, .stSelectbox * {
        border-radius: 0px !important;
    }
    
    .stSelectbox [role="listbox"], .stSelectbox [role="combobox"], select {
        font-family: 'Times New Roman', Times, serif !important;
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #ffffff !important;
        color: #000 !important;
    }
    
    /* Extra selectbox styling */
    .stSelectbox > div > button {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        font-family: 'Times New Roman', Times, serif !important;
    }

    /* Date Input Container */
    .stDateInput > div {
        border-radius: 0px !important;
    }
    
    .stDateInput > div > div > input {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        font-family: 'Times New Roman', Times, serif !important;
        background-color: #ffffff !important;
    }

    /* Time Slicer Section - Internal */
    .time-slicer-section {
        margin-top: 0;
    }

    /* Slider (range) styling - 90s style, no rounded corners */
    input[type="range"], .stSlider * {
        border-radius: 0px !important;
        border: 1px solid #000 !important;
        background-color: #ffffff !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stSlider > div > div {
        border-radius: 0px !important;
        border: none !important;
        padding: 4px !important;
        background-color: transparent !important;
    }
    /* Full width slider container */
    .full-width-slider {
        width: 100%;
        padding: 6px 8px;
        border: 1px solid #666;
        background-color: #ffffff;
        box-sizing: border-box;
        margin-bottom: 8px;
    }
    /* Ensure the Streamlit slider uses full width */
    .stSlider {
        width: 100% !important;
    }
    .stSlider > div {
        width: 100% !important;
    }
    /* Slimmer slider track (retro look) */
    .stSlider input[type="range"] {
        height: 8px !important;
        background: #e9e9e9 !important;
    }
    .stSlider .css-1x8cf1d {
        border-radius: 0px !important;
    }
    .slider-date-box {
        border: 1px solid #666;
        padding: 6px 10px;
        font-family: 'Times New Roman', Times, serif;
        font-size: 14px;
        color: #333333;
        background-color: #ffffff;
        text-align: center;
    }

    .fullwidth-title {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        font-size: 0.95em;
        color: #000;
        text-transform: uppercase;
        margin-top: 8px;
        margin-bottom: 6px;
    }

    .time-slicer-title {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        font-size: 0.95em;
        color: #000;
        text-transform: uppercase;
        margin-bottom: 10px;
        border-bottom: 1px solid #ccc;
        padding-bottom: 8px;
    }
    
    /* Config Box Section */
    .config-box {
        background-color: #ffffff;
        border: 1px solid #000;
        padding: 0px;
        margin-bottom: 15px;
    }
    
    .config-title {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        font-size: 1.0em;
        color: #000;
        text-transform: uppercase;
        background-color: #e0e0e0;
        padding: 8px 12px;
        margin: 0;
        border-bottom: 1px solid #000;
    }
    
    .config-content {
        padding: 12px;
    }
    
    .config-section {
        margin-bottom: 10px;
    }
    
    .config-section-divider {
        border-top: 1px solid #ccc;
        margin: 12px 0 0 0;
        padding-top: 0;
    }
    
    /* Guide Box Section */
    .guide-box {
        background-color: #ffffff;
        border: 1px solid #000;
        padding: 0px;
        margin-bottom: 15px;
    }
    
    .guide-title {
        font-family: 'Times New Roman', Times, serif;
        font-weight: bold;
        font-size: 1.0em;
        color: #000;
        text-transform: uppercase;
        background-color: #e0e0e0;
        padding: 8px 12px;
        margin: 0;
        border-bottom: 1px solid #000;
    }
    
    .guide-content {
        padding: 10px 12px 12px 12px;
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
        st.markdown('<div class="config-box"><div class="config-title">CONFIGURATION</div><div class="config-content">', unsafe_allow_html=True)
        
        # Target Asset
        st.markdown('<div class="config-section">', unsafe_allow_html=True)
        ticker = st.selectbox("Target Asset", ["BTC-USD", "ETH-USD"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Time Range Filter (Timeline Slicer) - title moved below the slider
        st.markdown('<div class="config-section-divider">', unsafe_allow_html=True)
        
        if st.button("Reload Dataset"):
            st.cache_data.clear()
        
        st.markdown('</div></div>', unsafe_allow_html=True)
            
    with col_legend:
        st.markdown('<div class="guide-box"><div class="guide-title">REFERENCE GUIDE</div><div class="guide-content">', unsafe_allow_html=True)
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
        st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown("---")

# Full-width date-range slider (90s style)
min_date = datetime(2009, 1, 3).date()
max_date = datetime.today().date()
default_start = datetime(2020, 1, 1).date()
default_end = max_date

# Layout: left date box, wide slider, right date box
col_l, col_m, col_r = st.columns([1, 10, 1])
with col_m:
    start_date, end_date = st.slider(
        "",
        min_value=min_date,
        max_value=max_date,
        value=(default_start, default_end),
        format="YYYY/MM/DD",
        help="Drag the handles to set start and end dates",
        key="date_range_slider"
    )
with col_l:
    st.markdown(f"<div class=\"slider-date-box\">{start_date}</div>", unsafe_allow_html=True)
with col_r:
    st.markdown(f"<div class=\"slider-date-box\">{end_date}</div>", unsafe_allow_html=True)

# Slicer Title (placed below slider for better balance)
st.markdown('<div class="fullwidth-title">TIME RANGE SLICER</div>', unsafe_allow_html=True)

# --- 5. Main Analysis ---

with st.spinner("Calculating metrics..."):
    df, note = get_data_and_calc(ticker)
    
    # Keep full data for validation and calculations
    df_full = df.copy()
    
    # Apply Date Range Filter for display only (use timestamps from slider)
    df_display = df.copy()
    if not df_display.empty:
        df_display = df_display[(df_display.index >= pd.to_datetime(start_date)) & (df_display.index <= pd.to_datetime(end_date))]
    
    if not df_full.empty and len(df_full) > 200:
        last = df_full.iloc[-1]
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
        # Calculate time range length to determine which buttons to display
        days_range = (end_date - start_date).days
        
        # Build dynamic button list based on selected time range
        range_buttons = [
            dict(count=1, label="1D", step="day", stepmode="backward"),
        ]
        
        if days_range >= 3:
            range_buttons.append(dict(count=3, label="3D", step="day", stepmode="backward"))
        if days_range >= 7:
            range_buttons.append(dict(count=7, label="1W", step="day", stepmode="backward"))
        if days_range >= 14:
            range_buttons.append(dict(count=14, label="2W", step="day", stepmode="backward"))
        if days_range >= 30:
            range_buttons.append(dict(count=1, label="1M", step="month", stepmode="backward"))
        if days_range >= 90:
            range_buttons.append(dict(count=3, label="3M", step="month", stepmode="backward"))
        if days_range >= 180:
            range_buttons.append(dict(count=6, label="6M", step="month", stepmode="backward"))
        if days_range >= 365:
            range_buttons.append(dict(count=1, label="1Y", step="year", stepmode="backward"))
        if days_range >= 1095:
            range_buttons.append(dict(count=3, label="3Y", step="year", stepmode="backward"))
        
        # Always add MAX button
        range_buttons.append(dict(step="all", label="MAX"))
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.12, 
            row_heights=[0.6, 0.4],
            subplot_titles=("Asset Value & Regression", "Deviation Index (DI)"),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )

        # Top Chart
        fig.add_trace(go.Scatter(
            x=df_display.index, y=df_display['Close'], 
            name="Value", 
            line=dict(color="#000080", width=1.5)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_display.index, y=df_display['GeoMean'], 
            name="Geo-Mean (200)", 
            line=dict(color="#555555", width=1.5, dash='dot')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=df_display.index, y=df_display['Predicted'], 
            name="Reg. Model", 
            line=dict(color="#800080", width=1.5, dash='dash')
        ), row=1, col=1)

        # Bottom Chart
        fig.add_trace(go.Scatter(
            x=df_display.index, y=df_display['AHR999'], 
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
            margin=dict(l=40, r=120, t=80, b=40),
            
            xaxis=dict(
                rangeselector=dict(
                    buttons=range_buttons,
                    bgcolor="#f0f0f0",
                    activecolor="#d0d0d0",
                    font=dict(color="#000", family="Times New Roman"),
                    y=1.15,
                    yanchor="top",
                    x=0.0,
                    xanchor="left"
                ),
                type="date",
                rangeslider=dict(visible=False),
                range=[df_full.index.min(), df_full.index.max()]
            ),
            
            yaxis=dict(title="Value (Log)", type="log", gridcolor="#f0f0f0", autorange=True),
            yaxis2=dict(title="Index (Log)", type="log", gridcolor="#f0f0f0", autorange=True),
            
            legend=dict(orientation="h", yanchor="top", y=0.99, xanchor="left", x=0.01, font=dict(family="Times New Roman", size=11), bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="#000", borderwidth=1)
        )
        
        fig.update_xaxes(rangeslider_visible=True, rangeselector_font_family="Times New Roman")

        st.plotly_chart(
            fig, 
            use_container_width=True,
            config={'scrollZoom': False, 'displayModeBar': False}
        )
        
        # Display Time Range Info - 90s Style
        st.markdown(f"""
        <div style="background-color: #f0f0f0; padding: 12px; border: 1px solid #000; margin-bottom: 10px; font-family: 'Times New Roman', serif;">
            <b>FILTER APPLIED:</b> {start_date} → {end_date} | <b>DATA POINTS:</b> {len(df_display)} | <b>PERIOD:</b> {(end_date - start_date).days} days
        </div>
        """, unsafe_allow_html=True)
        
        st.caption(f"Calculation: {note} | Source: Public Data Repository")

    else:
        if len(df_full) > 0 and len(df_full) <= 200:
            st.warning(f"⚠️ Insufficient data points. ({len(df_full)} points available, 200+ required)")
        else:
            st.error("Data unavailable.")
