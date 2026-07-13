import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Sales Intelligence — Command Deck",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================
# DESIGN SYSTEM — dark "command deck" theme
# Palette:
#   bg base      #0B1120
#   surface      #131B2E
#   surface-2    #1B2540
#   border       #263252
#   text primary #E8ECF4
#   text muted   #8891A7
#   accent teal  #2DD9C7   (growth / primary)
#   accent coral #FF6B7A   (alerts / anomalies)
#   accent violet#8E7DFF   (segments)
#   accent amber #FFC061   (forecast)
# Fonts: Space Grotesk (display), Inter (body), JetBrains Mono (numbers)
# =============================================================

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500;600;700&display=swap" rel="stylesheet">

<style>
:root{
    --bg:#0B1120;
    --surface:#131B2E;
    --surface2:#1B2540;
    --border:#263252;
    --text:#E8ECF4;
    --muted:#8891A7;
    --teal:#2DD9C7;
    --coral:#FF6B7A;
    --violet:#8E7DFF;
    --amber:#FFC061;
}

/* base app */
.stApp{
    background:
        radial-gradient(circle at 15% 0%, rgba(45,217,199,0.07), transparent 40%),
        radial-gradient(circle at 85% 100%, rgba(142,125,255,0.07), transparent 40%),
        var(--bg);
    font-family:'Inter', sans-serif;
    color: var(--text);
}

/* sidebar */
section[data-testid="stSidebar"]{
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stRadio label{
    font-family:'Inter', sans-serif;
    color: var(--text) !important;
}

/* headers */
h1, h2, h3{
    font-family:'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing:-0.01em;
}

/* hero title block */
.hero-wrap{
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    border-radius: 18px;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface2) 100%);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}
.hero-wrap::after{
    content:"";
    position:absolute; top:0; right:0; bottom:0; width:6px;
    background: linear-gradient(180deg, var(--teal), var(--violet));
}
.hero-eyebrow{
    font-family:'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    color: var(--teal);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.hero-title{
    font-family:'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--text);
    margin: 0;
}
.hero-sub{
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.35rem;
}

/* KPI cards */
.kpi-row{ display:flex; gap:14px; margin-bottom: 1.4rem; flex-wrap: wrap;}
.kpi-card{
    flex:1; min-width: 190px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    position: relative;
    transition: border-color .2s ease, transform .2s ease;
}
.kpi-card:hover{ border-color: var(--accent-color, var(--teal)); transform: translateY(-2px); }
.kpi-label{
    font-family:'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.kpi-value{
    font-family:'JetBrains Mono', monospace;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--text);
}
.kpi-delta{
    font-size: 0.8rem;
    margin-top: 0.3rem;
    font-family:'JetBrains Mono', monospace;
}
.kpi-dot{
    position:absolute; top: 1.1rem; right: 1.1rem;
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent-color, var(--teal));
    box-shadow: 0 0 8px var(--accent-color, var(--teal));
}

/* section labels */
.section-label{
    font-family:'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--violet);
    margin: 1.6rem 0 0.6rem 0;
    display:flex; align-items:center; gap:8px;
}
.section-label::before{
    content:""; width:14px; height:1px; background: var(--violet);
}

/* chart container feel */
div[data-testid="stPlotlyChart"]{
    background: #F7F8FC;
    border-radius: 14px;
    padding: 0.6rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}

/* selects / multiselect / slider */
.stMultiSelect [data-baseweb="select"], .stSelectbox [data-baseweb="select"]{
    background: var(--surface2);
    border-radius: 10px;
}
.stSlider{ padding-top: 0.4rem; }

/* dataframe */
[data-testid="stDataFrame"]{
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}

/* info box */
div[data-testid="stAlertContainer"]{
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px;
    color: var(--text) !important;
}

/* hide default streamlit chrome */
#MainMenu, footer{ visibility:hidden; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#F7F8FC",
        plot_bgcolor="#F7F8FC",
        font=dict(family="Inter, sans-serif", color="#1B2540", size=13),
        title_font=dict(family="Space Grotesk, sans-serif", size=17, color="#0B1120"),
        colorway=["#0EA89A", "#6C5CE7", "#E8960B", "#E0475B", "#2E63E7", "#1FA85C"],
        xaxis=dict(gridcolor="#E3E7F0", zerolinecolor="#D7DCE8", linecolor="#D7DCE8", color="#4A5578"),
        yaxis=dict(gridcolor="#E3E7F0", zerolinecolor="#D7DCE8", linecolor="#D7DCE8", color="#4A5578"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#1B2540")),
        margin=dict(t=60, l=20, r=20, b=20),
    )
)


def style_fig(fig, height=420):
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height)
    return fig


def kpi_card(label, value, delta=None, accent="#2DD9C7", delta_color=None):
    delta_html = ""
    if delta is not None:
        dc = delta_color or ("#4ADE80" if delta >= 0 else "#FF6B7A")
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="kpi-delta" style="color:{dc}">{arrow} {abs(delta):.1f}%</div>'
    return f"""
    <div class="kpi-card" style="--accent-color:{accent}">
        <div class="kpi-dot"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


# =============================================================
# DATA
# =============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format='%d/%m/%Y', errors='coerce')
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format='%d/%m/%Y', errors='coerce')
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df


df = load_data()

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.4rem 0 1.2rem 0;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.15rem; font-weight:700; color:#E8ECF4;">
                ◆ Command Deck
            </div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#8891A7; letter-spacing:0.08em;">
                SALES INTELLIGENCE SYSTEM
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    menu = st.radio(
        "Navigation",
        ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"],
        label_visibility="collapsed"
    )
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#8891A7; line-height:1.6;">
            RECORDS &nbsp;<span style="color:#E8ECF4">{len(df):,}</span><br>
            RANGE &nbsp;&nbsp;&nbsp;<span style="color:#E8ECF4">{df['Order Date'].min().date()} → {df['Order Date'].max().date()}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================
# PAGE 1 — SALES OVERVIEW
# =============================================================
if menu == "Sales Overview":

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Live Overview</div>
            <p class="hero-title">Sales Performance Deck</p>
            <p class="hero-sub">Revenue trends, regional spread, and category mix at a glance.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    total_sales = df["Sales"].sum()
    total_orders = df["Order ID"].nunique() if "Order ID" in df.columns else len(df)
    avg_order = total_sales / total_orders if total_orders else 0
    latest_year = int(df["Year"].max())
    prev_year = latest_year - 1
    ys = df[df.Year == latest_year]["Sales"].sum()
    yp = df[df.Year == prev_year]["Sales"].sum()
    yoy = ((ys - yp) / yp * 100) if yp else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card("Total Revenue", f"${total_sales:,.0f}", accent="#2DD9C7"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Total Orders", f"{total_orders:,}", accent="#8E7DFF"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Avg Order Value", f"${avg_order:,.2f}", accent="#FFC061"), unsafe_allow_html=True)
    k4.markdown(kpi_card(f"{latest_year} YoY Growth", f"{yoy:+.1f}%", delta=yoy, accent="#FF6B7A"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Revenue Trends</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.4])

    with c1:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Total Sales by Year")
        fig.update_traces(marker_color="#0EA89A", marker_line_width=0)
        st.plotly_chart(style_fig(fig), use_container_width=True, theme=None)

    with c2:
        monthly = df.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum().reset_index()
        fig2 = px.area(monthly, x="Order Date", y="Sales", title="Monthly Sales Trend")
        fig2.update_traces(line_color="#6C5CE7", fillcolor="rgba(108,92,231,0.12)")
        st.plotly_chart(style_fig(fig2), use_container_width=True, theme=None)

    st.markdown('<div class="section-label">Filter & Explore</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        region = st.multiselect("Select Region", df.Region.unique(), default=list(df.Region.unique()))
    with col2:
        category = st.multiselect("Select Category", df.Category.unique(), default=list(df.Category.unique()))

    filtered = df[df.Region.isin(region) & df.Category.isin(category)]

    c3, c4 = st.columns(2)
    with c3:
        region_chart = filtered.groupby("Region")["Sales"].sum().reset_index()
        fig3 = px.bar(region_chart, x="Region", y="Sales", title="Sales by Region", color="Region",
                      color_discrete_sequence=["#0EA89A", "#6C5CE7", "#E8960B", "#E0475B", "#2E63E7"])
        st.plotly_chart(style_fig(fig3), use_container_width=True, theme=None)

    with c4:
        cat_chart = filtered.groupby("Category")["Sales"].sum().reset_index()
        fig4 = px.pie(cat_chart, names="Category", values="Sales", title="Category Mix", hole=0.55,
                      color_discrete_sequence=["#0EA89A", "#6C5CE7", "#E8960B", "#E0475B"])
        fig4.update_traces(textfont_color="#1B2540")
        st.plotly_chart(style_fig(fig4), use_container_width=True, theme=None)


# =============================================================
# PAGE 2 — FORECAST EXPLORER
# =============================================================
elif menu == "Forecast Explorer":

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Predictive View</div>
            <p class="hero-title">Forecast Explorer</p>
            <p class="hero-sub">Project near-term demand using recent growth momentum.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        segment_type = st.selectbox("Forecast Level", ["Category", "Region"])
    with c2:
        selected = st.selectbox("Select Segment", df[segment_type].unique())
    with c3:
        horizon = st.slider("Forecast Horizon (Months)", 1, 3, 3)

    data = df[df[segment_type] == selected]
    monthly = data.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum().reset_index()

    avg_growth = monthly.Sales.pct_change().mean()
    future = []
    last_value = monthly.Sales.iloc[-1]
    for i in range(horizon):
        last_value = last_value * (1 + avg_growth)
        future.append(last_value)

    forecast_dates = pd.date_range(monthly["Order Date"].max(), periods=horizon + 1, freq="M")[1:]
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": future})

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("Segment", selected, accent="#8E7DFF"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Avg Monthly Growth", f"{avg_growth*100:+.1f}%", accent="#2DD9C7"), unsafe_allow_html=True)
    k3.markdown(kpi_card(f"Projected +{horizon}mo", f"${future[-1]:,.0f}", accent="#FFC061"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Historical + Projected Demand</div>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["Order Date"], y=monthly["Sales"], name="Historical",
                              line=dict(color="#2E63E7", width=2)))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], name="Forecast",
                              line=dict(color="#E8960B", width=2, dash="dash"),
                              mode="lines+markers"))
    fig.update_layout(title="Future Demand Forecast")
    st.plotly_chart(style_fig(fig, height=460), use_container_width=True, theme=None)

    st.info("Production version can load SARIMA / Prophet / XGBoost saved models.")


# =============================================================
# PAGE 3 — ANOMALY REPORT
# =============================================================
elif menu == "Anomaly Report":

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Signal Detection</div>
            <p class="hero-title">Sales Anomaly Report</p>
            <p class="hero-sub">Isolation Forest flags weeks that break from the normal pattern.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    weekly = df.resample("W", on="Order Date")["Sales"].sum().reset_index()

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["Anomaly"] = iso.fit_predict(weekly[["Sales"]]) == -1

    anomaly_points = weekly[weekly.Anomaly]

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("Weeks Analyzed", f"{len(weekly)}", accent="#2DD9C7"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Anomalies Found", f"{len(anomaly_points)}", accent="#FF6B7A"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Anomaly Rate", f"{len(anomaly_points)/len(weekly)*100:.1f}%", accent="#FFC061"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Weekly Sales — Flagged Points</div>', unsafe_allow_html=True)

    fig = px.line(weekly, x="Order Date", y="Sales", title="Weekly Sales Anomalies")
    fig.update_traces(line_color="#2E63E7")
    fig.add_scatter(
        x=anomaly_points["Order Date"],
        y=anomaly_points["Sales"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="#E0475B", size=10, line=dict(color="#F7F8FC", width=1))
    )
    st.plotly_chart(style_fig(fig, height=460), use_container_width=True, theme=None)

    st.markdown('<div class="section-label">Detected Anomalies</div>', unsafe_allow_html=True)
    st.dataframe(anomaly_points, use_container_width=True)


# =============================================================
# PAGE 4 — DEMAND SEGMENTS
# =============================================================
elif menu == "Demand Segments":

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-eyebrow">Clustering</div>
            <p class="hero-title">Product Demand Segmentation</p>
            <p class="hero-sub">K-Means groups sub-categories by volume, value, and volatility.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    product = df.groupby("Sub-Category").agg(
        Total_Sales=("Sales", "sum"),
        Avg_Order_Value=("Sales", "mean")
    )
    product["Volatility"] = df.groupby("Sub-Category")["Sales"].std()
    product.fillna(0, inplace=True)

    scaled = StandardScaler().fit_transform(product)

    kmeans = KMeans(n_clusters=4, random_state=42)
    product["Cluster"] = kmeans.fit_predict(scaled)

    pca = PCA(2)
    components = pca.fit_transform(scaled)

    plot_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Cluster": product.Cluster.astype(str),
        "Product": product.index
    })

    k1, k2, k3 = st.columns(3)
    k1.markdown(kpi_card("Sub-Categories", f"{len(product)}", accent="#8E7DFF"), unsafe_allow_html=True)
    k2.markdown(kpi_card("Clusters", "4", accent="#2DD9C7"), unsafe_allow_html=True)
    k3.markdown(kpi_card("Top Sub-Category", product.Total_Sales.idxmax(), accent="#FFC061"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Cluster Map (PCA)</div>', unsafe_allow_html=True)

    fig = px.scatter(
        plot_df, x="PC1", y="PC2", color="Cluster", hover_name="Product",
        title="Demand Clusters",
        color_discrete_sequence=["#0EA89A", "#6C5CE7", "#E8960B", "#E0475B"]
    )
    fig.update_traces(marker=dict(size=13, line=dict(width=1, color="#F7F8FC")))
    st.plotly_chart(style_fig(fig, height=460), use_container_width=True, theme=None)

    st.markdown('<div class="section-label">Segment Details</div>', unsafe_allow_html=True)
    st.dataframe(product, use_container_width=True)