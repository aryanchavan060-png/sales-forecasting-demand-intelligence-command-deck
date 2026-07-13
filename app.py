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
    layout="wide"
)

# =============================================================
# DATA LOADING (FIXED + SAFE)
# =============================================================
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], errors="coerce")

    df = df.dropna(subset=["Order Date"])

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month

    return df


df = load_data()

# =============================================================
# SIDEBAR
# =============================================================
menu = st.sidebar.radio(
    "Navigation",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Demand Segments"]
)

# =============================================================
# PAGE 1 — SALES OVERVIEW
# =============================================================
if menu == "Sales Overview":

    st.title("📊 Sales Overview")

    total_sales = df["Sales"].sum()
    total_orders = len(df)
    avg_order = total_sales / total_orders

    st.metric("Total Sales", f"${total_sales:,.0f}")
    st.metric("Total Orders", total_orders)
    st.metric("Avg Order Value", f"${avg_order:,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby("Year")["Sales"].sum().reset_index()
        fig = px.bar(yearly, x="Year", y="Sales", title="Yearly Sales")
        st.plotly_chart(fig, width="stretch")

    with col2:
        monthly = (
            df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
            .sum()
            .reset_index()
            .sort_values("Order Date")
        )

        fig2 = px.area(monthly, x="Order Date", y="Sales", title="Monthly Trend")
        st.plotly_chart(fig2, width="stretch")

# =============================================================
# PAGE 2 — FORECAST
# =============================================================
elif menu == "Forecast Explorer":

    st.title("📈 Forecast Explorer")

    segment = st.selectbox("Select Category", df["Category"].unique())
    horizon = st.slider("Months Forecast", 1, 6, 3)

    data = df[df["Category"] == segment]

    monthly = (
        data.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"]
        .sum()
        .reset_index()
        .sort_values("Order Date")
    )

    growth = monthly["Sales"].pct_change().mean()

    future = []
    last = monthly["Sales"].iloc[-1]

    for _ in range(horizon):
        last = last * (1 + growth)
        future.append(last)

    future_dates = pd.date_range(
        monthly["Order Date"].max(),
        periods=horizon + 1,
        freq="ME"
    )[1:]

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": future
    })

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly["Order Date"],
        y=monthly["Sales"],
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast"],
        name="Forecast",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig, width="stretch")

# =============================================================
# PAGE 3 — ANOMALY DETECTION
# =============================================================
elif menu == "Anomaly Report":

    st.title("🚨 Anomaly Detection")

    weekly = df.resample("W", on="Order Date")["Sales"].sum().reset_index()

    model = IsolationForest(contamination=0.05)
    weekly["Anomaly"] = model.fit_predict(weekly[["Sales"]])

    anomalies = weekly[weekly["Anomaly"] == -1]

    fig = px.line(weekly, x="Order Date", y="Sales", title="Weekly Sales")

    fig.add_scatter(
        x=anomalies["Order Date"],
        y=anomalies["Sales"],
        mode="markers",
        name="Anomaly"
    )

    st.plotly_chart(fig, width="stretch")
    st.dataframe(anomalies, width="stretch")

# =============================================================
# PAGE 4 — CLUSTERING
# =============================================================
elif menu == "Demand Segments":

    st.title("🧠 Demand Segmentation")

    product = df.groupby("Sub-Category").agg({
        "Sales": ["sum", "mean", "std"]
    })

    product.columns = ["Total", "Avg", "Std"]
    product.fillna(0, inplace=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(product)

    kmeans = KMeans(n_clusters=4, random_state=42)
    product["Cluster"] = kmeans.fit_predict(X)

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    plot_df = pd.DataFrame({
        "x": comp[:, 0],
        "y": comp[:, 1],
        "Cluster": product["Cluster"].astype(str),
        "Product": product.index
    })

    fig = px.scatter(plot_df, x="x", y="y", color="Cluster", hover_name="Product")
    st.plotly_chart(fig, width="stretch")

    st.dataframe(product, width="stretch")
