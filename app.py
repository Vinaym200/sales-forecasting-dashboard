import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
    df['Year'] = df['Order Date'].dt.year
    return df


df = load_data()

page = st.sidebar.radio(
    "Navigate",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments", "All Charts (Tasks 1-6)"]
)

# ================= PAGE 1: SALES OVERVIEW =================
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    st.subheader("Total Sales by Year")
    st.bar_chart(df.groupby('Year')['Sales'].sum())

    st.subheader("Monthly Sales Trend")
    monthly = df.set_index('Order Date').resample('MS')['Sales'].sum()
    st.line_chart(monthly)

    # interactive filters - both region and category, as the task requires
    st.subheader("Sales by Region and Category (Interactive Filters)")
    col1, col2 = st.columns(2)
    with col1:
        region_choice = st.multiselect("Select Region(s)", df['Region'].unique(), default=list(df['Region'].unique()))
    with col2:
        category_choice = st.multiselect("Select Category(s)", df['Category'].unique(), default=list(df['Category'].unique()))

    filtered = df[df['Region'].isin(region_choice) & df['Category'].isin(category_choice)]
    st.bar_chart(filtered.groupby('Category')['Sales'].sum())
    st.caption(f"Showing {len(filtered)} orders matching the selected filters.")

# ================= PAGE 2: FORECAST EXPLORER =================
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    option_type = st.selectbox("Select Type", ["Category", "Region"])
    if option_type == "Category":
        selected = st.selectbox("Select Category", df['Category'].unique())
        subset = df[df['Category'] == selected]
    else:
        selected = st.selectbox("Select Region", df['Region'].unique())
        subset = df[df['Region'] == selected]

    # slider lets the user pick how many months ahead to forecast
    horizon = st.slider("Forecast Horizon (months ahead)", min_value=1, max_value=3, value=3)

    monthly = subset.set_index('Order Date').resample('MS')['Sales'].sum()

    # cached so re-selecting the same segment doesn't refit the model every time -
    # cache key is based on the function's inputs (the monthly series + horizon)
    @st.cache_data
    def fit_forecast(series, test_size=3):
        train = series[:-test_size]
        test = series[-test_size:]
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=test_size)
        mae = np.mean(np.abs(forecast.values - test.values))
        rmse = np.sqrt(np.mean((forecast.values - test.values) ** 2))
        return forecast, test, mae, rmse

    with st.spinner("Fitting SARIMA model..."):
        forecast, test, mae, rmse = fit_forecast(monthly)

    st.subheader(f"{horizon}-Month Forecast for {selected}")
    result_df = pd.DataFrame({
        "Month": test.index[:horizon].strftime('%b %Y'),
        "Forecast": forecast.values[:horizon].round(2),
        "Actual": test.values[:horizon].round(2)
    })
    st.dataframe(result_df)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index[-12:], monthly.values[-12:], label="Historical", marker='o')
    ax.plot(test.index[:horizon], forecast.values[:horizon], label="Forecast", marker='x', color='red')
    ax.legend()
    ax.set_title(f"SARIMA Forecast — {selected}")
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")

# ================= PAGE 3: ANOMALY REPORT =================
elif page == "Anomaly Report":
    st.title("Anomaly Report")

    weekly_df = df.set_index('Order Date').resample('W')['Sales'].sum().reset_index()
    weekly_df.columns = ['Week', 'Sales']

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly_df['anomaly'] = iso.fit_predict(weekly_df[['Sales']])
    anomalies = weekly_df[weekly_df['anomaly'] == -1]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly_df['Week'], weekly_df['Sales'], label='Weekly Sales')
    ax.scatter(anomalies['Week'], anomalies['Sales'], color='red', label='Anomaly', zorder=5)
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies")
    st.pyplot(fig)

    st.subheader("Detected Anomaly Weeks")
    st.dataframe(anomalies[['Week', 'Sales']].reset_index(drop=True))

# ================= PAGE 4: PRODUCT SEGMENTS =================
elif page == "Product Segments":
    st.title("Product Demand Segments")

    agg = df.groupby('Sub-Category').agg(
        total_sales=('Sales', 'sum'),
        avg_order_value=('Sales', 'mean')
    ).reset_index()

    yearly = df.groupby(['Sub-Category', 'Year'])['Sales'].sum().reset_index()
    pivot = yearly.pivot(index='Sub-Category', columns='Year', values='Sales')
    growth_rate = ((pivot[2018] - pivot[2015]) / pivot[2015]).rename('growth_rate')
    volatility = pivot.std(axis=1).rename('volatility')
    final = agg.set_index('Sub-Category').join(growth_rate).join(volatility).reset_index()

    features = ['total_sales', 'avg_order_value', 'growth_rate', 'volatility']
    X_scaled = StandardScaler().fit_transform(final[features])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    final['cluster'] = kmeans.fit_predict(X_scaled)

    pca_result = PCA(n_components=2).fit_transform(X_scaled)
    final['pca1'], final['pca2'] = pca_result[:, 0], pca_result[:, 1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(final['pca1'], final['pca2'], c=final['cluster'], cmap='viridis', s=100)
    for i, row in final.iterrows():
        ax.annotate(row['Sub-Category'], (row['pca1'], row['pca2']), fontsize=8)
    ax.set_title("Sub-Category Clusters (PCA Projection)")
    st.pyplot(fig)

    st.subheader("Cluster Assignments")
    st.dataframe(final[['Sub-Category', 'total_sales', 'growth_rate', 'volatility', 'cluster']].reset_index(drop=True))

# ================= PAGE 5: ALL CHARTS GALLERY (bonus/extra) =================
elif page == "All Charts (Tasks 1-6)":
    st.title("Full Chart Gallery — Tasks 1 to 6")

    chart_folder = "charts"
    if os.path.exists(chart_folder):
        chart_files = sorted([f for f in os.listdir(chart_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
        if len(chart_files) == 0:
            st.warning("No images found in the 'charts' folder.")
        else:
            for chart_file in chart_files:
                readable_name = chart_file.replace('.png', '').replace('_', ' ').title()
                st.subheader(readable_name)
                st.image(os.path.join(chart_folder, chart_file))
    else:
        st.warning("'charts' folder not found next to app.py.")
