
import streamlit as st
import pandas as pd
from pathlib import Path

# ------------------------------------
# 1. Page Configuration
# ------------------------------------

st.set_page_config(
    page_title="Retail Business Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------
# 2. Custom Dashboard Styling
# ------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 34px;
    font-weight: bold;
    color: #38BDF8;
}

.subtitle {
    font-size: 16px;
    color: #94A3B8;
}

div[data-testid="stMetric"] {
    background-color: #1E293B;
    border: 1px solid #334155;
    padding: 20px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #CBD5E1;
}

div[data-testid="stMetricValue"] {
    color: #38BDF8;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------
# 3. Project Paths
# ------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PREDICTION_FILE = (
    PROJECT_ROOT / "reports" / "test_forecast_predictions.csv"
)

TEST_FILE = (
    PROJECT_ROOT / "data" / "deployment" / "test_metadata.csv"
)

HISTORICAL_FILE = (
    PROJECT_ROOT / "data" / "deployment" / "historical_daily_sales.csv"
)

# ------------------------------------
# 4. Verify Required Files
# ------------------------------------

required_files = [
    PREDICTION_FILE,
    TEST_FILE,
    HISTORICAL_FILE
]

missing_files = [
    file for file in required_files
    if not file.exists()
]

if missing_files:
    st.error("Required dashboard datasets are missing:")

    for file in missing_files:
        st.write(str(file))

    st.stop()

# ------------------------------------
# 5. Dashboard Heading
# ------------------------------------

st.markdown(
    '<div class="main-title">Retail Business Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-Powered Sales Analytics and Forecasting</div>',
    unsafe_allow_html=True
)

st.divider()

# ------------------------------------
# 6. Load Forecast Data
# ------------------------------------

@st.cache_data
def load_forecast_data():

    predictions = pd.read_csv(PREDICTION_FILE)

    test = pd.read_csv(
        TEST_FILE,
        usecols=["id", "date", "store_nbr", "family"]
    )

    if test["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in test metadata!")

    if predictions["id"].duplicated().any():
        raise ValueError("Duplicate IDs found in forecast predictions!")

    data = test.merge(
        predictions,
        on="id",
        how="left",
        validate="one_to_one"
    )

    data["date"] = pd.to_datetime(data["date"])

    if data["sales"].isna().any():
        raise ValueError("Missing forecast predictions found!")

    return data


data = load_forecast_data()

# ------------------------------------
# 7. Load Historical Data
# ------------------------------------

@st.cache_data
def load_historical_data():

    historical = pd.read_csv(
        HISTORICAL_FILE,
        usecols=["date", "sales"]
    )

    historical["date"] = pd.to_datetime(
        historical["date"]
    )

    historical["sales"] = pd.to_numeric(
        historical["sales"],
        errors="raise"
    )

    if historical["date"].duplicated().any():
        raise ValueError(
            "Historical daily dataset contains duplicate dates!"
        )

    return historical.sort_values("date")


historical = load_historical_data()

# ------------------------------------
# 8. Interactive Dashboard Filters
# ------------------------------------

st.sidebar.title("Dashboard Filters")

# Date filter

min_date = data["date"].min().date()
max_date = data["date"].max().date()

selected_dates = st.sidebar.date_input(
    "Select Forecast Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Store filter

available_stores = sorted(
    data["store_nbr"].unique().tolist()
)

selected_stores = st.sidebar.multiselect(
    "Select Stores",
    options=available_stores,
    default=available_stores
)

# Product family filter

available_families = sorted(
    data["family"].unique().tolist()
)

selected_families = st.sidebar.multiselect(
    "Select Product Families",
    options=available_families,
    default=available_families
)

# ------------------------------------
# 9. Apply Filters
# ------------------------------------

filtered_data = data.copy()

if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_data = filtered_data[
        (filtered_data["date"].dt.date >= start_date) &
        (filtered_data["date"].dt.date <= end_date)
    ]

elif hasattr(selected_dates, "year"):

    filtered_data = filtered_data[
        filtered_data["date"].dt.date == selected_dates
    ]

else:

    filtered_data = filtered_data.iloc[0:0]

filtered_data = filtered_data[
    filtered_data["store_nbr"].isin(selected_stores) &
    filtered_data["family"].isin(selected_families)
]

# ------------------------------------
# 10. Sidebar Filter Summary
# ------------------------------------

st.sidebar.divider()

st.sidebar.subheader("Filter Summary")

st.sidebar.write(
    f"Filtered records: {len(filtered_data):,}"
)

st.sidebar.write(
    f"Selected stores: {len(selected_stores)}"
)

st.sidebar.write(
    f"Selected product families: {len(selected_families)}"
)

if not filtered_data.empty:

    st.sidebar.metric(
        "Filtered Predicted Sales",
        f"{filtered_data['sales'].sum():,.2f}"
    )

    st.sidebar.metric(
        "Average Sales per Record",
        f"{filtered_data['sales'].mean():,.2f}"
    )

else:

    st.sidebar.warning(
        "No records match the selected filters."
    )

# ------------------------------------
# 11. Calculate Filtered KPIs
# ------------------------------------

total_sales = filtered_data["sales"].sum()

total_records = len(filtered_data)

forecast_days = filtered_data["date"].nunique()

total_stores = filtered_data["store_nbr"].nunique()

total_families = filtered_data["family"].nunique()

average_sales = (
    filtered_data["sales"].mean()
    if not filtered_data.empty else 0
)

# ------------------------------------
# 12. Dashboard Navigation
# ------------------------------------

overview_tab, historical_tab, forecast_tab, product_tab, store_tab = st.tabs(
    [
        "Overview",
        "Historical Analysis",
        "Forecast Analysis",
        "Product Performance",
        "Store Performance"
    ]
)

# ====================================
# TAB 1: OVERVIEW
# ====================================

with overview_tab:

    st.header("Business Performance Overview")

    # KPI Cards

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Predicted Sales",
            f"{total_sales:,.0f}"
        )

    with col2:
        st.metric(
            "Forecast Records",
            f"{total_records:,}"
        )

    with col3:
        st.metric(
            "Forecast Days",
            forecast_days
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric(
            "Total Stores",
            total_stores
        )

    with col5:
        st.metric(
            "Product Families",
            total_families
        )

    with col6:
        st.metric(
            "Average Predicted Sales",
            f"{average_sales:,.2f}"
        )

    st.divider()

    # Business Insights

    st.subheader("Key Business Insights")

    if not filtered_data.empty:

        family_totals = (
            filtered_data.groupby("family")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        store_totals = (
            filtered_data.groupby("store_nbr")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        top_family = family_totals.index[0]
        top_store = store_totals.index[0]

        top_family_sales = family_totals.iloc[0]
        top_store_sales = store_totals.iloc[0]

        insight_col1, insight_col2 = st.columns(2)

        with insight_col1:

            st.markdown("**Leading Product Family**")

            st.subheader(top_family)

            st.write(
                f"Predicted sales: {top_family_sales:,.2f}"
            )

            st.write(
                f"Contribution: "
                f"{(top_family_sales / total_sales) * 100:.2f}%"
            )

        with insight_col2:

            st.markdown("**Leading Store**")

            st.subheader(f"Store {top_store}")

            st.write(
                f"Predicted sales: {top_store_sales:,.2f}"
            )

            st.write(
                f"Contribution: "
                f"{(top_store_sales / total_sales) * 100:.2f}%"
            )

    else:

        st.info(
            "Select valid filters to view business insights."
        )

    st.divider()

    # Filtered Dataset Preview

    st.subheader("Filtered Forecast Dataset Preview")

    st.dataframe(
        filtered_data.head(20),
        width="stretch"
    )

    st.divider()

    # Dashboard Export and Reports

    st.header("Download Business Reports")

    if not filtered_data.empty:

        forecast_export = filtered_data[
            ["id", "date", "store_nbr", "family", "sales"]
        ].copy()

        forecast_export["date"] = (
            forecast_export["date"].dt.strftime("%Y-%m-%d")
        )

        family_export = (
            filtered_data.groupby("family")["sales"]
            .sum()
            .reset_index()
            .sort_values("sales", ascending=False)
        )

        family_export["sales_percentage"] = (
            family_export["sales"] /
            family_export["sales"].sum()
        ) * 100

        store_export = (
            filtered_data.groupby("store_nbr")["sales"]
            .sum()
            .reset_index()
            .sort_values("sales", ascending=False)
        )

        store_export["sales_percentage"] = (
            store_export["sales"] /
            store_export["sales"].sum()
        ) * 100

        daily_export = (
            filtered_data.groupby("date")["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        daily_export["date"] = (
            daily_export["date"].dt.strftime("%Y-%m-%d")
        )

        download_col1, download_col2 = st.columns(2)

        with download_col1:

            st.download_button(
                label="Download Forecast CSV",
                data=forecast_export.to_csv(index=False),
                file_name="filtered_forecast_report.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Product Family CSV",
                data=family_export.to_csv(index=False),
                file_name="product_family_report.csv",
                mime="text/csv"
            )

        with download_col2:

            st.download_button(
                label="Download Store CSV",
                data=store_export.to_csv(index=False),
                file_name="store_performance_report.csv",
                mime="text/csv"
            )

            st.download_button(
                label="Download Daily Sales CSV",
                data=daily_export.to_csv(index=False),
                file_name="daily_forecast_report.csv",
                mime="text/csv"
            )

    else:

        st.warning(
            "Select valid filters before downloading reports."
        )

# ====================================
# TAB 2: HISTORICAL ANALYSIS
# ====================================

with historical_tab:

    st.header("Historical Sales Analysis")

    # Daily sales trend

    daily_sales = historical.copy()

    st.subheader("Daily Sales Trend")

    st.line_chart(
        daily_sales,
        x="date",
        y="sales",
        width="stretch"
    )

    # Monthly sales

    st.subheader("Monthly Sales Performance")

    historical_monthly = historical.copy()

    historical_monthly["month"] = (
        historical_monthly["date"].dt.month
    )

    monthly_sales = (
        historical_monthly.groupby("month")["sales"]
        .sum()
        .reset_index()
    )

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    monthly_sales["month_name"] = (
        monthly_sales["month"].map(month_names)
    )

    monthly_sales["month_name"] = pd.Categorical(
        monthly_sales["month_name"],
        categories=list(month_names.values()),
        ordered=True
    )

    monthly_sales = monthly_sales.sort_values("month")

    st.bar_chart(
        monthly_sales,
        x="month_name",
        y="sales",
        width="stretch"
    )

    # Weekday vs weekend

    st.subheader(
        "Average Daily Total Sales: Weekday vs Weekend"
    )

    historical_day = historical.copy()

    historical_day["day_of_week"] = (
        historical_day["date"].dt.dayofweek
    )

    historical_day["day_type"] = (
        historical_day["day_of_week"].apply(
            lambda day: "Weekend" if day >= 5 else "Weekday"
        )
    )

    day_comparison = (
        historical_day.groupby("day_type")["sales"]
        .mean()
        .reset_index()
    )

    st.bar_chart(
        day_comparison,
        x="day_type",
        y="sales",
        width="stretch"
    )

    # Historical dataset summary

    st.subheader("Historical Dataset Summary")

    st.dataframe(
        daily_sales.tail(10),
        width="stretch"
    )

# ====================================
# TAB 3: FORECAST ANALYSIS
# ====================================

with forecast_tab:

    st.header("Sales Forecast Analysis")

    if not filtered_data.empty:

        historical_daily = historical.copy()

        historical_daily["sales_type"] = "Historical Sales"

        forecast_daily = (
            filtered_data.groupby("date")["sales"]
            .sum()
            .reset_index()
        )

        forecast_daily["sales_type"] = "Forecast Sales"

        comparison_data = pd.concat(
            [historical_daily, forecast_daily],
            ignore_index=True
        )

        comparison_data = comparison_data.sort_values("date")

        st.subheader("Historical vs Predicted Sales")

        chart_data = comparison_data.pivot(
            index="date",
            columns="sales_type",
            values="sales"
        )

        st.line_chart(
            chart_data,
            width="stretch"
        )

        st.subheader("Forecast Period Summary")

        forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

        with forecast_col1:

            st.metric(
                "Forecast Start Date",
                forecast_daily["date"].min().strftime("%d %b %Y")
            )

        with forecast_col2:

            st.metric(
                "Forecast End Date",
                forecast_daily["date"].max().strftime("%d %b %Y")
            )

        with forecast_col3:

            st.metric(
                "Forecast Total Sales",
                f"{forecast_daily['sales'].sum():,.2f}"
            )

        st.subheader("Daily Forecast Details")

        st.dataframe(
            forecast_daily,
            width="stretch"
        )

    else:

        st.info(
            "Select valid filters to display forecast analysis."
        )

# ====================================
# TAB 4: PRODUCT PERFORMANCE
# ====================================

with product_tab:

    st.header("Product Family Performance Analysis")

    if not filtered_data.empty:

        family_sales = (
            filtered_data.groupby("family")["sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        family_sales["sales_percentage"] = (
            family_sales["sales"] /
            family_sales["sales"].sum()
        ) * 100

        st.subheader("Top 10 Product Families")

        top_10_families = family_sales.head(10)

        st.bar_chart(
            top_10_families,
            x="family",
            y="sales",
            horizontal=True,
            width="stretch"
        )

        st.subheader("Product Family Sales Ranking")

        ranking_data = family_sales.copy()

        ranking_data.index = range(1, len(ranking_data) + 1)

        st.dataframe(
            ranking_data,
            width="stretch"
        )

        st.subheader("Explore Individual Product Families")

        selected_family = st.selectbox(
            "Select a Product Family",
            family_sales["family"].tolist(),
            key="product_family_selector"
        )

        selected_family_data = family_sales[
            family_sales["family"] == selected_family
        ].iloc[0]

        family_col1, family_col2 = st.columns(2)

        with family_col1:

            st.metric(
                "Predicted Sales",
                f"{selected_family_data['sales']:,.2f}"
            )

        with family_col2:

            st.metric(
                "Sales Contribution",
                f"{selected_family_data['sales_percentage']:.2f}%"
            )

        selected_family_daily = (
            filtered_data[
                filtered_data["family"] == selected_family
            ]
            .groupby("date")["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        st.subheader("Selected Product Family Daily Trend")

        st.line_chart(
            selected_family_daily,
            x="date",
            y="sales",
            width="stretch"
        )

        st.write(
            f"{selected_family} contributes "
            f"{selected_family_data['sales_percentage']:.2f}% "
            "of filtered forecast sales."
        )

    else:

        st.info(
            "No product family data is available for the selected filters."
        )

# ====================================
# TAB 5: STORE PERFORMANCE
# ====================================

with store_tab:

    st.header("Store Performance Analysis")

    if not filtered_data.empty:

        store_sales = (
            filtered_data.groupby("store_nbr")["sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        store_sales["sales_percentage"] = (
            store_sales["sales"] /
            store_sales["sales"].sum()
        ) * 100

        st.subheader("Top 10 Stores by Predicted Sales")

        top_10_stores = store_sales.head(10)

        st.bar_chart(
            top_10_stores,
            x="store_nbr",
            y="sales",
            horizontal=True,
            width="stretch"
        )

        st.subheader("Store Sales Ranking")

        st.dataframe(
            store_sales,
            width="stretch"
        )

        st.subheader("Explore Individual Store Performance")

        selected_store = st.selectbox(
            "Select a Store",
            store_sales["store_nbr"].tolist(),
            key="store_selector"
        )

        selected_store_data = store_sales[
            store_sales["store_nbr"] == selected_store
        ].iloc[0]

        store_col1, store_col2 = st.columns(2)

        with store_col1:

            st.metric(
                "Predicted Store Sales",
                f"{selected_store_data['sales']:,.2f}"
            )

        with store_col2:

            st.metric(
                "Sales Contribution",
                f"{selected_store_data['sales_percentage']:.2f}%"
            )

        selected_store_daily = (
            filtered_data[
                filtered_data["store_nbr"] == selected_store
            ]
            .groupby("date")["sales"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        st.subheader("Selected Store Daily Sales Trend")

        st.line_chart(
            selected_store_daily,
            x="date",
            y="sales",
            width="stretch"
        )

        st.write(
            f"Store {selected_store} generated predicted sales of "
            f"{selected_store_data['sales']:,.2f} "
            "during the selected forecast period."
        )

    else:

        st.info(
            "No store data is available for the selected filters."
        )

# ------------------------------------
# 13. Dashboard Completion
# ------------------------------------

st.sidebar.divider()

st.sidebar.caption(
    "Retail Business Intelligence Platform"
)
