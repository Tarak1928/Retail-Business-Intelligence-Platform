# Retail Business Intelligence & Forecasting Platform

An end-to-end retail sales analytics and forecasting application built
with **Python, Pandas, LightGBM, and Streamlit**. The platform analyses
historical store sales, generates model-based sales forecasts, and
presents business insights through an interactive dashboard.

## Live Application

**Deployed dashboard:**
<https://retail-business-intelligence-platform.streamlit.app/>

**GitHub repository:**
<https://github.com/Tarak1928/Retail-Business-Intelligence-Platform>

## Project Overview

Retail businesses work with sales information across multiple stores,
product families, and dates. Identifying historical patterns and
estimating future demand can be difficult when data is spread across
multiple files and reports.

This project builds a complete workflow to:

-   Prepare and validate retail time-series data.
-   Explore sales trends by date, store, product family, and other
    business dimensions.
-   Engineer calendar, lag, rolling, and categorical features.
-   Train and evaluate a LightGBM sales forecasting model.
-   Generate predictions for the supplied test period.
-   Present historical analysis and forecast results in a five-tab
    Streamlit dashboard.
-   Filter dashboard results and download CSV reports.

> **Prediction target:** The model forecasts sales. It does not
> calculate profit, margin, or cash flow.

## Dataset

The project uses the public **Store Sales -- Time Series Forecasting**
dataset from Corporación Favorita, available through Kaggle.

-   [Kaggle dataset and competition
    page](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

  Dataset detail                                    Value
  ---------------------------- --------------------------
  Training observations                         3,000,888
  Test observations                                28,512
  Historical training period     2013-01-01 to 2017-08-15
  Test/forecast period           2017-08-16 to 2017-08-31
  Stores                                               54
  Product families                                     33
  Target variable                                 `sales`

The dataset includes sales records and supporting information such as
store metadata, oil prices, holidays/events, and transactions.

The original raw Kaggle files are kept separate from the processed and
deployment datasets.

## Technology Stack

  Technology                  Purpose
  --------------------------- ----------------------------------------------------------
  Python 3.11                 Main programming language
  Pandas                      Data cleaning, transformation, aggregation, and analysis
  NumPy                       Numerical operations
  Matplotlib / Plotly         Data visualisation
  LightGBM                    Sales forecasting model
  Scikit-learn                Supporting machine-learning utilities and evaluation
  Streamlit                   Interactive web dashboard
  Git and GitHub              Version control and source-code hosting
  Streamlit Community Cloud   Application deployment

## Project Workflow

1.  **Data acquisition and preparation** -- load the Kaggle CSV files,
    inspect data quality, prepare dates, and combine relevant supporting
    information.
2.  **Exploratory Data Analysis** -- analyse historical sales patterns
    across product families, stores, dates, promotions, and day types.
3.  **Feature engineering** -- create calendar features, sales lags,
    rolling features, and encoded categorical variables.
4.  **Model development** -- establish a seasonal-naive baseline, train
    LightGBM experiments, and tune model parameters.
5.  **Model selection and forecasting** -- select the best validation
    model, train the final model using the labelled historical data, and
    generate predictions for the test records.
6.  **Forecast analysis** -- produce summaries and visualisations by
    date, store, and product family.
7.  **Dashboard development** -- build an interactive Streamlit
    application with filters, KPIs, charts, and downloadable reports.
8.  **Deployment** -- prepare compact dashboard datasets, push the
    project to GitHub, and deploy the application using Streamlit
    Community Cloud.

## Model Performance

A seasonal-naive baseline was compared with LightGBM experiments. The
best recorded LightGBM validation results were:

  Metric     Validation result
  -------- -------------------
  RMSLE              0.4335677
  MAE                    78.99
  RMSE                  268.79

The final model was trained on the labelled historical training data and
generated **28,512 test-period predictions**.

Recorded aggregate forecast statistics:

-   Sum of predictions: approximately **12,150,547.02**
-   Average prediction per test record: approximately **426.16**
-   Maximum prediction for an individual record: approximately
    **12,996.7**

**Important evaluation note:** The supplied Kaggle test dataset does not
contain actual sales labels. Therefore, the metrics above are validation
metrics, not measured test-set accuracy. The aggregate forecast values
are model estimates, not realised sales.

## Interactive Dashboard

The deployed Streamlit dashboard contains five main sections.

### 1. Overview

-   Total predicted sales and forecast record KPIs.
-   Number of forecast days, stores, and product families.
-   Leading product family and store within the current filters.
-   Filtered forecast data preview.
-   CSV downloads for forecast, product-family, store, and daily
    reports.

### 2. Historical Analysis

-   Historical daily sales trend.
-   Monthly historical sales.
-   Average daily total sales comparison for weekdays and weekends.
-   Recent historical daily sales records.

### 3. Forecast Analysis

-   Historical versus predicted daily sales comparison.
-   Forecast start and end dates.
-   Total forecast sales.
-   Daily forecast details.

### 4. Product Performance

-   Top 10 product families by predicted sales.
-   Product-family sales ranking.
-   Sales contribution.
-   Individual product-family selection and daily trend.

### 5. Store Performance

-   Top 10 stores by predicted sales.
-   Store sales ranking.
-   Sales contribution.
-   Individual store selection and daily trend.

### Interactive Filters

The sidebar allows users to filter forecast results by:

-   Forecast date range.
-   Store.
-   Product family.

Dashboard metrics, rankings, charts, and downloadable reports respond to
the selected forecast records.

## Repository Structure

The main project structure is organised as follows:

``` text
Business-AI-Platform/
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── raw/
│   │   └── store-sales-kaggle/
│   ├── processed/
│   └── deployment/
│       ├── test_metadata.csv
│       └── historical_daily_sales.csv
│
├── models/
│   └── final_lightgbm_model.txt
│
├── reports/
│   ├── eda_business_insights.md
│   ├── feature_importance.csv
│   ├── final_model_summary.json
│   ├── forecast_business_summary.json
│   ├── forecast_daily_analysis.csv
│   ├── forecast_family_analysis.csv
│   ├── forecast_store_analysis.csv
│   └── test_forecast_predictions.csv
│
├── src/
│   ├── app.py
│   ├── prepare_deployment_data.py
│   ├── analyze_forecast.py
│   ├── analyze_feature_importance.py
│   ├── forecast_test_sales.py
│   ├── select_final_model.py
│   └── tune_lightgbm.py
│
├── requirements.txt
├── runtime.txt
├── submission.csv
└── .gitignore
```

Large raw and processed datasets, virtual environments, caches, and
experimental model files are excluded from Git. The compact files under
`data/deployment/` are included because the hosted dashboard needs them.

## Run Locally

### Prerequisites

-   Python 3.11
-   Git
-   A local copy of this repository

### 1. Clone the repository

``` bash
git clone https://github.com/Tarak1928/Retail-Business-Intelligence-Platform.git
cd Retail-Business-Intelligence-Platform
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dashboard dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

``` bash
streamlit run src/app.py
```

Open the local address printed by Streamlit, usually:

``` text
http://localhost:8501
```

### Required dashboard files

The dashboard expects these files to be available:

``` text
reports/test_forecast_predictions.csv
data/deployment/test_metadata.csv
data/deployment/historical_daily_sales.csv
```

The raw Kaggle data and large processed training files are not included
in the Git repository. They are needed only if you intend to reproduce
the complete data-preparation and model-training workflow from scratch.

## Re-create the Compact Deployment Data

If the processed model-ready datasets are available locally, run:

``` bash
python src/prepare_deployment_data.py
```

The script creates:

-   `data/deployment/test_metadata.csv`
-   `data/deployment/historical_daily_sales.csv`

The verified deployment files contain 28,512 test metadata records and
1,684 historical daily records.

## Key Project Outputs

-   `models/final_lightgbm_model.txt` -- selected final LightGBM model
    artefact.
-   `reports/test_forecast_predictions.csv` -- test-record predictions
    used by the dashboard.
-   `submission.csv` -- competition-format prediction file.
-   `reports/feature_importance.csv` -- model feature-importance output.
-   `reports/forecast_daily_analysis.csv` -- forecast summary by date.
-   `reports/forecast_family_analysis.csv` -- forecast summary by
    product family.
-   `reports/forecast_store_analysis.csv` -- forecast summary by store.
-   `reports/eda_business_insights.md` -- exploratory analysis findings.

## Limitations

-   The current application forecasts sales, not profit or margin.
-   Actual labels for the supplied test period are unavailable, so
    test-set accuracy cannot be calculated from this dataset.
-   Validation performance may differ on future periods or in other
    retail environments.
-   Relationships found during EDA, such as differences between promoted
    and non-promoted records, do not establish causation.
-   Feature importance describes model behaviour; it does not prove that
    a feature causes sales changes.
-   The current release does not include prediction intervals, automated
    model-drift monitoring, inventory optimisation, or an AI chat
    assistant.

## Future Enhancements

Planned next development areas include:

1.  **AI Business Assistant** -- allow users to ask natural-language
    questions about sales, stores, product families, and forecast
    results.
2.  **Forecast uncertainty** -- add prediction intervals to communicate
    forecast risk.
3.  **Model monitoring** -- monitor data quality, feature drift, and
    forecast errors when actual sales become available.
4.  **Inventory planning** -- incorporate stock levels, supplier lead
    times, and costs before generating replenishment recommendations.
5.  **Model lifecycle management** -- add scheduled retraining, model
    version tracking, and performance comparisons.

These are future enhancements and are not part of the currently deployed
release.

## Author

**Taraka Ram Bayapaneni**

GitHub: [Tarak1928](https://github.com/Tarak1928)

## Acknowledgements

-   Corporación Favorita for providing the retail dataset through the
    Kaggle competition.
-   Kaggle for hosting the Store Sales -- Time Series Forecasting
    dataset and competition.

------------------------------------------------------------------------

*This README describes the project as currently developed and deployed.
Update the project status and results as new features are implemented.*
