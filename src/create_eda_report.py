from pathlib import Path


# --------------------------------------------------
# STEP 1: FIND THE PROJECT ROOT DIRECTORY
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# STEP 2: CREATE THE REPORTS DIRECTORY
# --------------------------------------------------

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# STEP 3: WRITE THE BUSINESS INSIGHTS REPORT
# --------------------------------------------------

report_content = """
# Retail Business Intelligence Platform
## Exploratory Data Analysis – Business Insights Report

---

## 1. Executive Summary

This report presents findings from an exploratory analysis of the
Corporación Favorita Store Sales dataset.

The dataset contains historical retail sales information from
54 stores across 33 product families.

The training data covers January 2013 to August 2017 and contains
approximately 3 million sales records.

Key findings:

- Total recorded sales: 1,073,644,952.20
- Average daily sales across all stores and product families: 637,556.38
- Grocery I is the highest-selling product family.
- Store 44 has the highest total sales.
- Promoted records have substantially higher average sales than
  non-promoted records.
- Sunday is the strongest day by average daily sales.
- December has the highest average daily sales by calendar month.

These findings can help retail managers investigate product demand,
store performance, promotional activity and seasonal patterns.

---

## 2. Overall Sales Performance

| Metric | Result |
|---|---:|
| Total recorded sales | 1,073,644,952.20 |
| Average daily sales | 637,556.38 |
| Highest daily sales | 1,463,083.96 |
| Date of highest daily sales | 2017-04-01 |
| Lowest daily sales | 2,511.62 |
| Number of stores | 54 |
| Number of product families | 33 |

The average daily sales figure represents the combined sales of all
stores and product families for a day. It is not the average sales
of an individual store.

### Business observation

Retail sales vary considerably over time. Managers can use daily
and monthly sales trends to support inventory planning and resource
allocation.

---

## 3. Product Family Performance

### Top-performing product families

| Product family | Total sales |
|---|---:|
| Grocery I | 343,462,734.89 |
| Beverages | 216,954,500 |
| Produce | 122,704,700 |
| Cleaning | 97,521,290 |
| Dairy | 64,487,710 |

Grocery I and Beverages are the leading product families by total
recorded sales.

### Lowest-performing product families

| Product family | Total sales |
|---|---:|
| Books | 6,438 |
| Baby Care | 10,051 |
| Home Appliances | 41,601 |
| Hardware | 103,470 |

### Business recommendations

1. Monitor stock availability for high-demand product families.
2. Investigate whether low-selling products need different
   merchandising or assortment strategies.
3. Study product demand at individual store level before making
   inventory decisions.
4. Avoid removing products based only on total sales; consider
   profit margins and strategic importance.

---

## 4. Store Performance

### Top-performing stores

| Store | Total sales |
|---|---:|
| 44 | 62,087,553.25 |
| 45 | 54,498,010 |
| 47 | 50,948,310 |
| 3 | 50,481,910 |
| 49 | 43,420,100 |

### Lowest-performing store

| Store | Total sales |
|---|---:|
| 52 | 2,696,169.73 |

Store 44 recorded the highest total sales, while Store 52 recorded
the lowest total sales in the analyzed period.

### Business recommendations

1. Investigate the business practices and product mix of high-sales
   stores.
2. Review the location, store type, operating conditions and demand
   patterns of lower-sales stores.
3. Compare stores using average daily sales and comparable operating
   periods.
4. Identify practices from high-performing stores that may be
   useful elsewhere.

A difference in total sales does not automatically mean that a
store is operationally successful or unsuccessful. Additional
business context is required.

---

## 5. Promotion Impact Analysis

| Metric | Not promoted | Promoted |
|---|---:|---:|
| Total sales | 378,139,800 | 695,505,200 |
| Average sales per record | 158.25 | 1,137.69 |
| Number of records | 2,389,559 | 611,329 |

Promoted records had approximately 7.19 times the average sales
of non-promoted records.

### Business recommendations

1. Analyze promotion performance separately for each product family.
2. Compare promotion performance across stores.
3. Monitor sales before, during and after promotional periods.
4. Evaluate promotional effectiveness using profit margin and
   promotion costs when those data become available.

### Important analytical note

This comparison identifies an association, not proof that
promotions cause higher sales.

Popular products, store characteristics, seasonal demand and
promotion selection may influence the results.

The promotion classification used here is:
- Promoted: onpromotion greater than zero.
- Not promoted: onpromotion equal to zero.

---

## 6. Time-Based Sales Analysis

### Weekday performance

| Day | Average daily sales |
|---|---:|
| Sunday | 825,218.12 |
| Saturday | 772,205.59 |
| Monday | 617,542.71 |
| Wednesday | 593,244.55 |
| Friday | 579,574.36 |
| Tuesday | 569,926.09 |
| Thursday | 505,269.20 |

Sunday has the highest average daily sales, while Thursday has
the lowest.

### Weekday vs weekend

| Day type | Average daily sales |
|---|---:|
| Weekday | 573,143.02 |
| Weekend | 798,656.75 |

Weekend average daily sales are approximately 39.3% higher than
weekday average daily sales.

### Month-of-year performance

| Month | Average daily sales |
|---|---:|
| December | 808,565.34 |
| November | 669,464.90 |
| July | 666,858.46 |
| October | 645,809.51 |
| September | 645,614.00 |
| June | 630,111.36 |
| March | 627,280.56 |
| January | 609,304.21 |
| May | 609,026.69 |
| April | 604,454.12 |
| August | 600,520.70 |
| February | 571,895.24 |

December has the highest average daily sales among calendar months,
while February has the lowest.

### Business recommendations

1. Consider higher inventory readiness for weekends.
2. Investigate Sunday demand and staffing requirements.
3. Plan seasonal inventory and promotions ahead of December.
4. Review weekday demand to identify opportunities for improving
   sales during slower periods.

These patterns are historical observations and should be checked
against future data before operational decisions are made.

---

## 7. Overall Business Recommendations

Based on the exploratory analysis, the following actions are worth
investigating:

1. **Inventory optimization:** Monitor demand for Grocery I,
   Beverages and other high-selling product families.

2. **Store performance monitoring:** Compare store-level demand
   and investigate the reasons behind large differences in sales.

3. **Promotion evaluation:** Measure promotional performance by
   product family and store, including profitability where possible.

4. **Seasonal planning:** Prepare inventory and staffing plans for
   weekends and high-demand months.

5. **Demand forecasting:** Develop forecasting models to estimate
   future sales at store and product-family level.

6. **Management dashboards:** Present sales trends, product
   performance, store comparisons and forecasts in an interactive
   dashboard.

---

## 8. Data Limitations

The following limitations should be considered when interpreting
the results:

- The analysis measures sales, not profit.
- Product costs, operating expenses and profit margins are not
  included in the analysis.
- Promotion comparisons do not establish causal relationships.
- Store sales differences may be influenced by location, store
  size, operating conditions and product mix.
- Historical patterns do not guarantee future performance.
- Transaction data was excluded from the final model-ready
  training and test datasets because test-period transaction
  values were unavailable.
- Oil-price interpolation across the historical and test date
  range should be reviewed carefully before predictive modeling
  to avoid future-data leakage.

---

## 9. Conclusion

The exploratory analysis identified important patterns in
product demand, store performance, promotional activity and
seasonal sales.

Grocery I and Beverages lead product-family sales, Store 44
has the highest recorded store sales, promoted records show
higher average sales, and weekends show stronger average daily
sales than weekdays.

These findings provide a foundation for the next stages of the
project: feature engineering, machine learning, sales forecasting
and interactive business intelligence dashboards.

---

**Project:** AI-Powered Retail Business Intelligence and Forecasting Platform

**Analysis:** Exploratory Data Analysis (EDA)

**Tools:** Python, Pandas, NumPy, Matplotlib and CSV datasets
"""

# --------------------------------------------------
# STEP 4: SAVE THE REPORT
# --------------------------------------------------

report_path = REPORTS_DIR / "eda_business_insights.md"

with open(report_path, "w", encoding="utf-8") as file:
    file.write(report_content)

# --------------------------------------------------
# STEP 5: DISPLAY SUCCESS MESSAGE
# --------------------------------------------------

print("=" * 60)
print("EDA BUSINESS INSIGHTS REPORT GENERATED")
print("=" * 60)

print(f"Report saved at: {report_path}")
print("Report format: Markdown (.md)")
print("Status: SUCCESS")