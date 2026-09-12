# Business Analysis Report

_Generated 2026-09-12 09:12:14_

## Executive Summary

- 'returned_at' has 90.0% missing values, which may bias any analysis using it.
- num_of_item is increasing overall (8016.67% total change).
- The sharpest drop in num_of_item was -89.11% in 2024-09-30.
- num_of_item moved from 13410.0 in 2024-08 to 1461.0 in 2024-09 (-89.11%).
- 'Returned' had the steepest decline (-90.61%) between the two periods.
- 6400 anomalies detected via IQR + Z-score (univariate).
- By num_of_item, 'Shipped' leads (54811.0) and 'Returned' trails (18039.0) among status.
- A baseline model predicts 'status' with 0.2187 accuracy; the top driver is 'numerical__order_id'.

## Key Metrics

- Rows: 125083
- Columns: 9
- Duplicate rows: 0
- Metric analyzed: num_of_item
- Date column used: created_at
- Grouping dimension: status

## Trends

- Overall direction for **num_of_item**: increasing
- Total change over the period: 8016.67%
- Largest single-period drop: -89.11% in 2024-09-30
- Largest single-period rise: 144.44% in 2019-02-28

## Period Comparison

- 2024-08: 13410.0
- 2024-09: 1461.0
- Change: -11949.0 (-89.11%, decrease)

Fastest-growing status:

- Cancelled: -88.42%
- Shipped: -88.98%
- Processing: -89.07%

Fastest-declining status:

- Returned: -90.61%
- Complete: -89.08%
- Processing: -89.07%

## Anomalies

- 6400 anomalies detected using IQR + Z-score (univariate).
- {'row_index': 22, 'value': 4.0, 'date': '2024-03-31 01:43:00 UTC'}
- {'row_index': 60, 'value': 4.0, 'date': '2024-02-25 15:17:00 UTC'}
- {'row_index': 71, 'value': 4.0, 'date': '2020-01-16 15:16:00 UTC'}
- {'row_index': 75, 'value': 4.0, 'date': '2024-08-06 03:06:00 UTC'}
- {'row_index': 78, 'value': 4.0, 'date': '2024-01-15 13:30:00 UTC'}

## Top / Bottom Performers

Top status by num_of_item (sum):

- Shipped: 54811.0
- Complete: 45297.0
- Processing: 36089.0
- Cancelled: 27519.0
- Returned: 18039.0

Bottom status by num_of_item:

- Returned: 18039.0
- Cancelled: 27519.0
- Processing: 36089.0
- Complete: 45297.0
- Shipped: 54811.0

## Predictive Signal

- Target column: status
- Task type: classification
- accuracy: 0.2187
- precision: 0.2253
- recall: 0.2187
- f1_score: 0.2214

Top features:
- numerical__order_id: 0.5348
- numerical__user_id: 0.4604
- numerical__num_of_item: 0.004
- categorical__gender_F: 0.0004
- categorical__gender_M: 0.0004

## Insights

- 'returned_at' has 90.0% missing values, which may bias any analysis using it.
- num_of_item is increasing overall (8016.67% total change).
- The sharpest drop in num_of_item was -89.11% in 2024-09-30.
- num_of_item moved from 13410.0 in 2024-08 to 1461.0 in 2024-09 (-89.11%).
- 'Returned' had the steepest decline (-90.61%) between the two periods.
- 6400 anomalies detected via IQR + Z-score (univariate).
- By num_of_item, 'Shipped' leads (54811.0) and 'Returned' trails (18039.0) among status.
- A baseline model predicts 'status' with 0.2187 accuracy; the top driver is 'numerical__order_id'.

## Recommendations

- Investigate why 'returned_at' has a high missing rate before using it in reporting or modeling.
- Review what happened around 2024-09-30 - it had the largest period-over-period drop in num_of_item.
- Prioritize investigating 'Returned' - it declined -90.61% between the two most recent periods.
- Manually review the flagged anomalous rows for data-entry errors or genuinely unusual business events.
- Examine why 'Returned' underperforms other status on num_of_item, and whether 'Shipped' has practices worth replicating.

## Limitations

- Generated automatically from the uploaded CSV only; it does not incorporate context outside the data.
- Date, metric, and grouping columns were auto-detected when not specified and may not match true business intent.
- Correlations, trends, and feature importance reflect association within this dataset, not proven causation.