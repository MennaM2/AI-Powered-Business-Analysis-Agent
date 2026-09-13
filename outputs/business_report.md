# Business Analysis Report

_Generated 2026-09-13 07:44:18_

## Executive Summary

- 'returned_at' has 90.0% missing values, which may bias any analysis using it.
- By order_id, 'Shipped' leads (2366887444.0) and 'Returned' trails (781806762.0) among status.

## Key Metrics

- Rows: 125083
- Columns: 9
- Duplicate rows: 0
- Metric analyzed: order_id
- Grouping dimension: status

## Trends

- No usable date/metric combination was found for trend analysis.

## Anomalies

- No significant anomalies detected.

## Top / Bottom Performers

Top status by order_id (sum):

- Shipped: 2366887444.0
- Complete: 1946028358.0
- Processing: 1548928853.0
- Cancelled: 1179289569.0
- Returned: 781806762.0

Bottom status by order_id:

- Returned: 781806762.0
- Cancelled: 1179289569.0
- Processing: 1548928853.0
- Complete: 1946028358.0
- Shipped: 2366887444.0

## Insights

- 'returned_at' has 90.0% missing values, which may bias any analysis using it.
- By order_id, 'Shipped' leads (2366887444.0) and 'Returned' trails (781806762.0) among status.

## Recommendations

- Investigate why 'returned_at' has a high missing rate before using it in reporting or modeling.
- Examine why 'Returned' underperforms other status on order_id, and whether 'Shipped' has practices worth replicating.

## Limitations

- Generated automatically from the uploaded CSV only; it does not incorporate context outside the data.
- Date, metric, and grouping columns were auto-detected when not specified and may not match true business intent.
- Correlations, trends, and feature importance reflect association within this dataset, not proven causation.