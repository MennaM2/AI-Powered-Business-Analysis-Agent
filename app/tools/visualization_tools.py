import os

import matplotlib

# Force a non-interactive backend. Streamlit runs matplotlib off the
# main thread in some environments, and the default backend can try to
# open a GUI window, which either errors or silently hangs. This must
# happen before pyplot is imported anywhere in the process.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def create_churn_plot(file_path: str) -> dict:

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    target_column = next(
        (
            column
            for column in df.columns
            if column.lower() == "churn"
        ),
        None
    )

    if target_column is None:
        return {
            "error": "No Churn column was found."
        }

    os.makedirs("outputs", exist_ok=True)

    counts = df[target_column].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    output_path = "outputs/churn_distribution.png"

    plt.savefig(output_path)
    plt.close()

    return {
        "status": "success",
        "file_path": output_path,
        "distribution": counts.to_dict()
    }


# ---------------------------------------------------------------------
# Visualization Skill
# ---------------------------------------------------------------------
# A single, generic visualization tool the agent can call for any chart
# type, instead of one hardcoded function per chart. This is what lets
# the "Visualization Skill" scale to new chart types without adding a
# new tool (and a new schema, and new latency) every time.

SUPPORTED_CHART_TYPES = {
    "histogram",
    "bar_chart",
    "box_plot",
    "correlation_heatmap",
    "scatter_plot",
    "target_distribution",
    "category_comparison"
}


def _safe_filename(*parts: str) -> str:
    """Turn chart-type + column names into a filesystem-safe filename."""
    raw = "_".join(part for part in parts if part)
    cleaned = "".join(
        char if char.isalnum() or char in ("_", "-") else "_"
        for char in raw
    )
    return cleaned or "chart"


def create_visualization(
    file_path: str,
    chart_type: str,
    column: str = None,
    column_x: str = None,
    column_y: str = None,
    category_column: str = None,
    target_column: str = None
) -> dict:
    """
    Create a chart from the dataset and save it as a PNG.

    chart_type must be one of:
      histogram            - needs `column` (numeric)
      bar_chart             - needs `column` (categorical)
      box_plot               - needs `column` (numeric), optional `category_column` to group by
      correlation_heatmap    - no columns needed, uses all numeric columns
      scatter_plot            - needs `column_x` and `column_y` (both numeric)
      target_distribution     - needs `target_column` (categorical/binary)
      category_comparison      - needs `category_column` and `target_column`

    Returns the saved file path plus compact metadata describing what
    was plotted, so the agent can explain the chart without needing to
    "see" the image.
    """

    if chart_type not in SUPPORTED_CHART_TYPES:
        return {
            "error": (
                f"Unsupported chart_type '{chart_type}'. "
                f"Supported types: {sorted(SUPPORTED_CHART_TYPES)}"
            )
        }

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        return {"error": f"Unable to read CSV: {exc}"}

    os.makedirs("outputs", exist_ok=True)

    def missing_column_error(name, value):
        return {"error": f"Column '{value}' ({name}) was not found."}

    plt.figure(figsize=(7, 5))
    metadata = {}

    try:
        if chart_type == "histogram":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            df[column].dropna().plot(kind="hist", bins=20)
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")

            metadata = {
                "column": column,
                "mean": round(float(df[column].mean()), 4),
                "std": round(float(df[column].std()), 4)
            }

        elif chart_type == "bar_chart":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            counts = df[column].value_counts().head(15)
            counts.plot(kind="bar")
            plt.title(f"Counts by {column}")
            plt.xlabel(column)
            plt.ylabel("Count")

            metadata = {
                "column": column,
                "top_values": counts.to_dict()
            }

        elif chart_type == "box_plot":
            if not column or column not in df.columns:
                plt.close()
                return missing_column_error("column", column)

            if category_column and category_column in df.columns:
                df.boxplot(column=column, by=category_column)
                plt.title(f"{column} by {category_column}")
                plt.suptitle("")
                plt.xlabel(category_column)
            else:
                df.boxplot(column=column)
                plt.title(f"Distribution of {column}")

            plt.ylabel(column)

            metadata = {
                "column": column,
                "category_column": category_column,
                "median": round(float(df[column].median()), 4),
                "q1": round(float(df[column].quantile(0.25)), 4),
                "q3": round(float(df[column].quantile(0.75)), 4)
            }

        elif chart_type == "correlation_heatmap":
            numeric_df = df.select_dtypes(include="number")

            if numeric_df.shape[1] < 2:
                plt.close()
                return {
                    "error": (
                        "Need at least two numerical columns "
                        "for a correlation heatmap."
                    )
                }

            corr = numeric_df.corr()

            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(
                range(len(corr.columns)),
                corr.columns,
                rotation=90
            )
            plt.yticks(
                range(len(corr.columns)),
                corr.columns
            )
            plt.title("Correlation Heatmap")

            # Compact metadata: only the strongest pairs, not the
            # whole matrix, to keep the tool result small.
            pairs = (
                corr.where(
                    ~corr.abs().eq(1.0)
                )
                .unstack()
                .dropna()
                .abs()
                .sort_values(ascending=False)
            )
            seen = set()
            top_pairs = []
            for (col_a, col_b), value in pairs.items():
                key = frozenset((col_a, col_b))
                if key in seen:
                    continue
                seen.add(key)
                top_pairs.append({
                    "columns": [col_a, col_b],
                    "correlation": round(float(corr.loc[col_a, col_b]), 4)
                })
                if len(top_pairs) >= 5:
                    break

            metadata = {"strongest_correlations": top_pairs}

        elif chart_type == "scatter_plot":
            if not column_x or column_x not in df.columns:
                plt.close()
                return missing_column_error("column_x", column_x)
            if not column_y or column_y not in df.columns:
                plt.close()
                return missing_column_error("column_y", column_y)

            plt.scatter(df[column_x], df[column_y], alpha=0.6)
            plt.title(f"{column_y} vs {column_x}")
            plt.xlabel(column_x)
            plt.ylabel(column_y)

            metadata = {
                "column_x": column_x,
                "column_y": column_y,
                "correlation": round(
                    float(df[column_x].corr(df[column_y])), 4
                )
            }

        elif chart_type == "target_distribution":
            if not target_column or target_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "target_column", target_column
                )

            counts = df[target_column].value_counts()
            counts.plot(kind="bar")
            plt.title(f"Distribution of {target_column}")
            plt.xlabel(target_column)
            plt.ylabel("Count")

            metadata = {
                "target_column": target_column,
                "distribution": counts.to_dict()
            }

        elif chart_type == "category_comparison":
            if not category_column or category_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "category_column", category_column
                )
            if not target_column or target_column not in df.columns:
                plt.close()
                return missing_column_error(
                    "target_column", target_column
                )

            grouped = df.groupby(category_column)[target_column].mean()
            grouped.plot(kind="bar")
            plt.title(f"Average {target_column} by {category_column}")
            plt.xlabel(category_column)
            plt.ylabel(f"Average {target_column}")

            metadata = {
                "category_column": category_column,
                "target_column": target_column,
                "averages": grouped.round(4).to_dict()
            }

        plt.tight_layout()

        filename = _safe_filename(
            chart_type, column, column_x, column_y,
            category_column, target_column
        )
        output_path = f"outputs/{filename}.png"

        plt.savefig(output_path)
        plt.close()

        return {
            "status": "success",
            "chart_type": chart_type,
            "file_path": output_path,
            "metadata": metadata
        }

    except Exception as exc:
        plt.close()
        return {"error": f"Failed to create visualization: {exc}"}