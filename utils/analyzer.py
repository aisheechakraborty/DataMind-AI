import pandas as pd
import numpy as np
from typing import Dict, Any, List

def classify_column_type(series: pd.Series) -> str:
    """
    Classifies a pandas Series into one of: 'numeric', 'categorical', 'datetime', 'boolean', or 'text'.
    """
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    elif pd.api.types.is_numeric_dtype(series):
        # If it has very few unique integers, it might be categorical, but let's classify as numeric for stats
        # If it's 0 and 1, it might be boolean
        unique_count = series.nunique()
        if unique_count == 2 and set(series.dropna().unique()).issubset({0, 1}):
            return "boolean"
        return "numeric"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    
    # Try parsing string to datetime to see if it is a datetime column
    # We check a sample to avoid performance hits on massive datasets
    sample = series.dropna().head(100)
    if len(sample) > 0:
        try:
            parsed = pd.to_datetime(sample, errors='coerce')
            if parsed.notna().sum() / len(sample) > 0.8:
                return "datetime"
        except (ValueError, TypeError):
            pass

    # Check unique value ratio to determine if it is free-text or categorical
    unique_ratio = series.nunique() / len(series) if len(series) > 0 else 0
    if unique_ratio > 0.5 and series.astype(str).str.len().mean() > 30:
        return "text"
    
    return "categorical"

def analyze_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a full metadata analysis, column descriptions, and stats for the given DataFrame.
    """
    if df.empty:
        return {
            "row_count": 0,
            "col_count": 0,
            "missing_cells": 0,
            "missing_percent": 0.0,
            "duplicate_rows": 0,
            "columns": [],
            "correlations": {}
        }

    total_cells = df.size
    missing_cells = int(df.isna().sum().sum())
    missing_percent = float((missing_cells / total_cells) * 100) if total_cells > 0 else 0.0
    duplicate_rows = int(df.duplicated().sum())

    columns_analysis = []
    
    for col_name in df.columns:
        series = df[col_name]
        col_type = classify_column_type(series)
        
        # Base column properties
        col_info = {
            "name": col_name,
            "type": col_type,
            "dtype": str(series.dtype),
            "null_count": int(series.isna().sum()),
            "null_percent": float((series.isna().sum() / len(df)) * 100) if len(df) > 0 else 0.0,
            "unique_count": int(series.nunique()),
            "sample_values": series.dropna().head(5).tolist()
        }

        # Type-specific stats
        stats = {}
        if col_type == "numeric":
            stats["min"] = float(series.min()) if not pd.isna(series.min()) else None
            stats["max"] = float(series.max()) if not pd.isna(series.max()) else None
            stats["mean"] = float(series.mean()) if not pd.isna(series.mean()) else None
            stats["median"] = float(series.median()) if not pd.isna(series.median()) else None
            stats["std"] = float(series.std()) if not pd.isna(series.std()) else None
        elif col_type == "categorical" or col_type == "boolean":
            value_counts = series.value_counts()
            if not value_counts.empty:
                stats["top_value"] = str(value_counts.index[0])
                stats["top_count"] = int(value_counts.iloc[0])
                stats["top_percent"] = float((value_counts.iloc[0] / len(df)) * 100) if len(df) > 0 else 0.0
        elif col_type == "datetime":
            try:
                parsed_series = pd.to_datetime(series, errors='coerce')
                min_val = parsed_series.min()
                max_val = parsed_series.max()
                stats["min"] = str(min_val.date()) if not pd.isna(min_val) else None
                stats["max"] = str(max_val.date()) if not pd.isna(max_val) else None
            except Exception:
                pass
                
        col_info["stats"] = stats
        columns_analysis.append(col_info)

    # Correlation Matrix (for numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_data = {}
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr(method='pearson')
        # Filter correlation data to avoid returning giant matrices, only keeping actual scores
        for c1 in corr_matrix.columns:
            correlation_data[c1] = {}
            for c2 in corr_matrix.index:
                val = corr_matrix.loc[c1, c2]
                if not pd.isna(val):
                    correlation_data[c1][c2] = float(val)

    return {
        "row_count": len(df),
        "col_count": len(df.columns),
        "missing_cells": missing_cells,
        "missing_percent": missing_percent,
        "duplicate_rows": duplicate_rows,
        "columns": columns_analysis,
        "correlations": correlation_data
    }

def get_textual_summary(analysis: Dict[str, Any], df_sample_rows: int = 5, df: pd.DataFrame = None) -> str:
    """
    Serializes dataset analysis into a compact markdown summary, ideal for Gemini API consumption.
    """
    summary = []
    summary.append(f"### Dataset General Metadata:")
    summary.append(f"- Rows: {analysis['row_count']}")
    summary.append(f"- Columns: {analysis['col_count']}")
    summary.append(f"- Total Missing Cells: {analysis['missing_cells']} ({analysis['missing_percent']:.2f}%)")
    summary.append(f"- Duplicate Rows: {analysis['duplicate_rows']}")
    
    summary.append(f"\n### Columns Details:")
    for col in analysis['columns']:
        t = col['type']
        nulls = f"{col['null_count']} missing"
        uniques = f"{col['unique_count']} unique values"
        
        stat_desc = ""
        if t == "numeric" and col['stats']:
            s = col['stats']
            stat_desc = f"(Min: {s.get('min')}, Max: {s.get('max')}, Mean: {s.get('mean'):.2f}, Median: {s.get('median'):.2f})"
        elif (t == "categorical" or t == "boolean") and col['stats']:
            s = col['stats']
            stat_desc = f"(Top Value: '{s.get('top_value')}' appearing {s.get('top_count')} times, {s.get('top_percent'):.1f}%)"
        elif t == "datetime" and col['stats']:
            s = col['stats']
            stat_desc = f"(Range: {s.get('min')} to {s.get('max')})"
            
        summary.append(f"- **{col['name']}** ({t}, {col['dtype']}): {uniques}, {nulls}. {stat_desc}")

    # Significant Correlations
    corrs = analysis.get("correlations", {})
    sig_corrs = []
    for c1, target_dict in corrs.items():
        for c2, score in target_dict.items():
            if c1 < c2 and abs(score) >= 0.4:  # Only report unique pairs with correlation >= 0.4
                sig_corrs.append(f"  - {c1} & {c2}: {score:.2f}")
                
    if sig_corrs:
        summary.append(f"\n### Key Linear Correlations (>= 0.40):")
        summary.extend(sig_corrs)

    if df is not None:
        summary.append(f"\n### First {df_sample_rows} Rows Sample:")
        # Render a clean string representation of sample
        sample_str = df.head(df_sample_rows).to_string()
        summary.append(f"```\n{sample_str}\n```")

    return "\n".join(summary)
