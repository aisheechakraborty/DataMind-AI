import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Theme configuration
THEME_BG = "#15151A"  # Dark glassmorphism card background
THEME_TEXT = "#E5E5EA"  # Light gray text
THEME_GRID = "#2C2C35"  # Dark gridlines
THEME_ACCENT = "#E50914"  # Crimson Red accent
COLOR_SEQUENCE = ["#E50914", "#F44336", "#FF7961", "#333333", "#555555", "#888888", "#CCCCCC"]

def apply_custom_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """
    Applies DataMind's dark SaaS theme (black/red/white) to a Plotly figure.
    """
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 18, 'color': '#FFFFFF', 'family': 'Inter, sans-serif'},
            'x': 0.05,
            'y': 0.95
        },
        paper_bgcolor='rgba(0,0,0,0)',  # transparent paper to allow container styling
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color=THEME_TEXT, family='Inter, sans-serif'),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(
            bgcolor='rgba(26,26,34,0.6)',
            bordercolor=THEME_GRID,
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(
            gridcolor=THEME_GRID,
            linecolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor=THEME_GRID,
            linecolor=THEME_GRID,
            zerolinecolor=THEME_GRID,
            tickfont=dict(size=11)
        )
    )
    # If the chart has traces, adjust colors if using standard plotly colorways
    for trace in fig.data:
        if hasattr(trace, 'type') and trace.type not in ['heatmap']:
            try:
                trace.update(marker=dict(line=dict(color='rgba(0,0,0,0)', width=0)))
            except Exception:
                pass
    return fig

def plot_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, title: str = "") -> go.Figure:
    """
    Generates a stylized Bar Chart.
    """
    # Limit categories for visual clarity
    unique_vals = df[x_col].nunique()
    if unique_vals > 25:
        # Group by and aggregate to keep top 20
        grouped = df.groupby(x_col)[y_col].sum().reset_index()
        df_plot = grouped.sort_values(by=y_col, ascending=False).head(20)
        title_suffix = " (Top 20 Categories)"
    else:
        df_plot = df
        title_suffix = ""

    fig = px.bar(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col or x_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data=df_plot.columns
    )
    return apply_custom_theme(fig, title or f"{y_col} by {x_col}{title_suffix}")

def plot_line(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, title: str = "") -> go.Figure:
    """
    Generates a stylized Line Chart (best for time-series or sequential data).
    """
    # Check if x_col needs to be sorted (especially for dates/ordinals)
    df_sorted = df.copy()
    try:
        if pd.api.types.is_datetime64_any_dtype(df_sorted[x_col]) or df_sorted[x_col].dtype == 'object':
            df_sorted[x_col] = pd.to_datetime(df_sorted[x_col], errors='ignore')
    except Exception:
        pass
    
    df_sorted = df_sorted.sort_values(by=x_col)

    fig = px.line(
        df_sorted,
        x=x_col,
        y=y_col,
        color=color_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        markers=True
    )
    fig.update_traces(line=dict(width=3, color=THEME_ACCENT) if not color_col else {})
    return apply_custom_theme(fig, title or f"{y_col} Over {x_col}")

def plot_pie(df: pd.DataFrame, names_col: str, values_col: Optional[str] = None, title: str = "") -> go.Figure:
    """
    Generates a stylized Pie/Donut Chart.
    """
    if values_col is None:
        # Pie chart of frequencies/counts
        df_counts = df[names_col].value_counts().reset_index()
        df_counts.columns = [names_col, 'Count']
        values_col = 'Count'
        df_plot = df_counts
    else:
        df_plot = df.groupby(names_col)[values_col].sum().reset_index()
        
    # Limit slices to top 10 + "Other" if too many
    if df_plot[names_col].nunique() > 10:
        df_plot = df_plot.sort_values(by=values_col, ascending=False)
        top_n = df_plot.head(9).copy()
        others_val = df_plot.iloc[9:][values_col].sum()
        other_row = pd.DataFrame([{names_col: 'Other Categories', values_col: others_val}])
        df_plot = pd.concat([top_n, other_row], ignore_index=True)

    fig = px.pie(
        df_plot,
        names=names_col,
        values=values_col,
        hole=0.4,
        color_discrete_sequence=COLOR_SEQUENCE
    )
    return apply_custom_theme(fig, title or f"Distribution of {names_col} (by {values_col})")

def plot_histogram(df: pd.DataFrame, x_col: str, bins: int = 30, title: str = "") -> go.Figure:
    """
    Generates a stylized Histogram.
    """
    fig = px.histogram(
        df,
        x=x_col,
        nbins=bins,
        color_discrete_sequence=[THEME_ACCENT]
    )
    fig.update_layout(bargap=0.08)
    return apply_custom_theme(fig, title or f"Distribution of {x_col}")

def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: Optional[str] = None, size_col: Optional[str] = None, title: str = "") -> go.Figure:
    """
    Generates a stylized Scatter Plot.
    """
    # Sample down to 2000 points if dataset is massive to maintain rendering speed
    df_plot = df
    if len(df) > 2000:
        df_plot = df.sample(2000, random_state=42)
        title_suffix = " (Sampled 2,000 points)"
    else:
        title_suffix = ""

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        color_discrete_sequence=COLOR_SEQUENCE,
        hover_data=df_plot.columns
    )
    
    # Try adding regression trendline if no color grouping and variables are purely numeric
    if not color_col and np.issubdtype(df[x_col].dtype, np.number) and np.issubdtype(df[y_col].dtype, np.number):
        try:
            # We can draw standard line using numpy fit to avoid statsmodels dependency
            clean_df = df_plot[[x_col, y_col]].dropna()
            if len(clean_df) > 1:
                x = clean_df[x_col].values
                y = clean_df[y_col].values
                m, c = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = m * x_line + c
                fig.add_trace(
                    go.Scatter(
                        x=x_line, y=y_line, 
                        mode='lines', 
                        name='Trendline',
                        line=dict(color='#FFFFFF', width=1.5, dash='dash')
                    )
                )
        except Exception:
            pass

    return apply_custom_theme(fig, (title or f"{y_col} vs {x_col}") + title_suffix)

def plot_correlation_heatmap(corr_matrix: Dict[str, Dict[str, float]], title: str = "Correlation Matrix") -> go.Figure:
    """
    Generates a stylized Correlation Heatmap.
    """
    if not corr_matrix:
        # Return empty figure
        fig = go.Figure()
        return apply_custom_theme(fig, "No numerical data for correlations")
        
    cols = list(corr_matrix.keys())
    z_data = []
    for c1 in cols:
        row = []
        for c2 in cols:
            row.append(corr_matrix[c1].get(c2, 0.0))
        z_data.append(row)
        
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=cols,
        y=cols,
        colorscale=[[0.0, "#0F0F12"], [0.5, "#444444"], [1.0, THEME_ACCENT]],
        zmin=-1.0,
        zmax=1.0,
        colorbar=dict(title="Correlation", thickness=15)
    ))
    return apply_custom_theme(fig, title)

def generate_auto_charts(df: pd.DataFrame, columns_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reviews columns and auto-generates up to 3 highly appropriate charts for the dashboard.
    Returns a list of dicts: {"title": str, "fig": Figure, "description": str}
    """
    recommended = []
    
    # 1. Separate column types
    numeric_cols = [c['name'] for c in columns_analysis if c['type'] == 'numeric']
    categorical_cols = [c['name'] for c in columns_analysis if c['type'] in ('categorical', 'boolean')]
    datetime_cols = [c['name'] for c in columns_analysis if c['type'] == 'datetime']
    
    # Check if we can find date columns
    time_col = datetime_cols[0] if datetime_cols else None
    
    # Chart 1: Time Series Trend (if Date + Numeric columns exist)
    if time_col and numeric_cols:
        target_num = numeric_cols[0]
        # Let's check for sales, revenue, profit, count, or value
        priority_cols = ['sales', 'revenue', 'profit', 'amount', 'price', 'total', 'count', 'marks', 'score']
        for c in numeric_cols:
            if any(p in c.lower() for p in priority_cols):
                target_num = c
                break
        fig = plot_line(df, time_col, target_num)
        recommended.append({
            "title": f"Timeline Trend Analysis ({target_num} over time)",
            "fig": fig,
            "description": f"Automated line chart visualizing values of **{target_num}** mapped across the temporal column **{time_col}**."
        })
        
    # Chart 2: Categorical Distribution (Bar or Pie)
    if categorical_cols:
        cat_col = categorical_cols[0]
        # Try to find a column named 'category', 'segment', 'region', 'city', 'department', 'gender'
        priority_cats = ['category', 'segment', 'region', 'city', 'department', 'gender', 'country', 'status', 'type']
        for c in categorical_cols:
            if any(p in c.lower() for p in priority_cats):
                cat_col = c
                break
                
        # If numeric columns exist, make a Bar Chart of sums
        if numeric_cols:
            target_num = numeric_cols[0]
            priority_nums = ['sales', 'revenue', 'profit', 'amount', 'total', 'count']
            for c in numeric_cols:
                if any(p in c.lower() for p in priority_nums):
                    target_num = c
                    break
            
            # Aggregate to create a clean bar chart
            agg_df = df.groupby(cat_col)[target_num].sum().reset_index()
            fig = plot_bar(agg_df, cat_col, target_num, title=f"Sum of {target_num} by {cat_col}")
            recommended.append({
                "title": f"Categorical Distribution ({target_num} by {cat_col})",
                "fig": fig,
                "description": f"Automated bar chart showing aggregated **{target_num}** values segmented across the **{cat_col}** categories."
            })
        else:
            # Frequency Pie Chart
            fig = plot_pie(df, cat_col)
            recommended.append({
                "title": f"Proportional Share (Distribution of {cat_col})",
                "fig": fig,
                "description": f"Donut chart visualizing relative proportions of entries within categorical column **{cat_col}**."
            })
            
    # Chart 3: Correlation or Scatter Plot (If multiple numeric columns)
    if len(numeric_cols) >= 2:
        num1, num2 = numeric_cols[0], numeric_cols[1]
        
        # Check if we have strong correlations
        corr_matrix = df[numeric_cols].corr(method='pearson')
        max_corr = 0
        best_pair = (num1, num2)
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                c1, c2 = numeric_cols[i], numeric_cols[j]
                val = corr_matrix.loc[c1, c2]
                if not pd.isna(val) and abs(val) > abs(max_corr):
                    max_corr = val
                    best_pair = (c1, c2)
                    
        # Make a scatter plot of the highest correlated pair
        fig = plot_scatter(df, best_pair[0], best_pair[1])
        r_str = f"Pearson r = {max_corr:.2f}"
        recommended.append({
            "title": f"Relationship Analysis ({best_pair[1]} vs {best_pair[0]})",
            "fig": fig,
            "description": f"Scatter plot showing correlation between **{best_pair[1]}** and **{best_pair[0]}** ({r_str}). Dashed line represents linear trendline."
        })
        
    # Chart 4: Fallback Histogram (If only 1 numeric column and no categorical columns)
    if not recommended and numeric_cols:
        col = numeric_cols[0]
        fig = plot_histogram(df, col)
        recommended.append({
            "title": f"Distribution Analysis ({col})",
            "fig": fig,
            "description": f"Histogram visualizing frequency distribution and skewness of values in numerical column **{col}**."
        })
        
    return recommended[:3]  # Return at most 3 automated charts
