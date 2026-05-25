import streamlit as st
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

# Import local modules
from utils.cleaner import clean_data
from utils.analyzer import analyze_dataset, get_textual_summary
from utils.visualizer import (
    plot_bar, plot_line, plot_pie, plot_histogram, plot_scatter,
    plot_correlation_heatmap, generate_auto_charts
)
from utils.insights import generate_business_insights
from utils.chatbot import chat_with_csv
from utils.exporter import df_to_csv_bytes, figure_to_html_bytes, markdown_to_html_report

# Page Config
st.set_page_config(
    page_title="DataMind AI - CSV & Excel Analytics Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Theme Custom Styling (Dark SaaS Aesthetic with Crimson Red Accents)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Main body background & color styling */
    .stApp {
        background-color: #0B0B0F;
        color: #E5E5EA;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header logo styling */
    .logo-container {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
    }
    .logo-text {
        font-family: 'Outfit', sans-serif;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin: 0;
    }
    .logo-text span {
        color: #E50914; /* Crimson Red */
    }
    .tagline {
        color: #A0A0AB;
        font-size: 14px;
        margin-top: -5px;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Card Widgets (KPI) with Glassmorphic Red Accent borders */
    .kpi-row {
        display: flex;
        gap: 20px;
        margin-bottom: 25px;
        flex-wrap: wrap;
    }
    .kpi-card {
        background: linear-gradient(135deg, rgba(20, 20, 26, 0.7) 0%, rgba(13, 13, 18, 0.7) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(229, 9, 20, 0.15);
        border-top: 3px solid #E50914;
        padding: 20px;
        border-radius: 12px;
        flex: 1;
        min-width: 200px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border: 1px solid rgba(229, 9, 20, 0.4);
        border-top: 3px solid #E50914;
    }
    .kpi-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #A0A0AB;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom tab indicators */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #23232C;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(20, 20, 26, 0.5) !important;
        border: 1px solid #23232C !important;
        border-radius: 8px !important;
        color: #A0A0AB !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF !important;
        border-color: rgba(229, 9, 20, 0.3) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        border: 1px solid #E50914 !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 14px rgba(229, 9, 20, 0.3);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #13131A !important;
        border: 1px solid #23232C !important;
        border-radius: 8px !important;
    }
    
    /* Metric Cards Override */
    [data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif;
        color: #FFFFFF !important;
    }
    [data-testid="stMetricLabel"] {
        color: #A0A0AB !important;
    }
    
    /* Clean chat logs layout */
    .chat-code-block {
        border-left: 3px solid #E50914;
        background: #14141A;
        padding: 10px;
        border-radius: 4px;
        font-family: monospace;
        margin-top: 5px;
        font-size: 13px;
    }
    
    /* Styled buttons */
    .stButton>button {
        background-color: #E50914;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #B2070F;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.4);
    }
    
    /* Section dividers */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(229, 9, 20, 0) 0%, rgba(229, 9, 20, 0.5) 50%, rgba(229, 9, 20, 0) 100%);
        margin: 25px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State Variables
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'dataset_name' not in st.session_state:
    st.session_state.dataset_name = None
if 'clean_log' not in st.session_state:
    st.session_state.clean_log = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis' not in st.session_state:
    st.session_state.analysis = None

# Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="logo-container"><h1 class="logo-text">DATAMIND<span>AI</span></h1></div>', unsafe_allow_html=True)
    st.markdown("<p style='color:#A0A0AB; font-size:12px; margin-top:-20px; margin-bottom: 20px;'>CSV & Excel Analytics Agent Platform</p>", unsafe_allow_html=True)
    
    st.markdown("### 🔑 API Authentication")
    api_key_input = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Reads from local .env by default. Enter API key to override."
    )
    
    # Save override to env state
    if api_key_input:
        api_key = api_key_input
        st.caption("🟢 API key connected.")
    else:
        api_key = ""
        st.caption("🔴 Gemini API key missing. Insert to use AI features.")

    st.markdown("<div style='margin: 15px 0; border-bottom: 1px solid #23232C;'></div>", unsafe_allow_html=True)
    
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel dataset below",
        type=["csv", "xlsx", "xls"],
        help="Upload CSV or Excel files up to 200MB. Drag and drop supported."
    )

    # Load Mock Dataset Helper
    st.markdown("<p style='text-align:center; color:#A0A0AB; margin:5px 0 2px 0; font-size:12px;'>Or try the platform with demo data:</p>", unsafe_allow_html=True)
    if st.button("📊 Load Sample Sales Dataset", use_container_width=True):
        # Generate robust mock sales dataset
        np.random.seed(42)
        dates = pd.date_range(start="2026-01-01", periods=150, freq="D").strftime("%Y-%m-%d")
        categories = ["Electronics", "Apparel", "Home & Decor", "Fitness Gear", "Office Supplies"]
        cities = ["San Francisco", "New York", "Austin", "Seattle", "Miami"]
        
        sample_data = {
            "Date": np.random.choice(dates, size=150),
            "Category": np.random.choice(categories, size=150, p=[0.25, 0.20, 0.20, 0.15, 0.20]),
            "Sales": np.random.uniform(20.0, 1500.0, size=150).round(2),
            "Quantity": np.random.randint(1, 8, size=150),
            "Profit": np.random.uniform(-100.0, 450.0, size=150).round(2),
            "City": np.random.choice(cities, size=150),
            "Feedback_Score": np.random.choice([1, 2, 3, 4, 5], size=150, p=[0.05, 0.05, 0.15, 0.35, 0.40])
        }
        
        mock_df = pd.DataFrame(sample_data)
        
        # Inject null values and duplicates for cleaner testing
        mock_df.loc[12:15, "Sales"] = np.nan
        mock_df.loc[45:47, "Category"] = np.nan
        mock_df.loc[80:82, "Feedback_Score"] = np.nan
        # Add some duplicates
        mock_df = pd.concat([mock_df, mock_df.iloc[[10]], mock_df.iloc[[25]], mock_df.iloc[[10]]], ignore_index=True)
        
        st.session_state.df_raw = mock_df.copy()
        st.session_state.df = mock_df.copy()
        st.session_state.dataset_name = "demo_retail_sales.csv"
        st.session_state.clean_log = None
        st.session_state.ai_insights = None
        st.session_state.chat_history = []
        st.session_state.analysis = analyze_dataset(mock_df)
        st.rerun()

    # If file uploaded, process it
    if uploaded_file is not None:
        # Avoid reloading if same file is already in state
        if st.session_state.dataset_name != uploaded_file.name:
            try:
                # Read CSV or Excel
                if uploaded_file.name.endswith(".csv"):
                    raw_df = pd.read_csv(uploaded_file)
                else:
                    raw_df = pd.read_excel(uploaded_file)
                st.session_state.df_raw = raw_df.copy()
                st.session_state.df = raw_df.copy()
                st.session_state.dataset_name = uploaded_file.name
                st.session_state.clean_log = None
                st.session_state.ai_insights = None
                st.session_state.chat_history = []
                st.session_state.analysis = analyze_dataset(raw_df)
                st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    if st.session_state.df is not None:
        st.markdown("<div style='margin: 15px 0; border-bottom: 1px solid #23232C;'></div>", unsafe_allow_html=True)
        st.markdown("### 🧹 Auto-Data Cleaning")
        
        clean_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
        
        num_clean_strategy = st.selectbox(
            "Numerical Missing Strategy",
            ["median", "mean", "mode", "zero", "drop", "none"],
            index=0,
            help="How to resolve empty cells in numeric columns"
        )
        
        cat_clean_strategy = st.selectbox(
            "Categorical Missing Strategy",
            ["mode", "unknown", "drop", "none"],
            index=0,
            help="How to resolve empty cells in object/category columns"
        )
        
        if st.button("⚡ Run Cleaning Agent", use_container_width=True):
            with st.spinner("Cleaning dataset and coercing types..."):
                cleaned_df, logs = clean_data(
                    st.session_state.df_raw,
                    remove_duplicates=clean_duplicates,
                    fill_numeric=num_clean_strategy,
                    fill_categorical=cat_clean_strategy,
                    clean_strings=True,
                    coerce_types=True
                )
                st.session_state.df = cleaned_df
                st.session_state.clean_log = logs
                st.session_state.analysis = analyze_dataset(cleaned_df)
                st.toast("Dataset cleaned successfully!", icon="✅")
                st.rerun()

        st.markdown("<div style='margin: 15px 0; border-bottom: 1px solid #23232C;'></div>", unsafe_allow_html=True)
        st.markdown("### 📥 Export Reports")
        
        # Cleaned CSV Export
        csv_data = df_to_csv_bytes(st.session_state.df)
        st.download_button(
            label="📄 Download Cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{st.session_state.dataset_name}",
            mime="text/csv",
            use_container_width=True
        )

        # Standalone HTML Insights Report Export (only if generated)
        if st.session_state.ai_insights:
            html_report = markdown_to_html_report(
                st.session_state.ai_insights,
                st.session_state.dataset_name,
                len(st.session_state.df),
                len(st.session_state.df.columns)
            )
            st.download_button(
                label="👑 Download Premium HTML Report",
                data=html_report,
                file_name=f"datamind_insights_{st.session_state.dataset_name.split('.')[0]}.html",
                mime="text/html",
                use_container_width=True
            )


# Main Dashboard Panel
st.markdown('<div class="logo-container"><h1 class="logo-text">DATAMIND<span>AI</span></h1></div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Production-grade AI CSV & Excel Analytics Agent & Executive Dashboard</div>', unsafe_allow_html=True)

# ----------------- LANDING PAGE IF NO DATA -----------------
if st.session_state.df is None:
    # Beautiful welcome workspace
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.markdown(
            """
            <h2 style='font-family: Outfit, sans-serif; font-size:32px; color: #FFFFFF;'>
                Analyze your business data instantly with generative intelligence.
            </h2>
            <p style='color: #A0A0AB; font-size: 16px; line-height: 1.6;'>
                DataMind AI is an advanced, automated CSV and Excel Data Scientist agent. 
                Upload datasets, run multi-strategy cleaning pipelines, render adaptive visualizations, 
                generate boardroom-ready insights, and chat with your database in plain English.
            </p>
            """,
            unsafe_allow_html=True
        )
        
        # Features Grid
        st.markdown("### ⚙️ Engine capabilities")
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown(
                """
                <div style='background:rgba(20,20,26,0.6); padding:15px; border-radius:10px; border:1px solid #23232C; margin-bottom:15px;'>
                    <h4 style='color:#E50914; margin-top:0;'>🧹 Auto-Data Cleaning</h4>
                    <p style='font-size:12px; color:#A0A0AB; margin:0;'>Cleans duplicate rows, imputes numerical/categorical cells, handles formatting noises, and standardizes datatypes.</p>
                </div>
                <div style='background:rgba(20,20,26,0.6); padding:15px; border-radius:10px; border:1px solid #23232C;'>
                    <h4 style='color:#E50914; margin-top:0;'>💡 AI Business insights</h4>
                    <p style='font-size:12px; color:#A0A0AB; margin:0;'>Generates executive summaries, key trends, anomaly risks, and tactical growth recommendations via Gemini Pro.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with f_col2:
            st.markdown(
                """
                <div style='background:rgba(20,20,26,0.6); padding:15px; border-radius:10px; border:1px solid #23232C; margin-bottom:15px;'>
                    <h4 style='color:#E50914; margin-top:0;'>📊 Adaptive Visuals</h4>
                    <p style='font-size:12px; color:#A0A0AB; margin:0;'>Scans columns to automatically plot time trends, bar breakdowns, pie distributions, scatter relationships, and heatmaps.</p>
                </div>
                <div style='background:rgba(20,20,26,0.6); padding:15px; border-radius:10px; border:1px solid #23232C;'>
                    <h4 style='color:#E50914; margin-top:0;'>💬 Conversational Chatbot</h4>
                    <p style='font-size:12px; color:#A0A0AB; margin:0;'>Ask questions naturally. The AI translates questions to Python pandas statements, runs them safely, and explains the answers.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
    with col2:
        # Visual Mock Dashboard Card
        st.markdown(
            """
            <div style='background: linear-gradient(135deg, #1C1C24 0%, #111116 100%); padding: 30px; border-radius: 16px; border: 1px solid rgba(229,9,20,0.2); box-shadow: 0 10px 40px rgba(0,0,0,0.6); text-align: center; margin-top: 15px;'>
                <div style='font-size: 50px;'>📊</div>
                <h3 style='margin-top: 15px; font-family: Outfit, sans-serif;'>Ready to Analyze</h3>
                <p style='color: #A0A0AB; font-size: 13px; line-height: 1.5;'>
                    Drag and drop a <code>.csv</code> file onto the sidebar uploader or click <strong>Load Sample Sales Dataset</strong> to run instantly with mock retail values.
                </p>
                <div style='margin-top:20px; font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#E50914; font-weight:600;'>
                    Built for Business Intelligence & Operations
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ----------------- ACTIVE METRICS & DASHBOARD -----------------
else:
    df = st.session_state.df
    analysis = st.session_state.analysis
    
    # 1. Row of glassmorphic KPI cards
    null_percentage = analysis["missing_percent"]
    st.markdown(
        f"""
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-label">Dataset Filename</div>
                <div class="kpi-value" style="color: #E50914; font-size:20px; word-break: break-all;">{st.session_state.dataset_name}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Rows</div>
                <div class="kpi-value">{analysis["row_count"]:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Total Columns</div>
                <div class="kpi-value">{analysis["col_count"]}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Duplicate Rows</div>
                <div class="kpi-value">{analysis["duplicate_rows"]:,}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Missing Cells</div>
                <div class="kpi-value">{analysis["missing_cells"]:,} <span style="font-size:12px; color:#FF7961;">({null_percentage:.2f}%)</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # 2. Tabs panel
    tab_explore, tab_visuals, tab_insights, tab_chat = st.tabs([
        "📋 Dataset Explorer",
        "📊 Auto-Visualization",
        "💡 AI Business Insights",
        "💬 Conversational Chatbot"
    ])
    
    # --- TAB 1: DATA EXPLORER & CLEANER LOGS ---
    with tab_explore:
        st.markdown("### 📋 Interactive Dataset Grid")
        st.dataframe(df, use_container_width=True)
        
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        col_sch, col_logs = st.columns([1.1, 0.9])
        
        with col_sch:
            st.markdown("### 🧬 Column Schema & Properties")
            schema_data = []
            for col in analysis["columns"]:
                schema_data.append({
                    "Name": col["name"],
                    "Type": col["type"],
                    "Data Type": col["dtype"],
                    "Unique Values": col["unique_count"],
                    "Null Count": col["null_count"],
                    "Null %": f"{col['null_percent']:.1f}%"
                })
            st.dataframe(pd.DataFrame(schema_data), use_container_width=True)
            
        with col_logs:
            st.markdown("### 🛠️ Data Cleaning Audit Trail")
            if st.session_state.clean_log:
                logs = st.session_state.clean_log
                st.markdown(
                    f"""
                    <div style='background:rgba(20,20,26,0.6); border:1px solid #23232C; padding:20px; border-radius:10px;'>
                        <p style='color:#A0A0AB; margin-top:0; font-size:12px; text-transform:uppercase;'>System Log Reports</p>
                        <ul style='margin-bottom:0; font-size:14px; padding-left:15px;'>
                            <li><strong>Initial Dimensions:</strong> {logs['initial_rows']} rows x {logs['initial_cols']} cols</li>
                            <li><strong>Duplicates Removed:</strong> {logs['duplicates_removed']}</li>
                            <li><strong>Datatypes Coerced:</strong> {len(logs['errors_coerced'])} columns corrected</li>
                            <li><strong>Final Dimensions:</strong> {logs['final_rows']} rows x {logs['final_cols']} cols</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if logs['missing_filled']:
                    st.markdown("<p style='font-size:13px; font-weight:600; margin-top:15px; margin-bottom:5px;'>Missing Values Imputed:</p>", unsafe_allow_html=True)
                    impute_list = []
                    for col_name, info in logs['missing_filled'].items():
                        impute_list.append({"Column": col_name, "Null Cells": info['null_count'], "Cleaning Strategy": info['strategy']})
                    st.dataframe(pd.DataFrame(impute_list), use_container_width=True)
                else:
                    st.caption("No missing values were present or imputed.")
            else:
                st.info("No cleaning operations run yet. Configure data cleaning settings in the sidebar and click **Run Cleaning Agent** to execute.")

    # --- TAB 2: AUTOMATIC & CUSTOM VISUALIZATION ---
    with tab_visuals:
        # Automated recommended charts section
        st.markdown("### 🤖 Automated Recommended Visualizations")
        auto_charts = generate_auto_charts(df, analysis["columns"])
        
        if auto_charts:
            chart_cols = st.columns(len(auto_charts))
            for idx, chart in enumerate(auto_charts):
                with chart_cols[idx]:
                    st.plotly_chart(chart["fig"], use_container_width=True)
                    st.markdown(f"<p style='font-size:13px; color:#A0A0AB;'>{chart['description']}</p>", unsafe_allow_html=True)
        else:
            st.warning("Could not auto-generate recommended charts based on the dataset columns.")
            
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Custom Chart Builder Grid
        col_ctrl, col_chart = st.columns([0.3, 0.7])
        
        with col_ctrl:
            st.markdown("### 🛠️ Custom Chart Builder")
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart", "Pie Chart", "Histogram", "Scatter Plot"]
            )
            
            # Extract column options
            numeric_cols = [c['name'] for c in analysis["columns"] if c['type'] == 'numeric']
            all_cols = [c['name'] for c in analysis["columns"]]
            
            x_col = st.selectbox("X-Axis Column", all_cols)
            y_col = None
            color_col = None
            
            if chart_type != "Histogram" and chart_type != "Pie Chart":
                y_col = st.selectbox("Y-Axis Column", numeric_cols if numeric_cols else all_cols)
                
            if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                color_col = st.selectbox("Color Grouping (Optional)", ["None"] + all_cols)
                color_col = None if color_col == "None" else color_col
                
            if chart_type == "Pie Chart":
                y_col = st.selectbox("Value Column (Summed - Optional)", ["None"] + numeric_cols)
                y_col = None if y_col == "None" else y_col
                
            st.markdown("<br>", unsafe_allow_html=True)
            build_button = st.button("📊 Render Plot", use_container_width=True)
            
        with col_chart:
            if build_button or 'active_custom_fig' not in st.session_state:
                # Render default or user requested chart
                try:
                    if chart_type == "Bar Chart":
                        fig = plot_bar(df, x_col, y_col, color_col)
                    elif chart_type == "Line Chart":
                        fig = plot_line(df, x_col, y_col, color_col)
                    elif chart_type == "Pie Chart":
                        fig = plot_pie(df, x_col, y_col)
                    elif chart_type == "Histogram":
                        fig = plot_histogram(df, x_col)
                    elif chart_type == "Scatter Plot":
                        fig = plot_scatter(df, x_col, y_col, color_col)
                    
                    st.session_state.active_custom_fig = fig
                    st.session_state.active_custom_title = f"Custom {chart_type}"
                except Exception as e:
                    st.error(f"Error plotting chart: {str(e)}")
                    st.session_state.active_custom_fig = None
            
            if st.session_state.active_custom_fig:
                st.plotly_chart(st.session_state.active_custom_fig, use_container_width=True)
                
                # Chart Export Options
                html_chart = figure_to_html_bytes(st.session_state.active_custom_fig)
                st.download_button(
                    label="💾 Export Interactive Standalone Chart (HTML)",
                    data=html_chart,
                    file_name=f"datamind_chart_{x_col}.html",
                    mime="text/html"
                )

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        
        # Correlation Heatmap Section
        st.markdown("### 🧮 Numerical Variables Correlation")
        if analysis["correlations"]:
            fig_corr = plot_correlation_heatmap(analysis["correlations"])
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Correlation analysis requires at least two numerical columns in the dataset.")

    # --- TAB 3: AI BUSINESS INSIGHTS ---
    with tab_insights:
        st.markdown("### 💡 AI Executive Insights & Strategy Generator")
        st.markdown(
            "<p style='color:#A0A0AB;'>Using Google Gemini 1.5, DataMind scans your metadata, "
            "correlations, statistics, and sample rows to synthesize a premium strategy report.</p>",
            unsafe_allow_html=True
        )
        
        if st.button("🧠 Generate Strategy & Insights Report", use_container_width=True):
            with st.spinner("Analyzing dataset patterns and drafting boardroom recommendations..."):
                summary_text = get_textual_summary(analysis, df_sample_rows=5, df=df)
                insights_report = generate_business_insights(summary_text, api_key=api_key)
                st.session_state.ai_insights = insights_report
                st.rerun()
                
        if st.session_state.ai_insights:
            st.markdown("<div style='background:rgba(20,20,26,0.5); padding:30px; border:1px solid #23232C; border-radius:12px; margin-top:20px;'>", unsafe_allow_html=True)
            st.markdown(st.session_state.ai_insights)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Export Options for Report
            st.markdown("<br>", unsafe_allow_html=True)
            html_report = markdown_to_html_report(
                st.session_state.ai_insights,
                st.session_state.dataset_name,
                len(df),
                len(df.columns)
            )
            
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.download_button(
                    label="👑 Export Standalone HTML Report",
                    data=html_report,
                    file_name=f"datamind_insights_{st.session_state.dataset_name.split('.')[0]}.html",
                    mime="text/html",
                    use_container_width=True
                )
            with exp_col2:
                st.download_button(
                    label="📄 Export Raw Markdown (.md)",
                    data=st.session_state.ai_insights.encode('utf-8'),
                    file_name=f"datamind_insights_{st.session_state.dataset_name.split('.')[0]}.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        else:
            st.info("Click the button above to generate a strategy insights report. (Requires active Gemini API Key)")

    # --- TAB 4: NATURAL LANGUAGE CHAT PANEL ---
    with tab_chat:
        st.markdown("### 💬 Chat with your Dataset (Natural Language)")
        st.markdown(
            "<p style='color:#A0A0AB; font-size:13px;'>Ask specific questions like: "
            "<i>'Which product category generated the highest profit?'</i>, "
            "<i>'What is the average satisfaction score in Seattle?'</i>, or "
            "<i>'List the top 5 sales transactions'</i>.</p>",
            unsafe_allow_html=True
        )
        
        # Display Chat History
        for q, a in st.session_state.chat_history:
            # User Message bubble
            with st.chat_message("user"):
                st.write(q)
            # Assistant Message bubble
            with st.chat_message("assistant"):
                # Check if assistant output is a dictionary containing code and answer
                if isinstance(a, dict):
                    st.markdown(a["answer"])
                    if "code" in a and a["code"]:
                        with st.expander("🛠️ Execution Log"):
                            st.markdown(f"**Generated Pandas Script:**\n```python\n{a['code']}\n```")
                else:
                    st.markdown(a)
                    
        # Chat Input
        user_query = st.chat_input("Ask a question about this CSV...")
        
        if user_query:
            # Show User Message immediately
            with st.chat_message("user"):
                st.write(user_query)
                
            with st.spinner("AI is calculating response..."):
                chat_res = chat_with_csv(
                    df,
                    user_query,
                    chat_history=[(h[0], h[1]["answer"] if isinstance(h[1], dict) else h[1]) for h in st.session_state.chat_history],
                    api_key=api_key
                )
                
            # Render Response
            with st.chat_message("assistant"):
                st.markdown(chat_res["answer"])
                if chat_res.get("code"):
                    with st.expander("🛠️ Execution Log"):
                        st.markdown(f"**Generated Pandas Script:**\n```python\n{chat_res['code']}\n```")
                        
            # Store in History
            st.session_state.chat_history.append((user_query, chat_res))
            st.rerun()
            
        # Clean chat button
        if st.session_state.chat_history:
            if st.button("🧹 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
