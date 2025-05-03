# Main Streamlit application for Stock Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Mutual Funds Guide - Stock Analyzer", page_icon="📊")

# --- Branding --- #
st.title("Mutual Funds Guide")
st.caption("Created by Akhilesh Gururani ( akhilesh.gururani@gmail.com)")
st.header("📊 Customizable Stock Analyzer")
st.markdown("Upload your stock data (CSV or Excel). The app will suggest default parameters and weights, which you can customize to rank stocks.")

# --- Constants & Defaults --- #
ESSENTIAL_COLUMNS = ["Name"] # Must be present
DEFAULT_PARAMS_WEIGHTS = {
    # Higher is Better
    "Return on equity": 15,
    "Profit growth 3Years": 10,
    "Sales growth 3Years": 10,
    "OPM": 10, # Operating Profit Margin
    "Return on capital employed": 10,
    "Return on assets": 5,
    "Free cash flow last year": 5,
    # Lower is Better
    "Price to Earning": 10,
    "Price to book value": 10,
    "Debt to equity": 10,
    "Debt": 5,
} # Sum = 100

TERM_DEFINITIONS = {
    "Return on equity": "(ROE) Measures a corporation's profitability in relation to shareholders’ equity. Higher is generally better.",
    "Profit growth 3Years": "The compound annual growth rate (CAGR) of profit over the last 3 years. Higher indicates strong growth.",
    "Sales growth 3Years": "The compound annual growth rate (CAGR) of sales over the last 3 years. Higher indicates strong revenue growth.",
    "OPM": "(Operating Profit Margin) Measures how much profit a company makes on a dollar of sales, after paying for variable costs of production, but before paying interest or tax. Higher is better.",
    "Return on capital employed": "(ROCE) Measures a company's profitability and the efficiency with which its capital is employed. Higher is better.",
    "Return on assets": "(ROA) Indicates how profitable a company is relative to its total assets. Higher is better.",
    "Free cash flow last year": "(FCF) The cash a company produces after accounting for cash outflows to support operations and maintain its capital assets. Higher is generally better.",
    "Price to Earning": "(P/E Ratio) The ratio of a company's share price to the company's earnings per share. Lower can indicate undervaluation, but context is important.",
    "Price to book value": "(P/B Ratio) Compares a company's market capitalization to its book value. Lower can indicate undervaluation.",
    "Debt to equity": "(D/E Ratio) Measures a company's financial leverage, calculated by dividing its total liabilities by its stockholders' equity. Lower generally indicates less risk.",
    "Debt": "Total amount of debt held by the company. Lower is generally preferred.",
    "Market Capitalization": "The total market value of a company's outstanding shares of stock. Often used to size up corporations.",
    # Add more definitions as needed
}

# --- Helper Functions --- #

def infer_direction(col_name):
    """Infers if higher or lower is better based on column name keywords."""
    name_lower = col_name.lower()
    if any(keyword in name_lower for keyword in ["debt", "ratio", "price to", "p/e", "p/b"]):
        return "lower"
    if any(keyword in name_lower for keyword in ["growth", "return", "margin", "profit", "sales", "flow", "opm", "roe", "roa", "roce", "market capitalization"]):
        return "higher"
    return "neutral" # Default if unsure

def generate_rank_explanation(rank1_stock, df_all_ranked, selected_params, params_weights, params_direction, normalized_params_cols):
    """Generates a detailed explanation for Rank #1, comparing vs best performer on each param."""
    if rank1_stock is None or rank1_stock.empty:
        return "No data available for the top-ranked stock."

    stock_name = rank1_stock["Name"].iloc[0]
    explanation = f"**Analysis for Rank #1: {stock_name}**\n\n"
    explanation += f"{stock_name} achieved the highest Composite Score based on the selected parameters and weights. Here's a breakdown of the key factors:\n\n"

    contributions = {}
    total_weight = sum(params_weights.values())
    if total_weight <= 0: return "Weights are zero, cannot determine ranking factors."

    for param in selected_params:
        weight = params_weights.get(param, 0)
        norm_col = normalized_params_cols.get(param)
        if norm_col and norm_col in rank1_stock.columns and weight > 0:
            norm_score = rank1_stock[norm_col].iloc[0]
            contribution = norm_score * (weight / total_weight)
            contributions[param] = contribution

    sorted_contributions = sorted(contributions.items(), key=lambda item: item[1], reverse=True)

    explanation += "**Key Contributing Factors (Strongest First):**\n"

    for i, (param, contribution) in enumerate(sorted_contributions):
        if i >= 5 and contribution < 0.01: # Limit to top factors
            break

        raw_value = rank1_stock[param].iloc[0]
        direction = params_direction.get(param, "higher")
        weight_perc = (params_weights.get(param, 0) / total_weight) * 100
        term_def = TERM_DEFINITIONS.get(param, "")

        performance_desc = ""
        if direction == "higher":
            performance_desc = f"scored well with a high value of **{raw_value:.2f}**"
        elif direction == "lower":
            performance_desc = f"scored well with a low value of **{raw_value:.2f}**"
        else:
             performance_desc = f"had a value of {raw_value:.2f}"

        explanation += f"- **{param}**: {performance_desc}. "
        if term_def:
            explanation += f"*(Definition: {term_def})* "
        explanation += f"This factor had a weight of {weight_perc:.1f}%.\n"

        # Comparison with Best Performer for this parameter
        if param in df_all_ranked.columns:
            best_performer_row = None
            if direction == "higher":
                best_performer_row = df_all_ranked.loc[df_all_ranked[param].idxmax()]
            elif direction == "lower":
                best_performer_row = df_all_ranked.loc[df_all_ranked[param].idxmin()]

            if best_performer_row is not None:
                best_value = best_performer_row[param]
                best_name = best_performer_row["Name"]
                comparison = ""
                if best_name == stock_name:
                    comparison = f"was the **best performer** among all stocks for this parameter."
                else:
                    comparison = f"compared to the best performer ({best_name}: {best_value:.2f})."
                explanation += f"    - *Comparison:* {stock_name} {comparison}\n"
        explanation += "\n"

    explanation += f"\n*Note: The Composite Score is a weighted average of normalized values (0-1 scale) for all selected parameters. This explanation highlights the most influential factors.*"
    return explanation

def generate_data_summary(df):
    """Generates a dynamic summary of the loaded data."""
    summary = "**Data Summary:**\n\n"
    num_companies = len(df)
    summary += f"- **Total Companies:** {num_companies}\n"

    # Market Cap Analysis (if column exists)
    market_cap_col = next((col for col in df.columns if "market cap" in col.lower()), None)
    if market_cap_col:
        df[market_cap_col] = pd.to_numeric(df[market_cap_col], errors='coerce')
        df_cap = df.dropna(subset=[market_cap_col])
        if not df_cap.empty:
            summary += f"- **Market Capitalization ({market_cap_col}):**\n"
            bins = [0, 100, 1000, 10000, np.inf]
            labels = ["< 100 Cr", "100-1000 Cr", "1000-10000 Cr", "> 10000 Cr"]
            # Assuming Market Cap is in Cr (common in Indian context, adjust if needed)
            try:
                cap_dist = pd.cut(df_cap[market_cap_col], bins=bins, labels=labels, right=False).value_counts().sort_index()
                for label, count in cap_dist.items():
                    summary += f"    - {label}: {count} companies\n"
            except Exception as e:
                 summary += f"    - Could not calculate distribution (Error: {e})\n"
            avg_cap = df_cap[market_cap_col].mean()
            median_cap = df_cap[market_cap_col].median()
            summary += f"    - Average: {avg_cap:.2f} Cr\n"
            summary += f"    - Median: {median_cap:.2f} Cr\n"

    # Exchange Listing Analysis (if column exists)
    exchange_col = next((col for col in df.columns if "exchange" in col.lower() or "listed on" in col.lower()), None)
    if exchange_col:
        df[exchange_col] = df[exchange_col].astype(str).str.upper()
        exchange_counts = df[exchange_col].value_counts()
        if not exchange_counts.empty:
            summary += f"- **Exchange Listing ({exchange_col}):**\n"
            for ex, count in exchange_counts.items():
                summary += f"    - {ex}: {count} companies\n"

    # Add more dynamic summaries based on common column names if needed
    # e.g., Sector distribution
    sector_col = next((col for col in df.columns if "sector" in col.lower() or "industry" in col.lower()), None)
    if sector_col:
        sector_counts = df[sector_col].value_counts()
        if not sector_counts.empty:
            summary += f"- **Sector/Industry ({sector_col}):**\n"
            # Show top N sectors
            top_n_sectors = 5
            for sector, count in sector_counts.head(top_n_sectors).items():
                 summary += f"    - {sector}: {count} companies\n"
            if len(sector_counts) > top_n_sectors:
                 summary += f"    - ... ({len(sector_counts) - top_n_sectors} other sectors)\n"

    return summary

# --- Data Loading and Validation ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data, identifies numeric columns, handles essential checks."""
    if uploaded_file is None: return None, "Please upload a data file.", []
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format.", []

        missing_essential = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
        if missing_essential:
            return None, f"Error: Missing essential columns: {", ".join(missing_essential)}.", []

        if "Name" in df.columns: df.dropna(subset=["Name"], inplace=True)
        else: return None, "Critical Error: 'Name' column not found.", []

        # Convert potential numeric columns AFTER identifying them
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Also try to convert object columns that might be numeric
        for col in df.select_dtypes(include=["object"]).columns:
             # Skip essential columns like 'Name'
             if col in ESSENTIAL_COLUMNS: continue
             try:
                 converted_col = pd.to_numeric(df[col], errors='coerce')
                 if not converted_col.isnull().all():
                     df[col] = converted_col
                     if col not in numeric_cols:
                         numeric_cols.append(col)
             except Exception:
                 continue # Ignore columns that fail conversion

        # Re-identify numeric columns after potential conversions
        final_numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        return df, None, final_numeric_cols
    except Exception as e:
        return None, f"Error processing file: {e}", []

# --- Data Preprocessing --- #
def preprocess_data(df, selected_params):
    """Handles missing values by filling with 0."""
    df_processed = df.copy()
    warnings = []
    for param in selected_params:
        if param not in df_processed.columns:
             warnings.append(f"Warning: Parameter '{param}' selected but not found in data during preprocessing.")
             continue
        # Ensure column is numeric before filling NaN
        if pd.api.types.is_numeric_dtype(df_processed[param]):
            if df_processed[param].isnull().any():
                num_missing = df_processed[param].isnull().sum()
                df_processed[param].fillna(0, inplace=True)
                warnings.append(f"Missing values ({num_missing}) found in '{param}'. Filled with 0.")
        else:
             warnings.append(f"Warning: Parameter '{param}' is not numeric and cannot be used in calculations.")
             # Optionally remove non-numeric selected params here

    return df_processed, warnings

# --- Scoring Logic --- #
def calculate_scores(df, params_weights, params_direction):
    """Calculates normalized scores."""
    df_scored = df.copy()
    df_scored["Composite Score"] = 0
    total_weight = sum(params_weights.values())
    if total_weight <= 0:
        st.warning("Total weight is zero. Scores cannot be calculated.")
        df_scored["Composite Score"] = 0
        df_scored["Rank"] = 1
        return df_scored, {}

    normalized_params_cols = {}
    for param, weight in params_weights.items():
        if param not in df_scored.columns: continue
        # Ensure column is numeric before proceeding
        if not pd.api.types.is_numeric_dtype(df_scored[param]): continue
        if weight == 0: continue

        min_val = df_scored[param].min()
        max_val = df_scored[param].max()
        direction = params_direction.get(param, "higher")
        norm_col_name = f"{param}_norm"
        normalized_params_cols[param] = norm_col_name

        if max_val == min_val:
            df_scored[norm_col_name] = 0.5
        else:
            if direction == "higher":
                df_scored[norm_col_name] = (df_scored[param] - min_val) / (max_val - min_val)
            elif direction == "lower":
                df_scored[norm_col_name] = (max_val - df_scored[param]) / (max_val - min_val)
            else:
                 df_scored[norm_col_name] = 0.5

        normalized_weight = weight / total_weight
        df_scored["Composite Score"] += df_scored[norm_col_name].fillna(0) * normalized_weight

    df_scored["Rank"] = df_scored["Composite Score"].rank(ascending=False, method="min").astype(int)
    df_scored.sort_values("Rank", inplace=True)
    return df_scored, normalized_params_cols

# --- Main App Logic ---

# --- File Uploader --- #
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

df_raw = None
error_message = None
all_numeric_cols = []

if uploaded_file is not None:
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.defaults_applied = False
        st.session_state.current_file_name = uploaded_file.name

    df_raw, error_message, all_numeric_cols = load_data(uploaded_file)

if error_message:
    st.error(error_message)
    st.stop()
if df_raw is None:
    st.info("Please upload a data file to begin analysis.")
    st.stop()

st.success(f"Successfully loaded data from '{uploaded_file.name}' with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")

# --- Sidebar Configuration --- #
st.sidebar.header("⚙️ Analysis Configuration")

if not all_numeric_cols:
    st.error("No numeric columns detected in the uploaded file for analysis.")
    st.stop()

params_direction = {col: infer_direction(col) for col in all_numeric_cols}
default_params_available = [p for p in DEFAULT_PARAMS_WEIGHTS if p in all_numeric_cols]

if 'defaults_applied' not in st.session_state: st.session_state.defaults_applied = False
if 'selected_params' not in st.session_state: st.session_state.selected_params = default_params_available

if not st.session_state.defaults_applied:
     current_selection = default_params_available
     st.session_state.selected_params = current_selection
     st.session_state.defaults_applied = True
else:
     current_selection = st.session_state.selected_params if st.session_state.selected_params else default_params_available

selected_params = st.sidebar.multiselect(
    "Select Parameters for Analysis:",
    options=all_numeric_cols,
    default=current_selection,
    key="param_selector"
)
st.session_state.selected_params = selected_params

params_weights = {}
if selected_params:
    st.sidebar.subheader("Parameter Weights")
    normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 100?", True)
    total_weight_input = 0
    weight_sliders = {}
    for param in selected_params:
        direction_indicator = f" ({params_direction.get(param, 'neutral')})"
        default_weight = DEFAULT_PARAMS_WEIGHTS.get(param, 50)
        weight = st.sidebar.slider(f"Weight for {param}{direction_indicator}", 0, 100, default_weight, key=f"weight_{param}")
        weight_sliders[param] = weight
        total_weight_input += weight

    params_weights = weight_sliders

    if normalize_weights and total_weight_input > 0 and total_weight_input != 100:
        st.sidebar.info(f"Normalizing weights from {total_weight_input} to 100.")
        factor = 100 / total_weight_input
        params_weights = {p: w * factor for p, w in params_weights.items()}
        st.sidebar.markdown("**Normalized Weights:**")
        for param, weight in params_weights.items():
            st.sidebar.markdown(f"- *{param}*: {weight:.1f}")
    elif total_weight_input <= 0:
         st.sidebar.warning("Assign positive weights for ranking.")
else:
    st.sidebar.warning("Select at least one parameter.")

top_n = st.sidebar.number_input("Select Top N Stocks to Display:", min_value=1, max_value=len(df_raw), value=min(10, len(df_raw)))

# --- Main Area --- #
if selected_params and sum(params_weights.values()) > 0:
    df_processed, preprocess_warnings = preprocess_data(df_raw, selected_params)
    if preprocess_warnings:
        for warning in preprocess_warnings:
            st.warning(warning)

    # Filter out non-numeric columns from selected_params before scoring
    numeric_selected_params = [p for p in selected_params if p in df_processed.columns and pd.api.types.is_numeric_dtype(df_processed[p])]
    numeric_params_weights = {p: w for p, w in params_weights.items() if p in numeric_selected_params}

    if not numeric_selected_params:
         st.error("No valid numeric parameters selected for scoring.")
         st.stop()

    if not df_processed.empty:
        df_ranked, normalized_params_cols = calculate_scores(df_processed, numeric_params_weights, params_direction)

        st.header(f"🏆 Top {top_n} Ranked Stocks")
        st.markdown(f"Based on selected parameters and weights. Score ranges from 0 to 1 (higher is better).")

        cols_to_display = [col for col in ["Rank", "Name", "Composite Score"] + numeric_selected_params if col in df_ranked.columns]
        df_display = df_ranked.head(top_n)[cols_to_display]
        format_dict = {col: '{:.2f}' for col in df_display.select_dtypes(include=np.number).columns}
        format_dict['Rank'] = '{:d}'
        format_dict['Composite Score'] = '{:.3f}'
        df_styled = df_display.style.format(format_dict, na_rep='-')
        st.dataframe(df_styled)

        # --- Download Button (CSV Only) --- #
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Table as CSV", data=csv_data, file_name=f'top_{top_n}_stocks_ranked.csv', mime='text/csv')

        # --- Rank 1 Explanation --- #
        st.subheader("Rank #1 Analysis")
        if not df_ranked.empty:
            rank1_data = df_ranked[df_ranked['Rank'] == 1]
            # Pass the full ranked df for comparison
            explanation = generate_rank_explanation(rank1_data, df_ranked, numeric_selected_params, numeric_params_weights, params_direction, normalized_params_cols)
            st.markdown(explanation)
        else:
            st.info("No stocks ranked.")

        st.divider()

        # --- Dashboard --- #
        st.header("📊 Dashboard")
        tab_breakdown, tab_scatter, tab_compare = st.tabs(["Top N Parameter Breakdown", "Scatter Analysis", "Top N Comparison"])

        with tab_breakdown:
            st.subheader(f"Parameter Details for Top {top_n} Stocks")
            st.markdown("Shows raw values and normalized scores (0-1) for the selected parameters.")
            breakdown_cols_raw = {param: param for param in numeric_selected_params}
            breakdown_cols_norm = {norm_col: f"{param} (Norm)" for param, norm_col in normalized_params_cols.items() if param in numeric_selected_params}
            breakdown_cols_display = ['Rank', 'Name', 'Composite Score'] + list(breakdown_cols_raw.keys()) + list(breakdown_cols_norm.keys())
            breakdown_cols_display = [col for col in breakdown_cols_display if col in df_ranked.columns]
            df_breakdown = df_ranked.head(top_n)[breakdown_cols_display].rename(columns=breakdown_cols_norm)
            format_dict_breakdown = {col: '{:.2f}' for col in df_breakdown.columns if '(Norm)' in col or col in numeric_selected_params}
            format_dict_breakdown['Rank'] = '{:d}'
            format_dict_breakdown['Composite Score'] = '{:.3f}'
            st.dataframe(df_breakdown.style.format(format_dict_breakdown, na_rep='-'))

        with tab_scatter:
            st.subheader("Scatter Plot Analysis")
            if len(numeric_selected_params) >= 2:
                sc_col1, sc_col2 = st.columns(2)
                x_axis = sc_col1.selectbox("Select X-axis Parameter:", numeric_selected_params, index=0, key="scatter_x")
                y_axis = sc_col2.selectbox("Select Y-axis Parameter:", numeric_selected_params, index=1 if len(numeric_selected_params) > 1 else 0, key="scatter_y")
                if x_axis != y_axis:
                    if x_axis in df_ranked.columns and y_axis in df_ranked.columns and 'Name' in df_ranked.columns:
                        fig_scatter = px.scatter(df_ranked.head(top_n), x=x_axis, y=y_axis, hover_name='Name', text='Name', title=f'{y_axis} vs. {x_axis} for Top {top_n} Stocks', color='Composite Score' if 'Composite Score' in df_ranked.columns else None, color_continuous_scale=px.colors.sequential.Viridis)
                        fig_scatter.update_traces(textposition='top center')
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("Selected axis/Name column not found.")
                else:
                    st.warning("Select different X and Y axes.")
            else:
                st.info("Select at least two numeric parameters.")

        with tab_compare:
            st.subheader(f"Comparison of Key Metric for Top {top_n} Stocks")
            if numeric_selected_params:
                metric_to_compare = st.selectbox("Select Metric to Compare:", numeric_selected_params, key="bar_compare")
                if metric_to_compare in df_ranked.columns and 'Name' in df_ranked.columns:
                    df_top_n_bar = df_ranked.head(top_n)
                    fig_bar = px.bar(df_top_n_bar, x='Name', y=metric_to_compare, title=f'{metric_to_compare} for Top {top_n} Stocks', text_auto='.2f')
                    fig_bar.update_layout(xaxis_title="Stock Name", yaxis_title=metric_to_compare)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("Selected metric/Name column not found.")
            else:
                 st.info("Select numeric parameters in the sidebar.")

        st.divider()

        # --- Data Summary --- #
        st.header("Data Summary")
        summary_text = generate_data_summary(df_raw)
        st.markdown(summary_text)

        st.divider()

        # --- Data Views --- #
        st.header("Data Views")
        with st.expander("View Uploaded Raw Data"):
            st.dataframe(df_raw)
        with st.expander("View Processed & Ranked Data (including normalized values)"):
            try:
                norm_cols_exist = [col for col in normalized_params_cols.values() if col in df_ranked.columns]
                display_cols_processed = [col for col in ["Rank", "Name", "Composite Score"] + numeric_selected_params + norm_cols_exist if col in df_ranked.columns]
                df_processed_display = df_ranked[display_cols_processed]
                format_dict_processed = {col: '{:.3f}' for col in df_processed_display.select_dtypes(include=np.number).columns if col != 'Rank'}
                format_dict_processed['Rank'] = '{:d}'
                st.dataframe(df_processed_display.style.format(format_dict_processed, na_rep='-'))
            except Exception as e:
                st.error(f"Error displaying processed data table: {e}")
                if 'display_cols_processed' in locals() and all(col in df_ranked.columns for col in display_cols_processed):
                     st.dataframe(df_ranked[display_cols_processed])
                else:
                     st.warning("Could not display processed data.")

    else:
        st.warning("No data remaining after preprocessing.")
else:
    st.info("Configure analysis parameters in the sidebar.")

