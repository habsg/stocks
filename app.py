# Main Streamlit application for Stock Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import dataframe_image as dfi # Added for image export

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Mutual Funds Guide - Stock Analyzer", page_icon="📊")

# --- Branding --- #
st.title("Mutual Funds Guide")
st.caption("Created by Akhilesh Gururani ( akhilesh.gururani@gmail.com)")
st.header("📊 Customizable Stock Analyzer")
st.markdown("Upload your stock data (CSV or Excel) and rank stocks based on selected financial parameters and weights.")


# Define essential columns needed for the core analysis
ESSENTIAL_COLUMNS = [ 'Name' ]

# Define potential parameters and their 'ideal' direction
AVAILABLE_PARAMS_CONFIG = {
    'Price to Earning': 'lower',
    'Price to book value': 'lower',
    'Return on equity': 'higher',
    'Debt to equity': 'lower',
    'Sales growth 3Years': 'higher',
    'Profit growth 3Years': 'higher',
    'OPM': 'higher',
    'Return on capital employed': 'higher',
    'Return on assets': 'higher',
    'Sustainable Growth Rate': 'higher',
    'Free cash flow last year': 'higher',
    'Market Capitalization': 'higher',
    'Current Price': 'neutral',
    'Debt': 'lower',
    'Graham': 'lower',
    'Profit after tax': 'higher',
    'Sales latest quarter': 'higher',
    'Equity capital': 'neutral',
    'Working capital': 'neutral',
    'Free cash flow preceding year': 'higher'
}

# --- Helper Function for Image Export --- #
def convert_df_to_image(df_styled):
    """Converts a styled DataFrame to PNG image bytes."""
    try:
        img_buf = io.BytesIO()
        # Explicitly tell dfi where to find chrome if needed, though often it finds it automatically if installed.
        # You might need to configure this path in Streamlit Cloud if possible.
        # dfi.export(df_styled, img_buf, table_conversion='chrome', chrome_path='/path/to/chrome/executable')
        dfi.export(df_styled, img_buf, table_conversion='chrome')
        img_buf.seek(0)
        return img_buf.getvalue()
    except FileNotFoundError:
        st.error("Image Generation Failed: Chrome executable not found. Please ensure Chrome or Chromium is installed and accessible in the environment's PATH.")
        return None
    except Exception as e:
        st.error(f"Failed to generate image: {e}")
        return None

# --- Helper Function for Rank Explanation --- #
def generate_rank_explanation(rank1_stock, selected_params, params_weights, params_direction, normalized_params_cols):
    """Generates a simple explanation for the top-ranked stock."""
    if rank1_stock is None or rank1_stock.empty:
        return "No data available for the top-ranked stock."

    stock_name = rank1_stock['Name'].iloc[0]
    explanation = f"**Why is {stock_name} ranked #1?**\n\n"
    explanation += f"{stock_name} achieved the highest Composite Score based on the selected parameters and weights. Here's a breakdown:\n\n"

    # Sort parameters by their contribution to the score (weight * normalized_score)
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
    for param, contribution in sorted_contributions:
        raw_value = rank1_stock[param].iloc[0]
        direction = params_direction.get(param, 'higher')
        weight_perc = (params_weights.get(param, 0) / total_weight) * 100

        # Simple qualitative description based on direction
        performance_desc = ""
        if direction == 'higher':
            performance_desc = f"performed well with a value of {raw_value:.2f}"
        elif direction == 'lower':
            performance_desc = f"performed well with a low value of {raw_value:.2f}"
        else:
             performance_desc = f"had a value of {raw_value:.2f}"

        explanation += f"- **{param}**: {performance_desc}, contributing significantly due to its assigned weight ({weight_perc:.1f}%).\n"

    explanation += f"\n*Note: This is a simplified explanation based on normalized scores and weights. The actual Composite Score is a weighted average of normalized values (0-1 scale) for all selected parameters.*"
    return explanation

# --- Data Loading and Validation ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from uploaded file (CSV or Excel), handling potential errors."""
    if uploaded_file is None:
        return None, "Please upload a data file."
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."

        missing_essential_cols = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
        if missing_essential_cols:
            return None, f"Error: Missing essential columns: {', '.join(missing_essential_cols)}."

        if "Name" in df.columns:
            df.dropna(subset=["Name"], inplace=True)
        else:
             return None, "Critical Error: 'Name' column not found."

        potential_numeric_cols = list(AVAILABLE_PARAMS_CONFIG.keys()) + [
             'Market Capitalization', 'Current Price', 'Debt', 'Graham',
             'Profit after tax', 'Sales latest quarter', 'Equity capital',
             'Working capital', 'Free cash flow preceding year'
             ]
        for col in potential_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df, None
    except Exception as e:
        return None, f"Error processing file: {e}"

# --- Data Preprocessing --- #
def preprocess_data(df, selected_params):
    """Handles missing values for selected parameters by filling with 0."""
    df_processed = df.copy()
    warnings = []
    for param in selected_params:
        if df_processed[param].isnull().any():
            num_missing = df_processed[param].isnull().sum()
            df_processed[param].fillna(0, inplace=True)
            warnings.append(f"Missing values ({num_missing}) found in '{param}'. Filled with 0.")
    return df_processed, warnings

# --- Scoring Logic --- #
def calculate_scores(df, params_weights, params_direction):
    """Calculates normalized scores based on selected parameters and weights."""
    df_scored = df.copy()
    df_scored['Composite Score'] = 0
    total_weight = sum(params_weights.values())
    if total_weight <= 0:
        st.warning("Total weight is zero. Scores cannot be calculated.")
        df_scored['Composite Score'] = 0
        df_scored['Rank'] = 1
        return df_scored, {}

    normalized_params_cols = {}
    for param, weight in params_weights.items():
        if param not in df_scored.columns: continue
        if weight == 0: continue

        min_val = df_scored[param].min()
        max_val = df_scored[param].max()
        direction = params_direction.get(param, 'higher')
        norm_col_name = f'{param}_norm'
        normalized_params_cols[param] = norm_col_name

        if max_val == min_val:
            df_scored[norm_col_name] = 0.5
        else:
            if direction == 'higher':
                df_scored[norm_col_name] = (df_scored[param] - min_val) / (max_val - min_val)
            elif direction == 'lower':
                df_scored[norm_col_name] = (max_val - df_scored[param]) / (max_val - min_val)
            else:
                 df_scored[norm_col_name] = 0.5

        normalized_weight = weight / total_weight
        df_scored['Composite Score'] += df_scored[norm_col_name] * normalized_weight

    df_scored['Rank'] = df_scored['Composite Score'].rank(ascending=False, method='min').astype(int)
    df_scored.sort_values('Rank', inplace=True)
    return df_scored, normalized_params_cols

# --- Main App Logic ---

# --- File Uploader --- #
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

df_raw = None
error_message = None

if uploaded_file is not None:
    df_raw, error_message = load_data(uploaded_file)

if error_message:
    st.error(error_message)
    st.stop()
if df_raw is None:
    st.info("Please upload a data file to begin analysis.")
    st.stop()

st.success(f"Successfully loaded data from '{uploaded_file.name}' with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")

# --- Sidebar Configuration --- #
st.sidebar.header("⚙️ Analysis Configuration")
valid_numeric_cols_in_df = [col for col in AVAILABLE_PARAMS_CONFIG if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col])]
missing_potential_params = [col for col in AVAILABLE_PARAMS_CONFIG if col not in df_raw.columns]
if missing_potential_params:
    st.sidebar.warning(f"Note: Potential parameters not found: {', '.join(missing_potential_params)}")
if not valid_numeric_cols_in_df:
    st.error("No valid numeric parameters found for analysis.")
    st.stop()

selected_params = st.sidebar.multiselect(
    "Select Parameters for Analysis:",
    options=valid_numeric_cols_in_df,
    default=valid_numeric_cols_in_df[:min(len(valid_numeric_cols_in_df), 4)]
)

params_weights = {}
if selected_params:
    st.sidebar.subheader("Parameter Weights")
    normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 100?", True)
    total_weight_input = 0
    for param in selected_params:
        direction_indicator = f" ({AVAILABLE_PARAMS_CONFIG.get(param, 'neutral')})"
        weight = st.sidebar.slider(f"Weight for {param}{direction_indicator}", 0, 100, 50)
        params_weights[param] = weight
        total_weight_input += weight

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

    if not df_processed.empty:
        df_ranked, normalized_params_cols = calculate_scores(df_processed, params_weights, AVAILABLE_PARAMS_CONFIG)

        st.header(f"🏆 Top {top_n} Ranked Stocks")
        st.markdown(f"Based on selected parameters and weights. Score ranges from 0 to 1 (higher is better).")

        cols_to_display = [col for col in ["Rank", "Name", "Composite Score"] + selected_params if col in df_ranked.columns]
        df_display = df_ranked.head(top_n)[cols_to_display]
        format_dict = {col: '{:.2f}' for col in df_display.select_dtypes(include=np.number).columns}
        format_dict['Rank'] = '{:d}'
        format_dict['Composite Score'] = '{:.3f}'
        df_styled = df_display.style.format(format_dict, na_rep='-')
        st.dataframe(df_styled)

        # --- Download Buttons --- #
        col1, col2 = st.columns(2)
        csv_data = df_display.to_csv(index=False).encode('utf-8')
        col1.download_button(label="Download Table as CSV", data=csv_data, file_name=f'top_{top_n}_stocks_ranked.csv', mime='text/csv')
        try:
            image_data = convert_df_to_image(df_styled)
            if image_data:
                col2.download_button(label="Download Table as Image", data=image_data, file_name=f'top_{top_n}_stocks_ranked.png', mime='image/png')
            # else: # Error message is now handled inside convert_df_to_image
            #     col2.warning("Image generation failed. Check logs/messages above.")
        except Exception as img_e:
             col2.error(f"Error during image generation setup: {img_e}")

        # --- Rank 1 Explanation --- #
        st.subheader("Rank #1 Analysis")
        if not df_ranked.empty:
            rank1_data = df_ranked[df_ranked['Rank'] == 1]
            explanation = generate_rank_explanation(rank1_data, selected_params, params_weights, AVAILABLE_PARAMS_CONFIG, normalized_params_cols)
            st.markdown(explanation)
        else:
            st.info("No stocks ranked.")

        st.divider()

        # --- Dashboard --- #
        st.header("📊 Dashboard")
        # Redesigned Tabs
        tab_breakdown, tab_scatter, tab_compare = st.tabs(["Top N Parameter Breakdown", "Scatter Analysis", "Top N Comparison"])

        with tab_breakdown:
            st.subheader(f"Parameter Details for Top {top_n} Stocks")
            st.markdown("Shows raw values and normalized scores (0-1) for the selected parameters.")
            # Prepare breakdown table
            breakdown_cols_raw = {param: param for param in selected_params}
            breakdown_cols_norm = {norm_col: f"{param} (Norm)" for param, norm_col in normalized_params_cols.items() if param in selected_params}
            breakdown_cols_display = ['Rank', 'Name', 'Composite Score'] + list(breakdown_cols_raw.keys()) + list(breakdown_cols_norm.keys())
            # Ensure columns exist
            breakdown_cols_display = [col for col in breakdown_cols_display if col in df_ranked.columns]
            df_breakdown = df_ranked.head(top_n)[breakdown_cols_display].rename(columns=breakdown_cols_norm)

            # Formatting for breakdown table
            format_dict_breakdown = {col: '{:.2f}' for col in df_breakdown.columns if '(Norm)' in col or col in selected_params}
            format_dict_breakdown['Rank'] = '{:d}'
            format_dict_breakdown['Composite Score'] = '{:.3f}'
            st.dataframe(df_breakdown.style.format(format_dict_breakdown, na_rep='-'))

        # Kept Scatter and Comparison Tabs
        with tab_scatter:
            st.subheader("Scatter Plot Analysis")
            if len(selected_params) >= 2:
                sc_col1, sc_col2 = st.columns(2)
                x_axis = sc_col1.selectbox("Select X-axis Parameter:", selected_params, index=0, key="scatter_x")
                y_axis = sc_col2.selectbox("Select Y-axis Parameter:", selected_params, index=1 if len(selected_params) > 1 else 0, key="scatter_y")
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
                st.info("Select at least two parameters.")

        with tab_compare:
            st.subheader(f"Comparison of Key Metric for Top {top_n} Stocks")
            if selected_params:
                metric_to_compare = st.selectbox("Select Metric to Compare:", selected_params, key="bar_compare")
                if metric_to_compare in df_ranked.columns and 'Name' in df_ranked.columns:
                    df_top_n_bar = df_ranked.head(top_n)
                    fig_bar = px.bar(df_top_n_bar, x='Name', y=metric_to_compare, title=f'{metric_to_compare} for Top {top_n} Stocks', text_auto='.2f')
                    fig_bar.update_layout(xaxis_title="Stock Name", yaxis_title=metric_to_compare)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("Selected metric/Name column not found.")
            else:
                 st.info("Select parameters in the sidebar.")

        st.divider()

        # --- Data Views --- #
        st.header("Data Views")
        with st.expander("View Uploaded Raw Data"):
            st.dataframe(df_raw)
        with st.expander("View Processed & Ranked Data (including normalized values)"):
            try:
                norm_cols_exist = [col for col in normalized_params_cols.values() if col in df_ranked.columns]
                display_cols_processed = [col for col in ["Rank", "Name", "Composite Score"] + selected_params + norm_cols_exist if col in df_ranked.columns]
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

