# Main Streamlit application for Stock Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Analyzer", page_icon="📊")

# Define essential columns needed for the core analysis
# Adjust this list based on the absolute minimum columns required
ESSENTIAL_COLUMNS = [ 'Name' ] # Name is usually essential for identification

# Define potential parameters and their 'ideal' direction
# These are the columns the user *can* select for analysis if they exist
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
    # Add other potential numeric columns from typical financial datasets
    'Market Capitalization': 'higher', # Example: Higher might be preferred by some
    'Current Price': 'neutral', # Neutral, depends on context
    'Debt': 'lower',
    'Graham': 'lower', # Graham number comparison often implies lower is better value
    'Profit after tax': 'higher',
    'Sales latest quarter': 'higher',
    'Equity capital': 'neutral',
    'Working capital': 'neutral',
    'Free cash flow preceding year': 'higher'
}

# --- Data Loading and Validation ---
@st.cache_data
def load_data(uploaded_file):
    """Loads data from uploaded file (CSV or Excel), handling potential errors."""
    if uploaded_file is None:
        return None, "Please upload a data file."

    try:
        # Check file type
        if uploaded_file.name.endswith('.csv'):
            # Read CSV directly from the uploaded file object
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            # Read Excel directly from the uploaded file object
            df = pd.read_excel(uploaded_file)
        else:
            return None, "Unsupported file format. Please upload a CSV or Excel file."

        # --- Column Validation ---
        missing_essential_cols = [col for col in ESSENTIAL_COLUMNS if col not in df.columns]
        if missing_essential_cols:
            return None, f"Error: The uploaded file is missing essential columns: {', '.join(missing_essential_cols)}. Cannot proceed."

        # --- Basic Cleaning & Type Conversion ---
        df.dropna(subset=["Name"], inplace=True) # Drop rows where Name is missing

        potential_numeric_cols = list(AVAILABLE_PARAMS_CONFIG.keys()) + [ # Include other potential numerics
             'Market Capitalization', 'Current Price', 'Debt', 'Graham',
             'Profit after tax', 'Sales latest quarter', 'Equity capital',
             'Working capital', 'Free cash flow preceding year'
             ]

        converted_cols = []
        for col in potential_numeric_cols:
            if col in df.columns:
                original_dtype = df[col].dtype
                # Attempt conversion, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Check if conversion actually happened (useful for object columns)
                if df[col].dtype != original_dtype:
                    converted_cols.append(col)

        # Report columns that couldn't be fully converted (still contain NaNs after coercion)
        # Note: This doesn't guarantee they are *all* numeric, just that conversion was attempted.
        # A stricter check might be needed depending on requirements.

        return df, None # Return dataframe and no error message

    except Exception as e:
        return None, f"Error processing file: {e}"

# --- Data Preprocessing ---
def preprocess_data(df, selected_params):
    """Handles missing values for selected parameters."""
    df_processed = df.copy()
    warnings = []
    # Fill NaNs in selected numeric columns with the median
    for param in selected_params:
        if df_processed[param].isnull().any():
            median_val = df_processed[param].median()
            if pd.isna(median_val):
                 # If median is NaN (e.g., all values were NaN), fill with 0 or drop?
                 # Filling with 0 might skew results, dropping might lose data.
                 # Let's fill with 0 for now and warn.
                 df_processed[param].fillna(0, inplace=True)
                 warnings.append(f"Column 	'{param}	' contained only missing values after conversion; filled with 0.")
            else:
                df_processed[param].fillna(median_val, inplace=True)
                warnings.append(f"Missing values found in 	'{param}	'. Filled with median ({median_val:.2f}).")

    # Drop rows where any selected parameter is still NaN (shouldn't happen with above fill)
    # df_processed.dropna(subset=selected_params, inplace=True)
    return df_processed, warnings

# --- Scoring Logic ---
def calculate_scores(df, params_weights, params_direction):
    """Calculates normalized scores based on selected parameters and weights."""
    df_scored = df.copy()
    df_scored['Composite Score'] = 0
    total_weight = sum(params_weights.values())

    if total_weight <= 0:
        st.warning("Total weight is zero or negative. Scores cannot be calculated.")
        # Assign default score and rank if weights are zero
        df_scored['Composite Score'] = 0
        df_scored['Rank'] = 1
        return df_scored, {}

    normalized_params_cols = {}

    for param, weight in params_weights.items():
        if param not in df_scored.columns:
            # This check should ideally be done earlier
            st.error(f"Parameter 	'{param}	' not found in data during scoring.")
            continue
        if weight == 0:
             continue # Skip parameters with zero weight

        # Min-Max Scaling (0 to 1)
        min_val = df_scored[param].min()
        max_val = df_scored[param].max()

        # Check for valid direction, default to 'higher'
        direction = params_direction.get(param, 'higher')

        if max_val == min_val:
             # Avoid division by zero if all values are the same
             # Assign 0.5 as a neutral score
            df_scored[f'{param}_norm'] = 0.5
        else:
            if direction == 'higher':
                # Higher is better: (value - min) / (max - min)
                df_scored[f'{param}_norm'] = (df_scored[param] - min_val) / (max_val - min_val)
            elif direction == 'lower':
                # Lower is better: (max - value) / (max - min)
                df_scored[f'{param}_norm'] = (max_val - df_scored[param]) / (max_val - min_val)
            else: # Neutral or unrecognized direction
                 df_scored[f'{param}_norm'] = 0.5 # Assign neutral score

        # Apply weight (normalized weight)
        normalized_weight = weight / total_weight
        df_scored['Composite Score'] += df_scored[f'{param}_norm'] * normalized_weight
        normalized_params_cols[param] = f'{param}_norm'

    # Rank based on score
    df_scored['Rank'] = df_scored['Composite Score'].rank(ascending=False, method='min').astype(int)
    df_scored.sort_values('Rank', inplace=True)

    return df_scored, normalized_params_cols

# --- Main App Logic ---
st.title("📊 Customizable Stock Analyzer")
st.markdown("Upload your stock data (CSV or Excel) and rank stocks based on selected financial parameters and weights.")

# --- File Uploader --- #
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

df_raw = None
error_message = None

if uploaded_file is not None:
    df_raw, error_message = load_data(uploaded_file)

# --- Display based on Data Load Status --- #
if error_message:
    st.error(error_message)
    st.stop() # Stop execution if loading failed critically

if df_raw is None:
    st.info("Please upload a data file to begin analysis.")
    st.stop()

# --- Data Loaded Successfully --- #
st.success(f"Successfully loaded data from '{uploaded_file.name}' with {df_raw.shape[0]} rows and {df_raw.shape[1]} columns.")

# --- Sidebar Configuration --- #
st.sidebar.header("⚙️ Analysis Configuration")

# Filter available params to only those present and numeric in the *uploaded* data
valid_numeric_cols_in_df = [col for col in AVAILABLE_PARAMS_CONFIG if col in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[col])]

# Check for missing potential parameters
missing_potential_params = [col for col in AVAILABLE_PARAMS_CONFIG if col not in df_raw.columns]
if missing_potential_params:
    st.sidebar.warning(f"Note: The following potential analysis parameters were not found in the uploaded file: {', '.join(missing_potential_params)}")

if not valid_numeric_cols_in_df:
    st.error("No valid numeric parameters (from the predefined list) found for analysis in the uploaded data. Cannot proceed with ranking.")
    st.stop()

selected_params = st.sidebar.multiselect(
    "Select Parameters for Analysis:",
    options=valid_numeric_cols_in_df,
    default=valid_numeric_cols_in_df[:min(len(valid_numeric_cols_in_df), 4)] # Default to first 4 valid params
)

params_weights = {}
if selected_params:
    st.sidebar.subheader("Parameter Weights")
    normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 100?", True)
    total_weight_input = 0
    for param in selected_params:
        # Use direction info in slider label
        direction_indicator = f" ({AVAILABLE_PARAMS_CONFIG.get(param, 'neutral')})"
        weight = st.sidebar.slider(f"Weight for {param}{direction_indicator}", 0, 100, 50)
        params_weights[param] = weight
        total_weight_input += weight

    # Normalize weights if checkbox is ticked and total is not 100
    if normalize_weights and total_weight_input > 0 and total_weight_input != 100:
        st.sidebar.info(f"Normalizing weights from {total_weight_input} to 100.")
        factor = 100 / total_weight_input
        params_weights = {p: w * factor for p, w in params_weights.items()}
        # Display normalized weights (read-only)
        st.sidebar.markdown("**Normalized Weights:**")
        for param, weight in params_weights.items():
            st.sidebar.markdown(f"- *{param}*: {weight:.1f}")
    elif total_weight_input <= 0:
         st.sidebar.warning("All weights are zero or negative. Please assign positive weights to parameters for ranking.")

else:
    st.sidebar.warning("Please select at least one parameter for analysis.")

top_n = st.sidebar.number_input("Select Top N Stocks to Display:", min_value=1, max_value=len(df_raw), value=min(10, len(df_raw)))

# --- Main Area --- #
if selected_params and sum(params_weights.values()) > 0:
    # Preprocess data based on selected parameters
    df_processed, preprocess_warnings = preprocess_data(df_raw, selected_params)
    if preprocess_warnings:
        for warning in preprocess_warnings:
            st.warning(warning)

    if not df_processed.empty:
        # Calculate scores
        df_ranked, normalized_params_cols = calculate_scores(df_processed, params_weights, AVAILABLE_PARAMS_CONFIG)

        st.header(f"🏆 Top {top_n} Ranked Stocks")
        st.markdown(f"Based on selected parameters and weights. Score ranges from 0 to 1 (higher is better).")

        # Columns to display in the ranked table
        cols_to_display = [col for col in ["Rank", "Name", "Composite Score"] + selected_params if col in df_ranked.columns]
        st.dataframe(df_ranked.head(top_n)[cols_to_display].style.format({"Composite Score": "{:.3f}"}, na_rep='-'))

        st.divider()

        # --- Dashboard --- #
        st.header("📊 Dashboard")
        tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Parameter Correlation", "Scatter Analysis", "Top N Comparison"])

        with tab1:
            st.subheader("Distribution of Composite Scores")
            if 'Composite Score' in df_ranked.columns:
                fig_hist = px.histogram(df_ranked, x='Composite Score', nbins=20, title='Overall Score Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("Composite Score not available for distribution plot.")

        with tab2:
            st.subheader("Correlation Between Selected Parameters")
            if len(selected_params) > 1:
                # Ensure only numeric columns selected are used for correlation
                numeric_selected_params = [p for p in selected_params if pd.api.types.is_numeric_dtype(df_processed[p])]
                if len(numeric_selected_params) > 1:
                    corr = df_processed[numeric_selected_params].corr()
                    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Matrix')
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Need at least two numeric parameters selected to calculate correlation.")
            else:
                st.info("Select at least two parameters to view correlation.")

        with tab3:
            st.subheader("Scatter Plot Analysis")
            if len(selected_params) >= 2:
                col1, col2 = st.columns(2)
                x_axis = col1.selectbox("Select X-axis Parameter:", selected_params, index=0, key="scatter_x")
                y_axis = col2.selectbox("Select Y-axis Parameter:", selected_params, index=1 if len(selected_params) > 1 else 0, key="scatter_y")

                if x_axis != y_axis:
                    # Ensure columns exist before plotting
                    if x_axis in df_ranked.columns and y_axis in df_ranked.columns:
                        fig_scatter = px.scatter(
                            df_ranked.head(top_n),
                            x=x_axis,
                            y=y_axis,
                            hover_name='Name',
                            title=f'{y_axis} vs. {x_axis} for Top {top_n} Stocks',
                            color='Composite Score' if 'Composite Score' in df_ranked.columns else None,
                            color_continuous_scale=px.colors.sequential.Viridis
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    else:
                        st.warning("Selected axis parameter not found in ranked data.")
                else:
                    st.warning("Please select different parameters for X and Y axes.")
            else:
                st.info("Select at least two parameters to view a scatter plot.")

        with tab4:
            st.subheader(f"Comparison of Key Metric for Top {top_n} Stocks")
            if selected_params:
                metric_to_compare = st.selectbox("Select Metric to Compare:", selected_params, key="bar_compare")
                if metric_to_compare in df_ranked.columns:
                    df_top_n = df_ranked.head(top_n)
                    fig_bar = px.bar(df_top_n, x='Name', y=metric_to_compare, title=f'{metric_to_compare} for Top {top_n} Stocks	', text_auto=True)
                    fig_bar.update_layout(xaxis_title="Stock Name", yaxis_title=metric_to_compare)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning("Selected metric not found in ranked data.")
            else:
                 st.info("Select parameters in the sidebar to compare.")

        st.divider()

        # --- Data Views --- #
        st.header("Data Views")
        with st.expander("View Uploaded Raw Data"):
            st.dataframe(df_raw)
        with st.expander("View Processed & Ranked Data (including normalized values)"):
            # Ensure all columns exist before trying to display
            norm_cols_exist = [col for col in normalized_params_cols.values() if col in df_ranked.columns]
            display_cols_processed = [col for col in ["Rank", "Name", "Composite Score"] + selected_params + norm_cols_exist if col in df_ranked.columns]
            st.dataframe(df_ranked[display_cols_processed].style.format({"Composite Score": "{:.3f}"}, formatter={col: "{:.3f}" for col in norm_cols_exist}, na_rep='-'))

    else:
        st.warning("No data remaining after preprocessing with selected parameters. Check data quality or parameter selection.")
else:
    st.info("Configure analysis parameters in the sidebar to see results.")

