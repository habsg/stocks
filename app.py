# Main Streamlit application for Stock Analysis

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Stock Analyzer", page_icon="📊")

# --- Data Loading and Caching ---
@st.cache_data
def load_data(file_path):
    """Loads data from CSV, handling potential errors."""
    try:
        df = pd.read_csv(file_path)
        # Basic cleaning: Remove rows where Name is missing
        df.dropna(subset=['Name'], inplace=True)
        # Attempt to convert potential numeric columns, coercing errors to NaN
        potential_numeric_cols = [
            'Current Price', 'Market Capitalization', 'Price to Earning',
            'Price to book value', 'Return on equity', 'Debt',
            'Sustainable Growth Rate', 'Graham', 'Sales growth 3Years',
            'Profit growth 3Years', 'Profit after tax', 'OPM',
            'Return on capital employed', 'Sales latest quarter',
            'Return on assets', 'Debt to equity', 'Equity capital',
            'Working capital', 'Free cash flow last year',
            'Free cash flow preceding year'
        ]
        for col in potential_numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: Data file not found at {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Data Preprocessing ---
def preprocess_data(df, selected_params):
    """Handles missing values for selected parameters."""
    df_processed = df.copy()
    # For simplicity, fill NaNs in selected numeric columns with the median
    # A more robust approach might involve imputation or dropping rows/columns
    # based on the percentage of missing values.
    for param in selected_params:
        if df_processed[param].isnull().any():
            median_val = df_processed[param].median()
            df_processed[param].fillna(median_val, inplace=True)
            # st.warning(f"Missing values found in '{param}'. Filled with median ({median_val:.2f}).")
    # Drop rows where any selected parameter is still NaN (if median was NaN)
    df_processed.dropna(subset=selected_params, inplace=True)
    return df_processed

# --- Scoring Logic ---
def calculate_scores(df, params_weights, params_direction):
    """Calculates normalized scores based on selected parameters and weights."""
    df_scored = df.copy()
    df_scored['Composite Score'] = 0
    total_weight = sum(params_weights.values())

    if total_weight == 0:
        st.warning("Total weight is zero. Scores cannot be calculated.")
        return df_scored.assign(**{f'{p}_norm': 0 for p in params_weights.keys()}), df_scored

    normalized_params = {}

    for param, weight in params_weights.items():
        if param not in df_scored.columns:
            st.error(f"Parameter '{param}' not found in data.")
            continue

        # Min-Max Scaling (0 to 1)
        min_val = df_scored[param].min()
        max_val = df_scored[param].max()

        if max_val == min_val:
             # Avoid division by zero if all values are the same
            df_scored[f'{param}_norm'] = 0.5
        else:
            if params_direction.get(param, 'higher') == 'higher':
                # Higher is better: (value - min) / (max - min)
                df_scored[f'{param}_norm'] = (df_scored[param] - min_val) / (max_val - min_val)
            else:
                # Lower is better: (max - value) / (max - min)
                df_scored[f'{param}_norm'] = (max_val - df_scored[param]) / (max_val - min_val)

        # Apply weight (normalized weight)
        normalized_weight = weight / total_weight
        df_scored['Composite Score'] += df_scored[f'{param}_norm'] * normalized_weight
        normalized_params[param] = f'{param}_norm'

    # Rank based on score
    df_scored['Rank'] = df_scored['Composite Score'].rank(ascending=False, method='min').astype(int)
    df_scored.sort_values('Rank', inplace=True)

    return df_scored, normalized_params

# --- Main App Logic ---
data_path = '/home/ubuntu/stock_analyzer_app/data/steel_data.csv'
df_raw = load_data(data_path)

if df_raw is not None:
    st.title("📊 Stock Analysis Dashboard")
    st.markdown("Analyze and rank stocks based on customizable financial parameters.")

    # --- Sidebar --- #
    st.sidebar.header("⚙️ Analysis Configuration")

    # Define potential parameters and their 'ideal' direction
    # Add more parameters as needed from the CSV
    available_params = {
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
        'Free cash flow last year': 'higher'
        # Add other relevant numeric columns from your CSV here
    }

    # Filter available params to only those present in the loaded data
    valid_params = {k: v for k, v in available_params.items() if k in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[k])}

    if not valid_params:
        st.error("No valid numeric parameters found for analysis in the uploaded data.")
    else:
        selected_params = st.sidebar.multiselect(
            "Select Parameters for Analysis:",
            options=list(valid_params.keys()),
            default=list(valid_params.keys())[:4] # Default to first 4 valid params
        )

        params_weights = {}
        if selected_params:
            st.sidebar.subheader("Parameter Weights")
            normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 100?", True)
            total_weight_input = 0
            for param in selected_params:
                weight = st.sidebar.slider(f"Weight for {param}", 0, 100, 50)
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
            elif total_weight_input == 0:
                 st.sidebar.warning("All weights are zero. Please assign weights to parameters.")

        else:
            st.sidebar.warning("Please select at least one parameter for analysis.")

        top_n = st.sidebar.number_input("Select Top N Stocks to Display:", min_value=1, max_value=len(df_raw), value=min(10, len(df_raw)))

        # --- Main Area --- #
        if selected_params and sum(params_weights.values()) > 0:
            # Preprocess data based on selected parameters
            df_processed = preprocess_data(df_raw, selected_params)

            if not df_processed.empty:
                # Calculate scores
                df_ranked, normalized_params_cols = calculate_scores(df_processed, params_weights, valid_params)

                st.header(f"🏆 Top {top_n} Ranked Stocks")
                st.markdown(f"Based on selected parameters and weights. Score ranges from 0 to 1 (higher is better).")

                # Columns to display in the ranked table
                cols_to_display = ['Rank', 'Name', 'Composite Score'] + selected_params
                st.dataframe(df_ranked.head(top_n)[cols_to_display].style.format({'Composite Score': "{:.3f}"}, na_rep='-'))

                st.divider()

                # --- Dashboard --- #
                st.header("📊 Dashboard")
                tab1, tab2, tab3, tab4 = st.tabs(["Score Distribution", "Parameter Correlation", "Scatter Analysis", "Top N Comparison"])

                with tab1:
                    st.subheader("Distribution of Composite Scores")
                    fig_hist = px.histogram(df_ranked, x='Composite Score', nbins=20, title='Overall Score Distribution')
                    st.plotly_chart(fig_hist, use_container_width=True)

                with tab2:
                    st.subheader("Correlation Between Selected Parameters")
                    if len(selected_params) > 1:
                        corr = df_processed[selected_params].corr()
                        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title='Correlation Matrix')
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.info("Select at least two parameters to view correlation.")

                with tab3:
                    st.subheader("Scatter Plot Analysis")
                    if len(selected_params) >= 2:
                        col1, col2 = st.columns(2)
                        x_axis = col1.selectbox("Select X-axis Parameter:", selected_params, index=0)
                        y_axis = col2.selectbox("Select Y-axis Parameter:", selected_params, index=1 if len(selected_params) > 1 else 0)

                        if x_axis != y_axis:
                            fig_scatter = px.scatter(
                                df_ranked.head(top_n),
                                x=x_axis,
                                y=y_axis,
                                hover_name='Name',
                                title=f'{y_axis} vs. {x_axis} for Top {top_n} Stocks',
                                color='Composite Score', # Optional: color by score
                                color_continuous_scale=px.colors.sequential.Viridis
                            )
                            st.plotly_chart(fig_scatter, use_container_width=True)
                        else:
                            st.warning("Please select different parameters for X and Y axes.")
                    else:
                        st.info("Select at least two parameters to view a scatter plot.")

                with tab4:
                    st.subheader(f"Comparison of Key Metric for Top {top_n} Stocks")
                    if selected_params:
                        metric_to_compare = st.selectbox("Select Metric to Compare:", selected_params)
                        df_top_n = df_ranked.head(top_n)
                        fig_bar = px.bar(df_top_n, x='Name', y=metric_to_compare, title=f'{metric_to_compare} for Top {top_n} Stocks', text_auto=True)
                        fig_bar.update_layout(xaxis_title="Stock Name", yaxis_title=metric_to_compare)
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                         st.info("Select parameters in the sidebar to compare.")

                st.divider()

                # --- Data Views --- #
                st.header("Raw and Processed Data")
                with st.expander("View Raw Data"):
                    st.dataframe(df_raw)
                with st.expander("View Processed & Ranked Data (including normalized values)"):
                    display_cols_processed = ['Rank', 'Name', 'Composite Score'] + selected_params + list(normalized_params_cols.values())
                    st.dataframe(df_ranked[display_cols_processed].style.format({'Composite Score': "{:.3f}"}, formatter={col: "{:.3f}" for col in normalized_params_cols.values()}, na_rep='-'))

            else:
                st.warning("No data remaining after preprocessing with selected parameters. Check data quality or parameter selection.")
        else:
            st.info("Configure analysis parameters in the sidebar to see results.")
else:
    st.error("Failed to load data. Cannot start the application.")


