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
    "Debt to equity": "(D/E Ratio) Measures a company's financial leverage, calculated by dividing its total liabilities by its stockholders’ equity. Lower generally indicates less risk.",
    "Debt": "Total amount of debt held by the company. Lower is generally preferred.",
    "Market Capitalization": "The total market value of a company's outstanding shares of stock. Often used to size up corporations.",
    "Net Profit": "The actual profit after working expenses not included in the calculation of gross profit have been paid.",
    "Last Year Profit": "The profit figure from the previous financial year."
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
            # Ensure the normalized score is not NaN before calculating contribution
            norm_score = rank1_stock[norm_col].iloc[0]
            if pd.notna(norm_score):
                contribution = norm_score * (weight / total_weight)
                contributions[param] = contribution
            else:
                contributions[param] = 0 # Assign 0 contribution if score is NaN

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
        if pd.isna(raw_value):
            performance_desc = "had missing data (treated as 0 for ranking)"
        elif direction == "higher":
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
        if param in df_all_ranked.columns and pd.api.types.is_numeric_dtype(df_all_ranked[param]):
            best_performer_row = None
            df_param_valid = df_all_ranked.dropna(subset=[param]) # Consider only rows with valid data for this param
            if not df_param_valid.empty:
                if direction == "higher":
                    best_performer_row = df_param_valid.loc[df_param_valid[param].idxmax()]
                elif direction == "lower":
                    best_performer_row = df_param_valid.loc[df_param_valid[param].idxmin()]

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
        # Ensure column exists and is numeric before proceeding
        if market_cap_col in df.columns and pd.api.types.is_numeric_dtype(df[market_cap_col]):
            df_cap = df.dropna(subset=[market_cap_col])
            if not df_cap.empty:
                summary += f"- **Market Capitalization ({market_cap_col}):**\n"
                bins = [0, 100, 1000, 10000, np.inf]
                labels = ["< 100 Cr", "100-1000 Cr", "1000-10000 Cr", "> 10000 Cr"]
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
        else:
             summary += f"- **Market Capitalization ({market_cap_col}):** No valid data found.\n"

    # Exchange Listing Analysis (if column exists)
    exchange_col = next((col for col in df.columns if "exchange" in col.lower() or "listed on" in col.lower()), None)
    if exchange_col:
        if exchange_col in df.columns:
            exchange_counts = df[exchange_col].astype(str).str.upper().value_counts()
            if not exchange_counts.empty:
                summary += f"- **Exchange Listing ({exchange_col}):**\n"
                for ex, count in exchange_counts.items():
                    summary += f"    - {ex}: {count} companies\n"
            else:
                 summary += f"- **Exchange Listing ({exchange_col}):** No valid data found.\n"

    # Sector distribution
    sector_col = next((col for col in df.columns if "sector" in col.lower() or "industry" in col.lower()), None)
    if sector_col:
        if sector_col in df.columns:
            sector_counts = df[sector_col].value_counts()
            if not sector_counts.empty:
                summary += f"- **Sector/Industry ({sector_col}):**\n"
                top_n_sectors = 5
                for sector, count in sector_counts.head(top_n_sectors).items():
                     summary += f"    - {sector}: {count} companies\n"
                if len(sector_counts) > top_n_sectors:
                     summary += f"    - ... ({len(sector_counts) - top_n_sectors} other sectors)\n"
            else:
                 summary += f"- **Sector/Industry ({sector_col}):** No valid data found.\n"

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

        # DEBUG: Print original dtypes
        # print("Original dtypes:\n", df.dtypes)

        numeric_cols_initial = df.select_dtypes(include=np.number).columns.tolist()
        converted_cols = []
        for col in df.select_dtypes(include=["object"]).columns:
             if col in ESSENTIAL_COLUMNS: continue
             try:
                 # Attempt conversion, cleaning common currency/percentage symbols first
                 cleaned_col = df[col].astype(str).str.replace(r'[$,%,₹,Cr]', '', regex=True).str.strip()
                 converted_col = pd.to_numeric(cleaned_col, errors='coerce')
                 # Check if *any* value was successfully converted (not all NaN)
                 if not converted_col.isnull().all():
                     df[col] = converted_col
                     converted_cols.append(col)
                     # DEBUG: Print successful conversion
                     # print(f"Successfully converted '{col}' to numeric.")
                 # else:
                     # DEBUG: Print failed conversion
                     # print(f"Failed to convert '{col}' to numeric (all NaN after conversion).")
             except Exception as e:
                 # DEBUG: Print error during conversion
                 # print(f"Error converting '{col}': {e}")
                 continue

        # Combine initially numeric and successfully converted columns
        final_numeric_cols = list(set(numeric_cols_initial + converted_cols))

        # DEBUG: Print final dtypes and identified numeric cols
        # print("Final dtypes:\n", df.dtypes)
        # print("Identified numeric columns:", final_numeric_cols)

        return df, None, final_numeric_cols
    except Exception as e:
        return None, f"Error processing file: {e}", []

# --- Data Preprocessing --- #
def preprocess_data(df, selected_params):
    """Handles missing values by filling with 0. Returns warnings in a list of dicts."""
    df_processed = df.copy()
    warnings_list = [] # Changed from simple list to list of dicts
    for param in selected_params:
        if param not in df_processed.columns:
             # This case should ideally be handled by checking selected_params against df columns earlier
             continue
        if pd.api.types.is_numeric_dtype(df_processed[param]):
            if df_processed[param].isnull().any():
                num_missing = df_processed[param].isnull().sum()
                df_processed[param].fillna(0, inplace=True)
                # Append dict for table display
                warnings_list.append({"Parameter": param, "Missing Values": num_missing, "Action": "Filled with 0"})
        # else: # Non-numeric columns are filtered out before scoring now
            # warnings_list.append({"Parameter": param, "Missing Values": "N/A", "Action": "Skipped (Non-Numeric)"})

    return df_processed, warnings_list

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
        if not pd.api.types.is_numeric_dtype(df_scored[param]): continue
        if weight == 0: continue

        min_val = df_scored[param].min()
        max_val = df_scored[param].max()
        direction = params_direction.get(param, "higher")
        norm_col_name = f"{param}_norm"
        normalized_params_cols[param] = norm_col_name

        if max_val == min_val:
            df_scored[norm_col_name] = 0.5 # Assign mid-score if all values are the same
        else:
            if direction == "higher":
                df_scored[norm_col_name] = (df_scored[param] - min_val) / (max_val - min_val)
            elif direction == "lower":
                df_scored[norm_col_name] = (max_val - df_scored[param]) / (max_val - min_val)
            else: # neutral
                 df_scored[norm_col_name] = 0.5 # Assign mid-score for neutral params

        normalized_weight = weight / total_weight
        # Ensure normalized score is not NaN before adding to composite score
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
    # Reset state if file changes
    if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.clear() # Clear entire state for new file
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.defaults_applied = False

    df_raw, error_message, all_numeric_cols = load_data(uploaded_file)

if error_message:
    st.error(error_message)
    st.stop()
if df_raw is None:
    st.info("Please upload a data file to begin analysis.")
    st.stop()

st.success(f"Successfully loaded and processed '{uploaded_file.name}'. Found {len(df_raw)} companies and {len(all_numeric_cols)} potential numeric parameters.")

# --- Sidebar Controls --- #
params_direction = {col: infer_direction(col) for col in all_numeric_cols}
default_params_available = [p for p in DEFAULT_PARAMS_WEIGHTS if p in all_numeric_cols]

if 'defaults_applied' not in st.session_state: st.session_state.defaults_applied = False
# Initialize selected_params in state if it doesn't exist, using defaults
if 'selected_params' not in st.session_state:
     st.session_state.selected_params = default_params_available

# Apply defaults logic only once per file upload
if not st.session_state.defaults_applied:
     current_selection = default_params_available
     st.session_state.selected_params = current_selection
     st.session_state.defaults_applied = True
     # --- Display Default Suggestion Note --- #
     if current_selection:
         suggestion_note = "**Suggestion:** Based on your data, the following parameters have been pre-selected with default weights for analysis. You can customize the parameters and weights in the sidebar below.\n\n"
         suggestion_note += "**Default Parameters & Weights:**\n"
         default_weights_display = {p: w for p, w in DEFAULT_PARAMS_WEIGHTS.items() if p in current_selection}
         # Normalize if needed for display consistency
         total_default_weight = sum(default_weights_display.values())
         if total_default_weight > 0 and total_default_weight != 100:
              factor = 100 / total_default_weight
              default_weights_display = {p: w * factor for p, w in default_weights_display.items()}

         for param, weight in default_weights_display.items():
             suggestion_note += f"- {param}: {weight:.1f}%\n"
         st.info(suggestion_note)
     # --- End Suggestion Note ---
else:
     # Use selection from state if defaults already applied
     current_selection = st.session_state.selected_params if st.session_state.selected_params else default_params_available

# --- Parameter Selection --- #
st.sidebar.subheader("Analysis Parameters")
selected_params = st.sidebar.multiselect(
    "Select Parameters for Analysis:",
    options=all_numeric_cols,
    default=current_selection,
    key="param_selector"
)
st.session_state.selected_params = selected_params # Update state

# --- Dynamic Range Filter (Market Cap / Profit) --- #
filter_col = None
filter_col_name = ""
# Prioritize Market Cap
market_cap_cols = [col for col in df_raw.columns if "market cap" in col.lower() or "market capitalization" in col.lower()]
# DEBUG: Print found market cap columns
# print("Found potential market cap columns:", market_cap_cols)

if market_cap_cols:
    potential_col = market_cap_cols[0]
    # DEBUG: Print potential column and its dtype
    # print(f"Checking potential filter column: '{potential_col}', dtype: {df_raw[potential_col].dtype}")
    # Check if the column is actually numeric in the DataFrame *after* loading
    if pd.api.types.is_numeric_dtype(df_raw[potential_col]):
        filter_col = potential_col
        filter_col_name = "Market Capitalization" # Use consistent name
        # DEBUG: Print confirmation
        # print(f"Using '{filter_col}' for filtering.")
    # else:
        # DEBUG: Print why it wasn't selected
        # print(f"Column '{potential_col}' is not numeric, skipping for filter.")

# Fallback to Profit if Market Cap not found or not numeric
if filter_col is None:
    profit_cols = [col for col in df_raw.columns if "net profit" in col.lower() or "last year profit" in col.lower() or col.lower() == "profit after tax"]
    # DEBUG: Print found profit columns
    # print("Found potential profit columns:", profit_cols)
    if profit_cols:
        potential_col = profit_cols[0]
        # DEBUG: Print potential column and its dtype
        # print(f"Checking potential filter column: '{potential_col}', dtype: {df_raw[potential_col].dtype}")
        if pd.api.types.is_numeric_dtype(df_raw[potential_col]):
            filter_col = potential_col
            filter_col_name = potential_col # Use actual column name
            # DEBUG: Print confirmation
            # print(f"Using '{filter_col}' for filtering.")
        # else:
            # DEBUG: Print why it wasn't selected
            # print(f"Column '{potential_col}' is not numeric, skipping for filter.")

# Initialize filter range in session state
if filter_col and f'{filter_col}_range' not in st.session_state:
    min_val = float(df_raw[filter_col].min())
    max_val = float(df_raw[filter_col].max())
    # Ensure min/max are valid before setting state
    if pd.notna(min_val) and pd.notna(max_val) and min_val <= max_val:
        st.session_state[f'{filter_col}_range'] = (min_val, max_val)
    else:
        # Handle case where min/max couldn't be determined (e.g., all NaNs)
        st.session_state[f'{filter_col}_range'] = (0.0, 0.0) # Default or placeholder
        filter_col = None # Disable filter if range is invalid
        # print(f"Could not determine valid min/max for '{filter_col}', disabling filter.")

# Display the slider if a valid filter column is found
if filter_col:
    st.sidebar.subheader(f"Filter by {filter_col_name}")
    # Retrieve min/max again in case they were invalid initially
    min_val_raw = float(df_raw[filter_col].min())
    max_val_raw = float(df_raw[filter_col].max())
    # Ensure min/max are valid before creating slider
    if pd.notna(min_val_raw) and pd.notna(max_val_raw) and min_val_raw <= max_val_raw:
        # Round min down and max up to nearest 100 for slider steps
        min_val_slider = max(0.0, np.floor(min_val_raw / 100.0) * 100.0)
        max_val_slider = np.ceil(max_val_raw / 100.0) * 100.0
        if max_val_slider <= min_val_slider: # Handle cases where max rounds down below min
            max_val_slider = min_val_slider + 100.0

        # Get the stored range, default to rounded min/max if not set or invalid
        current_range_raw = st.session_state.get(f'{filter_col}_range', (min_val_slider, max_val_slider))
        # Round current range to nearest 100 and ensure bounds
        current_range_slider = (
            max(min_val_slider, round(current_range_raw[0] / 100.0) * 100.0),
            min(max_val_slider, round(current_range_raw[1] / 100.0) * 100.0)
        )

        selected_range = st.sidebar.slider(
            f"Select range for {filter_col} (Cr):".replace("Market Capitalization", "Market Cap"), # Shorten label
            min_value=float(min_val_slider),
            max_value=float(max_val_slider),
            value=(float(current_range_slider[0]), float(current_range_slider[1])),
            step=100.0, # Step by 100
            format="%.0f Cr", # Display as integer Cr
            key=f'{filter_col}_slider'
        )
        # Store the potentially float range selected by the slider
        st.session_state[f'{filter_col}_range'] = selected_range
    else:
        st.sidebar.warning(f"Could not determine a valid range for {filter_col}. Filtering disabled.")
        filter_col = None # Disable filtering if range is invalid
# --- End Dynamic Range Filter --- #

# --- Parameter Weights --- #
params_weights = {}
if selected_params:
    st.sidebar.subheader("Parameter Weights")
    normalize_weights = st.sidebar.checkbox("Normalize weights to sum to 100?", True)
    total_weight_input = 0
    weight_sliders = {}
    for param in selected_params:
        direction_indicator = f" ({params_direction.get(param, 'neutral')})"
        # Get default weight, ensure it's within 0-100
        default_weight = max(0, min(100, DEFAULT_PARAMS_WEIGHTS.get(param, 50)))
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

# --- Top N Selection --- #
top_n = st.sidebar.number_input("Select Top N Stocks to Display:", min_value=1, max_value=len(df_raw), value=min(10, len(df_raw)))

# --- Main Area Processing --- #
if selected_params and sum(params_weights.values()) > 0:
    # Filter out non-numeric columns from selected_params BEFORE preprocessing
    numeric_selected_params = [p for p in selected_params if p in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw[p])]
    if not numeric_selected_params:
         st.error("No valid numeric parameters selected for analysis.")
         st.stop()
    if len(numeric_selected_params) < len(selected_params):
         skipped_params = [p for p in selected_params if p not in numeric_selected_params]
         st.warning(f"Skipping non-numeric parameters: {', '.join(skipped_params)}")

    # Apply dynamic range filter if active
    df_filtered = df_raw.copy()
    if filter_col and f'{filter_col}_range' in st.session_state:
        min_select, max_select = st.session_state[f'{filter_col}_range']
        # Ensure filtering happens correctly even with NaNs
        # Make sure filter_col is numeric before attempting filter
        if pd.api.types.is_numeric_dtype(df_filtered[filter_col]):
            df_filtered = df_filtered[df_filtered[filter_col].between(min_select, max_select, inclusive='both')]
            st.info(f"Filtered data based on {filter_col} range: {min_select:.2f} - {max_select:.2f}. Showing {len(df_filtered)} companies.")
            if df_filtered.empty:
                st.warning("No companies match the selected filter range.")
                st.stop()
        else:
             st.warning(f"Filter column '{filter_col}' is not numeric. Cannot apply filter.")

    # Preprocess the potentially filtered data
    df_processed, preprocess_warnings_list = preprocess_data(df_filtered, numeric_selected_params)

    # --- Display Missing Value Warnings Table --- #
    if preprocess_warnings_list:
        st.subheader("Data Preprocessing Notes")
        warnings_df = pd.DataFrame(preprocess_warnings_list)
        st.table(warnings_df)
    # --- End Warnings Table ---

    numeric_params_weights = {p: w for p, w in params_weights.items() if p in numeric_selected_params}

    if not df_processed.empty:
        df_ranked, normalized_params_cols = calculate_scores(df_processed, numeric_params_weights, params_direction)

        st.header(f"🏆 Top {top_n} Ranked Stocks")
        st.markdown(f"Based on selected parameters and weights. Score ranges from 0 to 1 (higher is better).")

        cols_to_display = [col for col in ["Rank", "Name", "Composite Score"] + numeric_selected_params if col in df_ranked.columns]
        # Ensure filter_col is displayed if it was used
        if filter_col and filter_col not in cols_to_display:
            cols_to_display.insert(3, filter_col)

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
            # Add filter col if used
            if filter_col and filter_col not in breakdown_cols_display:
                 breakdown_cols_display.insert(3, filter_col)
            breakdown_cols_display = [col for col in breakdown_cols_display if col in df_ranked.columns]
            df_breakdown = df_ranked.head(top_n)[breakdown_cols_display].rename(columns=breakdown_cols_norm)
            format_dict_breakdown = {col: '{:.2f}' for col in df_breakdown.columns if '(Norm)' in col or col in numeric_selected_params or col == filter_col}
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
        # Generate summary based on the *original* raw data before filtering
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
                # Add filter col if used
                if filter_col and filter_col not in display_cols_processed:
                    display_cols_processed.insert(3, filter_col)
                df_processed_display = df_ranked[display_cols_processed]
                format_dict_processed = {col: '{:.3f}' for col in df_processed_display.select_dtypes(include=np.number).columns if col != 'Rank'}
                format_dict_processed['Rank'] = '{:d}'
                st.dataframe(df_processed_display.style.format(format_dict_processed, na_rep='-'))
            except Exception as e:
                st.error(f"Error displaying processed data table: {e}")
                # Fallback display attempt
                try:
                    st.dataframe(df_ranked)
                except Exception:
                    st.warning("Could not display processed data.")

    else:
        st.warning("No data remaining after preprocessing or filtering.")
else:
    st.info("Configure analysis parameters in the sidebar.")

