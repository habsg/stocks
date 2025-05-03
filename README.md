# Stock Analysis Streamlit App

This project provides a Streamlit web application for analyzing and ranking stocks based on user-selected financial parameters and customizable weights.

## Features

*   **Data Loading:** Loads stock data from a CSV file (`data/steel_data.csv`).
*   **Parameter Selection:** Allows users to choose which financial parameters (e.g., P/E Ratio, ROE, Debt-to-Equity) to include in the analysis.
*   **Weight Customization:** Enables users to assign weights to each selected parameter, reflecting their importance in the ranking.
*   **Weight Normalization:** Option to automatically normalize weights so they sum up to 100.
*   **Top N Selection:** Users can specify how many top-ranked stocks to display.
*   **Scoring & Ranking:** Calculates a composite score for each stock based on normalized parameter values, weights, and directionality (higher/lower is better). Ranks stocks accordingly.
*   **Interactive Dashboard:**
    *   Score Distribution Histogram.
    *   Parameter Correlation Heatmap.
    *   Scatter Plot for comparing two selected parameters.
    *   Bar Chart comparing a chosen metric for the top N stocks.
*   **Data Views:** Option to view the original raw data and the processed/ranked data.

## Project Structure

```
stock_analyzer_app/
├── data/
│   └── steel_data.csv    # Input stock data
├── app.py                # Main Streamlit application script
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How to Run Locally

1.  **Clone the repository (or download the files).**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
5.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

## Deployment

This app is structured for deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  Push the project files (including `requirements.txt`, `app.py`, and the `data` directory) to a GitHub repository.
2.  Connect your GitHub account to Streamlit Cloud.
3.  Deploy the app by selecting the repository and the main script (`app.py`).

