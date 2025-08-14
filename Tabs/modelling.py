import os
import main 
import utils
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import streamlit as st
import geopandas as gpd
from zipfile import ZipFile
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_folium import st_folium
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_historic_averages(src_df, f_id, metric, client_name, years):
    historic_avarages_cache_dir = './historic_avarages_cache'
    historic_avarages_cache_path = f'{historic_avarages_cache_dir}/historic_avarages_cache.joblib'
    historic_avarages_cache_clp_path = f'{historic_avarages_cache_dir}/historic_avarages_cache_clp.joblib'

    # Ensure cache files exist
    os.makedirs(historic_avarages_cache_dir, exist_ok=True)
    if os.path.exists(historic_avarages_cache_path):
        historic_avarages_cache = joblib.load(historic_avarages_cache_path)
    else:
        historic_avarages_cache = {}
    if os.path.exists(historic_avarages_cache_clp_path):
        historic_avarages_cache_clp = joblib.load(historic_avarages_cache_clp_path)
    else:
        historic_avarages_cache_clp = {}

    if client_name not in historic_avarages_cache:
        historic_avarages_cache[client_name] = {}
    if metric not in historic_avarages_cache[client_name]:
        historic_avarages_cache[client_name][metric] = {}
    if client_name not in historic_avarages_cache_clp:
        historic_avarages_cache_clp[client_name] = {}
    if 'CLP' not in historic_avarages_cache_clp[client_name]:
        historic_avarages_cache_clp[client_name]['CLP'] = {}

    # Loop through years
    for year in years:
        if f_id not in historic_avarages_cache[client_name][metric]:
            historic_avarages_cache[client_name][metric][f_id] = {}
        if f_id not in historic_avarages_cache_clp[client_name]['CLP']:
            historic_avarages_cache_clp[client_name]['CLP'][f_id] = {}

        found_in_cache = year in historic_avarages_cache[client_name][metric][f_id] \
                         and len(historic_avarages_cache[client_name][metric][f_id][year]) > 0
        found_in_cache_clp = year in historic_avarages_cache_clp[client_name]['CLP'][f_id] \
                             and len(historic_avarages_cache_clp[client_name]['CLP'][f_id][year]) > 0

        if found_in_cache and found_in_cache_clp:
            st.info(f"Year {year} found in cache — skipping calculation")
            continue

        st.info(f"Calculating historic averages for {year}...")
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        # Fetch available dates
        historic_dates_for_year = get_and_cache_available_dates(src_df, f_id, year, start_date, end_date)
        historic_dates_for_year = sorted(
            [datetime.strptime(date, "%Y-%m-%d") for date in historic_dates_for_year]
        )
        historic_dates_for_year = [datetime.strftime(date, "%Y-%m-%d") for date in historic_dates_for_year]
        num_dates = len(historic_dates_for_year)

        historic_values = []
        historic_dates = []
        historic_values_clp = []

        # Threaded fetching function
        def fetch_data_for_date(current_date):
            current_df = main.get_cuarted_df_for_field(src_df, f_id, current_date, metric, client_name)
            current_df_clp = main.get_cuarted_df_for_field(src_df, f_id, current_date, 'CLP', client_name)
            current_avg = current_df[f"{metric}_{current_date}"].mean()
            current_avg_clp = current_df_clp[f"CLP_{current_date}"].mean()
            return current_date, current_avg, current_avg_clp

        progress_bar = st.progress(0)
        with st.spinner(f"Calculating Historic Averages for {year}..."):
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(fetch_data_for_date, historic_dates_for_year))
            for i, (date, avg, avg_clp) in enumerate(results):
                historic_values.append(avg)
                historic_dates.append(date)
                historic_values_clp.append(avg_clp)
                progress_bar.progress((i + 1) / num_dates)

        # Save results to cache
        historic_avarages_cache[client_name][metric][f_id][year] = {
            "historic_avarages": historic_values,
            "historic_avarages_dates": historic_dates
        }
        historic_avarages_cache_clp[client_name]['CLP'][f_id][year] = {
            "historic_avarages_clp": historic_values_clp
        }

        joblib.dump(historic_avarages_cache, historic_avarages_cache_path)
        joblib.dump(historic_avarages_cache_clp, historic_avarages_cache_clp_path)

        st.success(f"Historic averages for {year} saved in cache")

    return historic_avarages_cache, historic_avarages_cache_clp




def train_test_model(src_df, f_id, metric, client_name):

    # 1. TRAINING DATA: 2020–2023
    years_train = ['2020', '2021', '2022', '2023']

    calculate_historic_averages(src_df, f_id, metric, client_name, years)

    historic_dates_train = []
    historic_values_train = []

    for year in years_train:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        dates_for_year = get_and_cache_available_dates(src_df, f_id, year, start_date, end_date)
        dates_for_year.sort()
        for date in dates_for_year:
            current_df = main.get_cuarted_df_for_field(src_df, f_id, date, metric, client_name)
            avg_val = current_df[f'{metric}_{date}'].mean()
            historic_dates_train.append(date)
            historic_values_train.append(avg_val)

    # 2. TEST DATA: 2024
    year_test = '2024'
    historic_dates_test = []
    historic_values_test = []

    dates_2024 = get_and_cache_available_dates(src_df, f_id, year_test, f'{year_test}-01-01', f'{year_test}-12-31')
    dates_2024.sort()
    for date in dates_2024:
        current_df = main.get_cuarted_df_for_field(src_df, f_id, date, metric, client_name)
        avg_val = current_df[f'{metric}_{date}'].mean()
        historic_dates_test.append(date)
        historic_values_test.append(avg_val)

    # 3. Create DataFrames
    df_train = pd.DataFrame({'date': pd.to_datetime(historic_dates_train), 'value': historic_values_train})
    df_test = pd.DataFrame({'date': pd.to_datetime(historic_dates_test), 'value': historic_values_test})

    # 4. Create lag features
    def create_lag_features(data, lag=5):
        for i in range(1, lag + 1):
            data[f"lag_{i}"] = data["value"].shift(i)
        return data

    df_train = create_lag_features(df_train, lag=5).dropna()
    df_test = create_lag_features(df_test, lag=5).dropna()

    X_train = df_train.drop(columns=["date", "value"])
    y_train = df_train["value"]
    X_test = df_test.drop(columns=["date", "value"])
    y_test = df_test["value"]

    # 5. Train XGBoost
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4
    )
    model.fit(X_train, y_train)

    # 6. Predict
    y_pred = model.predict(X_test)

    # 7. Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.subheader("XGBoost Model Evaluation")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")

    # 8. Plot results
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_test["date"], y=y_test,
        mode="lines+markers",
        name="Actual"
    ))
    fig.add_trace(go.Scatter(
        x=df_test["date"], y=y_pred,
        mode="lines+markers",
        name="Predicted"
    ))

    fig.update_layout(
        title="Historic Averages Forecast (2024 Test Data)",
        xaxis_title="Date",
        yaxis_title="Value"
    )

    st.plotly_chart(fig, use_container_width=True)



