import os
import main 
import utils
import joblib
import numpy as np
import pandas as pd
import streamlit as st
   
# Get the app config
CONFIG = utils.parse_app_config()


def model(src_df, f_id, date, year, start_date, end_date, metric): 

	years_train = ['2020', '2021', '2022', '2023']
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

	# Prepare Data for Forecasting

	df_train = pd.DataFrame({'date': pd.to_datetime(historic_dates_train), 'value': historic_values_train}).set_index('date')
	df_test = pd.DataFrame({'date': pd.to_datetime(historic_dates_test), 'value': historic_values_test}).set_index('date')
