import os
import main 
import utils
import joblib
import numpy as np
from . import modelling
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
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_absolute_error, mean_squared_error

@st.cache_data()
def get_and_cache_available_dates(_df, field_id, year, start_date, end_date):
    dates = main.get_available_dates_for_field(_df, field_id, year, start_date, end_date)
    print(f'Caching Dates for {field_id}')
    return dates


def app(metric):
    f_id = -1
    st.title(f"{metric} Analysis")

    # Get the app config
    CONFIG = utils.parse_app_config()

    # Get the GeoJSON file for the client
    data_path = CONFIG['Client']['DataPath'].get()
    src_df = gpd.read_file(data_path)
    client_name = CONFIG['Client']['Name'].get()

    #Convert LastUpdated to string
    src_df['LastUpdate'] = src_df['LastUpdate'].astype(str)
    
    # Create a choropleth map of the client Fields
    m = src_df.explore(
        column="Crop_Type",
        tooltip=["Field_Id", "Crop_Type"],
        popup=True,
        style_kwds=dict(color="black", fillOpacity=0.1), location=[54.55, -1.51], zoom_start=13)
    
    # Add Google Satellite as a base map  
    google_map = utils.basemaps['Google Satellite']
    google_map.add_to(m)

    # Display the map inside an expander
    with st.expander("Show Client Fields"):
        st_folium(m, key=f'Client Fields - {metric}') 

    
    st.markdown('---')
    st.header('Select Field')

    # Initlize field_name as None
    field_name = 'None'

    # Set field_name as Crop_Type and field_id as Field_Id
    field_names = src_df.Crop_Type.tolist()

    # Add None to the end of the list to be used as a default value
    field_names.append('None')

    # Display the dropdown menu
    field_name = st.selectbox(
        "Check Field (or click on the map)",
        field_names, index=len(field_names)-1,
        key=f'Select Field Dropdown Menu - {metric}',
        )
    
    # If a field is selected, display the field name and get the field_id
    if field_name != 'None':
        f_id = src_df[src_df.Crop_Type == field_name].Field_Id.values[0]
        f_id = int(f_id)
        st.write(f'You selected {field_name} (Field ID: {f_id})')
    else:
        st.write('Please Select A Field')



    st.markdown('---')
    st.header('Select Observation Date')

    # Initlize date as -1 and dates as an empty list
    dates = []
    date = -1

    # If dates and date are not in session state, set them to the default values, else get them from the session state
    if 'dates' not in st.session_state:
        st.session_state['dates'] = dates
    else:
        dates = st.session_state['dates']
    if 'date' not in st.session_state:
        st.session_state['date'] = date
    else:
        date = st.session_state['date']

    # If a field is selected, Get the dates with available data for that field
    if f_id != -1 :

        # Give the user the option to select year, start date and end date
        with st.expander('Select Year, Start Date and End Date'):
            # Get the year
            years = [f'20{i}' for i in range(20, 24)]
            year = st.selectbox('Select Year: ', years, index=0, key=f'Select Year Dropdown Menu - {metric}')

            # Set the min, max and default values for start and end dates
            min_val = f'{year}-01-01'
            max_val = f'{year}-12-31'
            default_val = f'{year}-07-01'
            min_val = datetime.strptime(min_val, '%Y-%m-%d')
            max_val = datetime.strptime(max_val, '%Y-%m-%d')
            default_val = datetime.strptime(f'{year}-07-01', '%Y-%m-%d')
            # Get the start and end dates
            start_date = st.date_input('Start Date', value=default_val, min_value=min_val, max_value=max_val, key=f'Start Date - {metric}')
            end_date = st.date_input('End Date', value=max_val, min_value=min_val, max_value=max_val, key=f'End Date - {metric}')


        # Get the dates with available data for that field when the user clicks the button
        get_dates_button = st.button(f'Get Dates for Field {field_name} (Field ID: {f_id}) in {year} (from {start_date} to {end_date})',
                                     key=f'Get Dates Button - {metric}',
                                     help='Click to get the dates with available data for the selected field',
                                     use_container_width=True, type='primary')
        if get_dates_button:
            dates = get_and_cache_available_dates(src_df, f_id, year, start_date, end_date)
            # Add None to the end of the list to be used as a default value
            dates.append(-1)
            #Add the dates to the session state
            st.session_state['dates'] = dates

        # Display the dropdown menu
        if len(dates) > 0:
            date = st.selectbox('Select Observation Date: ', dates, index=len(dates)-1, key=f'Select Date Dropdown Menu - {metric}')
            if date != -1:
                st.write('You selected:', date)
                #Add the date to the session state
                st.session_state['date'] = date
            else:
                st.write('Please Select A Date')
        else:
            st.info('No dates available for the selected field and dates range, select a different range or click the button to fetch the dates again')

    else:
        st.info('Please Select A Field')

    st.markdown('---')
    st.header('Show Field Data')

    # If a field and a date are selected, display the field data
    if (date != -1) and (f_id != -1):   

        # Get the field data at the selected date
        with st.spinner('Loading Field Data...'):
            # Get the metric data and cloud cover data for the selected field and date
            metric_data = main.get_cuarted_df_for_field(src_df, f_id, date, metric, client_name)
            cloud_cover_data = main.get_cuarted_df_for_field(src_df, f_id, date, 'CLP', client_name)
            
            #Merge the metric and cloud cover data on the geometry column
            field_data = metric_data.merge(cloud_cover_data, on='geometry')

        # Display the field data
        st.write(f'Field Data for {field_name} (Field ID: {f_id}) on {date}')
        st.write(field_data.head(2))

        #Get Avarage Cloud Cover
        avg_clp = field_data[f'CLP_{date}'].mean() *100

        # If the avarage cloud cover is greater than 80%, display a warning message
        if avg_clp > 80:
            st.warning(f'⚠️ The Avarage Cloud Cover is {avg_clp}%')
            st.info('Please Select A Different Date')

        ## Generate the field data Map ##

        #Title, Colormap and Legend

        title = f'{metric} for selected field {field_name} (Field ID: {f_id}) in {date}'

        cmap_map = {
        "NDVI": "Greens",
        "NDWI": "Blues",
        "EVI": "PuBu",
        }


        cmap = cmap_map.get(metric, "RdYlGn")

        # Create a map of the field data
        metric_data_map  = metric_data.explore(
            column=f'{metric}_{date}',
            cmap=cmap,
            legend=True,
            vmin=0,
            vmax=1,
            marker_type='circle', marker_kwds={'radius':5.3, 'fill':True})
        
        # Add Google Satellite as a base map
        google_map = utils.basemaps['Google Satellite']
        google_map.add_to(metric_data_map)

        # Display the map
        st_folium(metric_data_map, width = 725, key=f'Field Data Map - {metric}')


        #Dwonload Links

        # If the field data is not empty, display the download links
        if len(field_data) > 0:
            # Create two columns for the download links
            download_as_shp_col, download_as_tiff_col = st.columns(2)

            # Create a shapefile of the field data and add a download link
            with download_as_shp_col:

                #Set the shapefile name and path based on the field id, metric and date
                extension = 'shp'
                shapefilename = f"{f_id}_{metric}_{date}.{extension}"
                path = f'./shapefiles/{f_id}/{metric}/{extension}'

                # Create the target directory if it doesn't exist
                os.makedirs(path, exist_ok=True)
                
                # Save the field data as a shapefile
                field_data.to_file(f'{path}/{shapefilename}')

                # Create a zip file of the shapefile
                files = []
                for i in os.listdir(path):
                    if os.path.isfile(os.path.join(path,i)):
                        if i[0:len(shapefilename)] == shapefilename:
                            files.append(os.path.join(path,i))
                zipFileName = f'{path}/{f_id}_{metric}_{date}.zip'
                zipObj = ZipFile(zipFileName, 'w')
                for file in files:
                    zipObj.write(file)
                zipObj.close()

                # Add a download link for the zip file
                with open(zipFileName, 'rb') as f:
                    st.download_button('Download as ShapeFile', f,file_name=zipFileName)

            # Get the tiff file path and create a download link
            with download_as_tiff_col:
                #get the tiff file path
                tiff_path = utils.get_masked_location_img_path(client_name, metric, date, f_id)
                # Add a download link for the tiff file
                donwnload_filename = f'{metric}_{f_id}_{date}.tiff'
                with open(tiff_path, 'rb') as f:
                    st.download_button('Download as Tiff File', f,file_name=donwnload_filename)

    else:
        st.info('Please Select A Field and A Date')


    # --- XGBoost model ---
    if st.button("Run XGBoost Model"):
     with st.spinner("Training model, please wait..."):
        modelling.train_test_model(src_df, f_id, metric, client_name)
    st.success("Model finished running!")


    st.markdown('---')
    st.header('Show Historic Averages')

    # If a field is selected, display the historic averages
    if f_id != -1:

        # Let the user select the year, start date and end date
        with st.expander('Select Year, Start Date and End Date'):
            years = [f'20{i}' for i in range(20, 24)]
            year = st.selectbox('Select Year: ', years, index=0, key=f'Select Year Dropdown Menu - {metric}- Historic Averages')
            start_date = f'{year}-01-01'
            end_date = f'{year}-12-31'

        # Get the dates for historic averages
        historic_avarages_dates_for_field = get_and_cache_available_dates(src_df, f_id, year, start_date, end_date)
        historic_avarages_dates_for_field = sorted(
            [datetime.strptime(date, '%Y-%m-%d') for date in historic_avarages_dates_for_field]
        )
        historic_avarages_dates_for_field = [datetime.strftime(date, '%Y-%m-%d') for date in historic_avarages_dates_for_field]
        num_historic_dates = len(historic_avarages_dates_for_field)

        st.write(f' Found {num_historic_dates} dates for field {f_id} in {year} (from {start_date} to {end_date})')

        display_historic_avgs_button = st.button(
            f'Display Historic Averages for Field {field_name} (Field ID: {f_id}) in {year} (from {start_date} to {end_date})',
            key=f'Display Historic Averages Button - {metric}',
            help='Click to display the historic averages for the selected field',
            use_container_width=True, type='primary'
        )

        if display_historic_avgs_button:
            historic_avarages_cache_dir = './historic_avarages_cache'
            historic_avarages_cache_path = f'{historic_avarages_cache_dir}/historic_avarages_cache.joblib'
            historic_avarages_cache_clp_path = f'{historic_avarages_cache_dir}/historic_avarages_cache_clp.joblib'

            if os.path.exists(historic_avarages_cache_path):
                historic_avarages_cache = joblib.load(historic_avarages_cache_path)
            else:
                os.makedirs(historic_avarages_cache_dir, exist_ok=True)
                joblib.dump({}, historic_avarages_cache_path)
                historic_avarages_cache = {}

            if os.path.exists(historic_avarages_cache_clp_path):
                historic_avarages_cache_clp = joblib.load(historic_avarages_cache_clp_path)
            else:
                os.makedirs(historic_avarages_cache_dir, exist_ok=True)
                joblib.dump({}, historic_avarages_cache_clp_path)
                historic_avarages_cache_clp = {}

            client_name = CONFIG['Client']['Name'].get()
            found_in_cache = False
            if client_name not in historic_avarages_cache:
                historic_avarages_cache[client_name] = {}
            if metric not in historic_avarages_cache[client_name]:
                historic_avarages_cache[client_name][metric] = {}
            if f_id not in historic_avarages_cache[client_name][metric]:
                historic_avarages_cache[client_name][metric][f_id] = {}
            if year not in historic_avarages_cache[client_name][metric][f_id]:
                historic_avarages_cache[client_name][metric][f_id][year] = {}
            if len(historic_avarages_cache[client_name][metric][f_id][year]) > 0:
                found_in_cache = True

            found_in_cache_clp = False
            if client_name not in historic_avarages_cache_clp:
                historic_avarages_cache_clp[client_name] = {}
            if 'CLP' not in historic_avarages_cache_clp[client_name]:
                historic_avarages_cache_clp[client_name]['CLP'] = {}
            if f_id not in historic_avarages_cache_clp[client_name]['CLP']:
                historic_avarages_cache_clp[client_name]['CLP'][f_id] = {}
            if year not in historic_avarages_cache_clp[client_name]['CLP'][f_id]:
                historic_avarages_cache_clp[client_name]['CLP'][f_id][year] = {}
            if len(historic_avarages_cache_clp[client_name]['CLP'][f_id][year]) > 0:
                found_in_cache_clp = True

            if found_in_cache and found_in_cache_clp:
                st.info('Found Historic Averages in Cache')
                historic_avarages = historic_avarages_cache[client_name][metric][f_id][year]['historic_avarages']
                historic_avarages_dates = historic_avarages_cache[client_name][metric][f_id][year]['historic_avarages_dates']
                historic_avarages_clp = historic_avarages_cache_clp[client_name]['CLP'][f_id][year]['historic_avarages_clp']
            else:
                st.info('Calculating Historic Averages...')

                historic_avarages = []
                historic_avarages_dates = []
                historic_avarages_clp = []

                def fetch_data_for_date(current_date):
                    current_df = main.get_cuarted_df_for_field(src_df, f_id, current_date, metric, client_name)
                    current_df_clp = main.get_cuarted_df_for_field(src_df, f_id, current_date, 'CLP', client_name)
                    current_avg = current_df[f'{metric}_{current_date}'].mean()
                    current_avg_clp = current_df_clp[f'CLP_{current_date}'].mean()
                    return current_date, current_avg, current_avg_clp

                dates_for_field_bar = st.progress(0)
                with st.spinner('Calculating Historic Averages...'):
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        results = list(executor.map(fetch_data_for_date, historic_avarages_dates_for_field))
                    for i, (date, avg, avg_clp) in enumerate(results):
                        historic_avarages.append(avg)
                        historic_avarages_dates.append(date)
                        historic_avarages_clp.append(avg_clp)
                        dates_for_field_bar.progress((i + 1) / num_historic_dates)

                historic_avarages_cache[client_name][metric][f_id][year]['historic_avarages'] = historic_avarages
                historic_avarages_cache[client_name][metric][f_id][year]['historic_avarages_dates'] = historic_avarages_dates
                historic_avarages_cache_clp[client_name]['CLP'][f_id][year]['historic_avarages_clp'] = historic_avarages_clp

                joblib.dump(historic_avarages_cache, historic_avarages_cache_path)
                joblib.dump(historic_avarages_cache_clp, historic_avarages_cache_clp_path)
                st.info('Historic Averages Saved in Cache')
                st.write(f'Cache Path: {historic_avarages_cache_path}')
                st.write(f'Cache CLP Path: {historic_avarages_cache_clp_path}')

            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=historic_avarages_dates, y=historic_avarages, name=f'{metric} Historic Averages'), secondary_y=False)
            fig.add_trace(go.Scatter(x=historic_avarages_dates, y=historic_avarages_clp, name='Cloud Cover'), secondary_y=True)
            fig.update_layout(title_text=f'{metric} Historic Averages for {field_name} (Field ID: {f_id}) in {year}')
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text=f'{metric} Historic Averages', secondary_y=False)
            fig.update_yaxes(title_text='Cloud Cover', secondary_y=True)
            st.plotly_chart(fig)


    else:
        st.info('Please Select A Field')

        
        