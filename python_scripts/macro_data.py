from openbb import obb
import pandas as pd
import os
import datetime as dt
from python_scripts.utils import prepare_data_for_simulation

#Helper functions for OpenBB data

def get_interest_rate_data():
    """
    Fetches interest rate data for the United States from the OpenBB API.
    Returns:
        dict: A dictionary containing interest rate data for immediate, short, and long durations.
    """
    data = {}
    durations = {'immediate': 'Overnight Interbank Rate', 'short': '3-Month Rate', 'long': '10-Year Rate'}

    for duration, description in durations.items():
        data[description] = obb.economy.interest_rates(
            provider='oecd', 
            country='united_states', 
            duration=duration,
            frequency='monthly'
        )

    return data

def get_cpi_data():
    """
    Fetches Consumer Price Index (CPI) data for the United States from the OpenBB API.
    Returns:
        DataFrame: A DataFrame containing CPI data for the United States.
    """
    cpi_data = obb.economy.cpi(
        country='united_states', 
        transform='yoy',  # Year-over-Year change by default
        provider='oecd'
    )
    return cpi_data

def clean_df(df,start_date):
    # We don't need all the data, and we also want to resample to hour to match the trade_script
    
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    
    if start_date == 'Max':
        start_date = df_copy.index.max()
        
    df_copy = df_copy[df_copy.index >= start_date]
    return df_copy

def main(current_date,api=True):
    # today_utc = dt.datetime.now(dt.timezone.utc) 
    # formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')
    # Fetch Interest Rate Data
    if api == True:
        print("Fetching Interest Rate Data...")
        interest_rate_data = get_interest_rate_data()
        # for key, value in interest_rate_data.items():
        #     print(f"{key}:\n{value}\n")
    
        # Fetch CPI Data
        print("Fetching CPI Data...")
        cpi_data = get_cpi_data()
        # print(f"CPI Data:\n{cpi_data}\n")
    
        interest_rate_dict = {}
    
        for key in interest_rate_data.keys():
            interest_rate_dict[key] = interest_rate_data[key].to_df()
        print(interest_rate_dict)
    
        cpi_data_df = cpi_data.to_df()
        cpi_data_df
    
        cpi_data_df_clean = clean_df(cpi_data_df, 'Max')
    
        for key in interest_rate_dict.keys():
            interest_rate_dict[key] = clean_df(interest_rate_dict[key], 'Max')
        print(interest_rate_dict)
    
        cpi_data_df_clean_prepared = prepare_data_for_simulation(cpi_data_df_clean, str(cpi_data_df_clean.index.min()), current_date)
    
        cpi_data_df_clean_prepared
    
        for key in interest_rate_dict.keys():
            interest_rate_dict[key] = prepare_data_for_simulation(interest_rate_dict[key], str(interest_rate_dict[key].index.min()), current_date)
        print(interest_rate_dict)
    
        interest_rate_dict['10-Year Rate']

        with open('../data/interest_rate_dict.json', 'w') as f:
            json.dump(interest_rate_dict, f)
            
        cpi_data_df_clean_prepared.to_csv('..data/cpi_data_df_clean_prepared.csv',index=False)
    else:
        with open('../data/interest_rate_dict.json', 'r') as f:
            interest_rate_dict = json.load(f)
            
        cpi_data_df_clean_prepared = pd.read_csv('..data/interest_rate_dict.csv')

    return interest_rate_dict, cpi_data_df_clean_prepared





