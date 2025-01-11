from openbb import obb
import pandas as pd
import os
import json
import datetime as dt
from python_scripts.utils import prepare_data_for_simulation

def get_interest_rate_data():
    """
    Fetches interest rate data for the United States from the OpenBB API.
    Returns:
        dict: A dictionary containing interest rate data for immediate, short, and long durations.
    """
    data = {}
    durations = {
        'immediate': 'Overnight Interbank Rate',
        'short': '3-Month Rate',
        'long': '10-Year Rate'
    }

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

def clean_df(df, start_date):
    """
    Trims the DataFrame to rows starting from `start_date`.
    """
    df_copy = df.copy()
    df_copy.index = pd.to_datetime(df_copy.index)
    
    if start_date == 'Max':
        start_date = df_copy.index.max()
        
    df_copy = df_copy[df_copy.index >= start_date]
    return df_copy

def prepare_data_for_simulation(df, start_date, end_date):
    """
    Placeholder for your data prep function.
    Assume it returns a DataFrame with the same columns, just an example.
    """
    # Implement your data processing logic here
    return df

def main(current_date, api=True):
    """
    Main function that fetches/loads data, prepares it, and stores/loads CSV files.
    """
    durations = {
        'immediate': 'Overnight Interbank Rate',
        'short': '3-Month Rate',
        'long': '10-Year Rate'
    }

    if api:
        print("Fetching Interest Rate Data...")
        interest_rate_data = get_interest_rate_data()
        print("Fetching CPI Data...")
        cpi_data = get_cpi_data()

        # Convert fetched data into DataFrames
        interest_rate_dict = {}
        for key, dataset in interest_rate_data.items():
            df = dataset.to_df()
            interest_rate_dict[key] = df

        cpi_data_df = cpi_data.to_df()
        cpi_data_df_clean = clean_df(cpi_data_df, 'Max')

        # Clean each DF in interest_rate_dict
        for key in interest_rate_dict.keys():
            interest_rate_dict[key] = clean_df(interest_rate_dict[key], 'Max')

        # Prepare data for simulation
        cpi_data_df_clean_prepared = prepare_data_for_simulation(
            cpi_data_df_clean,
            str(cpi_data_df_clean.index.min()),
            current_date
        )
        for key in interest_rate_dict.keys():
            interest_rate_dict[key] = prepare_data_for_simulation(
                interest_rate_dict[key],
                str(interest_rate_dict[key].index.min()),
                current_date
            )

        # -----------------------------------------
        # SAVE TO CSV
        # -----------------------------------------
        # 1) Save each interest rate DataFrame to its own CSV
        #    Convert "Overnight Interbank Rate" -> "overnight_interbank_rate.csv", etc.
        for key, df in interest_rate_dict.items():
            filename = key.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            csv_name = f"interest_rate_dict_{filename}.csv"
            df.to_csv(csv_name)

        # 2) Save CPI data
        cpi_data_df_clean_prepared.to_csv('cpi_data_df_clean_prepared.csv', index=False)

        return interest_rate_dict, cpi_data_df_clean_prepared

    else:
        print("Loading data from CSVs...")

        # -----------------------------------------
        # LOAD FROM CSV
        # -----------------------------------------
        # Rebuild interest_rate_dict from CSVs. We'll rely on the known keys in durations.
        interest_rate_dict = {}
        for description in durations.values():
            filename = description.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
            csv_name = f"interest_rate_dict_{filename}.csv"
            if os.path.exists(csv_name):
                df = pd.read_csv(csv_name)
                # If you need the date as index, do something like:
                # df['Date'] = pd.to_datetime(df['Date'])
                # df.set_index('Date', inplace=True)
                interest_rate_dict[description] = df
            else:
                print(f"Warning: {csv_name} not found.")
                interest_rate_dict[description] = pd.DataFrame()

        # Load CPI data
        if os.path.exists('cpi_data_df_clean_prepared.csv'):
            cpi_data_df_clean_prepared = pd.read_csv('cpi_data_df_clean_prepared.csv')
        else:
            print("Warning: cpi_data_df_clean_prepared.csv not found.")
            cpi_data_df_clean_prepared = pd.DataFrame()

        return interest_rate_dict, cpi_data_df_clean_prepared