from sql_queries.queries import token_classifier,token_prices
from python_scripts.utils import flipside_api_results
import pandas as pd
from datetime import timedelta
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

current_directory = os.getcwd()
print(f"Current Directory: {current_directory}")

def token_classifier_portfolio(api_key, network='arbitrum', name=None,days=60,backtest_period=4380,
                               volume_threshold=1,use_cached_data=False,
                               prices_only=True,start_date=None):
    print(f'use_cached_data: {use_cached_data}')
    classifier_path = f'data/{name}_results.csv'
    portfolio_path = f'data/{name}_prices.csv'

    print(f'volume_threshold: {volume_threshold}')

    if use_cached_data:
        data_struct = {'classifier':pd.read_csv(classifier_path).dropna(),
                          'portfolio':pd.read_csv(portfolio_path).dropna()}

    else:
        if prices_only:
            tokens = pd.read_csv(classifier_path).dropna()

            tokens['original_latest_hour'] = tokens['latest_hour']
        
            tokens['latest_hour'] = pd.to_datetime(tokens['latest_hour'])
            latest_hour = tokens['latest_hour'].max()

            data_start = latest_hour - timedelta(hours=backtest_period) # 6 Months backtesting
            data_start_str = str(data_start.tz_convert('UTC'))
            data_start_str = str(data_start)
            data_start_str = data_start_str.split('+')[0]

            portfolio = tokens['token_address'].unique()
            
            if start_date is not None:
                data_start_str = start_date
                
            token_prices_query = token_prices(portfolio,network,data_start_str)

            tokens_df = flipside_api_results(token_prices_query,api_key)

            tokens.to_csv(classifier_path,index=False)
            tokens_df.to_csv(portfolio_path,index=False)

            data_struct = {'classifier':tokens,
                                'portfolio':tokens_df,
                                'data start':data_start_str,
                                'latest_hour':latest_hour,
                                'original_hour':tokens['original_latest_hour'],
                                }
        else:
            classifier = token_classifier(network,days,volume_threshold,backtest_period)
            tokens = flipside_api_results(classifier,api_key)
            
            tokens['latest_hour'] = pd.to_datetime(tokens['latest_hour'])
            latest_hour = tokens['latest_hour'].max() 

            data_start = latest_hour - timedelta(hours=backtest_period) # 6 Months backtesting
            data_start_str = str(data_start.tz_localize(None))

            portfolio = tokens['token_address'].unique()
            token_prices_query = token_prices(portfolio,network,data_start_str)

            tokens_df = flipside_api_results(token_prices_query,api_key)

            tokens.to_csv(classifier_path,index=False)
            tokens_df.to_csv(portfolio_path,index=False)

            data_struct = {'classifier':tokens,
                                'portfolio':tokens_df,
                                'data start':data_start_str,
                                'latest_hour':latest_hour
                                }
    

    return data_struct 

@st.cache_data(ttl=timedelta(days=7))
def fetch_and_process_tbill_data(api_url, data_key, date_column, value_column, date_format='datetime'):
    api_key = os.getenv("FRED_API_KEY")
    api_url_with_key = f"{api_url}&api_key={api_key}"

    response = requests.get(api_url_with_key)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[data_key])
        
        if date_format == 'datetime':
            df[date_column] = pd.to_datetime(df[date_column])
        
        df.set_index(date_column, inplace=True)
        df[value_column] = df[value_column].astype(float)
        return df
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame in case of failure


    