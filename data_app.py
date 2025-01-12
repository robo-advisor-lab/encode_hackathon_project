from dotenv import load_dotenv
import os
from web3 import Web3
from eth_account import Account
import pandas as pd
import time
import random
import math
from diskcache import Cache
import datetime as dt
from datetime import timezone, timedelta
from uniswap import Uniswap
import json
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
import asyncio
from plotly.utils import PlotlyJSONEncoder

from python_scripts.web3_utils import *
from python_scripts.apis import token_classifier_portfolio, flipside_api_results
from python_scripts.macro_data import main as macro_main
from sql_queries.queries import latest_portfolio_metrics
from python_scripts.utils import fetch_and_process_tbill_data, prepare_data_for_simulation, composition_difference_exceeds_threshold

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import time

import logging
import json

import requests
import plotly.express as px

from diskcache import Cache

load_dotenv()
scheduler = BackgroundScheduler()

flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
FRED_API_KEY = os.getenv('FRED_API_KEY')

ACCOUNT_ADDRESS = os.getenv('ACCOUNT_ADDRESS')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
GATEWAY = os.getenv('ARBITRUM_GATEWAY')

cache = Cache('classifier_data')
data_cache = Cache('data_cache')

historical_data = cache.get(f'historical_data', pd.DataFrame())
historical_port_values = cache.get(f'historical_port_values', pd.DataFrame())

from python_scripts.web3_utils import *
from python_scripts.apis import token_classifier_portfolio, flipside_api_results
from python_scripts.macro_data import main as macro_main
from sql_queries.queries import latest_portfolio_metrics
from python_scripts.utils import fetch_and_process_tbill_data, prepare_data_for_simulation, calculate_cumulative_return

def get_latest_model_data(base_url='http://127.0.0.1:5012'):
    # pass https://llm-rebalancer.onrender.com as base_url?
    url = f'{base_url}/cached-data'
    response = requests.get(url)
    data = response.json()
    print(data)
    return data

def data_cleaning(df,dropna=True,ffill=False):
    clean_df = clean_prices(df)
    clean_df = to_time(clean_df)
    if dropna == True:
        # clean_df = clean_df.dropna(axis=1, how='any')
        clean_df = clean_df.dropna()
    if ffill == True:
        clean_df = clean_df.resample('h').ffill().bfill()

    if '__row_index' in clean_df.columns:
        clean_df.drop(columns=['__row_index'], inplace=True)

    return clean_df

def to_time(df):
    time_cols = ['date','dt','hour','time','day','month','year','week','timestamp','date(utc)','block_timestamp']
    for col in df.columns:
        if col.lower() in time_cols and col.lower() != 'timestamp':
            df[col] = pd.to_datetime(df[col])
            df.set_index(col, inplace=True)
        elif col.lower() == 'timestamp':
            df[col] = pd.to_datetime(df[col], unit='ms')
            df.set_index(col, inplace=True)
    print(df.index)
    return df 

def clean_prices(prices_df):
    print('cleaning prices')
    # Pivot the dataframe
    prices_df = prices_df.drop_duplicates(subset=['hour', 'symbol'])
    prices_df_pivot = prices_df.pivot(
        index='hour',
        columns='symbol',
        values='price'
    )
    prices_df_pivot = prices_df_pivot.reset_index()

    # Rename the columns by combining 'symbol' with a suffix
    prices_df_pivot.columns = ['hour'] + [f'{col}_Price' for col in prices_df_pivot.columns[1:]]
    
    print(f'cleaned prices: {prices_df_pivot}')
    return prices_df_pivot

# %%
def network_def(chain):
        if chain == 'gnosis':
            primary_gateway = GNOSIS_GATEWAY  # Replace with your Infura URL
            backup_gateway = 'https://lb.nodies.app/v1/406d8dcc043f4cb3959ed7d6673d311a'  # Your backup gateway
        elif chain == 'arbitrum':
            primary_gateway = GATEWAY  # Replace with your Infura URL
            backup_gateway = GATEWAY
        elif chain == 'optimism':
            primary_gateway = OPTIMISM_GATEWAY  # Replace with your Infura URL
            backup_gateway = OPTIMISM_GATEWAY
        elif chain == 'ethereum':
            primary_gateway = ETHEREUM_GATEWAY  # Replace with your Infura URL
            backup_gateway = ETHEREUM_GATEWAY

        print(f'Gateway: {primary_gateway}')

        for gateway in [primary_gateway, backup_gateway]:
            w3 = Web3(Web3.HTTPProvider(gateway))
            if w3.is_connected():
                try:
                    latest_block = w3.eth.get_block('latest')['number']  # Only try this if connected
                    print(f"Connected to {chain} via {gateway}: {latest_block} block")
                    return w3, gateway
                except Exception as e:
                    print(f"Connected to {gateway} but failed to fetch latest block. Error: {e}")
            else:
                print(f"Failed to connect to {chain} via {gateway}. Trying next gateway...")

        raise ConnectionError(f"Failed to connect to {chain} network using both primary and backup gateways.")

# %%
def pull_data(function,path,model_name, api=False,start_date=None):
    print(f'function:{function},start_date:{start_date},path:{path},api:{api},model_name: {model_name}')

    if api:
        print(f'api True')
        # Parse dates into datetime format for consistency
        start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        
        # Use formatted date strings as needed in the dao_advisor_portfolio and lst_portfolio_prices functions
        prices = function(start_date.strftime('%Y-%m-%d %H:%M:%S'))
        
        prices_df = flipside_api_results(prices, flipside_api_key)

        prices_df.to_csv(path)
    else:
        print(f'api False')
        prices_df = pd.read_csv(path)

    dataset = {
        f'portfolio': prices_df
    }

    return dataset

# %%
def prices_data_func(network,
                     api_key,use_cached_data,name,days=None,
                     function=None,start_date=None,
                     backtest_period=None,filtered_assets=None):
    
    if start_date is None and backtest_period is None:
        raise KeyError("Provide either a start date or backtest_period")
    
    print(f"backtest days: {(pd.to_datetime(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')) - pd.to_datetime(start_date)).days}")
    
    if backtest_period is None:
        backtest_period = (pd.to_datetime(dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00')) - pd.to_datetime(start_date)).days * 24
        if backtest_period < 1:
            backtest_period = 1

    if function is None:

        data = token_classifier_portfolio(
            network=network,
            days=days,
            name=name,
            api_key = api_key,
            use_cached_data=use_cached_data,
            start_date = start_date,
            prices_only=True
        )

        prices_df = data_cleaning(data['portfolio'])
        prices_df
    else: 
        data = pull_data(function=function,start_date=start_date, path=f'data/{name}.csv', api=not use_cached_data,model_name=name)
        prices_df = data_cleaning(data['portfolio'])
        prices_df = prices_df[prices_df.index >= start_date].dropna()
        prices_df

    # prices_df.columns = prices_df.columns.str.replace('_Price','')
    filtered_assets_with_price = [f"{asset}_Price" for asset in filtered_assets]


    return data, prices_df[filtered_assets_with_price]

# %%
def update_historical_data(live_comp):
    global historical_data
    new_data = pd.DataFrame([live_comp])
    historical_data = pd.concat([historical_data, new_data]).reset_index(drop=True)
    historical_data.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'historical_data', historical_data)

def update_portfolio_data(values):
    global historical_port_values
    print(f'values: {values}')
    values = pd.DataFrame([values])
    historical_port_values = pd.concat([historical_port_values, values]).reset_index(drop=True)
    historical_port_values.drop_duplicates(subset='date', keep='last', inplace=True)
    cache.set(f'historical_port_values', historical_port_values)

def update_price_data(values):
    global oracle_prices

    # Ensure the 'hour' column exists by resetting index if necessary
    if isinstance(values.index, pd.DatetimeIndex):
        values = values.reset_index().rename(columns={'index': 'hour'})
    
    if 'hour' not in values.columns:
        raise ValueError("The provided DataFrame must have a 'hour' column.")

    # Concatenate the new values with the existing oracle_prices
    oracle_prices = pd.concat([oracle_prices, values]).drop_duplicates(subset='hour', keep='last').reset_index(drop=True)
    
    # Cache the updated oracle_prices
    cache.set(f'oracle_prices', oracle_prices)

    print(f'Updated oracle_prices:\n{oracle_prices}')

def update_model_actions(actions):
    global model_actions
    print(f'model actions before update: {model_actions}')
    new_data = pd.DataFrame(actions)
    print(f'new data: {new_data}')
    model_actions = pd.concat([model_actions, new_data]).reset_index(drop=True)
    model_actions.drop_duplicates(subset='Date', keep='last', inplace=True)
    cache.set(f'actions', model_actions)

async def get_data():
    model_data = get_latest_model_data()

    next_run = model_data['next_run (UTC)']
    user_message = model_data['user_message']
    system_prompt = model_data['system_prompt']

    # %%
    last_run = model_data['last_run (UTC)']
    last_run

    # %%
    model_resonse = model_data['model_response'].get('rebalance')
    model_resonse

    # %%
    latest_target = model_data['target_comp']
    latest_target

    # %%
    prior_comp = model_data['prior_comp']
    prior_comp

    # %%
    network = 'arbitrum'

    model = f'{network}_classifier'

    params = cache.get(f'{model} params')
    classifier_data = cache.get(f'{model}_portfolio')
    original_prices_df = cache.get(f'{model}_prices')
    days = params['days']

    # %%
    data_start_date = dt.datetime.now(dt.timezone.utc) - timedelta(days=days)
    data_start_date = data_start_date.strftime('%Y-%m-%d %H:00:00')

    today_utc = dt.datetime.now(dt.timezone.utc) 
    formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

    data_version = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H-00-00')
    data_version_comp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

    # %%
    start_date = str(data_start_date)
    end_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

    start_date

    # %%
    filtered_assets = classifier_data['symbol'].unique()

    # %%
    data, prices_df = prices_data_func(
                                network=network, 
                                name=model,
                                api_key=flipside_api_key,
                                use_cached_data=False,
                                function=None,
                                start_date=start_date,
                                filtered_assets=filtered_assets
                                )
            
    prices_df = prepare_data_for_simulation(prices_df, start_date, end_date)

    # %%
    chain = params['network']

    w3, gateway = network_def(chain)

    account = Account.from_key(PRIVATE_KEY)
    w3.eth.default_account = account.address

    # %%
    portfolio = classifier_data[['symbol','token_address']]

    TOKEN_CONTRACTS = {
        row['symbol']: row['token_address'] for _, row in portfolio.iterrows()
    }

    TOKEN_DECIMALS = get_token_decimals(TOKEN_CONTRACTS,w3)

    # %%
    prices_df.columns = [col.replace('_Price', '') for col in prices_df.columns]
    prices_df.set_index('hour',inplace=True)
    prices_returns = prices_df.pct_change().dropna()
    prices_returns

    # %%
    latest_prices = {
        token: float(prices_df[f"{token}"].iloc[-1])
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}" in prices_df.columns
    }

    # %%
    model_balances = get_balance(TOKEN_CONTRACTS,TOKEN_DECIMALS,ACCOUNT_ADDRESS,w3)

    model_balances  

    model_balances_usd = convert_to_usd(model_balances,latest_prices,TOKEN_CONTRACTS)
    model_balances_usd

    available_balance = sum(model_balances_usd.values())
    available_balance

    comp_dict = {
        f"{token}": balance_usd / available_balance
        for token, balance_usd in model_balances_usd.items()
    }

    print(f'comp_dict: {comp_dict}')

    comp_dict["date"] = formatted_today_utc

    update_historical_data(comp_dict)

    portfolio_dict = {
        "Portfolio Value": available_balance,
        "date": formatted_today_utc
    }

    print(f'portfolio value: {portfolio_dict}')

    update_portfolio_data(portfolio_dict)

    # %%
    comp_df = pd.DataFrame([comp_dict]).set_index('date')

    # %%
    comp_df = comp_df.reset_index().drop(columns='date')

    # %%
    data = comp_df.iloc[0]
    data = data[data != 0]  # Filter out assets with zero weight

    # Convert the series to a DataFrame for plotting
    df_for_pie = pd.DataFrame({'Asset': data.index, 'Weight': data.values})

    # %%
    fig1 = px.pie(
        df_for_pie,
        names='Asset',               # Column name for labels
        values='Weight',             # Column name for values
        title=f"Portfolio Composition as of {formatted_today_utc}",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    # fig1.show()


    # %%
    fig2 = px.line(
        historical_port_values,
        x='date',
        y='Portfolio Value',
        title=f"Portfolio Value ($) Time Series Through {formatted_today_utc}",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # fig2.show()

    # %%
    melted = historical_data.melt(id_vars=['date'], 
                                var_name='Asset', 
                                value_name='Weight')

    # %%
    # melted = melted[melted['Weight'] > 0]

    # %%
    fig3 = px.bar(
        melted,
        x='date',
        y='Weight',
        color='Asset',
        title='Portfolio Composition Over Time',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    # fig3.show()

    results = {
        'last run (UTC)': last_run,
        'next run (UTC)': pd.to_datetime(next_run).strftime('%Y-%m-%d %H:00:00'),
        'rebalance frequency (days)':days,
        'address':ACCOUNT_ADDRESS,
        'portfolio balance': f"${available_balance:,.2f}",
        'chain':chain,
        'user_message': user_message,
        'system_prompt': system_prompt
    }

    graph_json_1 = json.dumps(fig1, cls=PlotlyJSONEncoder)
    graph_json_2 = json.dumps(fig2, cls=PlotlyJSONEncoder)
    graph_json_3 = json.dumps(fig3, cls=PlotlyJSONEncoder)
    
    cached_data = {"results": results, "graph_1": graph_json_1, "graph_2": graph_json_2,"graph_3":graph_json_3}

    data_cache.set(f'latest_data', cached_data)

    return jsonify(cached_data)

def create_app():
    app = Flask(__name__)

    def fetch_and_cache_data():
        with app.app_context():
            print("Scheduled task running.")
            asyncio.run(get_data())

    scheduler.add_job(
        fetch_and_cache_data,
        trigger='interval',   # Use an interval trigger
        days=1,               # Run every day
        id='data_fetch_job',  # Unique identifier for this job
        replace_existing=True # Replace existing job if it exists
    )

    fetch_and_cache_data()

    @app.route('/')
    def home():
        # Retrieve the cached data
        cached_data = data_cache.get('latest_data')
        
        # If cached_data exists, extract graphs and results
        if cached_data:
            results = cached_data.get('results', {})
            graph_1 = cached_data.get('graph_1', '{}')
            graph_2 = cached_data.get('graph_2', '{}')
            graph_3 = cached_data.get('graph_3', '{}')
        else:
            results = {}
            graph_1 = graph_2 = graph_3 = '{}'
        
        # Render the template with the data
        return render_template(
            'data_index.html',  # your HTML file name
            results=results,
            graph_1=graph_1,
            graph_2=graph_2,
            graph_3=graph_3
        )

    @app.route('/get_data')
    async def run_model_endpoint():
        # Optionally reuse run_model_logic here if needed
        # await get_data()
        cached_data = data_cache.get('latest_data')
        return jsonify(cached_data)

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        print('Clearing the cache...')
        data_cache.clear()
        return jsonify({"status": "Cache cleared successfully"})
    
    @app.route('/cached-data')
    def get_cached_data():
        cached_data = data_cache.get('model_results')
        if cached_data:
            return jsonify(cached_data)
        else:
            return jsonify({"error": "No cached data available"}), 404
        
    return app

if __name__ == "__main__":
    app = create_app()
    print('Starting Flask app...')
    scheduler.start()
    app.run(debug=True, use_reloader=False, port=5013)
    # Since app.run() is blocking, the following line will not execute until the app stops:
    print('Flask app has stopped.')
