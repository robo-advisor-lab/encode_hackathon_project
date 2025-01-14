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

from diskcache import Cache

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()

cache = Cache('classifier_data')
trade_cache = Cache('Trade Cache')

# model_results = cache.get(f'')

# target_comp = cache.get(f'target_comp', None)
# model_resp = cache.get(f'model_response', None)

cache_dict = {key: cache[key] for key in cache}
print('cache:',cache_dict)

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")
FRED_API_KEY = os.getenv('FRED_API_KEY')

ACCOUNT_ADDRESS = os.getenv('ACCOUNT_ADDRESS')
PRIVATE_KEY = os.getenv('PRIVATE_KEY')
GATEWAY = os.getenv('ARBITRUM_GATEWAY')

factory = '0x1F98431c8aD98523631AE4a59f267346ea31F984'
router = '0x5E325eDA8064b456f4781070C0738d849c824258'
version = 3
diff_tolerance_threshold = 0.05 #percent for diff between new portfolio action and current portfolio

openai_client = OpenAI(api_key=OPENAI_API_KEY)

class CompositionItem(BaseModel):
    asset: str = Field(..., description="The cryptocurrency symbol, e.g., ETH, BTC.")
    weight: float = Field(..., description="Fraction of total portfolio in [0,1]. The sum of all weights should be 1.")
    reasoning: str = Field(..., description="A brief explanation for why this allocation is chosen.")

class Rebalance(BaseModel):
    target_composition: List[CompositionItem] = Field(
        default_factory=list,
        description="A list of target allocations for each asset."
    )

class PositionReasoning(BaseModel):
    """
    This model wraps the Rebalance object under 'rebalance'.
    """
    rebalance: Rebalance = Field(
        default_factory=Rebalance,
        description="All target composition details."
    )

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

def rebalance_portfolio(
    uniswap, 
    token_contracts, 
    token_decimals, 
    target_compositions, 
    account_address, 
):
    """
    Rebalances the portfolio by selling all tokens into WETH and then buying target allocations using WETH.

    Parameters:
    - uniswap: Initialized Uniswap class instance.
    - token_contracts: Dict of token addresses.
    - token_decimals: Dict of token decimals.
    - target_compositions: Dict of target compositions as fractions summing to 1.
    - account_address: ETH wallet address.
    - web3: Initialized Web3 instance.
    """

    # WETH address and checksum
    WETH_ADDRESS = '0x82aF49447D8a07e3bd95BD0d56f35241523fBab1'
    checksum_weth_address = Web3.to_checksum_address(WETH_ADDRESS)

    # Step 1: Convert Token Addresses to Checksum Format
    checksum_addresses = {token: Web3.to_checksum_address(address) for token, address in token_contracts.items()}

    # Step 2: Sell All Current Token Holdings into WETH
    for token, address in checksum_addresses.items():
        try:
            balance_wei = uniswap.get_token_balance(address)
            balance = balance_wei / 10**token_decimals[token]
            
            # Adjust the balance to avoid precision issues (round down to 6 decimal places)
            adjusted_balance = math.floor(balance * 10**8) / 10**8
            
            if adjusted_balance > 0:
                amount_to_sell = int(adjusted_balance * 10**token_decimals[token])
                print(f"Selling {adjusted_balance:.6f} {token} for WETH")
                uniswap.make_trade(
                    checksum_addresses[token],
                    checksum_weth_address,  # WETH as output token
                    amount_to_sell
                )
                wait_time = random.randint(15, 30)
                print(f"Waiting {wait_time} seconds before the next call...")
                time.sleep(wait_time)
        except Exception as e:
            print(f"Error selling {token}: {e}")

    # Step 3: Get Current WETH Balance
    weth_balance_wei = uniswap.get_token_balance(checksum_weth_address)
    weth_balance = weth_balance_wei / 10**18
    print(f"Total WETH balance after selling: {weth_balance:.6f} WETH")

    # Step 4: Buy Target Tokens Based on Target Compositions
    for token, target_weight in target_compositions.items():
        if target_weight > 0:
            weth_to_spend = weth_balance * target_weight
            
            # Adjust the WETH amount to avoid precision issues (round down to 6 decimal places)
            adjusted_weth_to_spend = math.floor(weth_to_spend * 10**8) / 10**8

            if adjusted_weth_to_spend <= 0:
                continue

            try:
                print(f"Buying {token} with {adjusted_weth_to_spend:.6f} WETH")

                uniswap.make_trade(
                    checksum_weth_address,        # WETH as input token
                    checksum_addresses[token],    # Target token
                    int(adjusted_weth_to_spend * 10**18),  # Convert WETH amount to wei
                    fee=3000                      # Assuming 0.3% fee pool for Uniswap V3
                )

                wait_time = random.randint(15, 30)
                print(f"Waiting {wait_time} seconds before the next call...")
                time.sleep(wait_time)

            except Exception as e:
                print(f"Error buying {token}: {e}")

    # Step 5: Log the Rebalancing Info
    final_weth_balance = uniswap.get_token_balance(checksum_weth_address) / 10**18
    print(f"Final WETH balance: {final_weth_balance:.6f} WETH")

    rebal_info = {
        "account_address": account_address,
        "initial_weth_balance": weth_balance,
        "final_weth_balance": final_weth_balance,
        "purchases": target_compositions,
    }

    # Save rebalancing info to CSV
    rebal_df = pd.DataFrame([rebal_info])
    rebal_df.to_csv('data/live_rebal_results.csv', index=False)
    print("Rebalancing info saved to 'data/live_rebal_results.csv'.")

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

three_month_tbill_historical_api = "https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json"

global next_run_time

async def run_model():
    global next_run_time
    print(f'next_run_time: {next_run_time}')

    # import pdb; pdb.set_trace()

    network = 'arbitrum'

    model = f'{network}_classifier'

    params = cache.get(f'{model} params')
    classifier_data = cache.get(f'{model}_portfolio')
    original_prices_df = cache.get(f'{model}_prices')
    days = params['days']

    chain = params['network']

    w3, gateway = network_def(chain)

    account = Account.from_key(PRIVATE_KEY)
    w3.eth.default_account = account.address

    try: 
        three_month_tbill = fetch_and_process_tbill_data(api_url=three_month_tbill_historical_api, api_key=FRED_API_KEY,
                                                        data_key="observations",
                                                        date_column="date", 
                                                        value_column="value")
        three_month_tbill['decimal'] = three_month_tbill['value'] / 100
        current_risk_free = three_month_tbill['decimal'].iloc[-1]
        print(f"3-month T-bill data fetched: {three_month_tbill.tail()}")
    except Exception as e:
        print(f"Error in fetching tbill data: {e}")
    
    network = chain

    filtered_assets = classifier_data['symbol'].unique()

    data_start_date = dt.datetime.now(dt.timezone.utc) - timedelta(days=days)
    data_start_date = data_start_date.strftime('%Y-%m-%d %H:00:00')

    today_utc = dt.datetime.now(dt.timezone.utc) 
    formatted_today_utc = today_utc.strftime('%Y-%m-%d %H:00:00')

    data_version = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H-00-00')
    data_version_comp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

    interest_rate_dict, cpi_data_df_clean_prepared = macro_main(formatted_today_utc, api=False)

    start_date = str(data_start_date)
    end_date = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:00:00') 

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

    prices_df.columns = [col.replace('_Price', '') for col in prices_df.columns]
    prices_df.set_index('hour', inplace=True)
    # import pdb; pdb.set_trace()
    prices_returns = prices_df.pct_change().dropna()

    metrics_query = latest_portfolio_metrics(classifier_data['token_address'], network, days, dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d'))
    latest_metrics_df = flipside_api_results(metrics_query, flipside_api_key)

    three_month_tbill['hourly_rate'] = (1 + three_month_tbill['decimal']) ** (1 / 8760) - 1
    three_month_tbill = prepare_data_for_simulation(three_month_tbill[three_month_tbill.index>'2024-01-01'], data_start_date, end_date)

    three_month_tbill.set_index('hour',inplace=True)

    combined_data = prices_returns.merge(three_month_tbill[['hourly_rate']], left_index=True,right_index=True,how='left')
    asset_columns = prices_returns.columns  # List of asset symbols, e.g., ['LAVA', 'MAGIC', ...]
    excess_returns = combined_data[asset_columns].subtract(combined_data['hourly_rate'], axis=0)

    std_dev = excess_returns.std()

    # Mean excess return for each asset
    mean_excess_return = excess_returns.mean()

    # Hourly Sharpe ratio for each asset
    sharpe_ratio = mean_excess_return / std_dev

    # import pdb; pdb.set_trace()

    print("Hourly Sharpe Ratios for each asset:")
    print(sharpe_ratio)

    mean_excess_return_df = mean_excess_return.to_frame('Excess Returns')
    sharpe_ratios_df = sharpe_ratio.to_frame('Sharpe Ratios')

    # Map sharpe_ratios_df values to latest_metrics_df
    latest_metrics_df['sharpe_ratio'] = latest_metrics_df['symbol'].map(sharpe_ratios_df['Sharpe Ratios'])

    # Map mean_excess_return_df values to latest_metrics_df
    latest_metrics_df['excess_return'] = latest_metrics_df['symbol'].map(mean_excess_return_df['Excess Returns'])

    portfolio = classifier_data[['symbol','token_address']]

    TOKEN_CONTRACTS = {
        row['symbol']: row['token_address'] for _, row in portfolio.iterrows()
    }

    TOKEN_DECIMALS = get_token_decimals(TOKEN_CONTRACTS,w3)

    latest_prices = {
        token: float(prices_df[f"{token}"].iloc[-1])
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}" in prices_df.columns
    }

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

    # comp_dict["date"] = formatted_today_utc

    # update_historical_data(comp_dict)

    portfolio_dict = {
        "Portfolio Value": available_balance,
        "date": formatted_today_utc
    }

    # update_portfolio_data(portfolio_dict)

    identity_map = {
        "risk_level": "Moderate risk, willing to risk some money for the right investments but not chasing every new opportunity.",
        "token_preferences": "Only tokens appearing in the classifier portfolio.",
        "mission_statement": "Accumulate as much WETH as possible given the available funds.",
    }

    latest_overnight_interbank_rate = interest_rate_dict['Overnight Interbank Rate']

    macro_dict = {}

    for key in interest_rate_dict.keys():
        macro_dict[key] = f'{interest_rate_dict[key]["value"].iloc[0] * 100}%'
    print(macro_dict)

    macro_dict['cpi'] = f'{cpi_data_df_clean_prepared["value"].iloc[0] * 100}%'

    macro_summary = ", ".join(
        f"{k}: {macro_dict[k]}"
        for k in macro_dict
    )

    data_summary_list = []
    for symbol in latest_metrics_df['symbol'].unique():
        row = latest_metrics_df[latest_metrics_df['symbol'] == symbol].iloc[0]
        text = (
            f"{symbol} stats as of {pd.to_datetime(row['latest_hour']).strftime('%Y-%m-%d %H:%M')}: "
            f"Excess Return: {row['excess_return']}, "
            f"sharpe ratio: {row['sharpe_ratio']} "
            f"Latest Price: ${row['latest_price']}, "
            f"{days} day Return: {row['token_return']*100}% "
            f"7 day rolling average: ${row['rolling_7d_avg']} "
            f"30 day rolling average: ${row['rolling_30d_avg']} "
            f"{days} day average volume: ${row['avg_vol']} "
            f"{days} day total volume: ${row['sum_vol']} "
        )
        data_summary_list.append(text)

    # Join the list into a single string with newlines
    formatted_data_summary = "\n".join(data_summary_list)

    assets = classifier_data['symbol'].unique()

    user_message = (
        "# Instructions:\n"
        "Here are some details about my trading portfolio. Please help me make decisions to rebalance it based on the provided data.\n"
        "# Personality\n"
        f"{identity_map.get('chat_personality')}\n"
        "# Risk Level\n"
        f"{identity_map.get('risk_level')}\n"
        "This represents the total $USD value of the account, including positions, margin, and available funds.\n"
        "# Available Balance\n"
        f"{available_balance}\n"
        "Portions of this 'available_balance' can be used for placing new orders or modifying existing positions.\n"
        "Always leave a fraction of the total 'available_balance' as a safety buffer for unforeseen volatility.\n"
        "The 'available_balance' is shared by all positions, so it is important to keep track of the available value and adjust your position sizes accordingly.\n"
        "# Open Positions\n"
        f"{comp_dict}\n"
        "# Here is the most recent information I want to base my decisions on:\n"
        f"{formatted_data_summary}\n"
        "# Here is additional macro economic data for context\n"
        f"{macro_summary}\n"
        "\n"
        "# Please provide a JSON dictionary matching the following format:\n"
        "```json\n"
        "{\n"
        "  \"target_composition\": [\n"
        "    {\n"
        "      \"asset\": \"ETH\",\n"
        "      \"weight\": 0.50,\n"
        "      \"reasoning\": \"ETH has a strong Sharpe Ratio and good recent performance.\"\n"
        "    },\n"
        "    {\n"
        "      \"asset\": \"BTC\",\n"
        "      \"weight\": 0.25,\n"
        "      \"reasoning\": \"BTC remains a large-cap store of value with moderate performance.\"\n"
        "    },\n"
        "    {\n"
        "      \"asset\": \"ADA\",\n"
        "      \"weight\": 0.25,\n"
        "      \"reasoning\": \"ADA presents an opportunity for growth given recent metrics.\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "\n"
        "### Requirements:\n"
        "- The sum of all `weight` values **must equal 1**.\n"
        "- Provide a brief `reasoning` for each asset.\n"
        "- Provide **only** the JSON response. **No** extra text or explanation outside the JSON.\n"
        "- Each `weight` must be a decimal between 0 and 1.\n"
    )

    system_prompt = (
        "# Instructions:\n"
        "Act as a knowledgeable cryptocurrency assistant helping users manage and optimize their trading portfolio.\n"
        "Users understand that trading cryptocurrency is inherently risky and seek to make informed, strategic decisions to maximize their returns.\n"
        "You will be provided with the following data:\n"
        "- `available_balance`: Represents the user's total available USD value for making new trades.\n"
        "- `risk_level`: Indicates the user's risk tolerance (e.g., Low, Moderate, High).\n"
        "- `current_positions`: A dictionary mapping cryptocurrency symbols to their current USD value holdings (e.g., {'ETH': 1500, 'BTC': 1000}).\n"
        "- `market_data`: Recent performance metrics for each asset, including Sharpe Ratio, Excess Return, Latest Price, and 60-Day Return (e.g., {'ETH': {'sharpe_ratio': 1.5, 'excess_return': 0.05, 'latest_price': 3000, '60d_return': 20.0}, ...}).\n"
        "\n"
        "Your primary objective is to provide a rebalance strategy that aims to maximize the portfolio's returns while adhering to the following guidelines:\n"
        "\n"
        "## **Risk Management:**\n"
        "- Ensure that no individual asset exceeds the `available_balance` if you choose to buy or hold more of it.\n"
        "- Maintain a safety buffer within the `available_balance` to handle market volatility and unforeseen events.\n"
        "- Respect the user's `risk_level` by adjusting allocations accordingly.\n"
        "- No asset may be more than 80% of the portfolio.\n"
        "- Avoid allocations below 0.1 USD in absolute terms, to ensure meaningful positions.\n"
        "- There should only be one stablecoin per target composition.  For example, if it is best to include 50% in a stablecoin, choose just one stablecoin."
        "\n"
        "## **Performance Optimization:**\n"
        "- Prioritize assets with higher Sharpe Ratios and Excess Returns to maximize risk-adjusted returns.\n"
        "- Consider reallocating from underperforming assets to those with strong performance metrics.\n"
        "- Ensure diversification to spread risk while seeking high-return opportunities.\n"
        "\n"
        "## **Target Composition:**\n"
        "Instead of listing positions to open, modify, or maintain, **provide a single portfolio composition** where each asset has a `weight` (fraction of total portfolio) and a short `reasoning`.\n"
        "- **Asset**: The cryptocurrency symbol (e.g., ETH, BTC).\n"
        "- **Weight**: A decimal in [0, 1], where the sum of all weights must equal 1.0.\n"
        "- **Reasoning**: A concise explanation for why the asset is allocated that weight.\n"
        "\n"
        "**Remember, no asset may be more than 80% (0.8 composition) of the portfolio.**\n"
        "## **Output Format:**\n"
        "Provide a JSON dictionary with the following structure:\n"
        "```json\n"
        "{\n"
        "  \"target_composition\": [\n"
        "    {\n"
        "      \"asset\": \"ETH\",\n"
        "      \"weight\": 0.50,\n"
        "      \"reasoning\": \"ETH has a strong Sharpe Ratio and robust recent performance.\"\n"
        "    },\n"
        "    {\n"
        "      \"asset\": \"BTC\",\n"
        "      \"weight\": 0.30,\n"
        "      \"reasoning\": \"BTC remains a large-cap with moderate performance.\"\n"
        "    },\n"
        "    {\n"
        "      \"asset\": \"SOL\",\n"
        "      \"weight\": 0.20,\n"
        "      \"reasoning\": \"SOL presents growth potential given recent metrics.\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "\n"
        "### Requirements:\n"
        "- All `weight` values **must sum to 1.0**.\n"
        "- Provide a brief `reasoning` for each asset.\n"
        "- **No additional fields** other than `target_composition`.\n"
        "- Provide **only** the JSON response. **No** extra text or explanation outside the JSON.\n"
    )

    model = "gpt-4o-2024-08-06"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    completion = openai_client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=PositionReasoning,
    )

    try:
        # Access the content of the first choice
        response_content = completion.choices[0].message.content

        # Parse the JSON string into a Python dictionary
        result = json.loads(response_content)

        # Access the rebalance dictionary
        rebalance_dict = result.get("rebalance", {})

        print("Rebalance Strategy:", rebalance_dict)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        print("Raw response:", completion.choices[0].message.content)
    
    uniswap = Uniswap(address=ACCOUNT_ADDRESS, private_key=PRIVATE_KEY, version=version, provider=gateway,router_contract_addr=router,factory_contract_addr=factory)
    rebalance_list = rebalance_dict["target_composition"]

    new_compositions = {
        token: next(
            (item["weight"] for item in rebalance_list if item["asset"] == token),
            0.0  # default if not found
        )
        for token in TOKEN_CONTRACTS
    }

    print('new compositions:',new_compositions)

    if composition_difference_exceeds_threshold(comp_dict, new_compositions, diff_tolerance_threshold):
        print("Composition difference exceeds the threshold. Triggering immediate rebalance.")
        rebalance_portfolio(
            uniswap, 
            TOKEN_CONTRACTS, 
            TOKEN_DECIMALS, 
            new_compositions, 
            ACCOUNT_ADDRESS
        )
    
    model_results = {
        'prior_comp':comp_dict,
        'target_comp':new_compositions,
        'model_response': result,
        'last_run (UTC)': formatted_today_utc,
        'next_run (UTC)': next_run_time,
        'user_message': user_message,
        'system_prompt': system_prompt
    }
    
    trade_cache.set(f'model_results', model_results)

def create_app():
    app = Flask(__name__)
    global next_run_time

    def fetch_and_cache_data():
        with app.app_context():
            logger.info("Scheduled task running.")
            print("Scheduled task running.")
            asyncio.run(run_model())

    scheduler.add_job(
        fetch_and_cache_data,
        trigger='interval',   # Use an interval trigger
        days=7,               # Run every 7 days
        id='data_fetch_job',  # Unique identifier for this job
        replace_existing=True # Replace existing job if it exists
    )

    job = scheduler.get_job('data_fetch_job')
    if job is not None:
        # Calculate next run time using the trigger
        now = dt.datetime.now(dt.timezone.utc)
        next_run_time = job.trigger.get_next_fire_time(None, now)
        print(f"Next run time: {next_run_time}")
    else:
        print("Job not found.")

    fetch_and_cache_data()

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        print('Clearing the cache...')
        trade_cache.clear()
        return jsonify({"status": "Cache cleared successfully"})

    @app.route('/run-model')
    async def run_model_endpoint():
        # Optionally reuse run_model_logic here if needed
        await run_model()
        return jsonify({"status": "Model run complete"})

    @app.route('/cached-data')
    def get_cached_data():
        cached_data = trade_cache.get('model_results')
        if cached_data:
            return jsonify(cached_data)
        else:
            return jsonify({"error": "No cached data available"}), 404
        
    return app

if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app = create_app()
    print('Starting Flask app...')
    scheduler.start()
    logger.info("Scheduler started.")
    app.run(debug=True, use_reloader=False, port=5012)
    # Since app.run() is blocking, the following line will not execute until the app stops:
    logger.info("Flask app has stopped.")
    print('Flask app has stopped.')


