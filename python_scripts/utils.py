import pandas as pd
import random
import numpy as np
import requests
import os
import datetime as dt
from dotenv import load_dotenv
import json
import time

load_dotenv()
flipside_api_key = os.getenv("FLIPSIDE_API_KEY")

def flipside_api_results(query, api_key, attempts=10, delay=30):
    import requests
    import time
    import pandas as pd

    url = "https://api-v2.flipsidecrypto.xyz/json-rpc"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key
    }

    # Step 1: Create the query
    payload = {
        "jsonrpc": "2.0",
        "method": "createQueryRun",
        "params": [
            {
                "resultTTLHours": 1,
                "maxAgeMinutes": 0,
                "sql": query,
                "tags": {"source": "python-script", "env": "production"},
                "dataSource": "snowflake-default",
                "dataProvider": "flipside"
            }
        ],
        "id": 1
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Query creation failed. Status: {response.status_code}, Response: {response.text}")
        raise Exception("Failed to create query.")

    try:
        response_data = response.json()
    except json.JSONDecodeError as e:
        print(f"Error decoding query creation response: {e}. Response text: {response.text}")
        raise

    query_run_id = response_data.get('result', {}).get('queryRun', {}).get('id')
    if not query_run_id:
        print(f"Query creation response: {response_data}")
        raise KeyError("Failed to retrieve query run ID.")

    # Step 2: Poll for query completion
    for attempt in range(attempts):
        status_payload = {
            "jsonrpc": "2.0",
            "method": "getQueryRunResults",
            "params": [
                {
                    "queryRunId": query_run_id,
                    "format": "json",
                    "page": {"number": 1, "size": 10000}
                }
            ],
            "id": 1
        }

        response = requests.post(url, headers=headers, json=status_payload)
        if response.status_code != 200:
            print(f"Polling error. Status: {response.status_code}, Response: {response.text}")
            time.sleep(delay)
            continue

        try:
            resp_json = response.json()
        except json.JSONDecodeError as e:
            print(f"Error decoding polling response: {e}. Response text: {response.text}")
            time.sleep(delay)
            continue

        if 'result' in resp_json and 'rows' in resp_json['result']:
            all_rows = []
            page_number = 1

            while True:
                status_payload["params"][0]["page"]["number"] = page_number
                response = requests.post(url, headers=headers, json=status_payload)
                resp_json = response.json()

                if 'result' in resp_json and 'rows' in resp_json['result']:
                    rows = resp_json['result']['rows']
                    if not rows:
                        break  # No more rows to fetch
                    all_rows.extend(rows)
                    page_number += 1
                else:
                    break

            return pd.DataFrame(all_rows)

        if 'error' in resp_json and 'not yet completed' in resp_json['error'].get('message', '').lower():
            print(f"Query not completed. Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Unexpected polling error: {resp_json}")
            raise Exception(f"Polling error: {resp_json}")

    raise TimeoutError(f"Query did not complete after {attempts} attempts.")

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


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

def calculate_cumulative_return(portfolio_values_df):
    """
    Calculate the cumulative return for each column in the portfolio.
    
    Parameters:
    portfolio_values_df (pd.DataFrame): DataFrame with columns representing portfolio values
    
    Returns:
    pd.DataFrame: DataFrame with cumulative returns for each column
    """
    cumulative_returns = {}

    for col in portfolio_values_df.columns:
        initial_value = portfolio_values_df[col].iloc[0]
        final_value = portfolio_values_df[col].iloc[-1]
        cumulative_return = (final_value / initial_value) - 1
        cumulative_returns[col] = cumulative_return

    # Convert the dictionary to a DataFrame
    cumulative_returns_df = pd.DataFrame(cumulative_returns, index=['Cumulative_Return'])
    
    return cumulative_returns_df

def calculate_cagr(history):
    print(f'cagr history: {history}')
    #print(f'cagr history {history}')
    initial_value = history.iloc[0]
    #print(f'cagr initial value {initial_value}')
    final_value = history.iloc[-1]
    #print(f'cagr final value {final_value}')
    number_of_hours = (history.index[-1] - history.index[0]).total_seconds() / 3600
    #print(f'cagr number of hours {number_of_hours}')
    number_of_years = number_of_hours / (365.25 * 24)  # Convert hours to years
    #print(f'cagr number of years {number_of_years}')

    if number_of_years == 0:
        return 0

    cagr = (final_value / initial_value) ** (1 / number_of_years) - 1
    cagr_percentage = cagr * 100
    return cagr

# def calculate_beta(data, columnx, columny):
#     X = data[f'{columnx}'].pct_change().dropna().values.reshape(-1, 1)  
#     Y = data[f'{columny}'].pct_change().dropna().values
  
#     # Check if X and Y are not empty
#     if X.shape[0] == 0 or Y.shape[0] == 0:
#         print("Input arrays X and Y must have at least one sample each.")
#         return 0

#     # Fit the linear regression model
#     model = LinearRegression()
#     model.fit(X, Y)

#     # Output the beta
#     beta = model.coef_[0]
#     return beta

def fetch_and_process_tbill_data(api_url, api_key, data_key, date_column, value_column, date_format='datetime'):
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
    
def set_global_seed(env, seed=20):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

def normalize_asset_returns(price_timeseries, start_date, end_date, normalize_value=1e4):
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter the data based on the start date and end date
    filtered_data = price_timeseries[(price_timeseries.index >= start_date) & (price_timeseries.index <= end_date)].copy()
    
    if filtered_data.empty:
        print("Filtered data is empty after applying the date filter.")
        return pd.DataFrame()

    # Initialize a dictionary to store normalized values for each asset (column)
    normalized_values = {col: [normalize_value] for col in filtered_data.columns}
    dates = [filtered_data.index[0]]  # Use the original start date for labeling

    # Loop through the filtered data and calculate normalized returns for each asset
    for i in range(1, len(filtered_data)):
        for col in filtered_data.columns:
            prev_price = filtered_data[col].iloc[i-1]
            current_price = filtered_data[col].iloc[i]
            
            # Calculate log returns
            price_ratio = current_price / prev_price
            log_return = np.log(price_ratio)

            # Update the normalized value using the exponential of the log return
            normalized_values[col].append(normalized_values[col][-1] * np.exp(log_return))

        dates.append(filtered_data.index[i])

    # Create a DataFrame with normalized values for each asset
    normalized_returns_df = pd.DataFrame(normalized_values, index=dates)

    return normalized_returns_df

def prepare_data_for_simulation(price_timeseries, start_date, end_date):
    """
    Ensure price_timeseries has entries for start_date and end_date.
    If not, fill in these dates using the last available data.
    """
    # Ensure 'ds' is in datetime format
    # price_timeseries['hour'] = pd.to_datetime(price_timeseries['hour'])
    
    # Set the index to 'ds' for easier manipulation
    # if price_timeseries.index.name != 'hour':
    #     price_timeseries.set_index('hour', inplace=True)

    print(f'price index: {price_timeseries.index}')

    price_timeseries.index = price_timeseries.index.tz_localize(None)
    
    # Check if start_date and end_date exist in the data
    required_dates = pd.date_range(start=start_date, end=end_date, freq='H')
    all_dates = price_timeseries.index.union(required_dates)
    
    # Reindex the dataframe to ensure all dates from start to end are present
    price_timeseries = price_timeseries.reindex(all_dates)
    
    # Forward fill to handle NaN values if any dates were missing
    price_timeseries.fillna(method='ffill', inplace=True)

    # Reset index if necessary or keep the datetime index based on further needs
    price_timeseries.reset_index(inplace=True, drop=False)
    price_timeseries.rename(columns={'index': 'hour'}, inplace=True)
    
    return price_timeseries

def pull_data(start_date,function,path,model_name, api=False):
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

def set_global_seed(env, seed):
    print(f'seed: {seed}')
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.action_space.seed(seed)

def to_percent(df):
    df_copy = df.copy()

    # Identify the asset names based on the column suffix (e.g., 'Comp' and 'Price')
    assets = [col.replace(' Comp', '') for col in df.columns if ' Comp' in col]
    print(assets)

    # Calculate the value (Comp * Price) for each asset
    for asset in assets:
        comp_col = f"{asset} Comp"
        price_col = f"{asset}_Price"
        value_col = f"{asset}_Value"
        df_copy[value_col] = df_copy[comp_col] * df_copy[price_col]

    # Calculate the total portfolio value as the sum of all asset values
    df_copy['Total_Value'] = df_copy[[f"{asset}_Value" for asset in assets]].sum(axis=1)

    # Calculate the percentage composition for each asset
    for asset in assets:
        percentage_col = f"{asset}_Percentage"
        df_copy[percentage_col] = (df_copy[f"{asset}_Value"] / df_copy['Total_Value']) * 100

    # Display the resulting DataFrame with percentage columns
    percentage_cols = [f"{asset}_Percentage" for asset in assets]
    print(f'percentage_cols: {percentage_cols}')
    print(df_copy[percentage_cols])

    return df_copy[percentage_cols]

def calculate_excess_return(portfolio_return_df, asset_returns_df):
    """
    Calculate the excess return of the portfolio over each individual asset.
    
    Parameters:
    portfolio_return_df (pd.DataFrame): DataFrame with portfolio cumulative return (one row, one column)
    asset_returns_df (pd.DataFrame): DataFrame with cumulative returns for individual assets
    
    Returns:
    pd.DataFrame: DataFrame with excess returns for each asset
    """
    # Extract the portfolio cumulative return value
    portfolio_cumulative_return = portfolio_return_df.iloc[0, 0]
    print(f'portfolio_cumulative_return: {portfolio_cumulative_return}')
    print(f'asset_returns_df: {asset_returns_df}')
    
    # Calculate the excess return by subtracting asset cumulative returns from portfolio return
    excess_returns = portfolio_cumulative_return - asset_returns_df
    
    # Return the excess returns as a DataFrame
    return excess_returns
    
def convert_to_usd(balances, prices,TOKEN_CONTRACTS):
    """
    Convert token balances to their USD equivalent using token prices.

    Parameters:
    - balances (dict): Dictionary of token balances.
    - prices (dict): Dictionary of token prices.

    Returns:
    - dict: Dictionary of token balances converted to USD.
    """
    # Convert token keys to upper case for consistency
    balances = {token.upper(): balance for token, balance in balances.items()}

    print(f'balances: {balances.keys()}')
    print(f'TOKEN_CONTRACTS.keys(): {TOKEN_CONTRACTS.keys()}')

    for token in TOKEN_CONTRACTS.keys():
        if f"{token}_PRICE" not in prices:
            print(f"Missing price for token: {token}")

    usd_balances = {
        token: balances[token] * prices[f"{token}_PRICE"]
        for token in TOKEN_CONTRACTS.keys()
        if f"{token}_PRICE" in prices
    }
    return usd_balances

def calculate_sortino_ratio(df, risk_free, window_size):
    def sortino_ratio(returns, risk_free_rate):
        returns = pd.Series(returns)
        daily_risk_free_rate = (1 + risk_free_rate) ** (1/365) - 1

        excess_returns = returns - daily_risk_free_rate
        downside_returns = np.where(excess_returns < 0, excess_returns**2, 0)
        daily_downside_deviation = np.sqrt(downside_returns.mean())

        if np.isnan(daily_downside_deviation):
            daily_downside_deviation = 0.0

        active_days = returns.notna().sum()
        annual_factor = 365 / active_days if active_days != 0 else 0
        compounding_return = (1 + excess_returns).prod() ** annual_factor - 1 if active_days != 0 else 0
        annual_downside_deviation = daily_downside_deviation * np.sqrt(365)

        sortino = compounding_return / annual_downside_deviation if annual_downside_deviation != 0 else 0.0
        
        if np.isinf(sortino) or sortino > 1000:
            sortino = 0.0
            print("Unusual Sortino ratio detected, setting to 0.0")
            
        return sortino

    # Calculate rolling Sortino ratios for each column
    rolling_sortino = df.rolling(window=window_size).apply(lambda x: sortino_ratio(x, risk_free))

    # Calculate all-time Sortino ratios for each column
    all_time_sortino = df.apply(lambda col: sortino_ratio(col.dropna(), risk_free))

    return rolling_sortino, all_time_sortino