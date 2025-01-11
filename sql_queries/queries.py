import datetime as dt

def token_prices(token_addresses, network, start_date):
    start_date = dt.datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    """
    Generate a SQL query to get historical price data for given token addresses from a specific start date.

    Parameters:
    - token_addresses (list): List of token addresses.
    - start_date (str): Start date in 'YYYY-MM-DD' format.

    Returns:
    - str: The SQL query string.
    """
    # Format the addresses into the SQL VALUES clause
    addresses_clause = ", ".join(f"(LOWER('{address}'))" for address in token_addresses)

    beginning = f"'{start_date.strftime('%Y-%m-%d %H:%M:%S')}'"
    print('Beginning:', beginning)
    
    prices_query = f"""
    WITH addresses AS (
        SELECT column1 AS token_address 
        FROM (VALUES
            {addresses_clause}
        ) AS tokens(column1)
    )

    SELECT 
        hour,
        symbol,
        price
    FROM 
        {network}.price.ez_prices_hourly
    WHERE 
        token_address IN (SELECT token_address FROM addresses)
        AND hour >= DATE_TRUNC('hour', TO_TIMESTAMP({beginning}, 'YYYY-MM-DD HH24:MI:SS'))
    ORDER BY 
        hour DESC, symbol
    """

    return prices_query


def token_classifier(network,days,volume_threshold,backtest_period):
  
  hours_back = 8760 if backtest_period < 8760 else backtest_period #Ensures token data for least 1 year
  print(f'hours_back: {hours_back}')
   
  query = f"""
   
WITH rolling_averages AS (
    SELECT
        symbol,
        token_address,
        hour,
        price,
        AVG(price) OVER (
            PARTITION BY symbol, token_address
            ORDER BY hour
            ROWS BETWEEN 168 PRECEDING AND CURRENT ROW
        ) AS rolling_7d_avg,
        AVG(price) OVER (
            PARTITION BY symbol, token_address
            ORDER BY hour
            ROWS BETWEEN 720 PRECEDING AND CURRENT ROW
        ) AS rolling_30d_avg
    FROM
        {network}.price.ez_prices_hourly
    WHERE
        hour >= DATEADD(DAY, -{days}, CURRENT_DATE) -- Ensure a 60-day window
),
first_appearance AS (
    SELECT
        symbol,
        token_address,
        MIN(hour) AS first_hour
    FROM
        {network}.price.ez_prices_hourly
    GROUP BY
        symbol, token_address
),
latest_price AS (
    SELECT
        symbol,
        token_address,
        MAX(hour) AS latest_hour
    FROM
        rolling_averages
    WHERE
        hour >= CURRENT_DATE
    GROUP BY
        symbol, token_address
),
latest_price_details AS (
    SELECT
        lp.symbol,
        lp.token_address,
        lp.latest_hour,
        ra.price AS latest_price,
        ra.rolling_7d_avg,
        ra.rolling_30d_avg
    FROM
        latest_price lp
    JOIN
        rolling_averages ra
    ON
        lp.symbol = ra.symbol AND lp.token_address = ra.token_address AND lp.latest_hour = ra.hour
),
price_60_days_ago AS (
    SELECT
        symbol,
        token_address,
        MIN(hour) AS sixty_days_hour
    FROM
        {network}.price.ez_prices_hourly
    WHERE
        hour >= DATEADD(DAY, -{days}, CURRENT_DATE)
    GROUP BY
        symbol, token_address
),
price_60_days_details AS (
    SELECT
        p60.symbol,
        p60.token_address,
        p60.sixty_days_hour,
        eph.price AS sixty_day_price
    FROM
        price_60_days_ago p60
    JOIN
        {network}.price.ez_prices_hourly eph
    ON
        p60.symbol = eph.symbol AND p60.token_address = eph.token_address AND p60.sixty_days_hour = eph.hour
),
token_volumes AS (
    SELECT
        symbol_out,
        token_out,
        AVG(amount_out_usd) AS avg_vol,
        SUM(amount_out_usd) AS sum_vol
    FROM
        {network}.defi.ez_dex_swaps
    WHERE
        DATE_TRUNC('day', block_timestamp) >= DATEADD(DAY, -{days}, CURRENT_DATE)
    GROUP BY
        symbol_out, token_out
),
dataset_avg AS (
    SELECT
        AVG(sum_vol) AS overall_avg_vol
    FROM
        token_volumes
),
combined_agg AS (
    SELECT
        tv.symbol_out,
        tv.token_out,
        tv.avg_vol,
        tv.sum_vol
    FROM
        token_volumes tv
    CROSS JOIN
        dataset_avg da
    WHERE
        tv.sum_vol > (da.overall_avg_vol * {volume_threshold})
)
SELECT
    lpd.symbol,
    lpd.token_address,
    lpd.latest_hour,
    lpd.latest_price,
    p60d.sixty_day_price,
    (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price AS sixty_d_return,
    lpd.rolling_7d_avg,
    lpd.rolling_30d_avg,
    tkv.sum_vol as Volume,
    tkv.avg_vol as Average_Order
FROM
    latest_price_details lpd
JOIN
    price_60_days_details p60d
ON
    lpd.symbol = p60d.symbol AND lpd.token_address = p60d.token_address
JOIN
    first_appearance fa
ON
    lpd.symbol = fa.symbol AND lpd.token_address = fa.token_address
JOIN 
    token_volumes tkv
ON 
    tkv.symbol_out = lpd.symbol AND tkv.token_out = lpd.token_address
WHERE
    lpd.token_address IN (SELECT DISTINCT token_out FROM combined_agg)
    AND sixty_d_return > 0.001
    AND lpd.rolling_7d_avg > lpd.rolling_30d_avg -- Uncomment to filter tokens trending up
    AND fa.first_hour <= DATEADD(HOUR, -{hours_back}, CURRENT_DATE) -- Ensure data goes back at least 1 year
    AND p60d.sixty_day_price > 0.0001
ORDER BY
    sixty_d_return DESC;

   """
  return query

def sharpe_classifier(start_date,network,days,volume_threshold):
   print(f'start_date: {start_date}')
   query = f"""
WITH RECURSIVE date_series AS (
  SELECT
    TIMESTAMP '{start_date} 00:00:00' AS DT
  UNION
  ALL
  SELECT
    DT + INTERVAL '1 HOUR'
  FROM
    date_series
  WHERE
    DT + INTERVAL '1 HOUR' <= CURRENT_TIMESTAMP
),
tbill AS (
  SELECT
    livequery.live.udf_api(
      'https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json&api_key=af3aeb14543cb05941f1b87abc3e3b7b&observation_start={start_date}'
    ) AS response
),
PARSED_tbill AS (
  SELECT
    PARSE_JSON(response) AS DATA
  FROM
    tbill
),
flattened_tbill AS (
  SELECT
    -- Convert date string to timestamp and set the time to '00:00:00'
    TO_TIMESTAMP(
      SUBSTRING(f.value:date :: STRING, 1, 10),
      'YYYY-MM-DD'
    ) AS date,
    f.value:value :: FLOAT AS tbill
  FROM
    PARSED_tbill,
    LATERAL FLATTEN(input => DATA:data:observations) f
),
daily_tbill AS (
  SELECT
    ds.DT,
    (
      (
        LAST_VALUE(ft.tbill IGNORE NULLS) OVER (
          ORDER BY
            ds.DT ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ) / 100
      )
    ) AS tbill_annualized,
    (
      (
        LAST_VALUE(ft.tbill IGNORE NULLS) OVER (
          ORDER BY
            ds.DT ROWS BETWEEN UNBOUNDED PRECEDING
            AND CURRENT ROW
        ) / 100
      ) / 365
    ) AS tbill_hourly
  FROM
    date_series ds
    LEFT JOIN flattened_tbill ft ON ds.DT :: DATE = ft.date :: DATE
),
rolling_averages AS (
  SELECT
    symbol,
    token_address,
    hour,
    price,
    AVG(price) OVER (
      PARTITION BY symbol,
      token_address
      ORDER BY
        hour ROWS BETWEEN 168 PRECEDING
        AND CURRENT ROW
    ) AS rolling_7d_avg,
    AVG(price) OVER (
      PARTITION BY symbol,
      token_address
      ORDER BY
        hour ROWS BETWEEN 720 PRECEDING
        AND CURRENT ROW
    ) AS rolling_30d_avg
  FROM
    {network}.price.ez_prices_hourly
  WHERE
    hour >= DATEADD(DAY, - {days}, CURRENT_DATE) -- Ensure a 60-day window
),
first_appearance AS (
  SELECT
    symbol,
    token_address,
    MIN(hour) AS first_hour
  FROM
    {network}.price.ez_prices_hourly
  GROUP BY
    symbol,
    token_address
),
latest_price AS (
  SELECT
    symbol,
    token_address,
    MAX(hour) AS latest_hour
  FROM
    rolling_averages
  WHERE
    hour >= CURRENT_DATE
  GROUP BY
    symbol,
    token_address
),
latest_price_details AS (
  SELECT
    lp.symbol,
    lp.token_address,
    lp.latest_hour,
    ra.price AS latest_price,
    ra.rolling_7d_avg,
    ra.rolling_30d_avg
  FROM
    latest_price lp
    JOIN rolling_averages ra ON lp.symbol = ra.symbol
    AND lp.token_address = ra.token_address
    AND lp.latest_hour = ra.hour
  WHERE
    ra.price > 0.001
),
price_60_days_ago AS (
  SELECT
    symbol,
    token_address,
    MIN(hour) AS sixty_days_hour
  FROM
    {network}.price.ez_prices_hourly
  WHERE
    hour >= DATEADD(DAY, - {days}, CURRENT_DATE)
  GROUP BY
    symbol,
    token_address
),
price_60_days_details AS (
  SELECT
    p60.symbol,
    p60.token_address,
    p60.sixty_days_hour,
    eph.price AS sixty_day_price
  FROM
    price_60_days_ago p60
    JOIN {network}.price.ez_prices_hourly eph ON p60.symbol = eph.symbol
    AND p60.token_address = eph.token_address
    AND p60.sixty_days_hour = eph.hour
    AND eph.price > 0.0001
),
adjusted_risk_free_rate AS (
  SELECT
    tf.DT,
    POWER(
      1 + tf.tbill_annualized,
      {days} / 365.0
    ) - 1 AS tbill_60d
  FROM
    daily_tbill tf
),
token_volumes AS (
  SELECT
    symbol_out,
    token_out,
    AVG(amount_out_usd) AS avg_vol,
    SUM(amount_out_usd) AS sum_vol
  FROM
    {network}.defi.ez_dex_swaps
  WHERE
    DATE_TRUNC('day', block_timestamp) >= DATEADD(DAY, - {days}, CURRENT_DATE)
  GROUP BY
    symbol_out,
    token_out
),
dataset_avg AS (
  SELECT
    AVG(sum_vol) AS overall_avg_vol
  FROM
    token_volumes
),
combined_agg AS (
  SELECT
    tv.symbol_out,
    tv.token_out,
    tv.avg_vol,
    tv.sum_vol
  FROM
    token_volumes tv
    CROSS JOIN dataset_avg da
  WHERE
    tv.sum_vol > (
      da.overall_avg_vol * {volume_threshold}
    )
),
excess_return_calc AS (
  SELECT
    lpd.symbol,
    lpd.token_address,
    lpd.latest_hour,
    lpd.latest_price,
    tkv.avg_vol,
    tkv.sum_vol,
    (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price AS token_return,
    ad.tbill_60d AS risk_free_rate,
    (
      (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price
    ) - ad.tbill_60d AS excess_return
  FROM
    latest_price_details lpd
    JOIN price_60_days_details p60d ON lpd.symbol = p60d.symbol
    AND lpd.token_address = p60d.token_address
    JOIN token_volumes tkv ON tkv.symbol_out = lpd.symbol
    AND tkv.token_out = lpd.token_address
    left JOIN adjusted_risk_free_rate ad ON DATE_TRUNC('hour', lpd.latest_hour) = DATE_TRUNC('hour', ad.DT)
  WHERE
    lpd.token_address IN (
      SELECT
        DISTINCT token_out
      FROM
        combined_agg
    )
    AND (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price > 0.001
    AND lpd.rolling_7d_avg > lpd.rolling_30d_avg
),
rolling_std AS (
  SELECT
    symbol,
    token_address,
    hour,
    STDDEV(price) OVER (
      PARTITION BY symbol,
      token_address
      ORDER BY
        hour ROWS BETWEEN 720 PRECEDING
        AND CURRENT ROW
    ) AS stddev_30d
  FROM
    {network}.price.ez_prices_hourly
  WHERE
    token_address in (
      select
        distinct token_address
      from
        excess_return_calc
    )
),
latest_std_dev AS (
  SELECT
    symbol,
    token_address,
    stddev_30d,
    hour
  FROM
    (
      SELECT
        symbol,
        token_address,
        stddev_30d,
        hour,
        ROW_NUMBER() OVER (
          PARTITION BY symbol,
          token_address
          ORDER BY
            hour DESC
        ) AS row_num
      FROM
        rolling_std
    )
  WHERE
    row_num = 1
),
sharpe_ratio_calc AS (
  SELECT
    erc.symbol,
    erc.token_address,
    erc.excess_return,
    erc.latest_price,
    erc.latest_hour,
    lsd.stddev_30d,
    erc.token_return,
    erc.avg_vol,
    erc.sum_vol,
    erc.excess_return / NULLIF(lsd.stddev_30d, 0) AS sharpe_ratio
  FROM
    excess_return_calc erc
    JOIN latest_std_dev lsd ON erc.token_address = lsd.token_address
)
SELECT
  symbol,
  token_address,
  sharpe_ratio,
  excess_return,
  latest_price,
  latest_hour,
  stddev_30d,
  sum_vol as Volume,
  avg_vol as Average_Order,
  token_return as sixty_d_return
FROM
  sharpe_ratio_calc
ORDER BY
  sharpe_ratio DESC
limit
  10;

"""
   return query

def latest_portfolio_metrics(token_addresses, network, days, start_date):
    addresses_clause = ", ".join(f"(LOWER('{address}'))" for address in token_addresses)
    query = f"""
    WITH RECURSIVE date_series AS (
      SELECT
        TIMESTAMP '{start_date} 00:00:00' AS DT
      UNION
      ALL
      SELECT
        DT + INTERVAL '1 HOUR'
      FROM
        date_series
      WHERE
        DT + INTERVAL '1 HOUR' <= CURRENT_TIMESTAMP
    ),
    addresses AS (
        SELECT column1 AS token_address 
        FROM (VALUES
            {addresses_clause}
        ) AS tokens(column1)
    ),
    tbill AS (
      SELECT
        livequery.live.udf_api(
          'https://api.stlouisfed.org/fred/series/observations?series_id=TB3MS&file_type=json&api_key=af3aeb14543cb05941f1b87abc3e3b7b&observation_start=2024-01-01'
        ) AS response
    ),
    PARSED_tbill AS (
      SELECT
        PARSE_JSON(response) AS DATA
      FROM
        tbill
    ),
    flattened_tbill AS (
      SELECT
        TO_TIMESTAMP(
          SUBSTRING(f.value:date :: STRING, 1, 10),
          'YYYY-MM-DD'
        ) AS date,
        f.value:value :: FLOAT AS tbill
      FROM
        PARSED_tbill,
        LATERAL FLATTEN(input => DATA:data:observations) f
    ),
    daily_tbill AS (
      SELECT
        ds.DT,
        (
          (
            LAST_VALUE(ft.tbill IGNORE NULLS) OVER (
              ORDER BY
                ds.DT ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) / 100
          )
        ) AS tbill_annualized,
        (
          (
            LAST_VALUE(ft.tbill IGNORE NULLS) OVER (
              ORDER BY
                ds.DT ROWS BETWEEN UNBOUNDED PRECEDING
                AND CURRENT ROW
            ) / 100
          ) / 365
        ) AS tbill_hourly
      FROM
        date_series ds
        LEFT JOIN flattened_tbill ft ON ds.DT :: DATE = ft.date :: DATE
    ),
    rolling_averages AS (
      SELECT
        symbol,
        token_address,
        hour,
        price,
        AVG(price) OVER (
          PARTITION BY symbol,
          token_address
          ORDER BY
            hour ROWS BETWEEN 168 PRECEDING
            AND CURRENT ROW
        ) AS rolling_7d_avg,
        AVG(price) OVER (
          PARTITION BY symbol,
          token_address
          ORDER BY
            hour ROWS BETWEEN 720 PRECEDING
            AND CURRENT ROW
        ) AS rolling_30d_avg
      FROM
        {network}.price.ez_prices_hourly
      WHERE
        hour >= DATEADD(DAY, - {days}, CURRENT_DATE) 
      AND TOKEN_ADDRESS IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES)
    ),
    first_appearance AS (
      SELECT
        symbol,
        token_address,
        MIN(hour) AS first_hour
      FROM
        {network}.price.ez_prices_hourly
      GROUP BY
        symbol,
        token_address
    ),
    latest_price AS (
      SELECT
        symbol,
        token_address,
        MAX(hour) AS latest_hour
      FROM
        rolling_averages
      WHERE
        hour >= CURRENT_DATE
      GROUP BY
        symbol,
        token_address
    ),
    latest_price_details AS (
      SELECT
        lp.symbol,
        lp.token_address,
        lp.latest_hour,
        ra.price AS latest_price,
        ra.rolling_7d_avg,
        ra.rolling_30d_avg
      FROM
        latest_price lp
        JOIN rolling_averages ra ON lp.symbol = ra.symbol
        AND lp.token_address = ra.token_address
        AND lp.latest_hour = ra.hour
      WHERE
        LP.TOKEN_ADDRESS IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES) 
    ),
    price_60_days_ago AS (
      SELECT
        symbol,
        token_address,
        MIN(hour) AS sixty_days_hour
      FROM
        {network}.price.ez_prices_hourly
      WHERE
        hour >= DATEADD(DAY, - {days}, CURRENT_DATE)
      GROUP BY
        symbol,
        token_address
    ),
    price_60_days_details AS (
      SELECT
        p60.symbol,
        p60.token_address,
        p60.sixty_days_hour,
        eph.price AS sixty_day_price
      FROM
        price_60_days_ago p60
      JOIN {network}.price.ez_prices_hourly eph ON p60.symbol = eph.symbol
        AND p60.token_address = eph.token_address
        AND p60.sixty_days_hour = eph.hour
      WHERE
        p60.TOKEN_ADDRESS IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES)
       
    ),
    adjusted_risk_free_rate AS (
      SELECT
        tf.DT,
        POWER(
          1 + tf.tbill_annualized,
          {days} / 365.0
        ) - 1 AS tbill_60d
      FROM
        daily_tbill tf
    ),
    token_volumes AS (
      SELECT
        symbol_out,
        token_out,
        AVG(amount_out_usd) AS avg_vol,
        SUM(amount_out_usd) AS sum_vol
      FROM
        {network}.defi.ez_dex_swaps
      WHERE
        DATE_TRUNC('day', block_timestamp) >= DATEADD(DAY, - {days}, CURRENT_DATE)
      AND TOKEN_OUT IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES)
      GROUP BY
        symbol_out,
        token_out
    ),
    excess_return_calc AS (
      SELECT
        lpd.symbol,
        lpd.token_address,
        lpd.latest_hour,
        lpd.latest_price,
        lpd.rolling_7d_avg,
        lpd.rolling_30d_avg,
        (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price AS token_return,
        ad.tbill_60d AS risk_free_rate,
        (
          (lpd.latest_price - p60d.sixty_day_price) / p60d.sixty_day_price
        ) - ad.tbill_60d AS excess_return
      FROM
        latest_price_details lpd
        JOIN price_60_days_details p60d ON lpd.symbol = p60d.symbol
        AND lpd.token_address = p60d.token_address
        left JOIN adjusted_risk_free_rate ad ON DATE_TRUNC('hour', lpd.latest_hour) = DATE_TRUNC('hour', ad.DT)
      WHERE
        lpd.TOKEN_ADDRESS IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES)
    
    ),
    rolling_std AS (
      SELECT
        symbol,
        token_address,
        hour,
        STDDEV(price) OVER (
          PARTITION BY symbol,
          token_address
          ORDER BY
            hour ROWS BETWEEN 720 PRECEDING
            AND CURRENT ROW
        ) AS stddev_30d
      FROM
        {network}.price.ez_prices_hourly
       WHERE
        TOKEN_ADDRESS IN (SELECT DISTINCT TOKEN_ADDRESS FROM ADDRESSES)
    
    ),
    latest_std_dev AS (
      SELECT
        symbol,
        token_address,
        stddev_30d,
        hour
      FROM
        (
          SELECT
            symbol,
            token_address,
            stddev_30d,
            hour,
            ROW_NUMBER() OVER (
              PARTITION BY symbol,
              token_address
              ORDER BY
                hour DESC
            ) AS row_num
          FROM
            rolling_std
        )
      WHERE
        row_num = 1
    ),
    sharpe_ratio_calc AS (
      SELECT
        erc.symbol,
        erc.token_address,
        erc.excess_return,
        erc.latest_price,
        erc.latest_hour,
        erc.token_return,
        lsd.stddev_30d,
        erc.rolling_7d_avg,
        erc.rolling_30d_avg,
        erc.excess_return / NULLIF(lsd.stddev_30d, 0) AS sharpe_ratio
      FROM
        excess_return_calc erc
        JOIN latest_std_dev lsd ON erc.token_address = lsd.token_address
    )
    SELECT
      s.symbol,
      s.token_address,
      s.sharpe_ratio,
      s.excess_return,
      s.latest_price,
      s.latest_hour,
      s.stddev_30d,
      s.rolling_7d_avg,
      s.token_return,
      s.rolling_30d_avg,
      t.avg_vol,
      t.sum_vol
    FROM
      sharpe_ratio_calc s
    LEFT JOIN token_volumes t ON s.token_address = t.token_out
    ORDER BY
      sharpe_ratio DESC
    """
    return query 
    