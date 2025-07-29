import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from pandas.tseries.offsets import BDay
import alpaca_trade_api as tradeapi
from alpaca_key import alpaca_keys

# Alpaca 
API_KEY = alpaca_keys.get("api_key") or alpaca_keys.get("API_KEY")
API_SECRET = alpaca_keys.get("api_secret") or alpaca_keys.get("API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("API_KEY and API_SECRET ain't where its supposed to be")

# Strategy parameters
SYMBOL = "SPY"
TIMEFRAME = "5Min"
RSI_PERIOD = 7
BOLL_WINDOW = 25
BOLL_STD_DEV = 1.5
DAYS_BACK = 50
START_BALANCE = 10_000
MARGIN = 2

# Alpaca API client
DATA_BASE_URL = "https://data.alpaca.markets/v2"
api = tradeapi.REST(
    key_id=API_KEY,
    secret_key=API_SECRET,
    base_url=DATA_BASE_URL,
    api_version='v2'
)

def run_backtest_trading_days(n_days: int = 5, **kwargs):
    end = datetime.now(timezone.utc)
    # Subtract n business days
    start = (end.tz_convert('US/Pacific') - BDay(n_days)).tz_convert(timezone.utc)
    raw = fetch_data(SYMBOL, start, end)
    ind = compute_indicators(raw, kwargs['boll_window'], kwargs['boll_std_dev'])
    return backtest(ind)

def fetch_data(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    start_str = start.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    end_str = end.replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    records = [
        bar._raw for bar in api.get_bars_iter(
            symbol, TIMEFRAME, start=start_str, end=end_str,
            limit=1000, adjustment='raw', feed='iex'
        )
    ]
    if not records:
        raise RuntimeError(f"No data returned for {symbol} between {start_str} and {end_str}")
    df = pd.DataFrame(records)
    df.rename(columns={'t':'time', 'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}, inplace=True)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df.index = df.index.tz_convert('US/Pacific')
    return df.between_time('09:30', '16:00')

def compute_indicators(df: pd.DataFrame, boll_window=20, boll_std_dev=2) -> pd.DataFrame:
    df = df.copy()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    sma = df['close'].rolling(boll_window).mean()
    std = df['close'].rolling(boll_window).std()
    df['upper'] = sma + (std * boll_std_dev)
    df['lower'] = sma - (std * boll_std_dev)
    return df.dropna()

def generate_signals(
    df: pd.DataFrame,
    buy_rsi_low=25, buy_rsi_high=50,
    sell_rsi_low=60, sell_rsi_high=65,
    cooldown=0
) -> pd.Series:
    buy = (df['rsi'] >= buy_rsi_low) & (df['rsi'] <= buy_rsi_high) & (df['close'] < df['lower'])
    sell = (df['rsi'] >= sell_rsi_low) & (df['rsi'] <= sell_rsi_high) & (df['close'] < df['upper'])
    signals = pd.Series(0, index=df.index)
    last_trade_idx = -cooldown
    for i in range(len(df)):
        if i - last_trade_idx < cooldown:
            continue
        if buy.iloc[i]:
            signals.iloc[i] = 1
            last_trade_idx = i
        elif sell.iloc[i]:
            signals.iloc[i] = -1
            last_trade_idx = i
    return signals

def backtest(
    df: pd.DataFrame,
    buy_rsi_low=25, buy_rsi_high=50,
    sell_rsi_low=60, sell_rsi_high=65
) -> pd.DataFrame:
    df = df.copy()
    df['signal'] = generate_signals(df, buy_rsi_low, buy_rsi_high, sell_rsi_low, sell_rsi_high)
    df['position'] = df['signal'].replace(0, pd.NA).ffill().fillna(0)
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['equity_curve'] = (1 + df['strategy_returns']).cumprod()
    return df

def run_backtest_days(days: int = DAYS_BACK, boll_window=20, boll_std_dev=2) -> pd.DataFrame:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    raw = fetch_data(SYMBOL, start, end)
    ind = compute_indicators(raw, boll_window, boll_std_dev)
    results = backtest(ind)
    return results

def summarize_trades(bt: pd.DataFrame) -> list:
    trades = bt[bt['signal'] != 0].copy()
    trades['trade_type'] = trades['signal'].map({1: 'BUY', -1: 'SELL'})
    trades['date'] = trades.index.date
    trade_summaries = []
    trade_indices = trades.index.tolist()
    for i, idx in enumerate(trade_indices):
        entry_price = trades.loc[idx, 'close']
        entry_type = trades.loc[idx, 'trade_type']
        entry_date = trades.loc[idx, 'date']
        entry_rsi = trades.loc[idx, 'rsi']
        entry_upper = trades.loc[idx, 'upper']
        entry_lower = trades.loc[idx, 'lower']
        reason = (
            f"RSI={entry_rsi:.2f} in [25, 50] and close<{entry_lower:.2f} (lower band)"
            if entry_type == 'BUY'
            else f"RSI={entry_rsi:.2f} in [60, 65] and close<{entry_upper:.2f} (upper band)"
        )

        # Find exit: first bar after entry where exit condition is met, or EOD
        day_data = bt[bt.index.date == entry_date]
        entry_loc = day_data.index.get_loc(idx)
        exit_idx = None
        if entry_type == 'BUY':
            # Exit at first bar where close >= upper band
            after_entry = day_data.iloc[entry_loc+1:]
            exit_candidates = after_entry[after_entry['close'] >= after_entry['upper']]
        else:
            # Exit at first bar where close <= lower band
            after_entry = day_data.iloc[entry_loc+1:]
            exit_candidates = after_entry[after_entry['close'] <= after_entry['lower']]
        if not exit_candidates.empty:
            exit_idx = exit_candidates.index[0]
        else:
            exit_idx = day_data.index[-1]  # EOD if never touched

        exit_price = bt.loc[exit_idx, 'close']
        pnl = (exit_price - entry_price) if entry_type == 'BUY' else (entry_price - exit_price)
        trade_summaries.append({
            'date': entry_date,
            'entry_time': idx,
            'exit_time': exit_idx,
            'trade_type': entry_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'reason': reason
        })
    return trade_summaries

def save_run_summary(pl, max_drawdown, params, filename='run_summaries.csv'):
    summary_data = {
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        **params,
        'P/L': pl,
        'Max_Drawdown': max_drawdown
    }
    write_header = not os.path.isfile(filename)
    with open(filename, 'a') as f:
        if write_header:
            f.write(','.join(summary_data.keys()) + '\n')
        f.write(','.join(str(summary_data[k]) for k in summary_data.keys()) + '\n')
    print(f"Run summary appended to {filename}")

def plot_trade(bt, trade):
    entry_time = trade['entry_time']
    exit_time = trade['exit_time']
    entry_type = trade['trade_type']
    reason = trade['reason']
    entry_date = trade['date']
    day_mask = bt.index.date == entry_date
    day_data = bt.loc[day_mask]
    plt.figure(figsize=(14, 8))
    plt.subplot(2, 1, 1)
    plt.plot(day_data.index, day_data['close'], label='Close', color='black')
    plt.plot(day_data.index, day_data['upper'], label='Upper Band', linestyle='--', color='red')
    plt.plot(day_data.index, day_data['lower'], label='Lower Band', linestyle='--', color='blue')
    plt.fill_between(day_data.index, day_data['lower'], day_data['upper'], color='gray', alpha=0.1)
    plt.scatter(entry_time, trade['entry_price'], color='green' if entry_type == 'BUY' else 'red', marker='^' if entry_type == 'BUY' else 'v', s=100, label='Entry')
    plt.scatter(exit_time, trade['exit_price'], color='orange', marker='o', s=100, label='Exit')
    plt.title(f"Trade on {entry_date} | {entry_type} | {reason}")
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(day_data.index, day_data['rsi'], label='RSI', color='purple')
    plt.axhline(25, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(50, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(60, color='red', linestyle='--', alpha=0.5)
    plt.axhline(65, color='red', linestyle='--', alpha=0.5)
    plt.scatter(entry_time, bt.loc[entry_time, 'rsi'], color='green' if entry_type == 'BUY' else 'red', s=100, label='Entry RSI')
    plt.scatter(exit_time, bt.loc[exit_time, 'rsi'], color='orange', s=100, label='Exit RSI')
    plt.ylabel('RSI')
    plt.xlabel('Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_equity_curve(bt):
    plt.figure(figsize=(12, 6))
    plt.plot(bt.index, bt['equity_curve_margin'], label='Equity Curve (Margin)')
    # 1) Resample your margin‐adjusted equity to end-of-day values
    daily_equity = bt['equity_curve_margin'] \
    .resample('1D') \
    .last() \
    .dropna()

    # 2) Compute daily returns
    daily_returns = daily_equity.pct_change().dropna()

    # 3) Calculate annualized Sharpe (assumes 0% risk-free)
    trading_days = 252
    sharpe_daily = daily_returns.mean() / daily_returns.std() * (trading_days ** 0.5)

    print(f"Annualized Daily Sharpe: {sharpe_daily:.2f}")
    plt.title('Stock Backtest Equity Curve with Margin')
    plt.xlabel('Time')
    plt.ylabel('Equity ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    params = {
        'buy_rsi_low': 25,
        'buy_rsi_high': 30,
        'sell_rsi_low': 60,
        'sell_rsi_high': 65,
        'boll_window': 25,
        'boll_std_dev': 1.5,
        'margin': MARGIN,
        'start_balance': START_BALANCE
    }
    bt = run_backtest_days(boll_window=params['boll_window'], boll_std_dev=params['boll_std_dev'])
    if not bt.empty:
        bt['strategy_returns_margin'] = bt['strategy_returns'] * MARGIN
        bt['equity_curve_margin'] = START_BALANCE * (1 + bt['strategy_returns_margin']).cumprod()
        max_drawdown = (bt['equity_curve_margin'].cummax() - bt['equity_curve_margin']).max() / bt['equity_curve_margin'].cummax().max()
        pl = (bt['equity_curve_margin'].iloc[-1] - START_BALANCE) / START_BALANCE
        print(f"P/L: {pl:.4%}")
        print(f"Max Drawdown: {max_drawdown:.4%}")
        trade_summaries = summarize_trades(bt)
        trade_summary_df = pd.DataFrame(trade_summaries)
        daily_summary = trade_summary_df.groupby('date').agg(
            num_trades=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            avg_pnl=('pnl', 'mean')
        )
        print("\nDaily Trade Summary:")
        print(daily_summary)
        trade_summary_df.to_csv('trade_details.csv', index=False)
        daily_summary.to_csv('daily_trade_summary.csv')
        print("Trade details exported to trade_details.csv")
        print("Daily summary exported to daily_trade_summary.csv")
        save_run_summary(pl, max_drawdown, params)
        for trade in trade_summaries:
            plot_trade(bt, trade)
        plot_equity_curve(bt)
    else:
        print("No trades or results for these parameters.")
