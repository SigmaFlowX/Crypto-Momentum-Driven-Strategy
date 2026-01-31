import pandas as pd
import numpy as np
import optuna
from numba import njit
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import matplotlib

optuna.logging.set_verbosity(optuna.logging.CRITICAL)
matplotlib.use("Agg")
training_period = 6 #month
test_period = 3 #month


@njit
def backtest_numba(price_arr, ema_arr, roc_arr, roc_std_arr, vol_arr, vol_ma_arr, ema_std_arr, roc_threshold, trailing_pct, ema_threshold):
    balance = 10000
    risk_percent = 10
    fee =0.02

    pos = 0
    n = len(price_arr)

    for i in range(n):
        curr_price = price_arr[i]
        curr_ema = ema_arr[i]
        curr_roc = roc_arr[i]
        curr_roc_std = roc_std_arr[i]
        curr_vol = vol_arr[i]
        curr_vol_ma = vol_ma_arr[i]
        roc_prev = roc_arr[i-1] if i > 0 else roc_arr[i]
        ema_std = ema_std_arr[i]


        long_cond = (
                pos == 0 and
                curr_roc > roc_threshold * curr_roc_std and
                curr_price > curr_ema and
                curr_vol > curr_vol_ma and
                curr_roc > roc_prev and
                curr_price > curr_ema + ema_threshold * ema_std
        )

        short_cond = (
                pos == 0 and
                curr_roc < -roc_threshold * curr_roc_std and
                curr_price < curr_ema and
                curr_vol > curr_vol_ma and
                curr_roc < roc_prev and
                curr_price < curr_ema - ema_threshold * ema_std
        )
        if long_cond:
            long_price = curr_price
            pos = 1

            long_trailing_stop = long_price * (1 - trailing_pct / 100)

        elif short_cond:
            short_price = curr_price
            pos = -1

            short_trailing_stop = short_price * (1 + trailing_pct / 100)


        if pos == 1:
            long_trailing_stop = max(long_trailing_stop, curr_price * (1 - trailing_pct / 100))

            if (curr_price < curr_ema or curr_price < long_trailing_stop):
                pnl = curr_price / long_price - 1 - 2 * fee / 100
                balance *= 1 + pnl * risk_percent / 100
                pos = 0


        elif pos == -1:
            short_trailing_stop = min(short_trailing_stop, curr_price * (1 + trailing_pct / 100))

            if (curr_price > curr_ema or curr_price > short_trailing_stop):
                pnl = short_price / curr_price - 1 - 2 * fee / 100
                balance *= 1 + pnl * risk_percent / 100
                pos = 0


    if pos == 1:
        pnl = curr_price / long_price - 1 - 2 * fee / 100
        balance *= 1 + pnl * risk_percent / 100
    elif pos == -1:
        pnl = short_price / curr_price - 1 - 2 * fee / 100
        balance *= 1 + pnl * risk_percent / 100

    return balance/10000-1

def test(price_arr, ema_arr, roc_arr, roc_std_arr, vol_arr, vol_ma_arr, ema_std_arr, roc_threshold, trailing_pct, ema_threshold, timestamps=None, show_plots=True):
    balance = 10000
    risk_percent = 10
    fee = 0.02

    pnl_array = []
    trade_count = 0
    win_count = 0
    pos = 0

    equity_curve = [balance]
    long_entries, short_entries = [], []
    long_entry_prices, short_entry_prices = [], []
    exit_points, exit_prices = [], []

    n = len(price_arr)
    for i in range(n):
        curr_price = price_arr[i]
        curr_ema = ema_arr[i]
        curr_roc = roc_arr[i]
        curr_roc_std = roc_std_arr[i]
        curr_vol = vol_arr[i]
        curr_vol_ma = vol_ma_arr[i]
        roc_prev = roc_arr[i - 1] if i > 0 else curr_roc
        ema_std = ema_std_arr[i]

        long_cond = (
                pos == 0 and
                curr_roc > roc_threshold * curr_roc_std and
                curr_price > curr_ema and
                curr_vol > curr_vol_ma and
                curr_roc > roc_prev and
                curr_price > curr_ema + ema_threshold * ema_std
        )

        short_cond = (
                pos == 0 and
                curr_roc < -roc_threshold * curr_roc_std and
                curr_price < curr_ema and
                curr_vol > curr_vol_ma and
                curr_roc < roc_prev and
                curr_price < curr_ema - ema_threshold * ema_std
        )

        if long_cond:
            long_price = curr_price
            pos = 1
            long_trailing_stop = long_price * (1 - trailing_pct / 100)
            long_entries.append(i)
            long_entry_prices.append(curr_price)

        elif short_cond:
            short_price = curr_price
            pos = -1
            short_trailing_stop = short_price * (1 + trailing_pct / 100)
            short_entries.append(i)
            short_entry_prices.append(curr_price)

        if pos == 1:
            long_trailing_stop = max(long_trailing_stop, curr_price * (1 - trailing_pct / 100))
            if curr_price < curr_ema or curr_price < long_trailing_stop:
                pnl = curr_price / long_price - 1 - 2 * fee / 100
                balance *= 1 + pnl * risk_percent / 100
                pos = 0

                pnl_array.append(pnl)
                trade_count += 1
                if pnl > 0:
                    win_count += 1
                exit_points.append(i)
                exit_prices.append(curr_price)

        elif pos == -1:
            short_trailing_stop = min(short_trailing_stop, curr_price * (1 + trailing_pct / 100))
            if curr_price > curr_ema or curr_price > short_trailing_stop:
                pnl = short_price / curr_price - 1 - 2 * fee / 100
                balance *= 1 + pnl * risk_percent / 100
                pos = 0

                pnl_array.append(pnl)
                trade_count += 1
                if pnl > 0:
                    win_count += 1
                exit_points.append(i)
                exit_prices.append(curr_price)

        equity_curve.append(balance)

    if pos == 1:
        pnl = curr_price / long_price - 1 - 2 * fee / 100
        balance *= 1 + pnl * risk_percent / 100
        pnl_array.append(pnl)
        trade_count += 1
        if pnl > 0:
            win_count += 1
        exit_points.append(n - 1)
        exit_prices.append(price_arr[-1])
    elif pos == -1:
        pnl = short_price / curr_price - 1 - 2 * fee / 100
        balance *= 1 + pnl * risk_percent / 100
        pnl_array.append(pnl)
        trade_count += 1
        if pnl > 0:
            win_count += 1
        exit_points.append(n - 1)
        exit_prices.append(price_arr[-1])

    equity_curve = np.array(equity_curve[:n])
    returns = np.diff(np.log(equity_curve))

    if len(returns) > 1 and np.std(returns) > 0:
        periods_per_year = 365.25 * 24 * 4  # 15-минутные бары
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    win_ratio = win_count / trade_count if trade_count else 0

    if show_plots:
        time = np.arange(n) if timestamps is None else timestamps

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax1.plot(time, price_arr, label='Price', alpha=0.7)
        ax1.plot(time, ema_arr, label='EMA', color='orange', alpha=0.8)
        ax1.scatter(np.array(time)[long_entries], np.array(long_entry_prices),
                    marker='^', color='g', label='Long Entry', s=70)
        ax1.scatter(np.array(time)[short_entries], np.array(short_entry_prices),
                    marker='v', color='purple', label='Short Entry', s=70)
        ax1.scatter(np.array(time)[exit_points], np.array(exit_prices),
                    marker='x', color='r', label='Exit', s=70)
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)

        ax2.plot(time, equity_curve, label='Equity Curve', color='blue', linewidth=2)
        ax2.set_ylabel('Equity')
        ax2.set_xlabel('Time')
        ax2.legend(loc='upper left')
        ax2.grid(True)

        plt.suptitle('Price + EMA + Entries/Exits & Equity Curve')
        plt.savefig(f"equity_{test_start.date()}_{test_end.date()}.png")
        plt.close()

    return balance / 10000 - 1, trade_count, sharpe, win_ratio

def objective(trial, df):
    df = df.copy()

    EMA_length = trial.suggest_int('EMA_length', 10, 300)
    volume_ma_period = trial.suggest_int('volume_ma_period', 10, 300)
    roc_period = trial.suggest_int('roc_period', 10, 300)
    roc_threshold = trial.suggest_float('roc_threshold', 0.5, 10)
    roc_std_window = trial.suggest_int('roc_std_window', 10, 300)
    trailing_pct = trial.suggest_float('trailing_pct', 0.5, 10)
    ema_threshold = trial.suggest_float('ema_threshold', 1, 5)

    df['EMA'] = df['close'].ewm(span=EMA_length, adjust=False).mean()
    df['Volume_MA'] = df['volume'].rolling(volume_ma_period).mean()
    df['ROC'] = (df['close'] - df['close'].shift(roc_period)) / df['close'].shift(roc_period) * 100
    df['ROC_std'] = df['ROC'].rolling(roc_std_window).std()
    df['EMA_std'] = df['close'].rolling(EMA_length).std()
    df = df.dropna()

    return backtest_numba(
        df['close'].values,
        df['EMA'].values,
        df['ROC'].values,
        df['ROC_std'].values,
        df['volume'].values,
        df['Volume_MA'].values,
        df['EMA_std'].values,
        roc_threshold,
        trailing_pct,
        ema_threshold
    )

def generate_walkforward_windows(df, train_months=6, test_months=3):
    windows = []
    start_date = df.index.min()
    end_date = df.index.max()

    current_start = start_date

    while True:
        train_start = current_start
        train_end = train_start + relativedelta(months=train_months)
        test_start = train_end
        test_end = test_start + relativedelta(months=test_months)

        if test_end > end_date:
            break

        windows.append((train_start, train_end, test_start, test_end))
        current_start = train_start + relativedelta(months=test_months)

    return windows



df = pd.read_csv("BTCUSDT_1m.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
df = df.resample('15min').agg({
    'open':'first',
    'high':'max',
    'low':'min',
    'close':'last',
    'volume':'sum'
}).dropna()


windows = generate_walkforward_windows(df)

test_results = []
for train_start, train_end, test_start, test_end in windows:
    print(f"\nПериод: {train_start.date()} — {train_end.date()} (train), {test_start.date()} — {test_end.date()} (test)")

    train_df = df.loc[train_start:train_end].copy()
    test_df = df.loc[test_start:test_end].copy()

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_df), n_trials=1000, n_jobs=1)

    best_params = study.best_params
    print(best_params)
    EMA_length = best_params['EMA_length']
    volume_ma_period = best_params['volume_ma_period']
    roc_period = best_params['roc_period']
    roc_threshold = best_params['roc_threshold']
    roc_std_window = best_params['roc_std_window']
    trailing_pct = best_params['trailing_pct']
    ema_threshold = best_params['ema_threshold']
    test_df['EMA'] = test_df['close'].ewm(span=EMA_length, adjust=False).mean()
    test_df['Volume_MA'] = test_df['volume'].rolling(volume_ma_period).mean()
    test_df['ROC'] = (test_df['close'] - test_df['close'].shift(roc_period)) / test_df['close'].shift(roc_period) * 100
    test_df['ROC_std'] = test_df['ROC'].rolling(roc_std_window).std()
    test_df['EMA_std'] = test_df['close'].rolling(EMA_length).std()
    test_df = test_df.dropna()

    test_return, n_trades, sharpe, win_ratio = test(
        test_df['close'].values,
        test_df['EMA'].values,
        test_df['ROC'].values,
        test_df['ROC_std'].values,
        test_df['volume'].values,
        test_df['Volume_MA'].values,
        test_df['EMA_std'].values,
        roc_threshold,
        trailing_pct,
        ema_threshold,
        timestamps=test_df.index,
        show_plots=True
    )
    print("Доходность на тсесте:", test_return*100, "%")
    print(f"Годовая доходность на тесте:", ((1 + test_return) ** (1 / 0.25) - 1)*100, "%")
    print(f"Коэффициент Шарпа на тесте: {sharpe}")
    print(f"Число сделок на тесте: {n_trades}")
    print(f"Процент выигрышных сделок на тесте: {win_ratio * 100}%")

    test_results.append({
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'annualized test_return_%': ((1 + test_return) ** (1 / 0.25) - 1)*100,
        'n_trades': n_trades,
        'sharpe': sharpe,
        'win_rate_%': win_ratio,
        **best_params
    })


results_df = pd.DataFrame(test_results)
results_df.to_csv("wolk_forward_optimization.csv")
print("\nРезультаты Walk-Forward:")
print(results_df[['train_start', 'train_end', 'test_start', 'test_end', 'annualized test_return_%']])
print(f"\nСредняя годовая доходность по всем тестам: {results_df['annualized test_return_%'].mean():.2f}%")


