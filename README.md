This strategy uses EMA and ROC indicators to generate long/short signals. 
Optimization of parametrs is done using optuna and numba for faster cycles. 
Walk-forwart optimization principle is used. Parametrs are tuned on larger windows (6 month by default) and then the strategy is tested on the next 3 month window.
