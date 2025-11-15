This strategy uses EMA and ROC indicators to generate long/short signals. 

Optimization of parametrs is done using optuna and numba for faster cycles.   

Walk-forwart optimization principle is used. Parametrs are tuned on larger windows (6 month by default) and then the strategy is tested on the next 3 month window.

Link to the BTCUSDT 1min timeframe data that can be used for the strategy: https://drive.google.com/file/d/1U6JLJEKK35SG_QGbY6LM-XSjFVR4HZuj/view?usp=sharing
