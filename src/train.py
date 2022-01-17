import pandas as pd
import config
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima


class ARIMA:
    def __init__(self, timeseries):
        self.timeseries = timeseries

    def arimamodel(self):
        """
        AR(p): Linear combination Lags of Y
        MA(q): Linear combination of Lagged forecast errors
        D(d): Number of differencing required to make the time series stationary
        """
        model = auto_arima(
            train,
            start_p=1,
            start_q=1,
            test="adf",
            d=None,
            max_p=2,
            max_q=2,
            trace=True,
        )
        return model

    def prediction(self, timeseries):
        test = pd.DataFrame(timeseries)
        test["ARIMA"] = self.arimamodel(train).predict(len(test), index=test.index)
        return test

if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE_CLEAN)
    train, test = train_test_split(df, test_size=0.20, shuffle=False)
    pred = ARIMA.prediction(test)
    print(f"MAE:{mean_squared_error(test.target, test.ARIMA):.2f}%")