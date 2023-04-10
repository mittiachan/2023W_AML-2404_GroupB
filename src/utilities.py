from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import os
import time

import prophet as prophet
from prophet.plot import plot_plotly, plot_components_plotly

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


class Dataset:
    def __init__(self):
        self.socket = None
        self.dataset = None

    def build_dataset(self):
        start_date = datetime(2010, 1, 1).date()
        end_date = datetime.now().date()

        try:
            self.dataset = self.socket.history(start=start_date, end=end_date, interval="1d").reset_index()
            self.dataset['Date'] = self.dataset['Date'].dt.tz_convert(None)
            print('dataset downloaded')
            self.dataset.drop(columns=["Dividends", "Stock Splits", "Volume"], inplace=True)
        except Exception as e:
            print("Exception raised while building dataset: ", e)
            return False
        else:
            return True


class FeatureEngineering(Dataset):
    def create_features(self):
        status = self.build_dataset()
        if status:
            self.create_lag_fetaures()
            self.impute_missing_values()
            self.dataset.drop(columns=["Open", "High", "Low"], inplace=True)
            print(self.dataset.tail(3))
            return True
        else:
            raise Exception("Dataset creation failed!")

    def create_lag_fetaures(self, periods=12):
        for i in range(1, periods+1):
            self.dataset[f"Close_lag_{i}"] = self.dataset.Close.shift(periods=i, axis=0)
        return True

    def impute_missing_values(self):
        self.dataset.fillna(0, inplace=True)
        return True


class MasterProphet(FeatureEngineering):
    def __init__(self, ticker):
        self.ticker = ticker
        self.socket = yf.Ticker(self.ticker)
        self.validation_period = 30

    def build_model(self):
        additonal_features = [col for col in self.dataset.columns if "lag" in col]
        try:
            self.model = prophet.Prophet(yearly_seasonality=True, weekly_seasonality=True, seasonality_mode="additive")
            for name in additonal_features:
                self.model.add_regressor(name)
        except Exception as e:
            print("Exception raised while running MasterProphet.build_model", e)
            return False
        else:
            return True

    def add_forecast_date(self, date):
        present_date = date
        day_number = pd.to_datetime(present_date).isoweekday()
        if day_number in [5, 6]:
            forecast_date = present_date + timedelta(days=(7 - day_number) + 1)
        else:
            forecast_date = present_date + timedelta(days=1)
        return forecast_date

    def create_future_dates(self, model, last_set, n_days=10):
        prev = last_set
        for i in range(n_days):
            pred = model.predict(prev).yhat[0]
            next_day = prev.shift(periods=1, axis=1)
            next_day['ds'] = self.add_forecast_date(prev.ds[0])
            next_day['y'] = pred
            prev = next_day
        return next_day

    def train_and_forecast(self):
        n = self.validation_period
        df = self.dataset.reset_index()
        df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)
        train, validation = df.iloc[:-n], df.iloc[-n:]
        self.model.fit(df=train)
        validation = pd.concat([validation, self.create_future_dates(self.model,
                                                                     last_set=validation.tail(1).reset_index(
                                                                         drop=True), n_days=1)])
        #         return self.model.predict(validation[[col for col in df if col != "Close"]].rename(columns={"Date": "ds"}))
        return self.model.predict(validation)

    def create_plot(self, prediction):
        # Plotting predictions and uncertainity interval with respect to actuals
        plt.figure(figsize=(10, 6))

        # Plot actuals
        plt.plot(self.dataset.Date.iloc[-30:], self.dataset.Close.iloc[-30:])

        # Plot forecasts for the latest 365 days as validation period
        plt.plot(prediction.ds, prediction.yhat)

        # Plot uncertainty - lower and upper bound for the forecasts
        plt.fill_between(prediction.ds, prediction.yhat_lower, prediction.yhat_upper, alpha=0.2)

        plt.xlabel("Date")
        plt.ylabel("Close price of Stock")
        plt.legend(["Actual", "Forecast", "Forecast Uncertainty"])
        plt.title(f"{self.ticker}_Prophet_Model_Forecast_Analysis")
        new_graph_name = "graph_fb_" + str(time.time_ns()) + ".png"

        for filename in os.listdir('static/images/'):
            if filename.startswith('graph_fb_'):  # not to remove other images
                os.remove('static/images/' + filename)

        plt.savefig('static/images/' + new_graph_name)
        return 'static/images/' + new_graph_name

    def forecast(self):
        self.create_features()
        self.build_model()
        return self.train_and_forecast()


class MasterRegression(FeatureEngineering):
    def __init__(self, ticker):
        self.ticker = ticker
        self.socket = yf.Ticker(self.ticker)
        self.validation_period = 30

    def build_model(self):
        self.additonal_features = [col for col in self.dataset.columns if "Close_lag" in col]
        try:
            self.lin_model = LinearRegression()
            self.rfg_model = RandomForestRegressor()
        except Exception as e:
            print("Exception raised while running MasterRegression.build_model", e)
            return False
        else:
            return True

    def add_forecast_date(self, date):
        present_date = date
        day_number = pd.to_datetime(present_date).isoweekday()
        if day_number in [5, 6]:
            forecast_date = present_date + timedelta(days=(7-day_number) + 1)
        else:
            forecast_date = present_date + timedelta(days=1)
        return forecast_date

    def create_future_dates_reg(self, model, last_set, n_days=10, m=1):
        prev = last_set
        next_week = []
        for i in range(n_days):
            pred = model.predict([prev])
            print('--------------------------------------\n', pred[0], prev[:-1])
            if m == 1: next_day = np.concatenate((pred[0], prev[:-1]))
            if m == 2: next_day = np.concatenate((pred, prev[:-1]))
            next_week.append(next_day)
            prev = next_day
        return next_week

    def fit_train(self):
        x = np.concatenate([np.array(self.dataset[x]).reshape(-1, 1) for x in self.additonal_features], axis=1)
        y = np.array(self.dataset['Close']).reshape(-1, 1)
        n = self.validation_period
        self.X_train, self.X_test, self.y_train, self.y_test = x[:-n], x[-n:], y[:-n], y[-n:]
        self.lin_model.fit(self.X_train, self.y_train)
        self.rfg_model.fit(self.X_train, self.y_train)

    def create_plot(self, prediction):
        n = self.validation_period
        # Plotting predictions and uncertainity interval with respect to actuals
        plt.figure(figsize=(10, 6))

        # Plot actuals
        plt.plot(self.dataset.Date.iloc[-n:], self.y_test, label='Actual')

        # Plot forecasts for the LinearRegression model
        plt.plot(self.dataset.Date.iloc[-n:], prediction[0][:-1], label='Linear Regression')

        # Plot forecasts for the RandomForestRegressor model
        plt.plot(self.dataset.Date.iloc[-n:], prediction[1][:-1], label='Random Forest Regressor')

        plt.xlabel("Date")
        plt.ylabel("Close price of Stock")
        plt.legend()
        plt.title(f"{self.ticker}_Regression_Model_Forecast_Analysis")
        new_graph_name = "graph_reg_" + str(time.time_ns()) + ".png"

        for filename in os.listdir('static/images/'):
            if filename.startswith('graph_reg_'):  # not to remove other images
                os.remove('static/images/' + filename)

        plt.savefig('static/images/' + new_graph_name)
        return 'static/images/' + new_graph_name

    def reg_forecast(self):
        self.create_features()
        self.build_model()
        self.fit_train()
        lin_Xtest = np.concatenate((self.X_test, self.create_future_dates_reg(model=self.lin_model, last_set=self.X_test[-1], n_days=1)))
        rfg_Xtest = np.concatenate((self.X_test, self.create_future_dates_reg(model=self.rfg_model, last_set=self.X_test[-1], n_days=1, m=2)))
        self.lin_pred = self.lin_model.predict(lin_Xtest)
        self.rfg_pred = self.rfg_model.predict(rfg_Xtest)
        return (self.lin_pred, self.rfg_pred)


def plot_resampled_data(data):
    y = data[["Date", "Close"]]
    y["Date"] = pd.to_datetime(y['Date'])
    y.set_index("Date", inplace=True)
    plt.figure(figsize=(20, 10))
    plt.plot(y, marker='.', linestyle='-', linewidth=0.5, label='Daily')
    plt.plot(y.resample('W').mean(), marker='*', markersize=8, linestyle='-', label='Weekly Mean Resample')
    plt.plot(y.resample('M').mean(), marker='o', markersize=8, linestyle='-', label='Monthly Mean Resample')
    plt.xlabel("Date")
    plt.ylabel("Close price")
    plt.title(f"Resampled Mean Data Plot")
    plt.legend()

    new_graph_name = "graph_rmd_" + str(time.time_ns()) + ".png"
    for filename in os.listdir('static/images/'):
        if filename.startswith('graph_rmd_'):  # not to remove other images
            os.remove('static/images/' + filename)
    plt.savefig('static/images/' + new_graph_name)
    return 'static/images/' + new_graph_name
