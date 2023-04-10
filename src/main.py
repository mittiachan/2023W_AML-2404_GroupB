from flask import Flask, render_template, request
import pandas as pd
from src.utilities import MasterProphet, MasterRegression, plot_resampled_data

app = Flask(__name__)

# @app.after_request
# def add_header(response):
#     response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
#     response.headers["Cache-Control"] = "public, max-age=0"
#     return response

# define options for the dropdown
# tickers = pd.read_excel('static/YahooTickerSymbols.xlsx', sheet_name=0, header=3, usecols="A:E")
# org_list = [(row['Ticker'], row['Name']) for index, row in tickers[tickers['Country'] == 'Canada'].iterrows()]


@app.route('/', methods=['GET', 'POST'])
def dropdown():
    # render the dropdown template with the options
    # return render_template('home.html', options=org_list)
    return render_template('index.html')


@app.route("/predict", methods=["POST", "GET"])
def predict():
    ticker = request.form["ticker"]
    master_prophet = MasterProphet(ticker)
    master_reg = MasterRegression(ticker)

    fb_forecast = master_prophet.forecast()
    reg_forecast = master_reg.reg_forecast()

    latest_news = master_prophet.socket.news
    news_toDisplay = []
    for news in latest_news:
        data = {'title': news['title'], 'link': news['link'], 'relatedTickers': news['relatedTickers']}
        news_toDisplay.append(data)

    # FB Prophet model
    actual_forecast = round(fb_forecast.yhat[0], 2)
    lower_bound = round(fb_forecast.yhat_lower[0], 2)
    upper_bound = round(fb_forecast.yhat_upper[0], 2)
    bound = round(((upper_bound - actual_forecast) + (actual_forecast - lower_bound) / 2), 2)

    fb_graph_image = master_prophet.create_plot(fb_forecast)

    # Regression model
    reg_graph_image = master_reg.create_plot(reg_forecast)

    return render_template(
        "output.html",
        ticker=ticker.upper(),
        rmd_img_url=plot_resampled_data(master_prophet.dataset),
        tables=[master_reg.socket.major_holders.to_html(classes='data', header=False, index=False)],
        news=news_toDisplay,
        min_date=master_prophet.dataset.Date[0].date(),
        max_date=master_prophet.dataset.Date[len(master_prophet.dataset.Date)-1].date(),
        forecast_date=fb_forecast.ds[len(fb_forecast)-1].date(),
        actual_cp=round(master_prophet.dataset.iloc[len(master_prophet.dataset.Date)-1].Close, 2),
        forecast_cp=round(fb_forecast.iloc[len(fb_forecast)-2].yhat, 2),
        forecast_nd_cp=round(fb_forecast.iloc[len(fb_forecast)-1].yhat, 2),
        bound=bound,
        fb_img_url=fb_graph_image,
        lin_forecast_cp=round(reg_forecast[0][-2][0], 2),
        rfg_forecast_cp=round(reg_forecast[1][-2], 2),
        lin_forecast_nd_cp=round(reg_forecast[0][-1][0], 2),
        rfg_forecast_nd_cp=round(reg_forecast[1][-1], 2),
        reg_img_url=reg_graph_image
    )


if __name__ == '__main__':
    app.run(debug=True)
