import numpy as np
import pandas as pd
import dash_html_components as html
import colorlover as cl
from fbprophet import Prophet

df = pd.read_csv('top_5_product_recommender.csv', low_memory=False).iloc[:,1:]
products = list(df.Product_name)
jan_forecast = pd.read_csv('jan_forecast.csv')
feb_forecast = pd.read_csv('feb_forecast.csv')

product_dropdown = []
for product in products:
    product_dict = {'label': product, 'value': product}
    product_dropdown.append(product_dict)


#colorscale
rdpu = cl.scales['6']['seq']['PuRd']
ryb_nums = cl.to_numeric(rdpu)
rgb = 'rgb'
rgb_nums = [rgb+str(num) for num in ryb_nums]
percents = [0,.2,.4,.6,.8,1.0]
custom_colorscale = list(map(list,list(zip(percents,rgb_nums))))

colorscale_8_colors = [[0,'#003f5c'],
[0.14, '#2f4b7c'],
[0.28, '#665191'],
[0.42, '#a05195'],
[0.57, '#d45087'],
[0.70, '#f95d6a'],
[0.84, '#ff7c43'],
[1.0, '#ffa600']]

colorscale_6_colors = [[0,'#003f5c'],[0.2, '#444e86'],[0.4, '#955196'],[0.6, '#dd5182'],[0.8, '#ff6e54'],[1.0, '#ffa600']]


################################################################################################
#High Orders/Recommender EDA
categories = pd.read_csv('product_categories.csv')
categories.columns = ['product_category', 'revenue', 'quantity']
categories=categories.drop([4])

################################################################################################
#Web Traffic forecasting EDA

df = pd.read_csv('webtraffic_time.csv')
datetime = pd.DataFrame(pd.date_range(start = '2017-03-01', freq = 'H', end = '2019-01-30'))
hourly_traffic = pd.concat([df, datetime], axis =1)
hourly_traffic.drop(hourly_traffic.index[-1], inplace=True)
hourly_traffic = hourly_traffic.drop(columns = ['Hour Index'])
hourly_traffic.columns = (['Users', 'Datetime'])
hourly_traffic = hourly_traffic.set_index('Datetime')
hourly_traffic.Users = pd.to_numeric(hourly_traffic['Users'])

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df = df.copy()
    df['date'] = df.index
    df['weekday'] = df['date'].dt.day_name()
    df['weekday'] = pd.Categorical(df['weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)
    df['users'] = df.Users
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear

    X = df[['weekday','users','hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    return X

full_dates = create_features(hourly_traffic)

################################################################################################
#Web Traffic Forecasting Timeseries Using Prophet

# split_date = '01-Sept-2018'
# traffic_train = full_dates.loc[full_dates.index <= split_date].copy()
# traffic_test = full_dates.loc[full_dates.index > split_date].copy()
# # Format data for prophet model using ds and y
# traffic_train.reset_index().rename(columns={'Datetime':'ds', 'users':'y'}).head()
# #Set up and train model
# model = Prophet()
# model.fit(traffic_train.reset_index().rename(columns={'Datetime':'ds', 'users':'y'}))
# # Predict on training set with model
# traffic_test_forecast = model.predict(df=traffic_test.reset_index().rename(columns={'Datetime':'ds'}))
# jan_forecast = traffic_test_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][(traffic_test_forecast.ds > '2019-01-01')]

jantrace1 = {
  "x": list(jan_forecast.ds),
  "y": list(jan_forecast.users),
  "line": {"color": "plum"},
  "mode": "markers",
  "name": "actual no. of users",
  "type": "scatter"
}
jantrace2 = {
  "x": list(jan_forecast.ds),
  "y": list(jan_forecast.yhat_upper),
  "fill": "tonexty",
  "line": {"color": "#57b8ff"},
  "mode": "lines",
  "name": "upper_band",
  "type": "scatter"
}
jantrace3 = {
  "x": list(jan_forecast.ds),
  "y": list(jan_forecast.yhat_lower),
  "fill": "tonexty",
  "line": {"color": "#57b8ff"},
  "mode": "lines",
  "name": "lower_band",
  "type": "scatter"
}
jantrace4 = {
  "x": list(jan_forecast.ds),
  "y": list(jan_forecast.yhat),
  "line": {"color": "mediumvioletred"},
  "mode": "lines",
  "name": "model line of best fit",
  "type": "scatter"
}

#Forecast first week of February
# m = Prophet()
# m.fit(full_dates.reset_index().rename(columns={'Datetime':'ds', 'users':'y'}))
# future = m.make_future_dataframe(periods=240, freq="1h")
# future_forecast = m.predict(future)
# feb_forecast = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][(future_forecast.ds > '2019-01-22')]

febtrace1 = {
  "x": list(feb_forecast.ds),
  "y": list(full_dates['2019-01-22':]['users']),
  "line": {"color": "plum"},
  "mode": "markers",
  "name": "actual no. of users",
  "type": "scatter"
}
febtrace2 = {
  "x": list(feb_forecast.ds),
  "y": list(feb_forecast.yhat_upper),
  "fill": "tonexty",
  "line": {"color": "#57b8ff"},
  "mode": "lines",
  "name": "upper_band",
  "type": "scatter"
}
febtrace3 = {
  "x": list(feb_forecast.ds),
  "y": list(feb_forecast.yhat_lower),
  "fill": "tonexty",
  "line": {"color": "#57b8ff"},
  "mode": "lines",
  "name": "lower_band",
  "type": "scatter"
}
febtrace4 = {
  "x": list(feb_forecast.ds),
  "y": list(feb_forecast.yhat),
  "line": {"color": "mediumvioletred"},
  "mode": "lines",
  "name": "model line of best fit",
  "type": "scatter"
}

################################################################################################
#Revenue forecasting EDA

df = pd.read_csv('revenue_time.csv')
df['Day Index'] = pd.to_datetime(df['Day Index'])
df['Revenue'] = df['Revenue'].replace('[\$,]', '', regex=True).astype(float)
df['Revenue'] = pd.to_numeric(df['Revenue'])
df['Weekday'] = df['Day Index'].dt.day_name()
df['Weekday'] = pd.Categorical(df['Weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'], ordered=True)
df['Month'] = df['Day Index'].dt.month
# df['Monthname'] = df['Day Index'].dt.month_name()
# df['Monthname'] = pd.Categorical(df['Monthname'], categories=['Jan','Feb','March','April','May','June', 'July', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec'], ordered=True)

revenue = df.set_index('Day Index')
revenue = revenue[:'2019-01-20']
revenue_2018 = revenue['2018']

################################################################################################
#Web Traffic Forecasting Timeseries Sarimax
rev_forecast = pd.read_csv('rev_forecast_and_actuals.csv')
rev_forecast.columns = ['date', 'predicted_revenue', 'revenue', 'lower_rev', 'upper_rev']
rev_forecast.date = pd.to_datetime(rev_forecast.date)


rev1 = {
  "x": list(rev_forecast.date),
  "y": list(rev_forecast.upper_rev),
  "fill": "tonexty",
  "line": {"color": "#ffd29f"},
  "mode": "lines",
  "name": "upper_band",
  "type": "scatter"
}
rev2 = {
  "x": list(rev_forecast.date),
  "y": list(rev_forecast.lower_rev),
  "fill": "tonexty",
  "line": {"color": "#ffd29f"},
  "mode": "lines",
  "name": "lower_band",
  "type": "scatter"
}
rev3 = {
  "x": list(rev_forecast.date),
  "y": list(rev_forecast.predicted_revenue),
  "line": {"color": "#bc5090"},
  "mode": "lines",
  "name": "model line of best fit",
  "type": "scatter"
}
rev4 = {
  "x": list(rev_forecast.date),
  "y": list(rev_forecast.revenue),
  "line": {"color": "#003f5c"},
  "mode": "markers",
  "name": "actual revenue",
  "type": "scatter"
}


################################################################################################
#Large Order analysis
transactions = pd.read_csv('transaction_source_channel_region_revenue.csv')
transactions['Transaction_revenue'] = transactions['Transaction_revenue'].replace('[\$,]', '', regex=True).astype(float)
transactions['Transaction ID'] = transactions['Transaction ID'].replace('[\#,]', '', regex=True).astype('float')
transactions['Transaction ID'] = transactions['Transaction ID'].replace('[\#,]', '', regex=True).astype('int64')
import re
under_20 = transactions[transactions['Transaction_revenue']<20]
under_75 = transactions[transactions['Transaction_revenue']<75]
over_75 = transactions[transactions['Transaction_revenue']>75]
over_200 = transactions[transactions['Transaction_revenue']>200]
