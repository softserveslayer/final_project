# https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import colorlover as cl
from staghen_queries import *
bupu = cl.scales['9']['seq']['PuRd']



external_stylesheets = ['https://codepen.io/softserveslayer/pen/bzdoLL.css']

df = pd.read_csv('top_5_product_recommender.csv', low_memory=False).iloc[:,1:]
new_df = df.T
new_df.columns = new_df.iloc[0]
new_df = new_df.iloc[1:,:]
products = list(df.Product_name)
app = dash.Dash(external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    children=[
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(id='product-to-product',
        label='Product Recommendation',
            children=[
                dcc.Dropdown(
                id = 'product_dropdown',
                options=product_dropdown,
                value= 'Pineapple Snack Plates',
                style={'color': '#665191', 'fontSize': 24}
            ),
            html.Div(
            html.P('The top 5 recommended products are:'),
                style={'color': '#003f5c', 'fontSize': 20}),
            html.Div(
            html.P([html.P(product) for product in new_df['Pineapple Snack Plates']],
            id = 'top_products'),
            style={'color': '#cd7eaf', 'fontSize': 20}
            ),
            dcc.Graph(
                figure = go.Figure(
                data = [
                    go.Bar(
                        x=list(categories.product_category),
                        y=list(categories.revenue),
                        name = 'product revenue',
                        marker = {'color' : '#003f5c'}
                        ),
                    go.Scatter(
                        x=list(categories.product_category),
                        y=list(categories.quantity),
                        name = 'product quantity',
                        yaxis = 'y2',
                        line = {'color' : '#ffa600'})
                        ],
                layout = go.Layout(
                    title='Product Category Revenue vs. Quantity',
                    titlefont = {'family':'Georgia', 'size': 18},
                    legend=dict(x=-.1, y=1.2),
                    yaxis=dict(
                        title='Revenue',
                        titlefont=dict(
                            size=16
                        ),
                        tickfont=dict(
                            size=14
                        )),
                    yaxis2 = dict(
                        title = 'Quantity',
                        titlefont=dict(
                            size=16
                        ),
                        tickfont=dict(
                            size=14
                        ),
                        overlaying = 'y',
                        side = 'right'
                    )
                )
            ))
            ])
            ,
        dcc.Tab(id = 'web_traffic_forecast',
        label = 'Web Traffic Forecast',
            children = [
                dcc.Graph(
                    figure = go.Figure(
                    data = [
                        go.Bar(
                            x=list(full_dates['2019']['users'].index),
                            y=list(full_dates['2019']['users'].values),
                            marker = {'color': list(full_dates['2019']['users'].values),
                                    'colorscale' : custom_colorscale})],
                    layout = go.Layout(
                        title='Jan 2019 Hourly Web Traffic',
                        titlefont = {'family':'Georgia', 'size': 18},
                        xaxis=dict(
                            title = 'Date',
                            tickfont=dict(
                                size=14,
                                color='rgb(107, 107, 107)'),
                            rangeselector=
                                    dict(
                                    buttons=list([
                                        dict(count=1,
                                             label='1 day',
                                             step='day',
                                             stepmode='backward'),
                                        dict(count=7,
                                             label='1 week',
                                             step='day',
                                             stepmode='backward'),
                                        dict(step='all')
                            ])
                        ),
                        rangeslider = dict(
                            visible = True
                        )

                        ),
                        yaxis=dict(
                            title='Users',
                            titlefont=dict(
                                size=16,
                                color='rgb(107, 107, 107)'
                            ),
                            tickfont=dict(
                                size=14,
                                color='rgb(107, 107, 107)'
                            )
                        )
                    )
                )),
                dcc.Graph(
                    figure = go.Figure(
                    data = [
                        go.Bar(
                            x=list(full_dates.groupby('hour').users.sum().index),
                            y=list(full_dates.groupby('hour').users.sum().values),
                            marker = {'color': list(full_dates.groupby('hour').users.sum().values),
                                    'colorscale' : custom_colorscale})],
                    layout = go.Layout(
                        title='Traffic Breakdown Per Hour',
                        titlefont = {'family':'Georgia', 'size': 18},
                        xaxis=dict(
                            title = 'Hour',
                            tickfont=dict(
                                size=14,
                                color='rgb(107, 107, 107)'),
                                ),
                        yaxis=dict(
                            title='Users',
                            titlefont=dict(
                                size=16
                            ),
                            tickfont=dict(
                                size=14
                            )
                        )
                    )
                )),
                dcc.Graph(
                    figure = go.Figure(
                    data = [
                        go.Bar(
                            x=list(full_dates.groupby('weekday').users.sum().index),
                            y=list(full_dates.groupby('weekday').users.sum().values),
                            marker = {'color': list(full_dates.groupby('weekday').users.sum().values),
                                    'colorscale' : custom_colorscale})],
                    layout = go.Layout(
                        title='Traffic Breakdown Per Day of Week',
                        titlefont = {'family':'Georgia', 'size': 18},
                        xaxis=dict(
                            title = 'Hour',
                            tickfont=dict(
                                size=14,
                                color='rgb(107, 107, 107)')

                        ),
                        yaxis=dict(
                            title='Users',
                            titlefont=dict(
                                size=16,
                                color='rgb(107, 107, 107)'
                            ),
                            tickfont=dict(
                                size=14,
                                color='rgb(107, 107, 107)'
                            )
                        )
                    )
                )),
                html.P("""The Root Mean Squared Error of our forecasts is 68.18. This tells us that our model was able to forecast
                the average hourly web traffic in the test set within 68 of the real number of users. The number of users on the site per hour ranges from 0-724.
                Upper and Lower bounds are set with uncertainty levels of 80%."""),
                dcc.Graph(
                    figure = go.Figure(
                    data = [jantrace1, jantrace2, jantrace3, jantrace4],
                    layout = go.Layout(
                        {
                        # "paper_bgcolor": "rgb(243, 243, 243)",
                        #   "plot_bgcolor": "rgb(243, 243, 243)",
                          "title": "FB Prophet Model: Forecast vs Actuals, Jan 2019",
                          "titlefont" : {'family':'Georgia', 'size': 18},
                          "xaxis": {
                            "gridcolor": "rgb(255, 255, 255)",
                            "gridwidth": 2,
                            "ticklen": 5,
                            "title": "Date",
                            "zerolinewidth": 1},
                          "yaxis": {
                            "gridcolor": "rgb(255, 255, 255)",
                            "gridwidth": 2,
                            "ticklen": 5,
                            "title": "Users",
                            "zerolinewidth": 1}
                          }
                    )
                )),
                dcc.Graph(
                    figure = go.Figure(
                    data = [febtrace1, febtrace2, febtrace3, febtrace4],
                    layout = go.Layout(
                        {
                        # "paper_bgcolor": "rgb(243, 243, 243)",
                        #   "plot_bgcolor": "rgb(243, 243, 243)",
                          "title": "FB Prophet Model Forecast Feb 1-8, 2019",
                          "titlefont" : {'family':'Georgia', 'size': 18},
                          "xaxis": {
                            "gridcolor": "LightGrey",
                            "gridwidth": 1,
                            "ticklen": 5,
                            "title": "Date",
                            "zerolinewidth": 1},
                          "yaxis": {
                            "gridcolor": "LightGrey",
                            "gridwidth": 1,
                            "ticklen": 5,
                            "title": "Users",
                            "zerolinewidth": 1}
                          }
                    )
                ))
                ])
            ,
            dcc.Tab(id='Revenue Forecast',
            label='Revenue Forecast',
                children=[
                    dcc.Graph(
                        figure = go.Figure(
                        data = [
                            go.Bar(
                                x=list(revenue.groupby('Weekday').Revenue.sum().index),
                                y=list(revenue.groupby('Weekday').Revenue.sum().values),
                                marker = {'color': list(revenue.groupby('Weekday').Revenue.sum().values),
                                        'colorscale' : colorscale_6_colors})],
                        layout = go.Layout(
                            title='Revenue Breakdown Per Day of Week',
                            titlefont = {'family':'Georgia', 'size': 18},
                            xaxis=dict(
                                title = 'Hour',
                                tickfont=dict(
                                    size=14,
                                    color='rgb(107, 107, 107)')

                            ),
                            yaxis=dict(
                                title='Revenue',
                                titlefont=dict(
                                    size=16,
                                    color='rgb(107, 107, 107)'
                                ),
                                tickfont=dict(
                                    size=14,
                                    color='rgb(107, 107, 107)'
                                )
                            )
                        )
                    )),
                    dcc.Graph(
                        figure = go.Figure(
                        data = [
                            go.Bar(
                                x=list(revenue_2018.groupby('Month').Revenue.sum().index),
                                y=list(revenue_2018.groupby('Month').Revenue.sum().values),
                                marker = {'color': list(revenue_2018.groupby('Month').Revenue.sum().values),
                                        'colorscale' : colorscale_8_colors})],
                        layout = go.Layout(
                            title='Revenue Breakdown Per Month of Year',
                            titlefont = {'family':'Georgia', 'size': 18},
                            xaxis=dict(
                                title = 'Month',
                                tickfont=dict(
                                    size=14,
                                    color='rgb(107, 107, 107)')

                            ),
                            yaxis=dict(
                                title='Revenue',
                                titlefont=dict(
                                    size=16,
                                    color='rgb(107, 107, 107)'
                                ),
                                tickfont=dict(
                                    size=14,
                                    color='rgb(107, 107, 107)'
                                )
                            )
                        )
                    )),
                    html.P("""The Root Mean Squared Error of our forecasts is 235.9. This tells us that our model was able to forecast
                    the average daily revenue in the test set within $235.9 of the real sales. Our daily revenue ranges from around 0 to 1,674.
                    Upper and Lower bounds are set with uncertainty levels of 80%."""),
                    dcc.Graph(
                        figure = go.Figure(
                        data = [rev1, rev2, rev3, rev4],
                        layout = go.Layout(
                            {
                            # "paper_bgcolor": "rgb(243, 243, 243)",
                            #   "plot_bgcolor": "rgb(243, 243, 243)",
                              "title": "SARIMAX Forecast",
                              "titlefont" : {'family':'Georgia', 'size': 18},
                              "xaxis": {
                                "gridcolor": "LightGrey",
                                "gridwidth": 1,
                                "ticklen": 5,
                                "title": "Date",
                                "zerolinewidth": 1,
                                "rangeselector":
                                dict(
                                buttons=list([
                                    dict(count=1,
                                         label='1m',
                                         step='month',
                                         stepmode='backward'),
                                    dict(count=3,
                                         label='3m',
                                         step='month',
                                         stepmode='backward'),
                                    dict(step='all')
                                ])
                            ),
                            "rangeslider":dict(
                                visible = True
                            ),
                            "type":'date'
                        },
                              "yaxis": {
                                "gridcolor": "LightGrey",
                                "gridwidth": 1,
                                "ticklen": 5,
                                "title": "Revenue",
                                "zerolinewidth": 1}
                              }
                        )
                    ))
                ]),
            dcc.Tab(id='Large Order Analysis',
            label='Large Order Analysis',
                children=[
                    dcc.Graph(
                        figure = go.Figure(
                        data = [{'x': list(transactions['Transaction_revenue']),
                                'type': 'histogram',
                                'xbins':{'start':0, 'end':3000, 'size':25},
                                'marker':{'color':'#58508d'}}],
                        layout = go.Layout(
                            title = "Histogram of Order Revenue Amounts",
                            titlefont = {'family':'Georgia', 'size': 18},
                            xaxis={"title":"Order Revenues"},
                            yaxis={'title':"Count"}
                        )
                        # data = go.Histogram(x=transactions['Product Revenue'])
                        )),
                    dcc.Graph(
                            figure = go.Figure(
                            data = [
                                {'values': list(pd.DataFrame(transactions.groupby(transactions['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).Transaction_revenue),
                                'labels':list(pd.DataFrame(transactions.groupby(transactions['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).index),
                                # 'domain':{'x': [.52, 1],'y': [.51, 1]},
                                # 'domain':{'x':[.75,.95]},
                                'name':'All Orders',
                                'hoverinfo':'label+percent+name',
                                'hole':.4,
                                'type': 'pie',
                                'title':'All orders',
                                'marker':{'colors':['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']}
                                }],
                            layout = go.Layout(
                                title = "Channel Breakdown of All Orders, March 2017 - Jan 2019",
                                titlefont = {'family':'Georgia', 'size': 18},
                                # annotations = [
                                #     {'text':'Orders under $25'},
                                #     {'text':'Orders under $75'},
                                #     {'text':'Orders over $75'},
                                #     {'text':'All orders'}
                                # ]
                                )
                            )
                            ),
                        html.P("""Overall, Google/Organic search brings in the majority of orders and revenue. These pie charts break down the overall order sum by channel.
                                    Looking at the bubble charts as the overall order amounts increase, you can see Organic Search increasing as the Order Size (by revenue) increases.
                                    Meanwhile, orders coming from Social and Direct channels see a steady decline.
                                    One possible explanation is that people are more likely to buy either more costly or a higher quantity of items if they are explicitly searching for them.
                                    In addition, people may be less likely to buy higher-priced or a higher quantity of items instinctually while scrolling through Instagram."""),
                        dcc.Graph(
                            figure = go.Figure(
                            data = [
                                {'values': list(pd.DataFrame(under_20.groupby(under_20['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).Transaction_revenue),
                                'labels':list(pd.DataFrame(under_20.groupby(under_20['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).index),
                                # 'domain':{'x': [0, .3],'y': [0, .49]},
                                'domain':{'x':[0,.20]},
                                'name':'Orders under $25',
                                'hoverinfo':'label+percent+name',
                                'hole':.4,
                                'type': 'pie',
                                'title':'Orders under $25',
                                'marker':{'colors':['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']}
                                },
                                {'values': list(pd.DataFrame(under_75.groupby(under_75['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).Transaction_revenue),
                                'labels':list(pd.DataFrame(under_75.groupby(under_75['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).index),
                                # 'domain':{'x': [.52, 1],'y': [0, .49]},
                                'domain':{'x':[.25,.45]},
                                'name':'Orders under $75',
                                'hoverinfo':'label+percent+name',
                                'hole':.4,
                                'type': 'pie',
                                'title':'Orders under $75',
                                'marker':{'colors':['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']}
                                },
                                {'values': list(pd.DataFrame(over_75.groupby(over_75['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).Transaction_revenue),
                                'labels':list(pd.DataFrame(over_75.groupby(over_75['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).index),
                                # 'domain':{'x': [0, .3],'y': [.51, 1]},
                                'domain':{'x':[.5,.70]},
                                'name':'Orders over $75',
                                'hoverinfo':'label+percent+name',
                                'hole':.4,
                                'type': 'pie',
                                'title':'Orders over $75',
                                'marker':{'colors':['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']}
                                },
                                {'values': list(pd.DataFrame(over_200.groupby(over_200['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).Transaction_revenue),
                                'labels':list(pd.DataFrame(over_200.groupby(over_200['Channel'])['Transaction_revenue'].sum().sort_values(ascending = False)).index),
                                # 'domain':{'x': [.52, 1],'y': [.51, 1]},
                                'domain':{'x':[.75,.95]},
                                'name':'All Orders',
                                'hoverinfo':'label+percent+name',
                                'hole':.4,
                                'type': 'pie',
                                'title':'Order over $200',
                                'marker':{'colors':['#003f5c','#444e86','#955196','#dd5182','#ff6e54','#ffa600']}
                                }],
                            layout = go.Layout(
                                title = "Channel Breakdown by Order Volume, March 2017 - Jan 2019",
                                titlefont = {'family':'Georgia', 'size': 18},
                                # annotations = [
                                #     {'text':'Orders under $25'},
                                #     {'text':'Orders under $75'},
                                #     {'text':'Orders over $75'},
                                #     {'text':'All orders'}
                                # ]
                                )
                            )
                            )
                            ])
                    ])
                    ])



@app.callback(
    Output(component_id='top_products', component_property='children'),
    [Input(component_id='product_dropdown', component_property='value')]
)
def update_output_div(input_value):
    return [html.P(product) for product in new_df[input_value]]

if __name__ == '__main__':
    app.run_server(debug=True)
