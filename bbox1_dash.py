import numpy as np
import json

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from plotly.subplots import make_subplots

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

with open('assets/bbox1/42/x_train_trf.json', 'r') as json_file:
    train = json.load(json_file)
with open('assets/bbox1/42/x_test_trf.json', 'r') as json_file:
    test = json.load(json_file)

data = {
    'train': train,
    'test': test
}

app.layout = html.Div([
    html.Div([
            html.H1(children='Black Box 1 anomaly detection with PAE')],
        style={'display': 'inline-block', 
               'padding': '20px 10px'}),
    
    html.Div([
        html.Div([
            html.H2(children='Feature Transformation')],
            style={'width': '60%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='features-dataset',
                options=[{'label': i, 'value': j} for i, j in zip(['Train', 'Test'], ['train', 'test'])],
                value='train'
            ),
            dcc.RadioItems(
                id='features-transformed',
                options=[{'label': i, 'value': i} for i in ['Original', 'Quantile Transformer']],
                value='Quantile Transformer',
                labelStyle={'display': 'inline-block', 'margin-right': '20px'}
            )
        ],
        style={'width': '40%', 'float': 'right', 'display': 'inline-block'}),

        # html.Div([
        #     dcc.Dropdown(
        #         id='crossfilter-yaxis-column',
        #         options=[{'label': i, 'value': i} for i in available_indicators],
        #         value='Life expectancy at birth, total (years)'
        #     ),
        #     dcc.RadioItems(
        #         id='crossfilter-yaxis-type',
        #         options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
        #         value='Linear',
        #         labelStyle={'display': 'inline-block'}
        #     )
        # ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '20px 10px'
    }),

    html.Div([
        dcc.Graph(
            id='features-plot'
            ,config={'displayModeBar': False}
        )
    ], style={'width': '99%', 'display': 'inline-block', 'padding': '0 0'}),

    # html.Div([
    #     dcc.Graph(id='x-time-series'),
    #     dcc.Graph(id='y-time-series'),
    # ], style={'display': 'inline-block', 'width': '49%'}),

    # html.Div(dcc.Slider(
    #     id='crossfilter-year--slider',
    #     min=df['Year'].min(),
    #     max=df['Year'].max(),
    #     value=df['Year'].max(),
    #     marks={str(year): str(year) for year in df['Year'].unique()},
    #     step=None
    # ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


@app.callback(
    dash.dependencies.Output('features-plot', 'figure'),
    [dash.dependencies.Input('features-transformed', 'value'),
     dash.dependencies.Input('features-dataset', 'value')])
def update_graph(transformed, data_key):
    dataset = data[data_key]
    n = len(dataset['counts'])
    cols = n//4
    rows = n//cols +1
    color = 'steelblue' if transformed == 'Quantile Transformer' else 'darkorange'

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=dataset['name'])
    for i in range(n):
        counts = dataset['counts'][i]
        bins = dataset['bins'][i]
        feature = dataset['name'][i]
        fig.add_trace(
            go.Bar(x=bins, y=counts,
                   name=feature, marker_color=color),
            row=i//cols+1, col=i%cols+1
        )
    #fig.update_traces(customdata=features)
    fig.update_layout(
                    margin={'l': 40, 'b': 40, 't': 40, 'r': 0},
                    bargap=0,
                    font=dict(size=11),
                    showlegend=False)
    return fig


# def create_time_series(dff, axis_type, title):

#     fig = px.scatter(dff, x='Year', y='Value')

#     fig.update_traces(mode='lines+markers')

#     fig.update_xaxes(showgrid=False)

#     fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

#     fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
#                        xref='paper', yref='paper', showarrow=False, align='left',
#                        bgcolor='rgba(255, 255, 255, 0.5)', text=title)

#     fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

#     return fig


# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)


# @app.callback(
#     dash.dependencies.Output('y-time-series', 'figure'),
#     [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
#      dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
#      dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#     dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#     dff = dff[dff['Indicator Name'] == yaxis_column_name]
#     return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run_server(debug=True)