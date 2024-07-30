from dash import html, dcc
from data import fetch_data

def create_layout():
    df = fetch_data()
    
    layout = html.Div([
        dcc.Location(id='url', refresh=False),

        html.Meta(name='viewport', content='width=device-width, initial-scale=3'),
        html.Link(rel='shortcut icon', href='/assets/favicon_io/favicon.ico'),
        html.H1("Alab Dashboard"),

        dcc.Dropdown(
            id='sample-name-dropdown',
            options=[{'label': name, 'value': name} for name in df['name'].unique()],
            multi=True
        ),
        
        html.Div([
            html.Div(id='target-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '45%'}),
            html.Div(id='similar-experiments-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '45%'}),
        ], style={'display': 'flex', 'justify-content': 'space-between'}),
        html.Div([
            dcc.Graph(id='precursors-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block' }),
            dcc.Graph(id='pie-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
        html.Div([
            dcc.Graph(id='temperature-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
            dcc.Graph(id='correlation-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
        html.H2("XRD characterization", style={'text-align': 'left'}),  
        html.Div([
            html.Div([
                dcc.Graph(id='xrd-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'})
            ], style={'width': '40%', 'display': 'inline-block'}),
            html.Div([
                html.Div(id='best_rwp_box', style={'padding': '10px', 'margin': '10px'}),
                html.Div(id='results_box', style={'padding': '10px', 'margin': '10px'})
            ], style={'display': 'flex', 'flex-direction': 'column', 'height': '80%', 'width': '80%'})
        ], style={'display': 'flex'}),
        html.H2("SEM characterization", style={'text-align': 'left'}),  
        html.Div([
            html.Img(id='image1', style={'width': '35%', 'display': 'inline-block'}),
            html.Img(id='image2', style={'width': '35%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'}),

    ])
    
    return layout