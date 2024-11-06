# from dash import html, dcc
# from data import fetch_data
# import re

# # Define the sorting key function outside create_layout (or inside if preferred)
# def natural_sort_key(name):
#     return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]


# def create_layout():
#     df = fetch_data()
#     sorted_names = sorted(df['name'].unique(), key=natural_sort_key)
#     layout = html.Div([
#         dcc.Location(id='url', refresh=False),

#         html.Meta(name='viewport', content='width=device-width, initial-scale=3'),
#         html.Link(rel='shortcut icon', href='/assets/favicon_io/favicon.ico'),
#         html.H1("Alab Dashboard"),

#         dcc.Dropdown(
#             id='sample-name-dropdown',
#             options=[{'label': name, 'value': name} for name in sorted_names],
#             multi=True
#         ),
        
#         html.Div([
#             html.Div(id='target-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '15%'}),#45
#             html.Div(id='similar-experiments-box', style={'padding': '10px', 'margin': '10px', 'display': 'inline-block', 'width': '45%'}),
#         ], style={'display': 'flex', 'justify-content': 'space-between'}),
#         html.Div([
#             dcc.Graph(id='precursors-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block' }),
#             dcc.Graph(id='pie-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
#         ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
#         html.Div([
#             dcc.Graph(id='temperature-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
#             dcc.Dropdown(
#                 id='comparable-sample-dropdown',
#                 options=[{'label': name, 'value': name} for name in df['name'].unique()],
#                 placeholder="Select a comparable sample",
#                 style={'width': '5%', 'height': '5%', 'font-size': '14px'}  
#             ),
#             dcc.Graph(id='correlation-plot', style={'padding': '10px', 'margin': '10px','width': '45%', 'display': 'inline-block'}),
#         ], style={'display': 'flex', 'flex-wrap': 'wrap','justify-content': 'space-between'}),
#         html.H2("XRD characterization", style={'text-align': 'left'}),  
#         html.Div([
#             html.Div([
#                 dcc.Graph(id='xrd-plot', style={'padding': '10px', 'margin': '10px', 'width': '100%'})
#             ], style={'width': '40%', 'display': 'inline-block'}),
#             html.Div([
#                 html.Div(id='best_rwp_box', style={'padding': '10px', 'margin': '10px'}),
#                 html.Div(id='results_box', style={'padding': '10px', 'margin': '10px'})
#             ], style={'display': 'flex', 'flex-direction': 'column', 'height': '80%', 'width': '80%'})
#         ], style={'display': 'flex'}),
#         html.H2("SEM characterization", style={'text-align': 'left'}),  
#         html.Div([
#             html.Img(id='image1', style={'width': '35%', 'display': 'inline-block'}),
#             html.Img(id='image2', style={'width': '35%', 'display': 'inline-block'}),
#         ], style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'space-between'}),

#     ])
    
#     return layout

from dash import html, dcc
from data import fetch_data
import re

def natural_sort_key(name):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]

def create_layout():
    
    
    layout = html.Div([
        dcc.Store(id='saved-plots', data=[]),  # Store for saved plots

        dcc.Location(id='url', refresh=False),

        html.Meta(name='viewport', content='width=device-width, initial-scale=3'),
        html.Link(rel='shortcut icon', href='/assets/favicon_io/favicon.ico'),
        html.H1("Alab Dashboard"),

        # Tabs for main dashboard and saved plots
        dcc.Tabs(id="tabs", value='main-tab', children=[
            dcc.Tab(label='Main Dashboard', value='main-tab'),
            dcc.Tab(label='Saved Plots', value='saved-plots-tab')
        ]),

        html.Div(id='tab-content'),  # Content will update based on the selected tab
    ])
    
    return layout