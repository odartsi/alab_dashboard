from dash import Dash
from layout import create_layout
from callbacks import register_callbacks

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Alab Data Dashboard"

# Set up the layout
app.layout = create_layout()

# Register callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run_server(debug=True)
