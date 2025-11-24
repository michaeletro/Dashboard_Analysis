from __future__ import annotations

import dash

from .layouts import create_root_layout, index_string
from .config import APP_TITLE

# create Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    title=APP_TITLE,
)

server = app.server

# use the factory function, not the result
# Dash will call this every time it needs a fresh layout
app.layout = create_root_layout

# custom index template from layouts
app.index_string = index_string

# if you have a callbacks module that registers all callbacks,
# import it here so they attach to `app`
try:
    from . import callbacks  # noqa: F401, E402
except ImportError:
    # ok if you do not have a callbacks module yet
    pass

if __name__ == "__main__":
    app.run_server(
        debug=True,
        host="0.0.0.0",
        port=8050,
    )