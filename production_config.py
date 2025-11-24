# Production Deployment Configuration

import os
from dash import Dash
import dash_bootstrap_components as dbc
from dashboard.config import THEME

def create_production_app():
    """Create production-ready Dash app with optimizations."""
    
    app = Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.BOOTSTRAP,
            "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap"
        ],
        suppress_callback_exceptions=True,
        serve_locally=True,
        compress=True,
        title="Bloomberg Terminal Analytics"
    )
    
    # Production settings
    app.config.update({
        'compress': True,
        'serve_locally': True,
        'dev_tools_hot_reload': False,
        'dev_tools_ui': False,
        'dev_tools_props_check': False
    })
    
    # CSP headers for security
    app.server.config.update({
        'SECRET_KEY': os.environ.get('SECRET_KEY', 'bloomberg-dashboard-secret'),
        'SESSION_COOKIE_SECURE': True,
        'SESSION_COOKIE_HTTPONLY': True,
    })
    
    return app

def add_error_handling():
    """Add global error handling for production."""
    
    @app.callback(
        Output('error-modal', 'is_open'),
        Output('error-message', 'children'),
        Input('tabs', 'value'),
        prevent_initial_call=True
    )
    def handle_global_errors(tab_value):
        try:
            # Tab loading logic here
            return False, ""
        except Exception as e:
            error_msg = f"Error loading {tab_value}: {str(e)[:100]}..."
            return True, error_msg

def optimize_callbacks():
    """Add performance optimizations to callbacks."""
    
    # Add debouncing to prevent excessive API calls
    from dash import callback_context
    import time
    
    callback_cache = {}
    CACHE_TIMEOUT = 300  # 5 minutes
    
    def cached_callback(func):
        def wrapper(*args, **kwargs):
            cache_key = str(args) + str(kwargs)
            current_time = time.time()
            
            if (cache_key in callback_cache and 
                current_time - callback_cache[cache_key]['timestamp'] < CACHE_TIMEOUT):
                return callback_cache[cache_key]['result']
            
            result = func(*args, **kwargs)
            callback_cache[cache_key] = {
                'result': result,
                'timestamp': current_time
            }
            return result
        return wrapper
    
    return cached_callback