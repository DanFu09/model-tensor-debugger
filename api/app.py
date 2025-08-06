import sys
import os
# Add the parent directory to the path so we can import from the main app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the Flask app from the main directory
from app import app

# Vercel expects a callable named 'app' or 'handler'
# The Flask app object serves as the WSGI application
handler = app

if __name__ == "__main__":
    app.run()