import os

# Proxy script that runs app.py code directly.
# This ensures compatibility with Streamlit Community Cloud deployments pointing to main.py.
app_path = os.path.join(os.path.dirname(__file__), "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()
exec(code, globals())
