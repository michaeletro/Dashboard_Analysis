from __future__ import annotations

from dashboard.app_instance import app  # adjust package name if needed
import dashboard.callbacks  # noqa F401  ensure callbacks register


if __name__ == "__main__":
    app.run(debug=True, port=8051)
