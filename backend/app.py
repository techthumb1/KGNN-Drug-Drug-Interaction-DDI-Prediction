"""Compatibility entry point for the audited application."""
from run import app

if __name__ == "__main__":
    raise RuntimeError(
        "Use python run.py or gunicorn run:app"
    )