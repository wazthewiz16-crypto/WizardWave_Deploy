import os

def get_project_root():
    """Returns the absolute path to the project root directory."""
    # Since this file is in src/utils/paths.py, the root is two levels up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def get_data_path(*args):
    """Returns absolute path to a file/dir in the data directory."""
    return os.path.join(get_project_root(), "data", *args)

def get_config_path(*args):
    """Returns absolute path to a file/dir in the config directory."""
    return os.path.join(get_project_root(), "config", *args)

def get_model_path(*args):
    """Returns absolute path to a file in data/models."""
    return get_data_path("models", *args)
