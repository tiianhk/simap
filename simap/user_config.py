import json
from pathlib import Path

from platformdirs import user_config_dir

USER_CONFIG_PATH = Path(user_config_dir("simap")) / "config.json"


def get_user_config():
    try:
        with open(USER_CONFIG_PATH, "r") as f:
            return json.load(f)
    except:
        return {}


def set_user_config(config: dict):
    if not USER_CONFIG_PATH.parent.exists():
        USER_CONFIG_PATH.parent.mkdir(parents=True)
    with open(USER_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
