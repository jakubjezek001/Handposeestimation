import os

MASTER_THESIS_DIR = os.environ.get("MASTER_THESIS_PATH")
DATA_PATH = os.environ.get("DATA_PATH")
FREIHAND_DATA = os.path.join(DATA_PATH, "raw", "FreiHAND_pub_v2")
TRAINING_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "training_config.json"
)
