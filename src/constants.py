import os

MASTER_THESIS_DIR = os.environ.get("MASTER_THESIS_PATH")
DATA_PATH = os.environ.get("DATA_PATH")
FREIHAND_DATA = os.path.join(DATA_PATH, "raw", "FreiHAND_pub_v2")
TRAINING_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "training_config.json"
)
MODEL_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "model_config.json"
)
ANGLES = [i for i in range(10, 360, 10)]
SAVED_MODELS_BASE_PATH = os.environ.get("SAVED_MODELS_BASE_PATH")
STD_LOGGING_FORMAT = "%(name)s -%(levelname)s - %(message)s"
