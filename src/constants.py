import os

MASTER_THESIS_DIR = os.environ.get("MASTER_THESIS_PATH")
DATA_PATH = os.environ.get("DATA_PATH")
FREIHAND_DATA = os.path.join(DATA_PATH, "freihand_dataset")
INTERHAND_DATA = os.path.join(DATA_PATH, "InterHand")
YOUTUBE_DATA = os.path.join(DATA_PATH, "youtube_3d_hands", "data")
MPII_DATA = os.path.join(DATA_PATH, "mpii_dataset")
TRAINING_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "training_config.json"
)
SUPERVISED_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "supervised_config.json"
)
SIMCLR_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "simclr_config.json"
)
SSL_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "semi_supervised_config.json"
)
PAIRWISE_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "pairwise_config.json"
)
HYBRID1_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "hybrid1_config.json"
)
HYBRID2_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "hybrid2_config.json"
)
NIPS_A1_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "nips_a1_config.json"
)
NIPS_A2_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "nips_a2_config.json"
)
DOWNSTREAM_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "downstream_config.json"
)
HYBRID1_AUGMENTATION_CONFIG = os.path.join(
    MASTER_THESIS_DIR,
    "src",
    "experiments",
    "config",
    "hybrid1_augmentation_config.json",
)
HRNET_CONFIG = os.path.join(
    MASTER_THESIS_DIR, "src", "models", "HRnet", "hrnet_config.yaml"
)
# TODO: Add appropriate path
HEATMAP_CONFIG_PATH = os.path.join(
    MASTER_THESIS_DIR, "src", "experiments", "config", "heatmap_config.json"
)

ANGLES = [i for i in range(10, 360, 10)]
SAVED_MODELS_BASE_PATH = os.environ.get("SAVED_MODELS_BASE_PATH")
SAVED_META_INFO_PATH = os.environ.get("SAVED_META_INFO_PATH")
STD_LOGGING_FORMAT = "%(name)s -%(levelname)s - %(message)s"
COMET_KWARGS = {
    "api_key": os.environ.get("COMET_API_KEY"),
    "project_name": "master-thesis",
    "workspace": "dahiyaaneesh",
    "save_dir": SAVED_META_INFO_PATH,
}
