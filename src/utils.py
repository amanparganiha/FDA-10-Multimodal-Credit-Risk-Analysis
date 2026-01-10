# src/utils.py
import os
import yaml
import logging
from pathlib import Path

def setup_logging(level=logging.INFO):
    import logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(message)s',
        level=level
    )
    return logging.getLogger()

def load_config(path="config/config.yaml"):
    import yaml
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    Path(cfg['paths']['processed']).mkdir(parents=True, exist_ok=True)
    Path(cfg['paths']['artifacts']).mkdir(parents=True, exist_ok=True)
    Path(cfg['paths']['models']).mkdir(parents=True, exist_ok=True)
    return cfg

def safe_join(root, *parts):
    return os.path.join(root, *parts)
