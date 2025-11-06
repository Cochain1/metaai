from pathlib import Path
import yaml, os, random
import numpy as np
import torch

def bootstrap(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["runtime"]["cache_dir"], exist_ok=True)
    os.makedirs(cfg["runtime"]["log_dir"], exist_ok=True)
    seed = cfg["runtime"]["seed"]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return cfg
