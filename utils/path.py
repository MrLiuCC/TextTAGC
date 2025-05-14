"""
    指定规范化的路径
"""
import os
from pathlib import Path

models_saved_dir = "./cache/saved_models"
checkpoint_saved_dir = "./cache/saved_checkpoints"

# New directory if not exist
Path(models_saved_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_saved_dir).mkdir(parents=True, exist_ok=True)


def get_checkpoint_path(prefix, data, epoch):
    return os.path.join(checkpoint_saved_dir, f"{prefix}-{data}-{str(epoch)}.pth")


def get_model_path(prefix, data):
    return os.path.join(models_saved_dir, f"whole-{prefix}-{data}.pth")
