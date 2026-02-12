import os
from datetime import datetime

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import make_circles


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config("config.yaml")
    print(config)


if __name__ == "__main__":
    main()
