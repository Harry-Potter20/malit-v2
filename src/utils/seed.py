from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class SeedManager:
    def __init__(self, seeds: list[int]):
        self.seeds = seeds

    def __iter__(self):
        for seed in self.seeds:
            set_seed(seed)
            yield seed

    def __len__(self) -> int:
        return len(self.seeds)
