from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class GaborConfig:
    n_orientations: int = 8
    n_scales: int = 4
    kernel_size: int = 15
    out_channels: int = 3


@dataclass
class EfficientNetConfig:
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    freeze_blocks: int = 4


@dataclass
class ModelConfig:
    name: str = "malit_v2"
    num_classes: int = 2
    image_size: int = 224
    gabor: GaborConfig = field(default_factory=GaborConfig)
    efficientnet: EfficientNetConfig = field(default_factory=EfficientNetConfig)
    lcci_reduction: int = 16
    attention_channel_reduction: int = 16
    multiscale_rfs: list[int] = field(default_factory=lambda: [1, 3, 5, 7])


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4


@dataclass
class SchedulerConfig:
    name: str = "cosine_annealing"
    T_max: int = 50
    eta_min: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    patience: int = 3
    monitor: str = "val_f1"


@dataclass
class TrainingConfig:
    seeds_tier1: list[int] = field(default_factory=lambda: [42, 123, 456])
    seeds_tier2: list[int] = field(default_factory=lambda: [42, 123])
    seeds_tier3: list[int] = field(default_factory=lambda: [42])
    active_tier: int = 1
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 50
    batch_size: int = 32
    amp: bool = True
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    grad_clip: float = 1.0
    num_workers: int = 4

    @property
    def active_seeds(self) -> list[int]:
        return {1: self.seeds_tier1, 2: self.seeds_tier2, 3: self.seeds_tier3}[self.active_tier]


@dataclass
class DeduplicationConfig:
    hash_size: int = 16
    hamming_threshold: int = 4


@dataclass
class DataSplits:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


@dataclass
class DataConfig:
    dataset: str = "cell-images-for-detecting-malaria"
    splits: DataSplits = field(default_factory=DataSplits)
    seed: int = 42
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)


@dataclass
class StatisticsConfig:
    mcnemar_per_seed: bool = True
    mcnemar_aggregate: bool = True
    tost_margin: float = 0.005
    tost_alpha: float = 0.05


@dataclass
class ResultPaths:
    models: str = "results/models/"
    plots: str = "results/plots/"
    gradcam: str = "results/gradcam/"
    attention: str = "results/attention_maps/"
    statistics: str = "results/statistics/"
    tables: str = "results/tables/"
    reproducibility: str = "results/reproducibility/"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    paths: ResultPaths = field(default_factory=ResultPaths)

    def ensure_dirs(self, base: str = ".") -> None:
        base_path = Path(base)
        for attr in vars(self.paths).values():
            (base_path / attr).mkdir(parents=True, exist_ok=True)


def _deep_update(base: dict, update: dict) -> dict:
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path | None = None) -> Config:
    defaults: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            raw = yaml.safe_load(f)
        if raw:
            defaults = raw

    cfg = Config()

    model_raw = defaults.get("model", {})
    if model_raw:
        g = model_raw.get("gabor", {})
        cfg.model.gabor = GaborConfig(**{k: v for k, v in g.items() if k in GaborConfig.__dataclass_fields__})
        e = model_raw.get("efficientnet", {})
        cfg.model.efficientnet = EfficientNetConfig(
            **{k: v for k, v in e.items() if k in EfficientNetConfig.__dataclass_fields__}
        )
        cfg.model.num_classes = model_raw.get("num_classes", cfg.model.num_classes)
        cfg.model.image_size = model_raw.get("image_size", cfg.model.image_size)
        lcci = model_raw.get("lcci", {})
        cfg.model.lcci_reduction = lcci.get("reduction", cfg.model.lcci_reduction)
        att = model_raw.get("attention", {})
        cfg.model.attention_channel_reduction = att.get("channel_reduction", cfg.model.attention_channel_reduction)
        ms = model_raw.get("multiscale", {})
        cfg.model.multiscale_rfs = ms.get("receptive_fields", cfg.model.multiscale_rfs)

    train_raw = defaults.get("training", {})
    if train_raw:
        seeds = train_raw.get("seeds", {})
        cfg.training.seeds_tier1 = seeds.get("tier1", cfg.training.seeds_tier1)
        cfg.training.seeds_tier2 = seeds.get("tier2", cfg.training.seeds_tier2)
        cfg.training.seeds_tier3 = seeds.get("tier3", cfg.training.seeds_tier3)
        cfg.training.active_tier = train_raw.get("active_tier", cfg.training.active_tier)
        cfg.training.epochs = train_raw.get("epochs", cfg.training.epochs)
        cfg.training.batch_size = train_raw.get("batch_size", cfg.training.batch_size)
        cfg.training.amp = train_raw.get("amp", cfg.training.amp)
        cfg.training.grad_clip = train_raw.get("grad_clip", cfg.training.grad_clip)
        cfg.training.num_workers = train_raw.get("num_workers", cfg.training.num_workers)
        opt = train_raw.get("optimizer", {})
        cfg.training.optimizer = OptimizerConfig(**{k: v for k, v in opt.items() if k in OptimizerConfig.__dataclass_fields__})
        sch = train_raw.get("scheduler", {})
        cfg.training.scheduler = SchedulerConfig(**{k: v for k, v in sch.items() if k in SchedulerConfig.__dataclass_fields__})
        es = train_raw.get("early_stopping", {})
        cfg.training.early_stopping = EarlyStoppingConfig(**{k: v for k, v in es.items() if k in EarlyStoppingConfig.__dataclass_fields__})

    data_raw = defaults.get("data", {})
    if data_raw:
        cfg.data.dataset = data_raw.get("dataset", cfg.data.dataset)
        cfg.data.seed = data_raw.get("seed", cfg.data.seed)
        sp = data_raw.get("splits", {})
        cfg.data.splits = DataSplits(**{k: v for k, v in sp.items() if k in DataSplits.__dataclass_fields__})
        dd = data_raw.get("deduplication", {})
        cfg.data.deduplication = DeduplicationConfig(
            **{k: v for k, v in dd.items() if k in DeduplicationConfig.__dataclass_fields__}
        )

    stats_raw = defaults.get("statistics", {})
    if stats_raw:
        mcn = stats_raw.get("mcnemar", {})
        tost = stats_raw.get("tost", {})
        cfg.statistics.mcnemar_per_seed = mcn.get("per_seed", cfg.statistics.mcnemar_per_seed)
        cfg.statistics.mcnemar_aggregate = mcn.get("aggregate", cfg.statistics.mcnemar_aggregate)
        cfg.statistics.tost_margin = tost.get("margin", cfg.statistics.tost_margin)
        cfg.statistics.tost_alpha = tost.get("alpha", cfg.statistics.tost_alpha)

    paths_raw = defaults.get("paths", {}).get("results", {})
    if paths_raw:
        cfg.paths = ResultPaths(**{k: v for k, v in paths_raw.items() if k in ResultPaths.__dataclass_fields__})

    return cfg
