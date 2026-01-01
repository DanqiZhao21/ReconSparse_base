from typing import Tuple
from pathlib import Path
import logging
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
# from pytorch_lightning.loggers import WandbLogger
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.planning.training.dataset import CacheOnlyDatasetTest
from navsim.planning.training.agent_lightning_module import AgentLightningModule
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def custom_collate_fn(batch):
    features, targets, metrics, token = zip(*batch)

    features = torch.utils.data.default_collate(features)
    targets = torch.utils.data.default_collate(targets)

    return features, targets, metrics, token

@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")
    logger.info(f"Path where all results are stored: {cfg.output_dir}")
    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building PDMEvaluationDataset...")
    logger.info(f"Loading test set features from: {cfg.test_cache_path}")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    logger.info("Building Datasets")   
    test_dataset = CacheOnlyDatasetTest(
        feature_cache_path=cfg.test_cache_path, # <-- 指向新生成的测试特征缓存
        metric_cache_path=cfg.metric_cache_path, # <-- 指向原始的 metric_cache
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=scene_filter.log_names,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=cfg.dataloader.params.num_workers,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )

    logger.info("Num test samples: %d", len(test_dataset))

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent
    )

    logger.info("Building Trainer")
    original_callbacks = agent.get_training_callbacks()
    callbacks = original_callbacks
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)
    logger.info("Starting Validation")
    trainer.validate(
        model=lightning_module,
        dataloaders=test_dataloader,
    )


if __name__ == "__main__":
    main()
