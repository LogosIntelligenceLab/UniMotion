"""Run with python -m src.unimotion.run [Hydra overrides]."""

import hydra
import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import DictConfig

from .data import PretrainingDataModule
from .training import PretrainingModule


@hydra.main(config_path="../../configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig):
    if cfg.action not in ("fit", "validate"):
        raise ValueError("Pretraining action must be fit or validate")
    if cfg.action == "validate" and not cfg.ckpt_path:
        raise ValueError("validate requires ckpt_path")
    L.seed_everything(cfg.seed, workers=True)
    data = PretrainingDataModule(tasks=cfg.tasks, **cfg.data)
    model = PretrainingModule(tasks=cfg.tasks, **cfg.model)
    logger = instantiate(cfg.logger) if cfg.logger else False
    callbacks = [ModelCheckpoint(dirpath=f"{cfg.output_dir}/checkpoints", filename="epoch_{epoch:03d}",
                                 auto_insert_metric_name=False, save_last=True, save_top_k=1,
                                 monitor="val/loss", mode="min")]
    if logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    trainer_options = dict(cfg.trainer)
    # Alternating tasks leave one output head unused on each training step.
    if len(cfg.tasks) > 1 and trainer_options.get("strategy", "auto") in ("auto", "ddp"):
        if (trainer_options.get("strategy") == "ddp"
                or trainer_options.get("devices", 1) != 1
                or trainer_options.get("num_nodes", 1) > 1):
            trainer_options["strategy"] = "ddp_find_unused_parameters_true"
    trainer = L.Trainer(**trainer_options, logger=logger, callbacks=callbacks,
                        default_root_dir=cfg.output_dir)
    if cfg.action == "fit":
        trainer.fit(model, datamodule=data, ckpt_path=cfg.ckpt_path)
    else:
        trainer.validate(model, datamodule=data, ckpt_path=cfg.ckpt_path)


if __name__ == "__main__":
    main()
