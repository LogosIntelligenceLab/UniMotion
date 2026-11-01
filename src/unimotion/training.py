"""Training/validation orchestration, independent of rollout and official SDKs."""

import math

import torch
from lightning import LightningModule

from .data import prepare_batch
from .losses import PretrainingLoss
from .prediction_metrics import prediction_ade
from .network import UniMotion


class PretrainingModule(LightningModule):
    def __init__(self, tasks, decoder, loss, optimizer):
        super().__init__()
        self.save_hyperparameters()
        self.model = UniMotion(tasks, decoder)
        self.objective = PretrainingLoss(**loss)
        self.register_buffer("ade_total", torch.zeros(2, dtype=torch.float64), persistent=False)

    def forward(self, tokenized_map, agent, task):
        return self.model(tokenized_map, agent, task)

    def training_task(self):
        # Same task on every rank and throughout gradient accumulation.
        # global_step is restored with the optimizer when resuming.
        return self.hparams.tasks[self.global_step % len(self.hparams.tasks)]

    def _step(self, batch, stage, tasks):
        tokenized_map, agent = prepare_batch(batch)
        losses = {}
        for task in tasks:
            output = self(tokenized_map, agent, task)
            task_losses = self.objective(agent, output)
            losses[f"{task}_loss"] = task_losses.pop("loss")
            losses.update(task_losses)
            if stage == "val" and task == "pred":
                total, count = prediction_ade(agent, output["pred"])
                self.ade_total += torch.stack((total, count)).to(self.ade_total)
        losses["loss"] = sum(losses[f"{task}_loss"] for task in tasks)
        for name, value in losses.items():
            self.log(f"{stage}/{name}", value, on_step=stage == "train",
                     on_epoch=True, sync_dist=True, batch_size=batch.num_graphs)
        return losses["loss"]

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train", [self.training_task()])

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val", self.hparams.tasks)

    def on_validation_epoch_start(self):
        self.ade_total.zero_()

    def on_validation_epoch_end(self):
        if "pred" in self.hparams.tasks:
            total = self.ade_total.clone()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(total)
            # Empty-interest validation is explicitly counted, not a NaN.
            self.log("val/prediction_agent_count", total[1], sync_dist=False)
            self.log("val/minADE", total[0] / total[1].clamp_min(1), sync_dist=False)

    def configure_optimizers(self):
        cfg = self.hparams.optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        def factor(_scheduler_step):
            # Epoch-based cosine schedule using epoch + 1, with zero warmup.
            progress = min(1.0, (self.current_epoch + 1) / cfg["epochs"])
            return cfg["min_ratio"] + 0.5 * (1 - cfg["min_ratio"]) * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
