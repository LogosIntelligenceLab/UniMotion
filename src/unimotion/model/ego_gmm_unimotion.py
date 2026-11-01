# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
from pathlib import Path

import hydra
import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.unimotion.metrics import GMMADE, EgoNLL, WOSACMetrics, minADE
from src.unimotion.metrics.utils import get_euclidean_targets
from src.unimotion.modules.ego_gmm_unimotion_decoder import EgoGMMUniMotionDecoder
from src.unimotion.tokens.token_processor import TokenProcessor
from src.unimotion.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class EgoGMMUniMotion(LightningModule):

    def __init__(self, model_config) -> None:
        super(EgoGMMUniMotion, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_closed_loop = model_config.val_closed_loop
        self.token_processor = TokenProcessor(**model_config.token_processor)

        self.encoder = EgoGMMUniMotionDecoder(**model_config.decoder)
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.wosac_metrics = WOSACMetrics("val_closed", ego_only=True)
        self.gmm_ade_pos = GMMADE()
        self.gmm_ade_head = GMMADE()
        self.training_loss = EgoNLL(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.video_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.video_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            pred = self.encoder.inference(
                tokenized_map,
                tokenized_agent,
                sampling_scheme=self.training_rollout_sampling,
            )

        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"][
                tokenized_agent["ego_mask"]
            ],  # [n_agent, 2]
            current_epoch=self.current_epoch,
        )
        self.log("train/loss", loss, on_step=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = self.token_processor(data)

        # ! open-loop vlidation
        pred = self.encoder(tokenized_map, tokenized_agent)
        loss = self.training_loss(
            **pred,
            token_agent_shape=tokenized_agent["token_agent_shape"][
                tokenized_agent["ego_mask"]
            ],
        )
        self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

        bc_target, bc_target_valid = get_euclidean_targets(
            pred_pos=pred["gt_pos_raw"],
            pred_head=pred["gt_head_raw"],
            pred_valid=pred["gt_valid_raw"],
            gt_pos=pred["gt_pos_raw"],
            gt_head=pred["gt_head_raw"],
            gt_valid=pred["gt_valid_raw"],
        )  # bc_target: [n_agent, 16, 3], x,y,yaw. bc_target_valid: [n_agent, 16]

        self.gmm_ade_pos.update(
            logits=pred["ego_next_logits"],  # [n_agent, 16, n_k_ego_gmm]
            pred=pred["ego_next_poses"][..., :2],  # [n_agent, 16, n_k_ego_gmm, 2]
            target=bc_target[..., :2],  # [n_agent, 16, 2]
            valid=bc_target_valid & pred["ego_next_valid"],  # [n_agent, 16]
        )
        bc_target_head = torch.stack(
            [bc_target[..., -1].cos(), bc_target[..., -1].sin()], dim=-1
        )  # [n_agent, 16, 2]
        ego_next_heads = torch.stack(
            [
                pred["ego_next_poses"][..., -1].cos(),
                pred["ego_next_poses"][..., -1].sin(),
            ],
            dim=-1,
        )  # [n_agent, 16, n_k_ego_gmm, 2]
        self.gmm_ade_head.update(
            logits=pred["ego_next_logits"],  # [n_agent, 16, n_k_ego_gmm]
            pred=ego_next_heads,  # [n_agent, 16, n_k_ego_gmm, 2]
            target=bc_target_head,  # [n_agent, 16, 2]
            valid=bc_target_valid & pred["ego_next_valid"],  # [n_agent, 16]
        )
        self.log(
            "val_open/gmm_ade_pos",
            self.gmm_ade_pos,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

        self.log(
            "val_open/gmm_ade_head",
            self.gmm_ade_head,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

        # ! closed-loop vlidation
        if self.val_closed_loop:
            pred_traj, pred_z, pred_head = [], [], []
            for _ in range(self.n_rollout_closed_val):
                pred = self.encoder.inference(
                    tokenized_map, tokenized_agent, self.validation_rollout_sampling
                )
                pred_traj.append(pred["pred_traj_10hz"])
                pred_z.append(pred["pred_z_10hz"])
                pred_head.append(pred["pred_head_10hz"])

            pred_traj = torch.stack(pred_traj, dim=1)  # [n_ag, n_rollout, n_step, 2]
            pred_z = torch.stack(pred_z, dim=1)  # [n_ag, n_rollout, n_step]
            pred_head = torch.stack(pred_head, dim=1)  # [n_ag, n_rollout, n_step]

            # ! WOSAC
            self.minADE.update(
                pred=pred_traj[tokenized_agent["ego_mask"]],
                target=data["agent"]["position"][
                    :, self.num_historical_steps :, : pred_traj.shape[-1]
                ][tokenized_agent["ego_mask"]],
                target_valid=data["agent"]["valid_mask"][
                    :, self.num_historical_steps :
                ][tokenized_agent["ego_mask"]],
            )

            # WOSAC metrics
            if batch_idx < self.n_batch_wosac_metric:
                device = pred_traj.device
                scenario_rollouts = get_scenario_rollouts(
                    scenario_id=get_scenario_id_int_tensor(data["scenario_id"], device),
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )
                self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.global_rank == 0 and batch_idx < self.n_vis_batch:
                device = pred_traj.device
                scenario_rollouts = get_scenario_rollouts(
                    scenario_id=get_scenario_id_int_tensor(data["scenario_id"], device),
                    agent_id=data["agent"]["id"][tokenized_agent["ego_mask"]],
                    agent_batch=data["agent"]["batch"][tokenized_agent["ego_mask"]],
                    pred_traj=pred_traj[tokenized_agent["ego_mask"]],
                    pred_z=pred_z[tokenized_agent["ego_mask"]],
                    pred_head=pred_head[tokenized_agent["ego_mask"]],
                )
                for _i_sc in range(self.n_vis_scenario):
                    _vis = VisWaymo(
                        scenario_path=data["tfrecord_path"][_i_sc],
                        save_dir=self.video_dir
                        / f"batch_{batch_idx:02d}-scenario_{_i_sc:02d}",
                    )
                    _vis.save_video_scenario_rollout(
                        scenario_rollouts[_i_sc], self.n_vis_rollout
                    )
                    for _path in _vis.video_paths:
                        self.logger.log_video("/".join(_path.split("/")[-3:]), [_path])

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            epoch_wosac_metrics = self.wosac_metrics.compute()
            epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
            if self.global_rank == 0:
                epoch_wosac_metrics["epoch"] = (
                    self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                )
                self.logger.log_metrics(epoch_wosac_metrics)

            self.wosac_metrics.reset()
            self.minADE.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        def lr_lambda(current_step):
            current_step = self.current_epoch + 1
            if current_step < self.lr_warmup_steps:
                return (
                    self.lr_min_ratio
                    + (1 - self.lr_min_ratio) * current_step / self.lr_warmup_steps
                )
            return self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * (
                1.0
                + math.cos(
                    math.pi
                    * min(
                        1.0,
                        (current_step - self.lr_warmup_steps)
                        / (self.lr_total_steps - self.lr_warmup_steps),
                    )
                )
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]
