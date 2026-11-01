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

# Legacy simulation model retained for reference.

import math
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR

from src.unimotion.metrics import (
    CrossEntropy,
    TokenCls,
    WOSACMetrics,
    WOSACSubmission,
    minADE,
)
from src.unimotion.modules.unimotion_decoder import UniMotionDecoder
from src.unimotion.utils.finetune import set_model_for_finetuning
from src.utils.vis_waymo import VisWaymo
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts


class UniMotion(LightningModule):

    def __init__(self, model_config) -> None:
        super(UniMotion, self).__init__()
        self.save_hyperparameters()
        self.lr = model_config.lr
        self.lr_warmup_steps = model_config.lr_warmup_steps
        self.lr_total_steps = model_config.lr_total_steps
        self.lr_min_ratio = model_config.lr_min_ratio
        self.num_historical_steps = model_config.decoder.num_historical_steps
        self.log_epoch = -1
        self.val_open_loop = model_config.val_open_loop
        self.val_closed_loop = model_config.val_closed_loop

        self.encoder = UniMotionDecoder(**model_config.decoder)
        self.encoder.agent_encoder.finetune = model_config.finetune
        set_model_for_finetuning(self.encoder, model_config.finetune)

        self.minADE = minADE()
        self.TokenCls = TokenCls(max_guesses=5)
        self.CLTokenCls = TokenCls(max_guesses=5)
        self.wosac_metrics = WOSACMetrics("val_closed")
        self.wosac_submission = WOSACSubmission(**model_config.wosac_submission)
        self.training_loss = CrossEntropy(**model_config.training_loss)

        self.n_rollout_closed_val = model_config.n_rollout_closed_val
        self.vis_active = model_config.vis_active
        self.n_vis_batch = model_config.n_vis_batch
        self.n_vis_scenario = model_config.n_vis_scenario
        self.n_vis_rollout = model_config.n_vis_rollout
        self.n_batch_wosac_metric = model_config.n_batch_wosac_metric

        self.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        self.video_dir = Path(self.output_dir) / "videos"

        self.training_rollout_sampling = model_config.training_rollout_sampling
        self.validation_rollout_sampling = model_config.validation_rollout_sampling
    
    def on_train_epoch_start(self):
        torch.cuda.empty_cache()

    def training_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = dict(data["pt_token"]), dict(data["agent"])
        map_token, agent_token = data["all_tokens"][0].map_token, data["all_tokens"][0].agent_token
        tokenized_map["token_traj_src"] = map_token.to(data["pt_token"]["batch"].device)
        tokenized_map["num_graphs"] = data.num_graphs
        tokenized_agent["trajectory_token_veh"] = agent_token["veh"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_ped"] = agent_token["ped"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_cyc"] = agent_token["cyc"].to(data["agent"]["batch"].device)
        tokenized_agent["num_graphs"] = data.num_graphs
        task = data["task"][0]
        tokenized_agent["task"] = task
        if self.training_rollout_sampling.num_k <= 0:
            pred = self.encoder(tokenized_map, tokenized_agent)
        else:
            oom_flag = torch.tensor([0], device=tokenized_agent["batch"].device)
            try:
                pred = self.encoder.inference(
                    tokenized_map,
                    tokenized_agent,
                    sampling_scheme=self.training_rollout_sampling,
                )
                loss = self.cal_loss(tokenized_agent, pred, task)
                self.log("train/loss", loss, on_step=True, batch_size=1)
            except:
                print(111111111111111)
                oom_flag[0] = 1
                loss = torch.tensor(0.0, requires_grad=True).to(tokenized_agent["batch"].device)
                torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.all_reduce(oom_flag, op=dist.ReduceOp.SUM)
            if oom_flag.item() > 0:
                loss = torch.tensor(0.0, requires_grad=True).to(tokenized_agent["batch"].device)
        return loss

    def validation_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = dict(data["pt_token"]), dict(data["agent"])
        map_token, agent_token = data["all_tokens"][0].map_token, data["all_tokens"][0].agent_token
        tokenized_map["token_traj_src"] = map_token.to(data["pt_token"]["batch"].device)
        tokenized_map["num_graphs"] = data.num_graphs
        tokenized_agent["trajectory_token_veh"] = agent_token["veh"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_ped"] = agent_token["ped"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_cyc"] = agent_token["cyc"].to(data["agent"]["batch"].device)
        tokenized_agent["num_graphs"] = data.num_graphs
        task = data["task"][0]
        tokenized_agent["task"] = task

        # ! open-loop vlidation
        if self.val_open_loop:
            pred = self.encoder(tokenized_map, tokenized_agent)
            cl_pred = self.encoder.inference(tokenized_map, tokenized_agent, self.validation_rollout_sampling)
            loss = self.cal_loss(tokenized_agent, pred, task)
            self.update_metric(tokenized_agent, pred, cl_pred)
            self.log("val_open/loss", loss, on_epoch=True, sync_dist=True, batch_size=1)

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
            scenario_rollouts = None
            if self.wosac_submission.is_active:  # ! save WOSAC submission
                self.wosac_submission.update(
                    scenario_id=data["scenario_id"],
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                    global_rank=self.global_rank,
                )
                _gpu_dict_sync = self.wosac_submission.compute()
                if self.global_rank == 0:
                    for k in _gpu_dict_sync.keys():  # single gpu fix
                        if type(_gpu_dict_sync[k]) is list:
                            _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                    scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
                    self.wosac_submission.aggregate_rollouts(scenario_rollouts)
                self.wosac_submission.reset()

            else:  # ! compute metrics, disable if save WOSAC submission
                # self.minADE.update(
                #     pred=pred_traj,
                #     target=data["agent"]["position"][
                #         :, self.num_historical_steps :, : pred_traj.shape[-1]
                #     ],
                #     target_valid=data["agent"]["valid_mask"][
                #         :, self.num_historical_steps :
                #     ],
                # )

                # WOSAC metrics
                # if batch_idx < self.n_batch_wosac_metric:
                device = pred_traj.device
                scenario_rollouts = get_scenario_rollouts(
                    scenario_id=get_scenario_id_int_tensor(
                        data["scenario_id"], device
                    ),
                    agent_id=data["agent"]["id"],
                    agent_batch=data["agent"]["batch"],
                    pred_traj=pred_traj,
                    pred_z=pred_z,
                    pred_head=pred_head,
                )
                    # self.wosac_metrics.update(data["tfrecord_path"], scenario_rollouts)

            # ! visualization
            if self.vis_active and self.global_rank == 0:   # and batch_idx < self.n_vis_batch
                if scenario_rollouts is not None:
                    for _i_sc in range(len(data["tfrecord_path"])):
                        s_id = data["scenario_id"][_i_sc]
                        print(
                            f"Visualizing scenario {s_id} in batch {batch_idx}"
                        )
                        _vis = VisWaymo(
                            scenario_path=data["tfrecord_path"][_i_sc],
                            save_dir=self.video_dir
                            / f"{s_id}",
                        )
                        _vis.save_video_scenario_rollout(
                            scenario_rollouts[_i_sc], self.n_vis_rollout
                        )
                        # for _path in _vis.video_paths:
                        #     self.logger.log_video(
                        #         "/".join(_path.split("/")[-3:]), [_path]
                        #     )
            pass

    def on_validation_epoch_end(self):
        if self.val_closed_loop:
            if not self.wosac_submission.is_active:
                epoch_wosac_metrics = self.wosac_metrics.compute()
                epoch_wosac_metrics["val_closed/ADE"] = self.minADE.compute()
                if self.global_rank == 0:
                    epoch_wosac_metrics["epoch"] = (
                        self.log_epoch if self.log_epoch >= 0 else self.current_epoch
                    )
                    self.logger.log_metrics(epoch_wosac_metrics)

                self.wosac_metrics.reset()
                self.minADE.reset()

            if self.global_rank == 0:
                if self.wosac_submission.is_active:
                    self.wosac_submission.save_sub_file()

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

    def test_step(self, data, batch_idx):
        tokenized_map, tokenized_agent = dict(data["pt_token"]), dict(data["agent"])
        map_token, agent_token = data["all_tokens"][0].map_token, data["all_tokens"][0].agent_token
        tokenized_map["token_traj_src"] = map_token.to(data["pt_token"]["batch"].device)
        tokenized_map["num_graphs"] = data.num_graphs
        tokenized_agent["trajectory_token_veh"] = agent_token["veh"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_ped"] = agent_token["ped"].to(data["agent"]["batch"].device)
        tokenized_agent["trajectory_token_cyc"] = agent_token["cyc"].to(data["agent"]["batch"].device)
        tokenized_agent["num_graphs"] = data.num_graphs
        task = data["task"][0]
        tokenized_agent["task"] = task

        # ! only closed-loop vlidation
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

        # ! WOSAC submission save
        self.wosac_submission.update(
            scenario_id=data["scenario_id"],
            agent_id=data["agent"]["id"],
            agent_batch=data["agent"]["batch"],
            pred_traj=pred_traj,
            pred_z=pred_z,
            pred_head=pred_head,
            global_rank=self.global_rank,
        )
        _gpu_dict_sync = self.wosac_submission.compute()
        if self.global_rank == 0:
            for k in _gpu_dict_sync.keys():  # single gpu fix
                if type(_gpu_dict_sync[k]) is list:
                    _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
            scenario_rollouts = get_scenario_rollouts(**_gpu_dict_sync)
            self.wosac_submission.aggregate_rollouts(scenario_rollouts)
        self.wosac_submission.reset()

    def on_test_epoch_end(self):
        if self.global_rank == 0:
            self.wosac_submission.save_sub_file()

    def update_metric(self, agent_data, pred, cl_pred):
        self.TokenCls.update(
            # action that goes from [(10->15), ..., (85->90)]
            pred=pred["next_token_logits"],  # [n_agent, 16, n_token]
            target=agent_data["gt_idx"][:, 2:],
            valid_mask=pred["next_token_valid"] & agent_data["valid_mask"][:, 2:],
        )
        self.log(
            "val_open/sim_acc",
            self.TokenCls,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )
        self.CLTokenCls.update(
            # action that goes from [(10->15), ..., (85->90)]
            pred=cl_pred["next_token_logits"],  # [n_agent, 16, n_token]
            target=agent_data["gt_idx"][:, 2:],
            valid_mask=cl_pred["next_token_valid"] & agent_data["valid_mask"][:, 2:],
        )
        self.log(
            "val_closed/sim_acc",
            self.CLTokenCls,
            on_epoch=True,
            sync_dist=True,
            batch_size=1,
        )

    def cal_loss(self, agent_data, pred, task):
        cls_loss = self.training_loss(
            **pred,
            token_agent_shape=agent_data["token_agent_shape"],  # [n_agent, 2]
            token_traj=agent_data["token_traj"],  # [n_agent, n_token, 4, 2]
            train_mask=agent_data.get("train_mask"),  # [n_agent]
            current_epoch=self.current_epoch,
            token_traj_all=agent_data["token_traj_all"],
            agent_mask=agent_data["role"][:, 2],
            task=task,
        )

        return cls_loss    
