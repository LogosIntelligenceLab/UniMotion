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

import random
import torch
import pickle
from pathlib import Path
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from src.unimotion.utils import transform_to_global

def get_intention_pts():
    category_keys = ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]
    intention_points_file = Path(__file__).resolve().parents[3] / "dataset" / "cluster_64_center_dict.pkl"
    with open(intention_points_file, "rb") as f:
        intention_points_dict = pickle.load(f)
    return torch.stack([torch.from_numpy(intention_points_dict[k]) for k in category_keys]).float()

class WaymoTargetBuilderTrain(BaseTransform):
    def __init__(self, tokenizer, max_num: int, task: list, use_intention: bool) -> None:
        super(WaymoTargetBuilderTrain, self).__init__()
        self.step_current = 10
        self.max_num = max_num
        self.tokenizer = tokenizer
        self.use_intention = use_intention
        if use_intention:
            self.int_pts = get_intention_pts()
        if isinstance(task, str):
            task = [task]
        assert len(set(task) - {"sim", "pred", "plan"}) == 0
        self.task = task

    def __call__(self, data) -> HeteroData:
        pos = data["agent"]["position"]
        data["agent"]["role"] = data["agent"]["role"][:, :3]
        av_index = torch.where(data["agent"]["role"][:, 0])[0].item()
        distance = torch.norm(pos - pos[av_index], dim=-1)

        # we do not believe the perception out of range of 150 meters
        data["agent"]["valid_mask"] = data["agent"]["valid_mask"] & (distance < 150)

        # we do not predict vehicle too far away from ego car
        role_train_mask = data["agent"]["role"].any(-1)
        extra_train_mask = (distance[:, self.step_current] < 100) & (
            data["agent"]["valid_mask"][:, self.step_current + 1 :].sum(-1) >= 5
        )

        train_mask = extra_train_mask | role_train_mask
        if train_mask.sum() > self.max_num:  # too many vehicle
            _indices = torch.where(extra_train_mask & ~role_train_mask)[0]
            selected_indices = _indices[
                torch.randperm(_indices.size(0))[: self.max_num - role_train_mask.sum()]
            ]
            data["agent"]["train_mask"] = role_train_mask
            data["agent"]["train_mask"][selected_indices] = True
        else:
            data["agent"]["train_mask"] = train_mask  # [n_agent]
        data = self.tokenizer(data)
        # data["task"] = random.choice(self.task)
        data['task'] = self.task
        if self.use_intention:
            token_pos_gt = data["agent"]["token_pos_gt"]
            token_pos_mask = data["agent"]["token_pos_mask"]
            n_a, n_s = token_pos_gt.shape[:2]
            final_idx = torch.sum(token_pos_mask, dim=-1).long() - 1
            final_pos = token_pos_gt.view(-1, *token_pos_gt.shape[-2:])[torch.arange(n_a * n_s), final_idx.view(-1)]
            final_pos = final_pos.reshape(token_pos_gt.size(0), -1, 2)
            int_pts = self.int_pts[data["agent"]["type"].long()].clone()
            dist = (final_pos[:, :, None] - int_pts[:, None]).norm(dim=-1)
            # cur_step = self.tokenizer.num_historical_steps - 1
            # cur_pos, cur_head = data["agent"]["position"][:, cur_step, :2], data["agent"]["heading"][:, cur_step]
            # gbl_int_pts, _ = transform_to_global(int_pts, None, cur_pos, cur_head)
            data["agent"]["int_pts"] = int_pts
            # data["agent"]["gbl_int_pts"] = gbl_int_pts
            data["agent"]["mode_idx"] = dist.argmin(dim=-1)
            # data["agent"]["token_pos_mask"] = token_pos_mask & (final_idx[..., None] >= 40)
        return HeteroData(data)


class WaymoTargetBuilderVal(BaseTransform):
    def __init__(self, tokenizer, task: str, use_intention: bool) -> None:
        super(WaymoTargetBuilderVal, self).__init__()
        self.tokenizer = tokenizer
        self.use_intention = use_intention
        if use_intention:
            self.int_pts = get_intention_pts()
        if isinstance(task, str):
            task = [task]
        assert len(set(task) - {"sim", "pred", "plan"}) == 0
        self.task = task

    def __call__(self, data) -> HeteroData:
        data = self.tokenizer(data)
        data["task"] = self.task
        if self.use_intention:
            token_pos_gt = data["agent"]["token_pos_gt"]
            token_pos_mask = data["agent"]["token_pos_mask"]
            n_a, n_s = token_pos_gt.shape[:2]
            final_idx = torch.sum(token_pos_mask, dim=-1).long() - 1
            final_pos = token_pos_gt.view(-1, *token_pos_gt.shape[-2:])[torch.arange(n_a * n_s), final_idx.view(-1)]
            final_pos = final_pos.reshape(token_pos_gt.size(0), -1, 2)
            int_pts = self.int_pts[data["agent"]["type"].long()].clone()
            dist = (final_pos[:, :, None] - int_pts[:, None]).norm(dim=-1)
            # cur_step = self.tokenizer.num_historical_steps - 1
            # cur_pos, cur_head = data["agent"]["position"][:, cur_step, :2], data["agent"]["heading"][:, cur_step]
            # gbl_int_pts, _ = transform_to_global(int_pts, None, cur_pos, cur_head)
            data["agent"]["int_pts"] = int_pts  
            # data["agent"]["gbl_int_pts"] = gbl_int_pts
            data["agent"]["mode_idx"] = dist.argmin(dim=-1)
            # data["agent"]["token_pos_mask"] = token_pos_mask & (final_idx[..., None] >= 40)
        return HeteroData(data)
