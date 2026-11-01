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

import torch
from torch import Tensor, tensor
from torchmetrics import Metric


class minADE(Metric):

    def __init__(self) -> None:
        super(minADE, self).__init__()
        self.add_state("sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        pred: Tensor,  # [n_agent, n_rollout, n_step, 2]
        target: Tensor,  # [n_agent, n_step, 2]
        target_valid: Tensor,  # [n_agent, n_step]
    ) -> None:

        # [n_agent, n_rollout, n_step]
        dist = torch.norm(pred - target.unsqueeze(1), p=2, dim=-1)
        dist = (dist * target_valid.unsqueeze(1)).sum(-1).min(-1).values  # [n_agent]

        dist = dist / (target_valid.sum(-1) + 1e-6)  # [n_agent]
        self.sum += dist.sum()
        self.count += target_valid.any(-1).sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
