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
from torchmetrics import Metric


class TokenCls(Metric):

    def __init__(self, max_guesses: int = 6, **kwargs) -> None:
        super(TokenCls, self).__init__(**kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,  # next_token_logits: [n_agent, 16, n_token]
        target: torch.Tensor,  # next_token_idx_gt: [n_agent, 16]
        valid_mask: torch.Tensor,  # [n_agent, 16]
    ) -> None:
        target = target[..., None]
        acc = (torch.topk(pred, k=self.max_guesses, dim=-1)[1] == target).any(dim=-1)
        acc = acc * valid_mask
        self.sum += acc.sum()
        self.count += valid_mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
