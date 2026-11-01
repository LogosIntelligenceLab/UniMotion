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

from src.unimotion.metrics.cross_entropy import CrossEntropy
from src.unimotion.metrics.ego_nll import EgoNLL
from src.unimotion.metrics.gmm_ade import GMMADE
from src.unimotion.metrics.min_ade import minADE
from src.unimotion.metrics.next_token_cls import TokenCls


def __getattr__(name):
    if name == "WOSACMetrics":
        from .wosac_metrics import WOSACMetrics
        return WOSACMetrics
    if name == "WOSACSubmission":
        from .wosac_submission import WOSACSubmission
        return WOSACSubmission
    raise AttributeError(name)
