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

from typing import Optional

from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater

from src.unimotion.datasets import MultiDataset
from src.unimotion.tokens.token_processor import TokenProcessor

from .target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal


class CustomCollater(Collater):
    def __call__(self, batch):
        num_a = [sam["agent"]["num_nodes"] for sam in batch]
        max_num = max([elem["agent"]["num_nodes"] for elem in batch])
        if "norm_agent_pos" in batch[0]["agent"]:
            for i in range(len(batch)):
                num_nodes = batch[i]["agent"]["num_nodes"]
                norm_pos, norm_h = \
                    batch[i]["agent"]["norm_agent_pos"], batch[i]["agent"]["norm_agent_head"]
                norm_idx = batch[i]["agent"]["norm_agent_idx"]
                norm_mask = batch[i]["agent"]["norm_agent_mask"]
                num_step = norm_pos.size(2)
                new_idx = norm_pos.new_zeros((num_nodes, max_num)).long() - 1
                new_pos = norm_pos.new_zeros((num_nodes, max_num, num_step, 2))
                new_h = norm_pos.new_zeros((num_nodes, max_num, num_step))
                new_mask = norm_pos.new_zeros((num_nodes, max_num, num_step)).bool()
                new_idx[:, :num_nodes] = norm_idx
                new_pos[:, :num_nodes] = norm_pos
                new_h[:, :num_nodes] = norm_h
                new_mask[:, :num_nodes] = norm_mask
                batch[i]["agent"]["norm_agent_idx"] = new_idx
                batch[i]["agent"]["norm_agent_pos"] = new_pos
                batch[i]["agent"]["norm_agent_head"] = new_h
                batch[i]["agent"]["norm_agent_mask"] = new_mask
        return super().__call__(batch)

class MultiDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        train_raw_dir: str,
        val_raw_dir: str,
        test_raw_dir: str,
        val_tfrecords_splitted: str,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
        persistent_workers: bool,
        train_max_num: int,
        train_interval: int,
        train_task: str,
        eval_task: str,
        token_processor: dict,
        use_intention: bool = False,
    ) -> None:
        super(MultiDataModule, self).__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.val_tfrecords_splitted = val_tfrecords_splitted
        self.train_interval = train_interval
        self.train_tokenizer = TokenProcessor(**token_processor, training=True)
        self.val_tokenizer = TokenProcessor(**token_processor, training=False)
        self.train_transform = WaymoTargetBuilderTrain(self.train_tokenizer, max_num=train_max_num,
                                                       task=train_task, use_intention=use_intention)
        self.val_transform = WaymoTargetBuilderVal(self.val_tokenizer, task=eval_task, use_intention=use_intention)
        self.test_transform = WaymoTargetBuilderVal(self.val_tokenizer, task=eval_task, use_intention=use_intention)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = MultiDataset(self.train_raw_dir, self.train_transform, interval=self.train_interval)
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "validate":
            self.val_dataset = MultiDataset(
                self.val_raw_dir,
                self.val_transform,
                tfrecord_dir=self.val_tfrecords_splitted,
            )
        elif stage == "test":
            self.test_dataset = MultiDataset(self.test_raw_dir, self.test_transform)
        else:
            raise ValueError(f"{stage} should be one of [fit, validate, test]")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=CustomCollater(self.train_dataset),
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=CustomCollater(self.val_dataset),
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,  # 0
            pin_memory=self.pin_memory,  # False
            persistent_workers=self.persistent_workers,
            drop_last=False,
            collate_fn=CustomCollater(self.test_dataset),
        )
