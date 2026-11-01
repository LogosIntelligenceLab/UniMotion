"""WOMD cache input, synthetic scenarios, and motion token features."""

import copy
import pickle
from pathlib import Path

import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from src.unimotion.tokens.token_processor import TokenProcessor, TokenDict
from src.unimotion.datamodules.target_builder import WaymoTargetBuilderTrain, WaymoTargetBuilderVal

class PretrainingTokenizer(TokenProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if (self.num_historical_steps, self.num_future_steps) != (11, 80):
            raise ValueError("Pretraining requires 11 history and 80 future frames")
        # Build five steps of position, heading vector and displacement features.
        self.all_tokens = TokenDict()
        self.all_tokens.map_token = self.map_token_traj_src
        for kind in ("veh", "ped", "cyc"):
            contour = getattr(self, f"agent_token_all_{kind}")
            position = contour.mean(2)
            direction = contour[:, 1:, 0] - contour[:, 1:, 3]
            heading = torch.atan2(direction[..., 1], direction[..., 0])
            self.all_tokens.agent_token[kind] = torch.cat((
                position[:, 1:], torch.stack((heading.cos(), heading.sin()), -1),
                position[:, 1:] - position[:, :-1],
            ), -1).flatten(1)


class ScenarioTransform:
    def __init__(self, tokenizer, tasks, training, max_agents):
        self.tasks = list(tasks)
        if not self.tasks or set(self.tasks) - {"sim", "pred"}:
            raise ValueError("tasks must contain sim, pred, or both")
        self.base = (WaymoTargetBuilderTrain(tokenizer, max_agents, self.tasks, False)
                     if training else WaymoTargetBuilderVal(tokenizer, self.tasks, False))

    def __call__(self, raw):
        data = self.base(copy.deepcopy(raw))
        return data


def synthetic_scenario(index):
    """Small WOMD-shaped input; no fixture ever stands in for real evaluation."""
    count = 3 + index % 2
    time = torch.arange(91, dtype=torch.float32) * 0.1
    pos = torch.zeros(count, 91, 3)
    pos[:, :, 0] = time * (1 + torch.arange(count)[:, None] * 0.2)
    pos[:, :, 1] = torch.arange(count)[:, None] * 2
    role = torch.zeros(count, 3, dtype=torch.bool)
    role[0, 0] = True
    role[:, 1:] = True
    valid = torch.ones(count, 91, dtype=torch.bool)
    valid[-1, 60:] = False
    valid[1, 30:33] = False
    lines = torch.tensor([[[0., 0.], [2.5, 0.], [5., 0.]],
                          [[0., 4.], [2.5, 4.], [5., 4.]]])
    return {"scenario_id": f"synthetic-{index}", "agent": {
        "num_nodes": count, "valid_mask": valid, "role": role,
        "id": torch.arange(count), "type": torch.arange(count) % 3,
        "position": pos, "heading": torch.zeros(count, 91),
        "velocity": torch.stack((torch.ones(count, 91), torch.zeros(count, 91)), -1),
        "shape": torch.tensor([[2., 4.8, 1.5]]).repeat(count, 1),
    }, "map_save": {"traj_pos": lines, "traj_theta": torch.zeros(2)},
        "pt_token": {"num_nodes": 2, "type": torch.zeros(2, dtype=torch.long),
                     "pl_type": torch.zeros(2, dtype=torch.long), "light_type": torch.zeros(2, dtype=torch.long)}}


class ScenarioDataset(Dataset):
    def __init__(self, transform, directory=None, interval=1, synthetic_count=0):
        self.transform = transform
        self.synthetic_count = synthetic_count
        if interval < 1:
            raise ValueError("train_interval must be positive")
        self.files = [] if synthetic_count else sorted(Path(directory).glob("*.pkl"))[::interval]
        if not synthetic_count and not self.files:
            raise FileNotFoundError(f"No WOMD pickle scenarios in {directory}")

    def __len__(self):
        return self.synthetic_count or len(self.files)

    def __getitem__(self, index):
        if self.synthetic_count:
            raw = synthetic_scenario(index)
        else:
            with self.files[index].open("rb") as handle:
                raw = pickle.load(handle)
        return self.transform(raw)


class PretrainingDataModule(LightningDataModule):
    def __init__(self, tasks, tokenizer, cache_root, train_batch_size=4, val_batch_size=10,
                 num_workers=4, train_interval=5, max_agents=32, synthetic_count=0):
        super().__init__()
        self.save_hyperparameters()
        self.tasks = list(tasks)
        self.tokenizer_config = dict(tokenizer)
        self.root = Path(cache_root)

    def setup(self, stage=None):
        for name, training in (("train", True), ("val", False)):
            if stage == "validate" and training:
                continue
            tokenizer = PretrainingTokenizer(**self.tokenizer_config, training=training)
            transform = ScenarioTransform(tokenizer, self.tasks, training,
                                          self.hparams.max_agents)
            dataset = ScenarioDataset(
                transform, self.root / "waymo" / ("training" if training else "validation"),
                interval=self.hparams.train_interval if training else 1,
                synthetic_count=self.hparams.synthetic_count,
            )
            setattr(self, f"{name}_dataset", dataset)

    def _loader(self, name, batch_size, shuffle):
        return DataLoader(getattr(self, f"{name}_dataset"), batch_size=batch_size,
                          shuffle=shuffle, num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0,
                          pin_memory=False)

    def train_dataloader(self):
        return self._loader("train", self.hparams.train_batch_size, True)

    def val_dataloader(self):
        return self._loader("val", self.hparams.val_batch_size, False)


def prepare_batch(batch):
    tokenized_map, agent = dict(batch["pt_token"]), dict(batch["agent"])
    vocabulary = batch["all_tokens"][0]
    device = agent["batch"].device
    tokenized_map["token_traj_src"] = vocabulary.map_token.to(device)
    for kind in ("veh", "ped", "cyc"):
        agent[f"trajectory_token_{kind}"] = vocabulary.agent_token[kind].to(device)
    agent["num_graphs"] = batch.num_graphs
    return tokenized_map, agent
