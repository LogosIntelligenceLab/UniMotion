"""Shared map context and agent decoder for UniMotion pretraining."""

from torch import nn

from src.unimotion.modules.map_decoder import UniMotionMapDecoder
from .decoder import AgentDecoder


class UniMotion(nn.Module):
    def __init__(self, tasks, decoder):
        super().__init__()
        config = dict(decoder)
        map_layers = config.pop("num_map_layers")
        agent_layers = config.pop("num_agent_layers")
        map_radius = config.pop("pl2pl_radius")
        self.map_encoder = UniMotionMapDecoder(
            **{key: config[key] for key in (
                "hidden_dim", "num_freq_bands", "num_heads", "head_dim", "dropout"
            )}, num_layers=map_layers, pl2pl_radius=map_radius,
        )
        self.agent_encoder = AgentDecoder(task=tasks, num_layers=agent_layers, **config)

    def forward(self, tokenized_map, tokenized_agent, task):
        return self.agent_encoder(tokenized_agent, self.map_encoder(tokenized_map), task)
