"""Lightweight prediction metrics; official Waymo evaluation stays optional."""

import torch


@torch.no_grad()
def select_modes(trajectories, scores, count=6, distance=2.5):
    """Greedy endpoint NMS on cumulative positions, with score-ordered fallback."""
    count = min(count, scores.shape[-1])
    order = scores.argsort(dim=-1, descending=True)
    chosen = []
    for row, ranked in enumerate(order):
        selected = []
        for index in ranked.tolist():
            if not selected or ((trajectories[row, selected, -1] - trajectories[row, index, -1])
                                .norm(dim=-1) > distance).all():
                selected.append(index)
            if len(selected) == count:
                break
        selected += [i for i in ranked.tolist() if i not in selected][:count - len(selected)]
        chosen.append(selected)
    indices = torch.tensor(chosen, device=scores.device, dtype=torch.long).reshape(scores.shape[0], count)
    rows = torch.arange(scores.shape[0], device=scores.device)[:, None]
    return trajectories[rows, indices], scores[rows, indices], indices


@torch.no_grad()
def prediction_ade(agent, pred):
    """Current-time minADE sum/count for agents of interest, not dense future tokens."""
    n_agent = agent["type"].size(0)
    trajectory = pred["pred_mm_token_logits"].reshape(n_agent, -1, *pred["pred_mm_token_logits"].shape[1:])[:, 0]
    scores = pred["pred_mm_prob"].reshape(n_agent, -1, pred["pred_mm_prob"].shape[-1])[:, 0]
    trajectory, _, _ = select_modes(trajectory, scores)
    gt = agent["token_pos_gt"][:, 1]
    valid = agent["token_pos_mask"][:, 1]
    interested = agent["role"][:, 2] & valid.any(-1)
    distance = (trajectory - gt[:, None]).norm(dim=-1).masked_fill(~valid[:, None], 0)
    ade = (distance.sum(-1) / valid.sum(-1).clamp_min(1)[:, None]).min(-1).values
    return ade.masked_fill(~interested, 0).sum(), interested.sum()
