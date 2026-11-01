# UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning

[Paper](https://arxiv.org/abs/2602.00566) · [Installation](#install) · [Pretraining](#train-validate-and-resume) · [Citation](#citation)

**Nan Song, Junzhe Jiang, Jingyu Li, [Xiatian Zhu](https://scholar.google.com/citations?user=ZbA-z1cAAAAJ&hl=en), [Li Zhang](https://lzrobots.github.io)**

**NeurIPS 2025**

## Abstract

Motion simulation, prediction and planning are foundational tasks in autonomous driving, each essential for modeling and reasoning about dynamic traffic scenarios. While often addressed in isolation due to their differing objectives, such as generating diverse motion states or estimating optimal trajectories, these tasks inherently depend on shared capabilities: understanding multi-agent interactions, modeling motion behaviors, and reasoning over temporal and spatial dynamics.
Despite this underlying commonality, existing approaches typically adopt specialized model designs, which hinders cross-task generalization and system scalability. More critically, this separation overlooks the potential mutual benefits among tasks. Motivated by these observations, we propose UniMotion, a unified motion framework that captures shared structures across motion tasks while accommodating their individual requirements. Built on a decoder-only Transformer architecture, UniMotion employs dedicated interaction modes and tailored training strategies to simultaneously support these motion tasks. This unified design not only enables joint optimization and representation sharing but also allows for targeted fine-tuning to specialize in individual tasks when needed. Extensive experiments on the Waymo Open Motion Dataset demonstrate that joint training leads to robust generalization and effective task integration. With further fine-tuning, UniMotion achieves state-of-the-art performance across a range of motion tasks, establishing it as a versatile and scalable solution for autonomous driving.

## Pipeline

<p align="center">
  <img src="asset/main.jpg" alt="UniMotion paper architecture: map encoding and agent decoding for simulation, prediction and planning." width="900" />
</p>

The figure illustrates the framework described in the paper. The implementation
status and verified training path in this repository are detailed below.


## Install

Use a separate environment, Python 3.10 or 3.11. A tested stack is PyTorch 2.5.1,
Lightning 2.4.0 and PyG 2.6.1. No TensorFlow or Waymo SDK is needed for pretraining
from existing pickle caches or for synthetic verification.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter==2.1.2 torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
pip install -r requirements.txt
```

## Assets and data

With the legacy preprocessing dependencies installed, prepare caches from
locally downloaded WOMD scenarios (paths are examples):

```bash
bash scripts/cache_womd.sh --input_dir /path/to/scenario --split training
bash scripts/cache_womd.sh --input_dir /path/to/scenario --split validation
# Optional: --output_dir /path/to/cache/waymo
```

For training, supply existing WOMD pickle caches:

```text
dataset/
  waymo/training/*.pkl
  waymo/validation/*.pkl
```

## Train, validate and resume

Run from the repository root with the chosen environment activated:

```bash
bash scripts/pretrain.sh
# All training scenarios instead of the original 20% subset:
bash scripts/pretrain.sh data.train_interval=1 data.cache_root=/path/to/cache
# Single-task ablations:
bash scripts/pretrain.sh 'tasks=[sim]'
bash scripts/pretrain.sh 'tasks=[pred]'
# Multiple GPUs:
python -m torch.distributed.run --standalone --nproc_per_node=8 -m src.unimotion.run trainer.devices=8 trainer.strategy=ddp
```


## Citation

```bibtex
@inproceedings{song2025unimotion,
 title={UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning},
 author={Song, Nan and Jiang, Junzhe and Li, Jingyu and Zhu, Xiatian and Zhang, Li},
 booktitle={NeurIPS},
 year={2025},
}
```

## Acknowledgements and licenses

This project builds on [CAT-K](https://github.com/NVlabs/catk) and
[SMART](https://github.com/rainmaker22/SMART). Please also credit these works
when using their contributions.
