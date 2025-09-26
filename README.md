# Explaining Grokking and Information Bottleneck through Neural Collapse Emergence

This repository contains the official implementation of the paper: [Explaining Grokking and Information Bottleneck through Neural Collapse Emergence](https://arxiv.org/abs/2509.20829).

<p align="center">
    <img src="https://github.com/keitaroskmt/collapse-dynamics/blob/1c4dc4f2a61d17a5202ffdce2f8d841e6cecc882/img/figure1.png">
<!p>

## Installation

The code uses Python 3.13 and PyTorch 2.7.
We recommend using [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to manage the environment.

To set up the environment, run:

```bash
uv sync
```

If you are new to [`wandb`](https://wandb.ai/site), please login first:

```bash
wandb login
```

## Reproducing Results

To reproduce the results, run:

```bash
uv run train.py seed=$seed \
    num_steps=100000 \
    optimizer=adamw \
    model=toy_mlp \ # or toy_cnn, toy_transformer, resnetXX
    dataset=mnist \ # or fashionmnist, cifar10, sst2, trec, agnews
    train_size=3000 \
    optimizer.weight_decay=1e-2 \
    calc_nhsic=false \ # or true to calculate the normalized HSIC
    calc_mi_estimation=false \ # or true to calculate MI estimation via KDE, which is not used in the paper
    calc_mi_estimation_compression=false \ # or true to calculate MI estimation via autoencoder
    save_model=false \ # or true to save the intermediate models
    wandb.job_type=$job_type
```

After training, you can plot the results by running `visualize.ipynb`.

## Citation

If you find our work useful for your research, please cite using this BibTeX:

```BibTeX
@article{sakamoto2025explaining,
  title={Explaining Grokking and Information Bottleneck through Neural Collapse Emergence},
  author={Sakamoto, Keitaro and Sato, Issei},
  journal={arXiv preprint arXiv:2509.20829},
  year={2025}
}
```
