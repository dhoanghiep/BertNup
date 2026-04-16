"""Attention score visualization utilities."""

from __future__ import annotations

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def visualize_token2token_scores(
    scores_mat: np.ndarray,
    x_label_name: str = "Head",
    head: int | None = None,
    xticks: range = range(149),
):
    """Visualize token-to-token attention scores as heatmaps.

    If head is None, plots all heads in a grid.
    Otherwise, plots a single head's attention matrix.
    """
    if head is None:
        fig = plt.figure(figsize=(20, 20))
        for idx, scores in enumerate(scores_mat):
            scores_np = np.array(scores)
            ax = fig.add_subplot(4, 3, idx + 1)
            im = ax.imshow(scores_np, cmap="viridis")
            ax.set_xticks(range(149))
            ax.set_yticks(range(149))
            ax.set_xlabel(f"{x_label_name} {idx}")
            fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=(20, 20))
        score_np = np.array(scores_mat[head])
        im = ax.imshow(score_np, cmap="viridis")
        ax.set_xticks(range(149))
        ax.set_yticks(range(149))
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(xticks)
        ax.set_xlabel(f"Head {head}")
        fig.colorbar(im, fraction=0.046, pad=0.04)
        plt.show()


def visualize_sequence_attention(
    attn_scores: np.ndarray,
    sequence: str,
    vmax: float = 0.2,
    figsize: tuple[int, int] = (30, 3),
):
    """Visualize attention scores for a single sequence as a heatmap."""
    plt.figure(figsize=figsize)
    sns.heatmap(
        attn_scores,
        cmap="YlGnBu",
        vmax=vmax,
        xticklabels=list(sequence.upper()),
    )
    plt.show()


def visualize_dataset_attention(
    pos_attn: np.ndarray,
    neg_attn: np.ndarray,
    vmin: float = 0,
    vmax: float = 0.02,
    figsize: tuple[int, int] = (20, 15),
):
    """Visualize aggregated attention heatmaps for positive and negative subsets."""
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    sns.heatmap(pos_attn, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[0])
    sns.heatmap(neg_attn, cmap="YlGnBu", vmin=vmin, vmax=vmax, ax=axs[1])
    plt.show()


def plot_average_attention_by_position(
    pos_attn: np.ndarray,
    neg_attn: np.ndarray,
):
    """Plot average attention score by sequence position for both classes."""
    ave_pos = np.sum(pos_attn, axis=0) / pos_attn.shape[0]
    ave_neg = np.sum(neg_attn, axis=0) / neg_attn.shape[0]
    plt.plot(ave_pos, label="Nucleosome")
    plt.plot(ave_neg, label="Linker")
    plt.legend()
    plt.show()
