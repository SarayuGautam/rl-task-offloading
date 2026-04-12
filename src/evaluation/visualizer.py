# =============================================================================
# evaluation/visualizer.py
# Module 6C: Visualization Techniques
#
# Generates 4 plots:
#   1. Learning curve (reward over episodes with moving average)
#   2. Comparison bar chart (avg latency — Q-Learning vs baselines)
#   3. Q-value heatmap (learned policy across state space)
#   4. Action distribution over training (how behaviour evolves)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import List, Optional


# Colour palette — consistent across all plots
COLORS = {
    'Q-Learning'   : '#2E86AB',
    'Always Local' : '#A23B72',
    'Always Edge'  : '#F18F01',
    'Always Cloud' : '#C73E1D',
    'Random'       : '#8E8E8E',
}


def plot_learning_curve(
    episode_rewards: List[float],
    window: int = 50,
    save_path: Optional[str] = None,
):
    """
    Plot 1: reward vs episode with a moving-average smoothing.
    Shows the agent improving from random exploration to stable policy.
    """
    rewards = np.array(episode_rewards)
    episodes = np.arange(1, len(rewards) + 1)

    # Moving average
    ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
    ma_x = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(episodes, rewards, alpha=0.25, color=COLORS['Q-Learning'], linewidth=0.8, label='Episode reward')
    ax.plot(ma_x, ma, color=COLORS['Q-Learning'], linewidth=2.0, label=f'{window}-ep moving avg')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('Q-Learning: reward convergence over training', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.show()
    plt.close()


def plot_comparison_bars(
    all_metrics: dict,
    metric: str = 'avg_latency',
    ylabel: str = 'Average Latency (s)',
    title: str = 'Average Latency: Q-Learning vs Baselines',
    save_path: Optional[str] = None,
):
    """
    Plot 2: bar chart comparing Q-Learning vs each baseline strategy.
    """
    names  = list(all_metrics.keys())
    values = [all_metrics[n][metric] for n in names]
    colors = [COLORS.get(n, '#555555') for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, values, color=colors, edgecolor='white', linewidth=0.5)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Highlight Q-Learning bar
    if 'Q-Learning' in names:
        idx = names.index('Q-Learning')
        bars[idx].set_edgecolor('#1a1a1a')
        bars[idx].set_linewidth(2)

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis='x', labelsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.show()
    plt.close()


def plot_qtable_heatmap(
    q_table: dict,
    save_path: Optional[str] = None,
):
    """
    Plot 3: heatmap of Q-values showing learned policy across state space.
    Each row = a state, each column = an action (Local/Edge/Cloud).
    The brightest cell = greedy action chosen.
    """
    if not q_table:
        print('Q-table is empty — skipping heatmap')
        return

    states = sorted(q_table.keys())
    q_matrix = np.array([q_table[s] for s in states])

    fig, ax = plt.subplots(figsize=(6, max(3, len(states) * 0.5)))
    im = ax.imshow(q_matrix, aspect='auto', cmap='Blues')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Local', 'Edge', 'Cloud'], fontsize=11)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([f'State {s}' for s in states], fontsize=9)
    ax.set_title('Q-table: learned action values per state\n(darker = higher value = preferred)', fontsize=11)

    # Mark the greedy (best) action per row with a white star
    for i, row in enumerate(q_matrix):
        best = int(np.argmax(row))
        ax.text(best, i, '★', ha='center', va='center', fontsize=14, color='white')

    plt.colorbar(im, ax=ax, label='Q-value')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.show()
    plt.close()


def plot_improvement_summary(
    all_metrics: dict,
    ql_key: str = 'Q-Learning',
    save_path: Optional[str] = None,
):
    """
    Plot 4: horizontal bar chart of % latency improvement over each baseline.
    Positive = Q-Learning wins.
    """
    from src.evaluation.metrics import improvement_over

    if ql_key not in all_metrics:
        print('Q-Learning metrics not found — skipping improvement plot')
        return

    baselines = [n for n in all_metrics if n != ql_key]
    ql = all_metrics[ql_key]
    improvements = [improvement_over(ql, all_metrics[b], 'avg_latency') for b in baselines]
    bar_colors   = [COLORS.get(b, '#555555') for b in baselines]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.barh(baselines, improvements, color=bar_colors, edgecolor='white')

    for bar, val in zip(bars, improvements):
        xpos = val + 0.5 if val >= 0 else val - 0.5
        align = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.1f}%', va='center', ha=align, fontsize=10, fontweight='bold')

    ax.axvline(0, color='black', linewidth=0.8)
    ax.axvline(10, color='green', linewidth=1.2, linestyle='--', label='10% target')
    ax.set_xlabel('Latency improvement (%)', fontsize=12)
    ax.set_title('Q-Learning latency improvement over each baseline', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f'Saved → {save_path}')
    plt.show()
    plt.close()
