# =============================================================================
# evaluation/visualizer.py — Charts for Module 6C
# =============================================================================
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def learning_curve(episode_rewards: list, save_path: str, window: int = 50):
    """Reward over training episodes with moving average."""
    fig, ax = plt.subplots(figsize=(9, 4))
    eps = range(1, len(episode_rewards) + 1)
    ax.plot(eps, episode_rewards, alpha=0.25, color='steelblue', linewidth=0.8, label='Episode reward')
    if len(episode_rewards) >= window:
        ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(episode_rewards)+1), ma, color='steelblue', linewidth=2,
                label=f'{window}-ep moving avg')
    ax.set_xlabel('Episode'); ax.set_ylabel('Total reward')
    ax.set_title('Q-Learning: reward convergence over training')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")


def comparison_bar(results: dict, metric: str, save_path: str):
    """Bar chart comparing agents on a single metric."""
    labels = list(results.keys())
    values = [results[k][metric] for k in labels]
    colors = ['#e07b54' if k == 'Q-Learning' else '#aab7c4' for k in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.55)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'Strategy comparison: {metric.replace("_"," ")}')
    ax.grid(axis='y', alpha=0.3); fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")


def qtable_heatmap(q_table: dict, save_path: str):
    """Heatmap of greedy actions across state dimensions."""
    import matplotlib.patches as mpatches
    action_names  = ['Local', 'Edge', 'Cloud']
    action_colors = ['#f0a500', '#2196F3', '#4caf50']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    queue_levels = [0, 1, 2, 3]; size_levels = [0, 1, 2]; net_levels = [0, 1, 2]
    for ax_idx, net_bin in enumerate(net_levels):
        grid = np.full((len(size_levels), len(queue_levels)), -1)
        for qi, q in enumerate(queue_levels):
            for si, s in enumerate(size_levels):
                st = (q, s, net_bin)
                if st in q_table:
                    grid[si, qi] = int(np.argmax(q_table[st]))
        im = axes[ax_idx].imshow(grid, cmap=matplotlib.colors.ListedColormap(action_colors),
                                  vmin=0, vmax=2, aspect='auto')
        axes[ax_idx].set_xticks(range(4)); axes[ax_idx].set_xticklabels(['empty','1-2','3-5','6+'])
        axes[ax_idx].set_yticks(range(3)); axes[ax_idx].set_yticklabels(['small','med','large'])
        axes[ax_idx].set_xlabel('Edge queue'); axes[ax_idx].set_ylabel('Task size')
        axes[ax_idx].set_title(f'Net quality: {["poor","ok","good"][net_bin]}')
        for qi in range(4):
            for si in range(3):
                if grid[si, qi] >= 0:
                    axes[ax_idx].text(qi, si, action_names[grid[si, qi]][0],
                                      ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    patches = [mpatches.Patch(color=c, label=n) for c, n in zip(action_colors, action_names)]
    fig.legend(handles=patches, loc='upper right', fontsize=9)
    fig.suptitle('Learned policy: greedy action per state (L=Local, E=Edge, C=Cloud)')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"Saved: {save_path}")
