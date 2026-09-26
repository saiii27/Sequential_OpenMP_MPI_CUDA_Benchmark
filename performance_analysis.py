"""
performance_analysis.py
=======================
Performance comparison of Parallel Matrix Multiplication implementations.

Implementations : Sequential, OpenMP (8 threads), MPI (4 processes), CUDA (RTX 5060 Ti)
Matrix Size     : 4000 × 4000
GPU             : NVIDIA GeForce RTX 5060 Ti

Usage:
    python3 performance_analysis.py

Output:
    - performance_comparison.png   (saved in the same directory)
    - Printed summary in the terminal
"""

import matplotlib
matplotlib.use("Agg")           # Non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Measured execution times (seconds) ──────────────────────────────────────
implementations = ["Sequential", "OpenMP\n(8 threads)", "MPI\n(4 processes)", "CUDA\n(RTX 5060 Ti)"]
times           = [329.438233,    99.856072,              101.025582,           0.331008]

labels_short    = ["Sequential", "OpenMP", "MPI", "CUDA"]

seq_time = times[0]
speedups = [seq_time / t for t in times]

# ── Terminal summary ─────────────────────────────────────────────────────────
print("=" * 60)
print("  Performance Comparison – Parallel Matrix Multiplication")
print("  Matrix Size: 4000 × 4000")
print("=" * 60)
print(f"  {'Implementation':<20} {'Time (s)':>14} {'Speedup':>10}")
print("-" * 60)
for i, (label, t, sp) in enumerate(zip(labels_short, times, speedups)):
    if i == 0:
        print(f"  {label:<20} {t:>14.6f} {'1.0000':>10}")
    else:
        print(f"  {label:<20} {t:>14.6f} {sp:>10.4f}x")
print("=" * 60)
print()

# ── Colour palette (academic / clean) ───────────────────────────────────────
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

x      = np.arange(len(implementations))
bars   = ax.bar(x, times, color=COLORS, width=0.55,
                edgecolor="white", linewidth=0.8, zorder=3)

# ── Annotate each bar with the exact time ────────────────────────────────────
for bar, t in zip(bars, times):
    height = bar.get_height()
    if height < 10:
        label_str = f"{t:.6f}s"
    else:
        label_str = f"{t:.6f}s"
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + seq_time * 0.012,
        label_str,
        ha="center", va="bottom",
        fontsize=9.5, fontweight="bold", color="#222222"
    )

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(implementations, fontsize=11)
ax.set_ylabel("Execution Time (seconds)", fontsize=12, labelpad=10)
ax.set_xlabel("Implementation", fontsize=12, labelpad=10)
ax.set_title(
    "Performance Comparison of Matrix Multiplication\n"
    "(4000 × 4000 Matrix | NVIDIA GeForce RTX 5060 Ti)",
    fontsize=13, fontweight="bold", pad=16
)

ax.set_ylim(0, seq_time * 1.20)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

# Grid on y-axis only
ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Legend / annotation box ───────────────────────────────────────────────────
speedup_lines = [
    "Sequential : 1.0000×  (baseline)",
    f"OpenMP     : {speedups[1]:.4f}×",
    f"MPI        : {speedups[2]:.4f}×",
    f"CUDA       : {speedups[3]:.4f}×",
]
annotation_text = "Speedup over Sequential\n" + "\n".join(speedup_lines)

ax.text(
    0.99, 0.97,
    annotation_text,
    transform=ax.transAxes,
    fontsize=8.5,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F4F8",
              edgecolor="#AAAAAA", alpha=0.9),
    family="monospace"
)

plt.tight_layout()

output_file = "performance_comparison.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"  Graph saved as: {output_file}")
print()
