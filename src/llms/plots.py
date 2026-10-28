"""Figures for the README. One function per plot; each reads plain dicts/lists from results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#7b8794", "#e4e7eb"
BLUE, ORANGE, GREEN, RED = "#2563eb", "#ea580c", "#16a34a", "#dc2626"


def _style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def learning_curve(points: list[dict], path: Path) -> None:
    pts = sorted(points, key=lambda p: p["n"])
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([p["n"] for p in pts], [p["dev_macro_f1"] for p in pts], marker="o", color=BLUE, linewidth=2)
    for p in pts:
        ax.annotate(f"{p['dev_macro_f1']:.3f}", (p["n"], p["dev_macro_f1"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=INK)
    ax.set_xscale("log")
    _style(ax, "Learning curve", "teacher-labeled training examples (log scale)", "dev macro-F1")
    _save(fig, path)


def reliability(bins_before: list[dict], bins_after: list[dict], ece_before: float, ece_after: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="perfect calibration")
    for bins, color, name, e in ((bins_before, ORANGE, "before", ece_before), (bins_after, BLUE, "after", ece_after)):
        ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], marker="o", color=color,
                linewidth=2, label=f"{name} temperature scaling (ECE {e:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style(ax, "Reliability diagram (test)", "mean confidence in bin", "accuracy in bin")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    _save(fig, path)


def coverage_accuracy(curve: list[dict], chosen: dict | None, teacher_acc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot([c["coverage"] for c in curve], [c["student_accuracy_covered"] for c in curve], color=BLUE,
            linewidth=2, label="student on covered items")
    ax.plot([c["coverage"] for c in curve], [c["cascade_accuracy"] for c in curve], color=GREEN,
            linewidth=2, label="cascade (student + teacher)")
    ax.axhline(teacher_acc, color=ORANGE, linestyle="--", linewidth=1.2, label=f"teacher only ({teacher_acc:.3f})")
    if chosen:
        ax.scatter([chosen["coverage"]], [chosen["cascade_accuracy"]], color=RED, zorder=5, label="cost-optimal threshold")
    _style(ax, "Coverage vs accuracy (dev)", "coverage: share handled by the student", "accuracy")
    ax.legend(frameon=False, fontsize=8, loc="best")
    _save(fig, path)


def non_inferiority_plot(paired: dict, ni: list[dict], path: Path) -> None:
    """Delta_hat with its two-sided CI and one-sided lower bound, against each margin -delta."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    d, lo, hi = paired["delta_hat"], paired["ci_low"], paired["ci_high"]
    lb = ni[0]["lower_bound"] if ni else lo
    ax.hlines(0, lo, hi, color=BLUE, linewidth=6, alpha=0.35, label=f"{paired['ci_level']:.0%} CI")
    ax.plot([lb], [0], marker="|", markersize=22, color=BLUE, label="one-sided lower bound")
    ax.plot([d], [0], marker="o", color=BLUE, markersize=8, label=f"Δ̂ = {d:+.4f}")
    for i, r in enumerate(sorted(ni, key=lambda r: r["margin"])):
        color = GREEN if r["verdict"] == "switch" else RED
        ax.axvline(-r["margin"], color=color, linestyle="--", linewidth=1.2)
        ax.annotate(f"δ={r['margin']:g}: {r['verdict']}", (-r["margin"], 0.85 - 0.22 * i), ha="right",
                    fontsize=8, color=color, xytext=(-3, 0), textcoords="offset points")
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    _style(ax, "Student vs teacher: non-inferiority (test)", "Δ = accuracy(student) − accuracy(teacher)", "")
    ax.legend(frameon=False, fontsize=8, loc="lower left", ncol=3)
    _save(fig, path)


def throughput(rows: list[dict], path: Path) -> None:
    """rows: {engine, concurrency, requests_per_s, p50_latency_s, ...}."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    engines = sorted({r["engine"] for r in rows})
    colors = [BLUE, ORANGE, GREEN, RED, MUTED]
    for eng, color in zip(engines, colors, strict=False):
        rs = sorted((r for r in rows if r["engine"] == eng), key=lambda r: r["concurrency"])
        x = [r["concurrency"] for r in rs]
        axes[0].plot(x, [r["requests_per_s"] for r in rs], marker="o", color=color, linewidth=2, label=eng)
        axes[1].plot(x, [r["p95_latency_s"] * 1000 for r in rs], marker="o", color=color, linewidth=2, label=eng)
    for ax, title, ylab in ((axes[0], "Throughput", "requests / s"), (axes[1], "Tail latency", "p95 latency (ms)")):
        ax.set_xscale("log", base=2)
        _style(ax, title, "concurrency (vLLM) / batch size (HF)", ylab)
        ax.legend(frameon=False, fontsize=8)
    _save(fig, path)
