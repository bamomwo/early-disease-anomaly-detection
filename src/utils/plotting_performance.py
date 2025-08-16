# # #!/usr/bin/env python3
# # """
# # plotting_performance.py — config-driven (lean)

# # Reads a single CONFIG JSON mapping participant labels -> { loss_path, train_seq, val_seq, test_seq }.
# # Computes best loss (min of val_losses, fallback train_losses) per participant and
# # plots a bar chart vs a chosen sequence count (default: train+val).

# # Example:
# #   python plotting_performance.py \
# #     --config participants_config.json \
# #     --out results/perf.png \
# #     --title "Best Loss vs #Sequences" \
# #     --seq train+val --pdf
# # """

# # from __future__ import annotations

# # import argparse
# # import json
# # from pathlib import Path
# # from typing import Dict, List, Optional, Tuple
# # import matplotlib.pyplot as plt


# # def parse_best_loss(data: dict) -> Tuple[float, int]:
# #     vals = data.get("val_losses")
# #     if not isinstance(vals, list) or not vals:
# #         vals = data.get("train_losses")
# #     if not isinstance(vals, list) or not vals:
# #         raise ValueError("Expected 'val_losses' or 'train_losses' as a non-empty list.")
# #     vals = [float(v) for v in vals]
# #     best_idx = min(range(len(vals)), key=lambda i: vals[i])
# #     return float(vals[best_idx]), int(best_idx)


# # def sequences_from_cfg(cfg_entry: Dict[str, str], mode: str) -> int:
# #     to_int = lambda k: int(cfg_entry.get(k, 0)) if str(cfg_entry.get(k, 0)).strip() != "" else 0
# #     tr, va, te = to_int("train_seq"), to_int("val_seq"), to_int("test_seq")
# #     if mode == "train":
# #         return tr
# #     if mode == "val":
# #         return va
# #     if mode == "test":
# #         return te
# #     if mode == "train+val":
# #         return tr + va
# #     # mode == "total"
# #     return tr + va + te


# # def plot_bar(losses: List[float], labels: List[str], nseqs: List[int], out: Path,
# #              title: Optional[str], add_pdf: bool) -> None:
# #     plt.rcParams.update({
# #         "axes.titlesize": 12,
# #         "axes.labelsize": 11,
# #         "xtick.labelsize": 9,
# #         "ytick.labelsize": 9,
# #         "legend.fontsize": 9,
# #         "figure.dpi": 300,
# #         "savefig.dpi": 300,
# #     })

# #     x = list(range(len(labels)))
# #     fig, ax = plt.subplots(figsize=(10, 4.2))

# #     bars = ax.bar(x, losses, width=0.65, color="white", edgecolor="black", linewidth=1.0)
# #     ax.set_ylabel("Best loss (min of val/train)")
# #     if title:
# #         ax.set_title(title)
# #     ax.set_xticks(x)
# #     ax.set_xticklabels(labels)
# #     ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
# #     ax.set_ylim(bottom=0)

# #     for r, v in zip(bars, losses):
# #         ax.text(r.get_x() + r.get_width()/2, r.get_height(), f"{v:.3g}", ha="center", va="bottom", fontsize=8)

# #     ax2 = ax.twinx()
# #     ax2.plot(x, nseqs, marker="o", linestyle="-", linewidth=1.2, color="black", alpha=0.9)
# #     ax2.set_ylabel("# sequences (right)")
# #     ax2.set_ylim(0, max(nseqs) * 1.2 if nseqs else 1)

# #     fig.tight_layout()

# #     out = out.with_suffix(".png")
# #     fig.savefig(out, bbox_inches="tight")
# #     print(f"Saved: {out}")

# #     if add_pdf:
# #         pdf_path = out.with_suffix(".pdf")
# #         fig.savefig(pdf_path, bbox_inches="tight")
# #         print(f"Saved: {pdf_path}")

# #     plt.close(fig)


# # def main() -> None:
# #     ap = argparse.ArgumentParser(description="Plot best loss vs sequences from a config JSON")
# #     ap.add_argument("--config", type=Path, required=True, help="Config JSON with per-participant metadata")
# #     ap.add_argument("--out", type=Path, required=True, help="Output image path (e.g., results/perf.png)")
# #     ap.add_argument("--title", type=str, default=None)
# #     ap.add_argument("--seq", choices=["train", "val", "test", "train+val", "total"], default="train+val",
# #                     help="Which sequence count to plot on the secondary axis (default: train+val)")
# #     ap.add_argument("--pdf", action="store_true", help="Also save a PDF alongside the PNG")
# #     args = ap.parse_args()

# #     with args.config.open("r") as f:
# #         cfg: Dict[str, dict] = json.load(f)

# #     labels = sorted(cfg.keys())  # stable display

# #     losses, nseqs = [], []
# #     for lab in labels:
# #         entry = cfg[lab]
# #         loss_path = Path(entry["loss_path"])  # required
# #         with loss_path.open("r") as lf:
# #             loss_json = json.load(lf)
# #         best, _ = parse_best_loss(loss_json)
# #         losses.append(best)
# #         nseqs.append(sequences_from_cfg(entry, args.seq))

# #     args.out.parent.mkdir(parents=True, exist_ok=True)
# #     plot_bar(losses, labels, nseqs, args.out, args.title, args.pdf)


# # if __name__ == "__main__":
# #     main()


# #!/usr/bin/env python3
# """
# plotting_performance.py — config-driven (sequences on Y; train/val loss as bars on right axis)

# Reads CONFIG JSON mapping participant -> { loss_path, train_seq, val_seq, test_seq }.
# Plots bar chart where the PRIMARY Y-AXIS is the chosen sequence count.
# Also shows TWO bars per participant on a SECONDARY Y-AXIS: min(train_losses) and min(val_losses).

# Example:
#   python plotting_performance.py \
#     --config participants_config.json \
#     --out results/perf.png \
#     --title "Sequences vs Train/Val Loss" \
#     --seq train+val --pdf
# """

# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple
# import matplotlib.pyplot as plt

# # ------------------------ helpers ------------------------ #

# def _min_or_none(xs):
#     if isinstance(xs, list) and xs:
#         return float(min(float(v) for v in xs))
#     return None


# def parse_min_losses(data: dict) -> Tuple[Optional[float], Optional[float]]:
#     """Return (min_train_loss, min_val_loss), either may be None if missing."""
#     tr = _min_or_none(data.get("train_losses"))
#     va = _min_or_none(data.get("val_losses"))
#     if tr is None and va is None:
#         raise ValueError("Expected 'train_losses' and/or 'val_losses' as non-empty lists.")
#     return tr, va


# def sequences_from_cfg(cfg_entry: Dict[str, str], mode: str) -> int:
#     to_int = lambda k: int(cfg_entry.get(k, 0)) if str(cfg_entry.get(k, 0)).strip() != "" else 0
#     tr, va, te = to_int("train_seq"), to_int("val_seq"), to_int("test_seq")
#     if mode == "train":
#         return tr
#     if mode == "val":
#         return va
#     if mode == "test":
#         return te
#     if mode == "train+val":
#         return tr + va
#     return tr + va + te  # total


# # ------------------------ plotting ------------------------ #

# def plot_sequences_with_losses(labels: List[str], seqs: List[int],
#                                train_losses: List[Optional[float]],
#                                val_losses: List[Optional[float]],
#                                out: Path, title: Optional[str], add_pdf: bool) -> None:
#     plt.rcParams.update({
#         "axes.titlesize": 12,
#         "axes.labelsize": 11,
#         "xtick.labelsize": 9,
#         "ytick.labelsize": 9,
#         "legend.fontsize": 9,
#         "figure.dpi": 300,
#         "savefig.dpi": 300,
#     })

#     x = list(range(len(labels)))
#     fig, ax = plt.subplots(figsize=(10, 4.6))

#     # Primary axis: sequences (wide bars)
#     seq_bars = ax.bar(x, seqs, width=0.6, color="white", edgecolor="black", linewidth=1.0, label="# Sequences")
#     ax.set_ylabel("# Sequences")
#     if title:
#         ax.set_title(title)
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
#     ax.set_ylim(0, (max(seqs) if seqs else 1) * 1.15)

#     # Secondary axis: min train/val loss as two slim bars per participant
#     ax2 = ax.twinx()
#     w2 = 0.22
#     val_bars = []
#     train_bars = []
#     # use hatch patterns for print-friendly distinction
#     for i in x:
#         if val_losses[i] is not None:
#             vb = ax2.bar(i - w2/2, val_losses[i], width=w2, edgecolor="black", facecolor="white",
#                          hatch="//", linewidth=1.0, label="Val loss" if i == 0 else None)
#             val_bars.append(vb)
#         if train_losses[i] is not None:
#             tb = ax2.bar(i + w2/2, train_losses[i], width=w2, edgecolor="black", facecolor="white",
#                          hatch="\\", linewidth=1.0, label="Train loss" if i == 0 else None)
#             train_bars.append(tb)
#     ax2.set_ylabel("Loss (right)")
#     max_loss = max([v for v in (train_losses + val_losses) if v is not None], default=1.0)
#     ax2.set_ylim(0, max_loss * 1.15)

#     # Legend (combine handles from both axes)
#     handles = [seq_bars]  # BarContainer is acceptable
#     labels_ = ["# Sequences"]
#     if val_losses and any(v is not None for v in val_losses):
#         handles.append(val_bars[0])
#         labels_.append("Val loss")
#     if train_losses and any(v is not None for v in train_losses):
#         handles.append(train_bars[0])
#         labels_.append("Train loss")
#     ax.legend(handles, labels_, loc="upper left", frameon=False)

#     fig.tight_layout()

#     out = out.with_suffix(".png")
#     fig.savefig(out, bbox_inches="tight")
#     print(f"Saved: {out}")
#     if add_pdf:
#         pdf_path = out.with_suffix(".pdf")
#         fig.savefig(pdf_path, bbox_inches="tight")
#         print(f"Saved: {pdf_path}")
#     plt.close(fig)


# # ------------------------ main ------------------------ #

# def main() -> None:
#     ap = argparse.ArgumentParser(description="Plot #sequences (primary Y) + min train/val loss bars (secondary Y) from config JSON")
#     ap.add_argument("--config", type=Path, required=True, help="Config JSON with per-participant metadata")
#     ap.add_argument("--out", type=Path, required=True, help="Output image path (e.g., results/perf.png)")
#     ap.add_argument("--title", type=str, default="Sequences vs Train/Val Loss")
#     ap.add_argument("--seq", choices=["train", "val", "test", "train+val", "total"], default="train+val",
#                     help="Which sequence count to use for the primary axis (default: train+val)")
#     ap.add_argument("--pdf", action="store_true", help="Also save a PDF alongside the PNG")
#     args = ap.parse_args()

#     with args.config.open("r") as f:
#         cfg: Dict[str, dict] = json.load(f)

#     labels = sorted(cfg.keys())

#     seqs: List[int] = []
#     tr_losses: List[Optional[float]] = []
#     va_losses: List[Optional[float]] = []

#     for lab in labels:
#         entry = cfg[lab]
#         loss_path = Path(entry["loss_path"])  # required
#         with loss_path.open("r") as lf:
#             loss_json = json.load(lf)
#         tr, va = parse_min_losses(loss_json)
#         tr_losses.append(tr)
#         va_losses.append(va)
#         seqs.append(sequences_from_cfg(entry, args.seq))

#     args.out.parent.mkdir(parents=True, exist_ok=True)
#     plot_sequences_with_losses(labels, seqs, tr_losses, va_losses, args.out, args.title, args.pdf)


# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
plotting_performance.py — config-driven (ONE Y-axis = sequences; two bars per participant)

Reads CONFIG JSON mapping participant -> { loss_path, train_seq, val_seq, test_seq }.
Plots two bars per participant (train_seq and val_seq) on a SINGLE sequences Y-axis.
Best train/val losses are shown as compact annotations above the corresponding bars.

Example:
  python plotting_performance.py \
    --config participants_config.json \
    --out results/perf.png \
    --title "Train/Val Sequences (loss annotated)" --pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

# ------------------------ helpers ------------------------ #

def _min_or_none(xs):
    if isinstance(xs, list) and xs:
        return float(min(float(v) for v in xs))
    return None


def parse_min_losses(data: dict) -> Tuple[Optional[float], Optional[float]]:
    """Return (min_train_loss, min_val_loss), either may be None if missing."""
    tr = _min_or_none(data.get("train_losses"))
    va = _min_or_none(data.get("val_losses"))
    if tr is None and va is None:
        raise ValueError("Expected 'train_losses' and/or 'val_losses' as non-empty lists.")
    return tr, va


# ------------------------ plotting ------------------------ #

def plot_seq_bars_with_loss_labels(labels: List[str], train_seqs: List[int], val_seqs: List[int],
                                   train_losses: List[Optional[float]], val_losses: List[Optional[float]],
                                   out: Path, title: Optional[str], add_pdf: bool) -> None:
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })

    n = len(labels)
    x = list(range(n))
    fig, ax = plt.subplots(figsize=(10, 4.6))

    w = 0.38
    bars_tr = ax.bar([i - w/2 for i in x], train_seqs, width=w, color="white", edgecolor="black", linewidth=1.0, label="Train seq")
    bars_va = ax.bar([i + w/2 for i in x], val_seqs,   width=w, color="white", edgecolor="black", linewidth=1.0, label="Val seq", hatch="//")

    ax.set_ylabel("# Sequences")
    if title:
        ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_ylim(0, max(train_seqs + val_seqs) * 1.15 if (train_seqs or val_seqs) else 1)

    # annotate losses above bars (if available)
    for r, loss in zip(bars_tr, train_losses):
        if loss is not None:
            ax.text(r.get_x() + r.get_width()/2, r.get_height(), f"tr:{loss:.3g}", ha="center", va="bottom", fontsize=8)
    for r, loss in zip(bars_va, val_losses):
        if loss is not None:
            ax.text(r.get_x() + r.get_width()/2, r.get_height(), f"val:{loss:.3g}", ha="center", va="bottom", fontsize=8)

    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()

    out = out.with_suffix(".png")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    if add_pdf:
        pdf_path = out.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    plt.close(fig)


# ------------------------ main ------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(description="Plot two bars per participant (train/val sequences), with loss annotations")
    ap.add_argument("--config", type=Path, required=True, help="Config JSON with per-participant metadata")
    ap.add_argument("--out", type=Path, required=True, help="Output image path (e.g., results/perf.png)")
    ap.add_argument("--title", type=str, default="Train/Val Sequences (loss annotated)")
    ap.add_argument("--pdf", action="store_true", help="Also save a PDF alongside the PNG")
    args = ap.parse_args()

    with args.config.open("r") as f:
        cfg: Dict[str, dict] = json.load(f)

    labels = sorted(cfg.keys())

    train_seqs: List[int] = []
    val_seqs: List[int] = []
    tr_losses: List[Optional[float]] = []
    va_losses: List[Optional[float]] = []

    for lab in labels:
        entry = cfg[lab]
        # sequences
        train_seqs.append(int(entry.get("train_seq", 0)))
        val_seqs.append(int(entry.get("val_seq", 0)))
        # losses for annotation
        loss_path = Path(entry["loss_path"])  # required
        with loss_path.open("r") as lf:
            loss_json = json.load(lf)
        tr, va = parse_min_losses(loss_json)
        tr_losses.append(tr)
        va_losses.append(va)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_seq_bars_with_loss_labels(labels, train_seqs, val_seqs, tr_losses, va_losses, args.out, args.title, args.pdf)


if __name__ == "__main__":
    main()

