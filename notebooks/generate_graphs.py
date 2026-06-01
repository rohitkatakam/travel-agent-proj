"""Generate evaluation graphs for the presentation slides.

Reads data/processed/eval_results.json and produces:
  graph_slot_accuracy.png   — per-slot DST accuracy (horizontal bar)
  graph_overall_metrics.png — all top-level metrics (vertical bar)
  graph_turns.png           — turns-to-completion distribution

All files saved to data/processed/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_PATH = Path(__file__).parent.parent / "data" / "processed" / "eval_results.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "processed"

NU_PURPLE = "#4E2A84"
NU_LIGHT  = "#836EAA"
NU_GREY   = "#B6ACD1"

SLOT_LABELS = {
  "origin":       "Origin",
  "destination":  "Destination",
  "depart_date":  "Depart Date",
  "return_date":  "Return Date",
  "budget_usd":   "Budget (USD)",
  "num_travelers":"# Travelers",
  "preferences":  "Preferences",
}


def graph_slot_accuracy(per_slot: dict) -> None:
  """Horizontal bar chart of per-slot DST accuracy, sorted descending."""
  pairs = sorted(per_slot.items(), key=lambda x: x[1], reverse=True)
  slots  = [SLOT_LABELS.get(k, k) for k, _ in pairs]
  values = [v * 100 for _, v in pairs]

  fig, ax = plt.subplots(figsize=(8, 4.5))
  colors = [NU_PURPLE if v >= 90 else NU_LIGHT if v >= 60 else NU_GREY for v in values]
  bars = ax.barh(slots, values, color=colors, edgecolor="white", height=0.6)

  for bar, val in zip(bars, values):
    ax.text(
      bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
      f"{val:.0f}%", va="center", ha="left", fontsize=10, color="#333333"
    )

  ax.set_xlim(0, 115)
  ax.set_xlabel("Accuracy (%)", fontsize=11)
  ax.set_title("DST Slot Accuracy (n=20 test dialogues)", fontsize=13, fontweight="bold", pad=12)
  ax.axvline(x=100, color="#cccccc", linewidth=0.8, linestyle="--")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.tick_params(axis="y", labelsize=10)

  legend_patches = [
    mpatches.Patch(color=NU_PURPLE, label="≥ 90%"),
    mpatches.Patch(color=NU_LIGHT,  label="60–89%"),
    mpatches.Patch(color=NU_GREY,   label="< 60%"),
  ]
  ax.legend(handles=legend_patches, loc="lower right", fontsize=9, framealpha=0.7)

  fig.tight_layout()
  out = OUT_DIR / "graph_slot_accuracy.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved {out}")


def graph_overall_metrics(results: dict) -> None:
  """Vertical bar chart of all top-level evaluation metrics."""
  dst       = results["dst"]
  retrieval = results["retrieval"]
  policy    = results["policy"]
  e2e       = results["end_to_end"]

  labels = [
    "Slot\nAccuracy",
    "Joint Goal\nAccuracy",
    "Flight\nRecall@5",
    "Hotel\nRecall@5",
    "Policy\nAccuracy",
    "Task\nCompletion",
  ]
  raw_values = [
    dst["slot_accuracy"],
    dst["joint_goal_accuracy"],
    retrieval["flight_recall_at_k"],
    retrieval["hotel_recall_at_k"],
    policy["action_accuracy"],
    e2e["task_completion_rate"],
  ]
  values = [v * 100 if v is not None else 0 for v in raw_values]

  fig, ax = plt.subplots(figsize=(10, 5.5))
  x = np.arange(len(labels))
  bar_colors = [NU_PURPLE if v >= 80 else NU_LIGHT if v >= 50 else NU_GREY for v in values]
  bars = ax.bar(x, values, color=bar_colors, edgecolor="white", width=0.6)

  for bar, val in zip(bars, values):
    ax.text(
      bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
      f"{val:.1f}%", ha="center", va="bottom", fontsize=10, color="#333333"
    )

  ax.set_xticks(x)
  ax.set_xticklabels(labels, fontsize=10)
  ax.set_ylim(0, 118)
  ax.set_ylabel("Score (%)", fontsize=11)
  ax.set_title("Evaluation Summary (n=20 test dialogues)", fontsize=13, fontweight="bold", pad=12)
  ax.axhline(y=100, color="#cccccc", linewidth=0.8, linestyle="--")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  legend_patches = [
    mpatches.Patch(color=NU_PURPLE, label="≥ 80%"),
    mpatches.Patch(color=NU_LIGHT,  label="50–79%"),
    mpatches.Patch(color=NU_GREY,   label="< 50%"),
  ]
  ax.legend(handles=legend_patches, loc="upper right", fontsize=9, framealpha=0.7)

  fig.tight_layout()
  out = OUT_DIR / "graph_overall_metrics.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved {out}")


def graph_turns(results: dict, test_dialogues: list) -> None:
  """Bar chart of turns-to-completion across all test dialogues."""
  from agent import run_agent

  turns_per_dialogue = []
  labels_list = []
  completed_flags = []

  for dlg in test_dialogues:
    user_turns = [t["text"] for t in dlg["turns"] if t["speaker"] == "user"]
    try:
      outcome = run_agent(turns=user_turns, prompt_optional=False)
      turns_per_dialogue.append(outcome["num_user_turns"])
      completed_flags.append(outcome["task_completed"])
    except Exception:
      turns_per_dialogue.append(0)
      completed_flags.append(False)
    labels_list.append(dlg["id"].replace("tc0", "tc").replace("tc", "TC"))

  avg = results["end_to_end"]["avg_turns"] or 0

  fig, ax = plt.subplots(figsize=(12, 5))
  x = np.arange(len(labels_list))
  bar_colors = [NU_PURPLE if c else NU_GREY for c in completed_flags]
  bars = ax.bar(x, turns_per_dialogue, color=bar_colors, edgecolor="white", width=0.7)

  ax.axhline(y=avg, color="#E84A27", linewidth=1.5, linestyle="--", label=f"Avg = {avg:.1f} turns")
  ax.set_xticks(x)
  ax.set_xticklabels(labels_list, rotation=45, ha="right", fontsize=8)
  ax.set_ylabel("User Turns", fontsize=11)
  ax.set_title("Turns to Completion per Test Dialogue", fontsize=13, fontweight="bold", pad=12)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  legend_patches = [
    mpatches.Patch(color=NU_PURPLE, label="Task completed"),
    mpatches.Patch(color=NU_GREY,   label="Not completed"),
  ]
  ax.legend(handles=legend_patches + [
    plt.Line2D([0], [0], color="#E84A27", linestyle="--", linewidth=1.5, label=f"Avg = {avg:.1f}")
  ], loc="upper right", fontsize=9, framealpha=0.7)

  fig.tight_layout()
  out = OUT_DIR / "graph_turns.png"
  fig.savefig(out, dpi=150, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved {out}")


def main() -> None:
  with open(RESULTS_PATH) as f:
    results = json.load(f)

  from pathlib import Path as P
  import json as js
  test_path = P(__file__).parent.parent / "data" / "test_dialogues.json"
  with open(test_path) as f:
    test_dialogues = js.load(f)

  graph_slot_accuracy(results["dst"]["per_slot_accuracy"])
  graph_overall_metrics(results)
  graph_turns(results, test_dialogues)
  print("All graphs saved.")


if __name__ == "__main__":
  main()
