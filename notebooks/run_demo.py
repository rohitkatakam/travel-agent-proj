"""Generate a formatted end-to-end transcript for the NYC→Miami happy-path demo.

Reads tc01_happy_path from data/test_dialogues.json, replays it through
run_agent() in batch mode, and writes a clean transcript to
data/processed/demo_transcript.txt.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agent import run_agent

TEST_PATH = Path(__file__).parent.parent / "data" / "test_dialogues.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "processed" / "demo_transcript.txt"


def main() -> None:
  with open(TEST_PATH) as f:
    dialogues = json.load(f)

  tc01 = next(d for d in dialogues if d["id"] == "tc01_happy_path")
  user_turns = [t["text"] for t in tc01["turns"] if t["speaker"] == "user"]

  result = run_agent(turns=user_turns, prompt_optional=True)

  lines = []
  lines.append("=== End-to-End Dialogue Demo: NYC → Miami ===")
  lines.append("")

  for msg in result["transcript"]:
    role = "User" if msg["role"] == "user" else "Agent"
    lines.append(f"{role}: {msg['content']}")
    lines.append("")

  lines.append(f"Action sequence: {' → '.join(result['action_sequence'])}")
  lines.append(f"Task completed:  {result['task_completed']}")
  lines.append(f"Turns:           {result['num_user_turns']}")

  transcript = "\n".join(lines)
  print(transcript)
  OUT_PATH.write_text(transcript)
  print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
  main()
