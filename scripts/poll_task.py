#!/usr/bin/env python3
"""Poll /agent/tasks/{id} and print compact status."""
from __future__ import annotations

import json
import sys
import urllib.request


def main() -> None:
    task_id = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:3001"
    with urllib.request.urlopen(f"{base}/agent/tasks/{task_id}") as resp:
        data = json.loads(resp.read())
    task = data.get("task", {})
    status = task.get("status")
    events = data.get("build_timeline") or []
    completed = [
        e.get("phase_id") or e.get("id")
        for e in events
        if e.get("status") == "completed"
    ]
    active = next(
        (e.get("phase_id") or e.get("id") for e in events if e.get("status") == "active"),
        None,
    )
    fr = data.get("final_report") or {}
    summary = (fr.get("summary") or fr.get("error") or "")[:200]
    print(f"status={status} completed={len(completed)} active={active}")
    if completed:
        print(f"last_completed={completed[-1]}")
    if summary:
        print(f"summary={summary}")
    sys.exit(0 if status in {"completed", "failed"} else 2)


if __name__ == "__main__":
    main()
