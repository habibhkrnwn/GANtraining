from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def append_markdown_run_report(
    report_path: str | Path,
    stage: str,
    summary: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
) -> Path:
    """Append one run report entry to a markdown tracking file."""
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(f"## {timestamp} | {stage}")
    lines.append("")
    lines.append("### Summary")
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    if details:
        lines.append("")
        lines.append("### Details")
        for key, value in details.items():
            lines.append(f"- {key}: {value}")

    lines.append("")
    text = "\n".join(lines)

    if out_path.exists():
        with out_path.open("a", encoding="utf-8") as file:
            file.write("\n")
            file.write(text)
            file.write("\n")
    else:
        header = "# Run Tracking Log\n\n"
        with out_path.open("w", encoding="utf-8") as file:
            file.write(header)
            file.write(text)
            file.write("\n")

    return out_path
