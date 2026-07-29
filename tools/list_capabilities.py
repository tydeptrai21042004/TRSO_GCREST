"""Print the repository capability matrix as JSON or Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capability_catalog import as_dict


def markdown(catalog: dict) -> str:
    lines = ["# Repository capability matrix", "", "## Tasks and metrics", ""]
    lines.append("| Task | Primary metric | Direction | Required metrics |")
    lines.append("|---|---|---|---|")
    for name, spec in catalog["tasks"].items():
        direction = "maximize" if spec["maximize"] else "minimize"
        lines.append(f"| `{name}` | `{spec['primary_metric']}` | {direction} | {', '.join(spec['required_metrics'])} |")
    lines += ["", "## Open datasets", "", "| Dataset | Task | Access | Download | Notes |", "|---|---|---|---|---|"]
    for row in catalog["open_datasets"]:
        lines.append(f"| `{row['name']}` | `{row['task']}` | {row['access']} | {row['download']} | {row['notes']} |")
    lines += ["", "## Backbone families", ""]
    for family, row in catalog["backbones"].items():
        lines.append(f"### {family.replace('_', ' ').title()}")
        lines.append(f"- Sources: {', '.join(row['sources'])}")
        lines.append(f"- Examples: {', '.join(row['examples'])}")
        lines.append("")
    lines += ["## Strict benchmark", ""]
    lines.append("- Reference controls: " + ", ".join(f"`{name}`" for name in catalog["reference_controls"]))
    lines.append("- Proposal: `trso`")
    lines.append("- Original-paper baselines: " + ", ".join(f"`{name}`" for name in catalog["strict_paper_baselines"]))
    lines.append("- Opt-in engineering controls: " + ", ".join(f"`{name}`" for name in catalog["engineering_controls_opt_in"]))
    lines.append("- Opt-in transferred controls: " + ", ".join(f"`{name}`" for name in catalog["transferred_controls_opt_in"]))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    catalog = as_dict()
    text = json.dumps(catalog, indent=2) if args.format == "json" else markdown(catalog)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
