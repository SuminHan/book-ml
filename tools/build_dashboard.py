#!/usr/bin/env python3
"""Regenerate progress data + build the dashboard HTML for publishing.
Usage: python3 tools/build_dashboard.py [output_path]
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("/home/smhan/book-ml")
DEFAULT_OUT = Path(
    "/tmp/claude-1002/-home-smhan/4fdcb316-9aa8-40bb-b9a3-9d42af7a1d1f/scratchpad/dashboard.html"
)


MDBOOK = "/home/smhan/miniconda3/bin/mdbook"


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT

    # Rebuild docs/kor so generate_progress_data.py can pull real rendered
    # HTML (correct links/lists/tables) for "done" sections' modal content.
    # mdbook rewrites ~every file (hashed asset names), so we revert docs/
    # afterward -- this is a read-only extraction pass, not a real publish.
    subprocess.run([MDBOOK, "build"], cwd=REPO / "kor", check=True,
                    capture_output=True, text=True)

    data_json = subprocess.run(
        ["python3", "tools/generate_progress_data.py"], cwd=REPO,
        capture_output=True, text=True, check=True,
    ).stdout

    subprocess.run(["git", "checkout", "--", "docs/"], cwd=REPO, check=True)
    subprocess.run(["git", "clean", "-fd", "docs/"], cwd=REPO, check=True)

    katex_assets = (REPO / "tools" / "katex_inline_assets.html").read_text()

    template = (REPO / "tools" / "dashboard_template.html").read_text()
    html = template.replace("__PROGRESS_JSON__", data_json)
    html = html.replace("__KATEX_ASSETS__", katex_assets)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    print(f"built {out_path} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
