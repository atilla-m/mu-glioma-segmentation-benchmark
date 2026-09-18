#!/usr/bin/env python3
"""Generate vector figures for the current manuscript."""

from __future__ import annotations

import csv
import html
import shutil
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_ROOT = PACKAGE_ROOT / "manuscript" / "figures"
TRACKER = PACKAGE_ROOT / "results" / "progress" / "MU_Glioma_35_Run_Progress.csv"


def esc(value: object) -> str:
    return html.escape(str(value))


def rect(x, y, width, height, fill, stroke="#17365d", radius=18, stroke_width=3):
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )


def text(x, y, value, size=24, weight=400, fill="#17202a", anchor="middle"):
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Liberation Sans, Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#506477"):
    return (
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{color}" '
        'stroke-width="4" marker-end="url(#arrowhead)"/>'
    )


def study_design_svg() -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="820" viewBox="0 0 1400 820">',
        '<defs><marker id="arrowhead" markerWidth="12" markerHeight="8" refX="10" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="#506477"/></marker></defs>',
        '<rect width="1400" height="820" fill="#ffffff"/>',
        text(700, 48, "Nested patient-grouped evaluation", 32, 700, "#17365d"),
        text(700, 82, "All longitudinal timepoints from one patient remain together", 20, 400, "#59636e"),
        rect(70, 125, 260, 130, "#eaf2f8"),
        text(200, 166, "Eligible dataset", 25, 700),
        text(200, 202, "203 patients", 23, 400),
        text(200, 234, "594 labeled scans", 23, 400),
        arrow(330, 190, 420, 190),
        rect(420, 120, 300, 140, "#eef1f4"),
        text(570, 160, "Five outer rotations", 25, 700),
        text(570, 197, "Patient grouped", 22, 400),
        text(570, 229, "Frozen seed 2026", 22, 400),
        arrow(720, 160, 820, 160),
        arrow(720, 220, 820, 405),
        rect(820, 105, 480, 145, "#fff4e5", "#a65f00"),
        text(1060, 145, "Locked outer-test fold", 25, 700, "#7a4300"),
        text(1060, 183, "40–41 patients; 118–119 scans", 22, 400),
        text(1060, 218, "Never used for optimization", 22, 700, "#9b1c31"),
        rect(820, 330, 480, 150, "#edf7f2", "#2d7d5e"),
        text(1060, 371, "Outer-training remainder", 25, 700, "#245f49"),
        text(1060, 410, "162–163 patients; 475–476 scans", 22, 400),
        text(1060, 446, "Split again by patient", 22, 400),
        '<path d="M 940 480 L 940 510 L 815 510 L 815 540" fill="none" stroke="#506477" stroke-width="4" marker-end="url(#arrowhead)"/>',
        '<path d="M 1180 480 L 1180 540" fill="none" stroke="#506477" stroke-width="4" marker-end="url(#arrowhead)"/>',
        rect(650, 540, 330, 130, "#eaf2f8"),
        text(815, 580, "Fit partition", 25, 700),
        text(815, 617, "130 patients", 22, 400),
        text(815, 650, "379–381 scans", 22, 400),
        rect(1030, 540, 300, 130, "#f3eef8", "#67408b"),
        text(1180, 580, "Inner tuning", 25, 700, "#553274"),
        text(1180, 617, "32–33 patients", 22, 400),
        text(1180, 650, "94–96 scans", 22, 400),
        rect(420, 540, 180, 130, "#edf7f2", "#2d7d5e"),
        text(510, 579, "Train", 25, 700, "#245f49"),
        text(510, 615, "5 models", 22, 400),
        text(510, 647, "per fold", 22, 400),
        arrow(650, 605, 600, 605),
        rect(70, 535, 280, 140, "#f3eef8", "#67408b"),
        text(210, 575, "Select checkpoint", 24, 700, "#553274"),
        text(210, 612, "Inner tuning only", 22, 400),
        text(210, 646, "Prespecified stopping", 20, 400),
        arrow(420, 605, 350, 605),
        '<path d="M 1030 605 L 1000 605 L 1000 510 L 210 510 L 210 535" fill="none" stroke="#506477" stroke-width="4" marker-end="url(#arrowhead)"/>',
        '<path d="M 210 675 L 210 690 L 570 690 L 570 708" fill="none" stroke="#506477" stroke-width="4" marker-end="url(#arrowhead)"/>',
        '<path d="M 1300 177 L 1360 177 L 1360 690 L 1190 690 L 1190 708" fill="none" stroke="#506477" stroke-width="4" marker-end="url(#arrowhead)"/>',
        rect(420, 708, 880, 92, "#17365d", "#17365d", 14),
        text(860, 744, "Selected checkpoint + locked outer fold", 20, 700, "#ffffff"),
        text(860, 778, "One-time full-volume inference → validated ZIP → patient-level analysis", 18, 700, "#ffffff"),
        '</svg>',
    ]
    return "".join(parts)


def matrix_status_svg() -> str:
    with TRACKER.open(newline="", encoding="utf-8") as handle:
        rows = {row["run_id"]: row for row in csv.DictReader(handle)}
    matrix = [
        ("3D U-Net", [1, 2, 3, 4, 5, 6, 7]),
        ("Residual 3D U-Net", [8, 9, 10, 11, 12, 13, 14]),
        ("3D V-Net", [15, 16, 17, 18, 19, 20, 21]),
        ("3D U-Net++", [22, 23, 24, 25, 26, 27, 28]),
        ("nnU-Net v2 3D fullres", [29, 30, 31, 32, 33, 34, 35]),
    ]
    headers = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Seed 2027", "Seed 2028"]
    colors = {
        "DONE_VALIDATED": ("#2d7d5e", "#ffffff", "Validated"),
        "RUNNING_RUNPOD": ("#2f6fad", "#ffffff", "Running"),
        "QUEUED_RUNPOD": ("#e0a33a", "#17202a", "Queued"),
    }
    x0, y0, cell_w, cell_h = 410, 150, 132, 86
    width, height = 1400, 730
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="1400" height="730" fill="#ffffff"/>',
        text(700, 48, "Prespecified experiment matrix", 32, 700, "#17365d"),
        text(700, 82, "Five primary outer folds plus two fold-1 training-seed sensitivity runs", 20, 400, "#59636e"),
    ]
    for column, header in enumerate(headers):
        x = x0 + column * cell_w + cell_w / 2
        parts.append(text(x, 128, header, 18, 700, "#334455"))
    for row_index, (label, run_numbers) in enumerate(matrix):
        y = y0 + row_index * cell_h
        parts.append(text(385, y + 52, label, 20, 700, "#263746", "end"))
        for column, run_number in enumerate(run_numbers):
            run_id = f"no{run_number}"
            status = rows[run_id]["status"]
            fill, font, label_status = colors.get(status, ("#c9ced3", "#17202a", "Pending"))
            x = x0 + column * cell_w
            parts.append(rect(x + 6, y + 7, cell_w - 12, cell_h - 14, fill, fill, 12, 1))
            parts.append(text(x + cell_w / 2, y + 40, run_id, 21, 700, font))
            parts.append(text(x + cell_w / 2, y + 65, label_status, 14, 400, font))
    legend_y = 625
    legend = [("#2d7d5e", "Validated"), ("#2f6fad", "Running"), ("#e0a33a", "Queued"), ("#c9ced3", "Other/pending")]
    start_x = 365
    for i, (color, label) in enumerate(legend):
        x = start_x + i * 205
        parts.append(rect(x, legend_y, 30, 30, color, color, 5, 1))
        parts.append(text(x + 42, legend_y + 23, label, 17, 400, "#334455", "start"))
    done = sum(row["status"] == "DONE_VALIDATED" for row in rows.values())
    parts.append(text(700, 700, f"Snapshot: {done}/35 archives formally validated", 20, 700, "#17365d"))
    parts.append('</svg>')
    return "".join(parts)


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    figures = {"study_design": study_design_svg()}
    for stem, content in figures.items():
        svg_path = FIGURE_ROOT / f"{stem}.svg"
        svg_path.write_text(content, encoding="utf-8")
        converter = shutil.which("convert")
        if converter:
            subprocess.run(
                [converter, "-background", "white", "-density", "150", str(svg_path), str(FIGURE_ROOT / f"{stem}.png")],
                check=True,
            )
    print(f"Wrote figures to {FIGURE_ROOT}")


if __name__ == "__main__":
    main()
