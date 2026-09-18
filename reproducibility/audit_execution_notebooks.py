#!/usr/bin/env python3
"""Static integrity audit for the execution notebooks published in this repository."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = REPOSITORY / "notebooks"
EXECUTION_MANIFEST = NOTEBOOK_ROOT / "execution_notebook_manifest.csv"
RUN_MANIFEST = REPOSITORY / "protocol" / "run_matrix_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def joined_code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def literal_assignments(source: str) -> dict[str, object]:
    values: dict[str, object] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return values
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except (TypeError, ValueError):
            continue
    return values


def main() -> None:
    with EXECUTION_MANIFEST.open(newline="", encoding="utf-8") as handle:
        executions = list(csv.DictReader(handle))
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        runs = {row["run_id"]: row for row in csv.DictReader(handle)}

    assert len(executions) == 36
    assert set(runs) == {f"no{number}" for number in range(1, 36)}
    assert Counter(row["run_id"] for row in executions) == {
        **{f"no{number}": 1 for number in range(1, 36) if number != 30},
        "no30": 2,
    }
    assert all(row["current_run_status"] == "DONE_VALIDATED" for row in executions)

    published_files = sorted(NOTEBOOK_ROOT.glob("*.ipynb"))
    assert len(published_files) == 36
    expected_names = {
        Path(row["packaged_notebook"]).name for row in executions
    }
    assert {path.name for path in published_files} == expected_names

    code_cells = 0
    for execution in executions:
        run_id = execution["run_id"]
        run = runs[run_id]
        path = NOTEBOOK_ROOT / Path(execution["packaged_notebook"]).name
        assert sha256(path) == execution["execution_notebook_sha256"], path

        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert isinstance(notebook.get("metadata"), dict)
        for index, cell in enumerate(notebook["cells"]):
            assert cell["cell_type"] in {"code", "markdown"}
            assert isinstance(cell["source"], list)
            if cell["cell_type"] != "code":
                continue
            code_cells += 1
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
            compile("".join(cell["source"]), f"{path.name}:cell{index}", "exec")

        source = joined_code(notebook)
        all_source = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"]
        )
        values = literal_assignments(source)
        assert values["RUN_ID"] == run_id, path
        assert int(values["CV_FOLD"]) == int(run["cv_fold"]), path
        assert int(values["TRAINING_SEED"]) == int(run["training_seed"]), path
        assert run["split_sha256"] in source, path
        assert run["expected_zip"] in all_source, path

    print("PASS: 36 published execution notebooks document all 35 experiment IDs")
    print("PASS: every notebook matches its recorded SHA-256 identity")
    print(f"PASS: all {code_cells} code cells are output-free and compile")
    print("PASS: run IDs, folds, training seeds, split hashes, and archive names match")


if __name__ == "__main__":
    main()
