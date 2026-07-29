"""Remove generated caches and build a source-only release archive.

The default clean operation removes interpreter/test/tool caches only. Training
outputs and checkpoints are removed only with ``--include-artifacts``. The ZIP
builder always excludes both caches and generated experiment artifacts.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

CACHE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".coverage_cache", ".ipynb_checkpoints", ".tox", ".nox",
}
CACHE_FILE_PATTERNS = ("*.pyc", "*.pyo", "*.pyd", ".coverage", "coverage.xml", "*.prof")
ARTIFACT_DIR_NAMES = {
    "outputs", "output", "runs", "checkpoints", "wandb", "tensorboard",
    "lightning_logs", "mlruns", "results_tmp", "tmp_results",
}
ARTIFACT_FILE_PATTERNS = ("*.pth", "*.pt", "*.ckpt", "*.onnx")


def is_cache_path(path: Path) -> bool:
    return any(part in CACHE_DIR_NAMES for part in path.parts) or any(
        fnmatch.fnmatch(path.name, pattern) for pattern in CACHE_FILE_PATTERNS
    )


def is_artifact_path(path: Path) -> bool:
    return any(part in ARTIFACT_DIR_NAMES for part in path.parts) or any(
        fnmatch.fnmatch(path.name, pattern) for pattern in ARTIFACT_FILE_PATTERNS
    )


def clean_tree(root: Path, *, include_artifacts: bool = False) -> list[str]:
    root = root.resolve()
    removed: list[str] = []
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        relative = path.relative_to(root)
        remove = is_cache_path(relative) or (include_artifacts and is_artifact_path(relative))
        if not remove:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
        removed.append(relative.as_posix())
    return removed


def source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if is_cache_path(relative) or is_artifact_path(relative):
            continue
        if any(part == ".git" for part in relative.parts):
            continue
        yield path, relative


def write_manifest(root: Path) -> tuple[Path, Path, Path]:
    rows = []
    checksum_lines = []
    metadata_names = {"SOURCE_MANIFEST.json", "SOURCE_MANIFEST.txt", "CHECKSUMS.sha256"}
    for path, relative in source_files(root):
        if relative.as_posix() in metadata_names:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest})
        checksum_lines.append(f"{digest}  {relative.as_posix()}")
    manifest_path = root / "SOURCE_MANIFEST.json"
    text_manifest_path = root / "SOURCE_MANIFEST.txt"
    checksum_path = root / "CHECKSUMS.sha256"
    manifest_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    text_manifest_path.write_text("\n".join(row["path"] for row in rows) + "\n", encoding="utf-8")
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest_path, text_manifest_path, checksum_path


def build_zip(root: Path, output: Path) -> Path:
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, relative in source_files(root):
            archive.write(path, Path(root.name) / relative)
    return output


def validate_clean(root: Path) -> list[str]:
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if is_cache_path(relative):
            violations.append(relative.as_posix())
    return sorted(violations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--include-artifacts", action="store_true")
    parser.add_argument("--zip", dest="zip_path", default="")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    removed = clean_tree(root, include_artifacts=args.include_artifacts)
    print(f"Removed {len(removed)} generated paths.")
    if args.manifest:
        manifest, text_manifest, checksums = write_manifest(root)
        print("Manifest:", manifest)
        print("Text manifest:", text_manifest)
        print("Checksums:", checksums)
    if args.check:
        violations = validate_clean(root)
        if violations:
            raise SystemExit("Cache paths remain:\n" + "\n".join(violations))
        print("Cache-free validation passed.")
    if args.zip_path:
        print("Release ZIP:", build_zip(root, Path(args.zip_path)))


if __name__ == "__main__":
    main()
