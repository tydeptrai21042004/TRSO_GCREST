from pathlib import Path

from tools.clean_release import build_zip, clean_tree, validate_clean


def test_cleaner_removes_caches_but_preserves_source(tmp_path: Path):
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    (tmp_path / "pkg" / "__pycache__" / "x.pyc").write_bytes(b"cache")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "state").write_text("x")
    (tmp_path / "pkg" / "module.py").write_text("VALUE = 1\n")
    clean_tree(tmp_path)
    assert (tmp_path / "pkg" / "module.py").is_file()
    assert validate_clean(tmp_path) == []


def test_release_zip_excludes_cache_and_checkpoints(tmp_path: Path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "main.py").write_text("print('ok')\n")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "main.pyc").write_bytes(b"x")
    (root / "outputs").mkdir()
    (root / "outputs" / "checkpoint.pth").write_bytes(b"x")
    archive = build_zip(root, tmp_path / "release.zip")
    import zipfile
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    assert any(name.endswith("main.py") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
    assert not any("outputs/" in name or name.endswith(".pth") for name in names)
