from __future__ import annotations

import json
import zipfile


def test_build_archive_excludes_secret_runtime_and_heavy_data(tmp_path):
    from scripts.package_btc_submission import build_archive

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "runs").mkdir()
    (root / "index").mkdir()
    (root / "scripts" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (root / ".env").write_text("GEMINI_API_KEY=AIza-real-secret-value\n")
    (root / ".env.example").write_text(
        "GEMINI_API_KEY=your_gemini_api_key_here\n",
        encoding="utf-8",
    )
    (root / "runs" / "result.json").write_text("{}", encoding="utf-8")
    (root / "index" / "frame_map.parquet").write_bytes(b"heavy")
    output = tmp_path / "submission.zip"

    manifest = build_archive(root=root, output=output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        parsed_manifest = json.loads(
            archive.read("AIC2026/SUBMISSION_MANIFEST.json")
        )
    assert "AIC2026/scripts/app.py" in names
    assert "AIC2026/.env.example" in names
    assert "AIC2026/.env" not in names
    assert "AIC2026/runs/result.json" not in names
    assert "AIC2026/index/frame_map.parquet" not in names
    assert parsed_manifest["file_count"] == manifest["file_count"] == 2


def test_build_archive_stops_when_secret_is_embedded_in_source(tmp_path):
    from scripts.package_btc_submission import build_archive

    root = tmp_path / "repo"
    root.mkdir()
    (root / "bad.py").write_text(
        "TOKEN = '" + "AIza" + "0" * 30 + "'\n",
        encoding="utf-8",
    )

    try:
        build_archive(root=root, output=tmp_path / "submission.zip")
    except RuntimeError as exc:
        assert "bad.py" in str(exc)
    else:
        raise AssertionError("Đóng gói phải dừng khi source chứa secret")
