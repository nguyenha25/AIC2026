"""Đóng gói mã nguồn AIC 2026 để nộp BTC, không kèm secret/dữ liệu nặng."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "AIC2026-BTC-source-code.zip"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv-qwen",
    "__pycache__",
    "build",
    "data",
    "dist",
    "node_modules",
    "outputs",
    "runs",
}
EXCLUDED_SUFFIXES = {
    ".7z",
    ".faiss",
    ".gz",
    ".index",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pem",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".tar",
    ".zip",
}
EXCLUDED_FILE_NAMES = {
    ".env",
    "pip_may_toi.txt",
}
SECRET_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{25,}"),
    re.compile(r"sk-ant-[0-9A-Za-z_-]{20,}"),
    re.compile(r"sk-(?:proj-)?[0-9A-Za-z_-]{24,}"),
    re.compile(
        r"(?im)^\s*(?:GEMINI|GOOGLE|OPENAI|ANTHROPIC)_API_KEY\s*=\s*"
        r"(?!your_|dan-|<|\$\{|$)([^\s#]+)"
    ),
)


def should_exclude(relative_path: Path) -> bool:
    parts = relative_path.parts
    if any(part in EXCLUDED_DIR_NAMES or part.startswith(".venv") for part in parts[:-1]):
        return True
    name = relative_path.name
    if name in EXCLUDED_FILE_NAMES:
        return True
    if name.startswith(".env.") and name != ".env.example":
        return True
    return relative_path.suffix.casefold() in EXCLUDED_SUFFIXES


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def find_secret(path: Path, payload: bytes) -> str | None:
    if b"\x00" in payload[:4096]:
        return None
    text = payload.decode("utf-8", errors="ignore")
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return pattern.pattern[:60]
    return None


def collect_source_files(root: Path) -> list[Path]:
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and not should_exclude(path.relative_to(root))
    ]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_archive(
    *,
    root: Path,
    output: Path,
    prefix: str = "AIC2026",
) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    files = collect_source_files(root)
    if not files:
        raise ValueError(f"Không tìm thấy source file trong {root}")

    entries: list[dict[str, object]] = []
    payloads: list[tuple[Path, bytes]] = []
    secret_hits: list[str] = []
    for path in files:
        payload = path.read_bytes()
        relative = path.relative_to(root)
        if find_secret(path, payload):
            secret_hits.append(relative.as_posix())
            continue
        payloads.append((relative, payload))
        entries.append({
            "path": relative.as_posix(),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
        })

    if secret_hits:
        raise RuntimeError(
            "Dừng đóng gói vì phát hiện chuỗi giống secret trong: "
            + ", ".join(secret_hits)
        )

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "package": "AIC2026-BTC-source-code",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": root.name,
        "file_count": len(entries),
        "files": entries,
        "exclusions": {
            "secret_env": True,
            "virtualenv_and_cache": True,
            "runs_and_outputs": True,
            "model_index_and_dataset_files": True,
            "previous_archives": True,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative, payload in payloads:
            archive.writestr(f"{prefix}/{relative.as_posix()}", payload)
        archive.writestr(
            f"{prefix}/SUBMISSION_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
    temporary.replace(output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tạo ZIP source code AIC 2026 đã loại secret và dữ liệu nặng."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_archive(root=args.root, output=args.output)
    print("BTC SOURCE PACKAGE: OK")
    print(f"Source : {args.root.resolve()}")
    print(f"Output : {args.output.resolve()}")
    print(f"Files  : {manifest['file_count']}")
    print(f"SHA256 : {sha256_bytes(args.output.read_bytes())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
