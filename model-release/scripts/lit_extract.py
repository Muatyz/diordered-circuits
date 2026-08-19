"""将论文 PDF 提取为可搜索的本地文本缓存。

优先使用 PyMuPDF；若当前环境未安装，则回退到系统中的 ``pdftotext``。
脚本用 PDF 的 SHA-256 判断缓存是否仍然有效，避免重复提取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256，避免一次性把大型 PDF 读入内存。"""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_with_pymupdf(pdf_path: Path) -> tuple[str, dict[str, object]]:
    """使用 PyMuPDF 提取逐页文本和 PDF 元数据。"""

    import fitz  # type: ignore[import-not-found]

    document = fitz.open(pdf_path)
    pages = []
    for page_index, page in enumerate(document):
        pages.append(f"\n--- PAGE {page_index + 1} ---\n{page.get_text('text')}")
    metadata = {key: value for key, value in document.metadata.items() if value}
    metadata["pages"] = document.page_count
    return "".join(pages).strip() + "\n", metadata


def parse_pdfinfo(pdf_path: Path) -> dict[str, object]:
    """调用 pdfinfo 读取基础元数据；不可用时返回空字典。"""

    executable = shutil.which("pdfinfo")
    if executable is None:
        return {}
    completed = subprocess.run(
        [executable, str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    metadata: dict[str, object] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip().lower().replace(" ", "_")] = value.strip()
    return metadata


def extract_with_pdftotext(pdf_path: Path) -> tuple[str, dict[str, object]]:
    """使用 TeX Live/Poppler 的 pdftotext 提取保留布局的文本。"""

    executable = shutil.which("pdftotext")
    if executable is None:
        raise RuntimeError("未找到 PyMuPDF 或 pdftotext，无法提取 PDF 文本。")
    completed = subprocess.run(
        [executable, "-layout", str(pdf_path), "-"],
        check=True,
        capture_output=True,
    )
    text = completed.stdout.decode("utf-8", errors="replace")
    return text, parse_pdfinfo(pdf_path)


def extract_pdf(pdf_path: Path) -> tuple[str, dict[str, object], str]:
    """选择当前环境可用的 PDF 文本提取后端。"""

    try:
        import fitz  # noqa: F401
    except ImportError:
        text, metadata = extract_with_pdftotext(pdf_path)
        return text, metadata, "pdftotext"
    text, metadata = extract_with_pymupdf(pdf_path)
    return text, metadata, "pymupdf"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="待提取的 PDF 路径")
    parser.add_argument(
        "--out",
        type=Path,
        help="文本输出路径；默认写入 PDF 同目录下的 paper.txt",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="提取元数据路径；默认写入文本同目录下的 extraction.json",
    )
    parser.add_argument("--force", action="store_true", help="忽略 SHA-256 缓存并重新提取")
    return parser


def main(argv: list[str] | None = None) -> int:
    """提取 PDF；内容未变化时直接复用已有缓存。"""

    args = build_parser().parse_args(argv)
    pdf_path = args.pdf.resolve()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

    output_path = (args.out or pdf_path.with_name("paper.txt")).resolve()
    metadata_path = (args.metadata or output_path.with_name("extraction.json")).resolve()
    pdf_hash = sha256_file(pdf_path)

    if not args.force and output_path.is_file() and metadata_path.is_file():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("sha256") == pdf_hash:
            print(f"cache hit: {output_path}")
            return 0

    text, pdf_metadata, backend = extract_pdf(pdf_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8", newline="\n")
    metadata = {
        "source_pdf": str(pdf_path),
        "output_text": str(output_path),
        "sha256": pdf_hash,
        "backend": backend,
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "characters": len(text),
        "pdf_metadata": pdf_metadata,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"extracted with {backend}: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
