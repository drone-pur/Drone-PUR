#!/usr/bin/env python3
"""Check that the code-only release contains no credentials, weights or host paths."""
import argparse
import ast
import io
from pathlib import Path
import re
import tokenize

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    issues, count = [], 0
    ignored = {".git", "__pycache__", ".pytest_cache", ".venv", "build", "dist"}
    for path in sorted(args.root.rglob("*")):
        relative = path.relative_to(args.root)
        if any(
            part in ignored or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        if path.is_symlink():
            issues.append(f"{relative}: symlink is not a self-contained release file")
            continue
        if not path.is_file():
            continue
        count += 1
        if path.suffix in {".safetensors", ".bin", ".pt", ".pth", ".zip", ".tar"}:
            issues.append(f"{relative}: weights/data/archive must not be bundled")
            continue
        if relative.parts[:2] == ("scripts", "assets") and path.suffix == ".png":
            try:
                with Image.open(path) as picture:
                    if picture.format != "PNG":
                        raise ValueError("expected PNG")
                    picture.verify()
            except (OSError, ValueError, SyntaxError):
                issues.append(f"{relative}: invalid documentation PNG")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append(f"{relative}: unexpected binary file")
            continue
        if re.search(r"sk-[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,}", text):
            issues.append(f"{relative}: possible credential")
        if re.search(
            r"/(?:root|mnt)/(?:project|pretrained|datasets|miniconda|nas-resource)[^\s]*",
            text,
        ):
            issues.append(f"{relative}: server-specific absolute path")
        if path.suffix == ".py":
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node) or ""
                    if re.search(r"[\u3400-\u9fff]", doc):
                        issues.append(f"{relative}: non-English docstring")
            for token in tokenize.generate_tokens(io.StringIO(text).readline):
                if token.type == tokenize.COMMENT and (
                    re.search(r"[\u3400-\u9fff]", token.string)
                    or re.fullmatch(r"#\s*[#=*-]{4,}\s*", token.string)
                ):
                    issues.append(
                        f"{relative}:{token.start[0]}: untranslated or separator-only comment"
                    )
    for issue in issues:
        print(issue)
    if issues:
        raise SystemExit(1)
    print(
        f"PASS: {count} release files; no credential patterns, machine paths, symlinks, weights or redundant/non-English comments"
    )


if __name__ == "__main__":
    main()
