from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import yaml


BOOK_ROOT = Path(__file__).resolve().parent
MYST_YML = BOOK_ROOT / "myst.yml"
CONTENT_ROOT = BOOK_ROOT / "content"
HEADING_RE = re.compile(r"^(#{2})\s+(.*?)\s*$")
CUSTOM_ID_RE = re.compile(r"\s*\{#([A-Za-z0-9][A-Za-z0-9\-_:]*)\}\s*$")
FENCE_RE = re.compile(r"^(```+|~~~+)")
NUMERIC_PREFIX_RE = re.compile(r"^\d+-")
NAV_IGNORE_TOKEN = "<!-- nav:ignore -->"


def strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            return "\n".join(lines[idx + 1 :])
    return text


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    ascii_text = ascii_text.replace("&", " and ")
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = re.sub(r"-{2,}", "-", ascii_text)
    return ascii_text.strip("-")


def clean_heading_title(value: str) -> str:
    value = re.sub(r"`([^`]*)`", r"\1", value)
    value = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", value)
    value = re.sub(r"[*_]+", "", value)
    return value.strip()


def page_url(rel_path: Path) -> str:
    parts = [NUMERIC_PREFIX_RE.sub("", part) for part in rel_path.with_suffix("").parts]
    return "/" + "/".join(parts)


def extract_section_links(markdown_path: Path) -> list[dict[str, str]] | None:
    text = markdown_path.read_text(encoding="utf-8")
    if NAV_IGNORE_TOKEN in text:
        return None

    text = strip_front_matter(text)
    page_href = page_url(markdown_path.relative_to(BOOK_ROOT))
    children: list[dict[str, str]] = []
    in_fence = False

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = HEADING_RE.match(line)
        if not match:
            continue

        heading_text = match.group(2).strip()
        custom_id_match = CUSTOM_ID_RE.search(heading_text)
        if custom_id_match:
            anchor = custom_id_match.group(1)
            heading_text = CUSTOM_ID_RE.sub("", heading_text).strip()
        else:
            anchor = slugify(heading_text)

        title = clean_heading_title(heading_text)
        if not title or not anchor:
            continue

        children.append({"title": title, "url": f"{page_href}#{anchor}"})

    return children


def is_auto_section_list(children: list[Any]) -> bool:
    return all(isinstance(child, dict) and "url" in child and "file" not in child for child in children)


def sync_items(items: list[Any], changes: list[str]) -> bool:
    changed = False

    for item in items:
        if not isinstance(item, dict):
            continue

        file_value = item.get("file")
        if (
            isinstance(file_value, str)
            and file_value.startswith("content/")
            and file_value.endswith(".md")
            and (
                "children" not in item
                or (isinstance(item["children"], list) and is_auto_section_list(item["children"]))
            )
        ):
            file_path = BOOK_ROOT / file_value
            if not file_path.exists():
                continue

            generated = extract_section_links(file_path)
            if generated is None:
                continue

            current_children = item.get("children")
            if generated:
                if current_children != generated:
                    item["children"] = generated
                    changes.append(file_value)
                    changed = True
            elif "children" in item:
                item.pop("children", None)
                changes.append(file_value)
                changed = True
            continue

        child_items = item.get("children")
        if isinstance(child_items, list):
            changed = sync_items(child_items, changes) or changed

    return changed


def write_config(data: dict[str, Any]) -> None:
    yaml_text = yaml.safe_dump(
        data,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=1000,
    )
    MYST_YML.write_text(yaml_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync MyST left-nav section links from Markdown headings.")
    parser.add_argument("--build", action="store_true", help="Build the book after syncing the navigation.")
    parser.add_argument("--dry-run", action="store_true", help="Show which files would update without writing myst.yml.")
    args = parser.parse_args()

    data = yaml.safe_load(MYST_YML.read_text(encoding="utf-8"))
    project = data.get("project", {})
    toc = project.get("toc")
    if not isinstance(toc, list):
        print("No project.toc list found in myst.yml", file=sys.stderr)
        return 1

    changes: list[str] = []
    changed = sync_items(toc, changes)

    if args.dry_run:
        if changes:
            print("Would update section nav for:")
            for file_value in changes:
                print(f"  - {file_value}")
        else:
            print("Navigation already matches markdown headings.")
        return 0

    if changed:
        write_config(data)
        print("Updated section nav for:")
        for file_value in changes:
            print(f"  - {file_value}")
    else:
        print("Navigation already matches markdown headings.")

    if args.build:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter_book", "build", "--html"],
            cwd=BOOK_ROOT,
            check=False,
        )
        return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
