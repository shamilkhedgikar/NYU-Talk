from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import yaml


BOOK_ROOT = Path(__file__).resolve().parent
HTML_ROOT = BOOK_ROOT / "_build" / "html"
CONTENT_ROOT = BOOK_ROOT / "content"
MYST_CONFIG = BOOK_ROOT / "myst.yml"


SCRIPT = r"""
<script id="relweights-sidebar-enhancer">
(() => {
  const normalizePath = (value) => {
    if (!value) return "/";
    const trimmed = value.replace(/\/+$/, "");
    return trimmed || "/";
  };

  const stripNumericPrefix = (value) => value.replace(/^\d+-/, "");

  const derivePagePath = (filePath, baseUrl) => {
    const noExt = filePath.replace(/\.(md|ipynb)$/i, "");
    const parts = noExt.split("/").filter(Boolean).map(stripNumericPrefix);
    return `${baseUrl}/${parts.join("/")}`;
  };

  const extractBaseUrl = (project) => {
    const candidates = [window.location.pathname];
    if (project?.pages) {
      for (const page of project.pages) {
        if (page?.url) candidates.push(page.url);
        if (page?.thumbnail) candidates.push(page.thumbnail);
      }
    }

    for (const candidate of candidates) {
      if (!candidate) continue;
      let pathname = "/";
      try {
        pathname = new URL(candidate, window.location.origin).pathname;
      } catch (_error) {
        continue;
      }
      if (pathname.includes("/content/")) {
        return normalizePath(pathname.split("/content/")[0]);
      }
      if (pathname.includes("/build/")) {
        return normalizePath(pathname.split("/build/")[0]);
      }
    }
    return "/";
  };

  const getProject = () => {
    try {
      return window.__remixContext?.state?.loaderData?.root?.project ?? null;
    } catch (_error) {
      return null;
    }
  };

  const createLink = (title, url) => {
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.textContent = title;
    anchor.className = [
      "block",
      "break-words",
      "focus:outline",
      "outline-blue-200",
      "outline-2",
      "rounded",
      "p-2",
      "my-1",
      "ml-2",
      "text-sm",
      "hover:bg-slate-300/30",
    ].join(" ");
    return anchor;
  };

  const resolveTargetElement = (targetId) => {
    if (!targetId) return null;

    const candidates = [targetId];
    if (targetId.startsWith("id-")) {
      candidates.push(targetId.slice(3));
    } else {
      candidates.push(`id-${targetId}`);
    }

    for (const candidate of candidates) {
      const element = document.getElementById(candidate);
      if (element) return element;
    }
    return null;
  };

  const syncCurrentPageHashLinks = () => {
    const currentOrigin = window.location.origin;
    const currentPath = normalizePath(window.location.pathname);

    document.querySelectorAll("a[href*='#']").forEach((link) => {
      const href = link.getAttribute("href");
      if (!href || href.startsWith("http")) return;

      const url = new URL(href, currentOrigin);
      if (!url.hash) return;
      if (normalizePath(url.pathname) !== currentPath) return;

      const targetId = decodeURIComponent(url.hash.slice(1));
      const targetEl = resolveTargetElement(targetId);
      if (!targetEl || !targetEl.id || targetEl.id === targetId) return;

      link.setAttribute("href", `${url.pathname}#${targetEl.id}`);
    });
  };

  const scrollToResolvedHash = (hash, behavior = "smooth") => {
    if (!hash) return false;

    const targetId = decodeURIComponent(hash.replace(/^#/, ""));
    const targetEl = resolveTargetElement(targetId);
    if (!targetEl) return false;

    if (targetEl.id && targetEl.id !== targetId) {
      history.replaceState(null, "", `${window.location.pathname}#${targetEl.id}`);
    }

    targetEl.scrollIntoView({ behavior, block: "start" });
    return true;
  };

  const populateSidebar = () => {
    const project = getProject();
    if (!project || !Array.isArray(project.toc)) return false;

    const currentOrigin = window.location.origin;
    const baseUrl = extractBaseUrl(project);

    const itemsByPath = new Map();
    const walk = (items) => {
      for (const item of items || []) {
        if (item && typeof item === "object" && item.file && Array.isArray(item.children) && item.children.length) {
          const path = normalizePath(derivePagePath(item.file, baseUrl));
          itemsByPath.set(path, item.children);
        }
        if (item && typeof item === "object" && Array.isArray(item.children)) {
          walk(item.children);
        }
      }
    };
    walk(project.toc);

    let populated = 0;
    document.querySelectorAll(".myst-primary-sidebar-toc .w-full").forEach((wrapper) => {
      const row = wrapper.querySelector(":scope > .myst-toc-item");
      const content = wrapper.querySelector(":scope > .collapsible-content");
      const link = row?.querySelector("a[href]");
      if (!row || !content || !link) return;
      if (content.dataset.relweightsPopulated === "true") return;

      const path = normalizePath(new URL(link.href, currentOrigin).pathname);
      const children = itemsByPath.get(path);
      if (!children || !children.length) return;

      content.replaceChildren();
      const fragment = document.createDocumentFragment();
      for (const child of children) {
        if (!child?.title || !child?.url) continue;
        fragment.appendChild(createLink(child.title, child.url));
      }
      if (fragment.childNodes.length > 0) {
        content.appendChild(fragment);
        content.dataset.relweightsPopulated = "true";
        populated += 1;
      }
    });

    syncCurrentPageHashLinks();
    return populated > 0;
  };

  const bindSamePageHashLinks = () => {
    document.addEventListener(
      "click",
      (event) => {
        const link = event.target.closest("a[href*='#']");
        if (!link) return;

        const href = link.getAttribute("href");
        if (!href || href.startsWith("http")) return;

        const url = new URL(href, window.location.origin);
        if (!url.hash) return;

        const currentPath = normalizePath(window.location.pathname);
        const targetPath = normalizePath(url.pathname);
        if (currentPath !== targetPath) return;

        const targetId = decodeURIComponent(url.hash.slice(1));
        const targetEl = resolveTargetElement(targetId);
        if (!targetEl) return;

        event.preventDefault();
        const resolvedHash = targetEl.id ? `#${targetEl.id}` : url.hash;
        history.replaceState(null, "", `${url.pathname}${resolvedHash}`);
        targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
      },
      true,
    );
  };

  const start = () => {
    bindSamePageHashLinks();
    let pending = false;
    const schedulePopulate = () => {
      if (pending) return;
      pending = true;
      window.requestAnimationFrame(() => {
        pending = false;
        populateSidebar();
      });
    };

    schedulePopulate();
    window.setTimeout(schedulePopulate, 250);
    window.setTimeout(schedulePopulate, 1000);
    window.addEventListener("popstate", schedulePopulate);
    window.addEventListener("hashchange", () => {
      schedulePopulate();
      scrollToResolvedHash(window.location.hash, "smooth");
    });

    const observer = new MutationObserver(() => {
      schedulePopulate();
    });
    observer.observe(document.body, { childList: true, subtree: true });

    if (window.location.hash) {
      window.requestAnimationFrame(() => {
        scrollToResolvedHash(window.location.hash, "auto");
      });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start, { once: true });
  } else {
    start();
  }
})();
</script>
"""


def inject_script(html_path: Path) -> bool:
    text = html_path.read_text(encoding="utf-8")
    script_pattern = re.compile(
        r'<script id="relweights-sidebar-enhancer">.*?</script>',
        flags=re.DOTALL,
    )
    if script_pattern.search(text):
        updated = script_pattern.sub(lambda _match: SCRIPT.strip(), text, count=1)
        if updated == text:
            return False
        html_path.write_text(updated, encoding="utf-8")
        return True
    if "</body>" not in text:
        return False
    text = text.replace("</body>", f"{SCRIPT}\n</body>")
    html_path.write_text(text, encoding="utf-8")
    return True


def strip_numeric_prefix(value: str) -> str:
    return value.split("-", 1)[1] if value[:2].isdigit() and "-" in value else value


def derive_page_path(file_path: str, base_url: str) -> str:
    no_ext = re.sub(r"\.(md|ipynb)$", "", file_path, flags=re.IGNORECASE)
    parts = [strip_numeric_prefix(part) for part in no_ext.split("/") if part]
    base = base_url.rstrip("/")
    return f"{base}/{'/'.join(parts)}" if base else f"/{'/'.join(parts)}"


def build_children_map() -> dict[str, list[dict[str, str]]]:
    config = yaml.safe_load(MYST_CONFIG.read_text(encoding="utf-8"))
    toc = config.get("project", {}).get("toc", [])
    base_url = os.environ.get("BASE_URL", "").rstrip("/")

    mapping: dict[str, list[dict[str, str]]] = {}

    def walk(items: list[dict]) -> None:
        for item in items or []:
            if not isinstance(item, dict):
                continue
            children = item.get("children")
            file_path = item.get("file")
            if file_path and isinstance(children, list) and children:
                page_path = derive_page_path(file_path, base_url)
                usable_children = [
                    {"title": child.get("title", ""), "url": child.get("url", "")}
                    for child in children
                    if isinstance(child, dict) and child.get("title") and child.get("url")
                ]
                if usable_children:
                    mapping[page_path] = usable_children
            if isinstance(children, list):
                walk(children)

    walk(toc)
    return mapping


def create_sidebar_links(children: list[dict[str, str]]) -> str:
    return "".join(
        (
            f'<a href="{child["url"]}" '
            'class="block break-words focus:outline outline-blue-200 outline-2 '
            'rounded p-2 my-1 ml-2 text-sm hover:bg-slate-300/30">'
            f'{child["title"]}</a>'
        )
        for child in children
    )


def inject_sidebar_children(html_path: Path, children_map: dict[str, list[dict[str, str]]]) -> bool:
    text = html_path.read_text(encoding="utf-8")
    changed = False

    pattern = re.compile(
        r'(?P<prefix><div data-state="(?:closed|open)" class="w-full"><div class="myst-toc-item.*?<a[^>]+href="(?P<href>[^"]+)"[^>]*>.*?</a>.*?</div><div[^>]*class="[^"]*\bcollapsible-content\b[^"]*"[^>]*>)(?P<content>.*?)(?P<suffix></div></div>)',
        flags=re.DOTALL,
    )

    def replacer(match: re.Match[str]) -> str:
        nonlocal changed
        href = match.group("href").rstrip("/")
        children = children_map.get(href)
        if not children:
            return match.group(0)
        changed = True
        return f'{match.group("prefix")}{create_sidebar_links(children)}{match.group("suffix")}'

    updated = pattern.sub(replacer, text)
    if changed:
        html_path.write_text(updated, encoding="utf-8")
    return changed


def mirror_interactive_html() -> int:
    copied = 0
    for interactive_dir in CONTENT_ROOT.glob("*/interactive"):
        section_dir = interactive_dir.parent
        route_section = strip_numeric_prefix(section_dir.name)
        page_files = [
            path
            for path in section_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".md", ".ipynb"}
        ]
        if not page_files:
            continue

        html_files = list(interactive_dir.glob("*.html"))
        if not html_files:
            continue

        for page_file in page_files:
            destination_dir = HTML_ROOT / "content" / route_section / page_file.stem / "interactive"
            destination_dir.mkdir(parents=True, exist_ok=True)
            for html_file in html_files:
                shutil.copy2(html_file, destination_dir / html_file.name)
                copied += 1
    return copied


def main() -> int:
    if not HTML_ROOT.exists():
        raise SystemExit(f"Build output not found: {HTML_ROOT}")

    copied = mirror_interactive_html()
    children_map = build_children_map()

    sidebar_fills = 0
    for html_path in HTML_ROOT.rglob("*.html"):
        if inject_sidebar_children(html_path, children_map):
            sidebar_fills += 1

    changed = 0
    for html_path in HTML_ROOT.rglob("*.html"):
        if inject_script(html_path):
            changed += 1

    print(f"Copied {copied} interactive HTML files into the built site.")
    print(f"Filled sidebar submenu HTML in {sidebar_fills} pages.")
    print(f"Injected sidebar enhancer into {changed} HTML files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
