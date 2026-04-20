<script>
(() => {
  const normalizePath = (value) => {
    if (!value) return "/";
    const trimmed = value.replace(/\/+$/, "");
    return trimmed || "/";
  };

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
      const targetEl = document.getElementById(targetId);
      if (!targetEl) return;

      event.preventDefault();
      history.replaceState(null, "", `${url.pathname}${url.hash}`);
      targetEl.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    true,
  );
})();
</script>
