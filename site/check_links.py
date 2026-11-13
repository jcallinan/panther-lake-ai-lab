"""Crawl a served copy of the site and report broken local links and images without alt text.

    cd site/_build && python -m http.server 8765      # in another shell
    python site/check_links.py http://127.0.0.1:8765/

External links are listed but not fetched. Exit code 1 if anything local is broken.
"""
from __future__ import annotations

import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlparse


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs: list[str] = []
        self.ids: set[str] = set()
        self.no_alt: list[str] = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("href", "src"):
            if a.get(k):
                self.refs.append(a[k])
        if tag == "img" and not (a.get("alt") or "").strip():
            self.no_alt.append(a.get("src", "?"))


def fetch(url: str) -> tuple[int, bytes, str]:
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return r.status, r.read(), r.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        return e.code, b"", ""
    except Exception as e:  # noqa: BLE001
        print(f"  error fetching {url}: {e}")
        return 0, b"", ""


def main(base: str) -> int:
    base = base.rstrip("/") + "/"
    host = urlparse(base).netloc
    todo, seen, pages = [base + "index.html"], set(), {}
    status: dict[str, int] = {}
    external: set[str] = set()
    problems = 0
    anchors: list[tuple[str, str]] = []
    while todo:
        url = todo.pop()
        if url in seen:
            continue
        seen.add(url)
        code, body, ctype = fetch(url)
        status[url] = code
        if code != 200 or "html" not in ctype:
            continue
        p = Links()
        p.feed(body.decode("utf-8", "replace"))
        pages[url] = p
        for src in p.no_alt:
            print(f"MISSING ALT  {url} -> {src}")
            problems += 1
        for ref in p.refs:
            absu = urljoin(url, ref)
            target, frag = urldefrag(absu)
            if urlparse(absu).netloc != host:
                if absu.startswith("http"):
                    external.add(absu)
                continue
            if frag:
                anchors.append((target, frag))
            if target not in seen:
                todo.append(target)
    for (target, frag) in anchors:
        if target in pages and frag not in pages[target].ids:
            print(f"BROKEN ANCHOR #{frag} on {target}")
            problems += 1
    for url, code in sorted(status.items()):
        if code != 200:
            print(f"HTTP {code}  {url}")
            problems += 1
    print(f"checked {len(status)} local URLs ({len(pages)} HTML pages), {len(external)} external links not fetched, "
          f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/"))
