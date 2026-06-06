"""
Scrapes the New American Bible from vatican.va and saves each book as a .txt file.
Output folder: ./bible_books/
"""

import re
import time
import os
import urllib.request
from html.parser import HTMLParser

BASE_URL = "https://www.vatican.va/archive/ENG0839/"
INDEX_URL = BASE_URL + "_INDEX.HTM"
OUTPUT_DIR = "bible_books"

# --------------------------------------------------------------------------- #
# Minimal HTML → plain-text parser                                            #
# --------------------------------------------------------------------------- #

class TextExtractor(HTMLParser):
    """Strips HTML tags and returns readable plain text."""
    SKIP_TAGS = {"script", "style", "head"}

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip += 1
        if tag in ("p", "br", "tr", "li"):
            self.text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self._skip -= 1
        if tag in ("p", "div", "td", "li"):
            self.text_parts.append("\n")

    def handle_data(self, data):
        if self._skip == 0:
            self.text_parts.append(data)

    def get_text(self):
        raw = "".join(self.text_parts)
        # Collapse runs of blank lines to at most two
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def fetch(url: str, retries: int = 3) -> str:
    """Fetch a URL and return the decoded HTML string."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; BibleScraper/1.0)"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read()
            # Try common encodings
            for enc in ("utf-8", "latin-1", "windows-1252"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    continue
            return raw.decode("utf-8", errors="replace")
        except Exception as exc:
            print(f"  [warn] attempt {attempt+1} failed for {url}: {exc}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()


# --------------------------------------------------------------------------- #
# Parse the index page to build a book → [chapter URLs] mapping              #
# --------------------------------------------------------------------------- #

class IndexParser(HTMLParser):
    """
    Walks the index page and groups chapter links under their parent book name.

    Structure on the page:
      * Book name (plain text list item, sometimes a link to an intro page)
        + chapter links  (nested list)
    We track the current book name and collect every __P*.HTM link.
    """

    def __init__(self):
        super().__init__()
        self.books: dict[str, list[str]] = {}   # book_name -> [url, ...]
        self._current_book: str | None = None
        self._in_li = False
        self._depth = 0          # nesting level of <ul>/<ol>
        self._pending_text = []

    # Book names are in top-level <li> items; chapter links are in nested lists.
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in ("ul", "ol"):
            self._depth += 1
        if tag == "li":
            self._in_li = True
            self._pending_text = []
        if tag == "a" and "href" in attrs_dict:
            href = attrs_dict["href"]
            # Only chapter pages (double-underscore prefix)
            if re.match(r"__P[0-9A-Za-z]+\.HTM", href, re.I):
                url = BASE_URL + href
                if self._current_book:
                    self.books.setdefault(self._current_book, []).append(url)

    def handle_endtag(self, tag):
        if tag in ("ul", "ol"):
            self._depth -= 1
        if tag == "li":
            if self._depth == 1:          # top-level list item = book name
                text = "".join(self._pending_text).strip()
                # Strip footnote numbers / leading punctuation
                text = re.sub(r"^\W+", "", text).strip()
                if text:
                    self._current_book = text
            self._in_li = False

    def handle_data(self, data):
        if self._in_li:
            self._pending_text.append(data)


# --------------------------------------------------------------------------- #
# Clean up raw chapter text                                                   #
# --------------------------------------------------------------------------- #

def clean_chapter(raw: str) -> str:
    """Remove navigation boilerplate that appears on every chapter page."""
    lines = raw.splitlines()
    cleaned = []
    skip_patterns = [
        r"The Holy See",
        r"IntraText",
        r"Click here to show",
        r"Previous.*Next",
        r"New American Bible.*Text",
        r"^\s*\d+\s*$",          # lone verse-number lines (optional)
    ]
    for line in lines:
        if any(re.search(p, line) for p in skip_patterns):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching index …")
    index_html = fetch(INDEX_URL)

    parser = IndexParser()
    parser.feed(index_html)
    books = parser.books

    if not books:
        raise RuntimeError("No books found — the index page structure may have changed.")

    print(f"Found {len(books)} books.\n")

    for book_name, chapter_urls in books.items():
        safe_name = re.sub(r'[\\/*?:"<>|]', "_", book_name)
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.txt")

        print(f"  [{book_name}]  {len(chapter_urls)} chapter page(s) …")

        book_parts = [book_name, "=" * len(book_name), ""]

        for i, url in enumerate(chapter_urls, 1):
            try:
                html = fetch(url)
                text = html_to_text(html)
                text = clean_chapter(text)
                book_parts.append(text)
                book_parts.append("")          # blank line between chapters
            except Exception as exc:
                print(f"    [error] skipping {url}: {exc}")
            # Be polite to the server
            time.sleep(0.3)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(book_parts))

        print(f"    → saved: {out_path}")

    print(f"\nDone! All books saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
