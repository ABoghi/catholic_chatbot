"""
Download all 73 books of the Catholic Bible (New American Bible)
from the Vatican website and save each as a PDF.
"""

import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

BASE_URL = "https://www.vatican.va/archive/ENG0839"
INDEX_URL = f"{BASE_URL}/_INDEX.HTM"
OUTPUT_DIR = Path("bible_books")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

BOOK_ORDER = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth",
    "1 Samuel", "2 Samuel", "1 Kings", "2 Kings",
    "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah",
    "Tobit", "Judith", "Esther", "1 Maccabees", "2 Maccabees",
    "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Songs",
    "Wisdom", "Sirach",
    "Isaiah", "Jeremiah", "Lamentations", "Baruch", "Ezekiel", "Daniel",
    "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah",
    "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John",
    "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians",
    "Ephesians", "Philippians", "Colossians",
    "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon",
    "Hebrews", "James",
    "1 Peter", "2 Peter",
    "1 John", "2 John", "3 John",
    "Jude", "Revelation",
]

# Words that indicate a nav/section link, not a book name
NAV_WORDS = {
    "back", "up", "help", "index", "preface", "credits", "statistics",
    "the pentateuch", "the historical books", "the wisdom books",
    "the prophetic books", "the gospels", "new testament letters",
    "catholic letters", "overview", "footnotes", "graphs",
}


def fetch(url, retries=3, delay=1.5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                print(f"  Failed to fetch {url}: {e}")
                return None


def is_nav_item(text):
    """Return True if text looks like a navigation/section item, not a book."""
    t = text.lower().strip()
    if not t or len(t) < 2:
        return True
    if t in NAV_WORDS:
        return True
    if any(t.startswith(w) for w in NAV_WORDS):
        return True
    return False


def parse_index(index_html):
    """
    Parse the index page and return:
    { book_name: [list of chapter URLs] }

    Strategy: walk all <li> elements. A book <li> contains a nested <ul>
    whose <li> children hold the numbered chapter links (__P*.HTM).
    The book name comes from either:
      (a) an <a> tag inside the <li> (before the nested <ul>), or
      (b) the plain text of the <li> before the nested <ul>.
    """
    soup = BeautifulSoup(index_html, "html.parser")
    books = {}

    for li in soup.find_all("li"):
        # Only process li items that have a nested ul (= book entries with chapters)
        nested_ul = li.find("ul", recursive=False)
        if not nested_ul:
            # Could be a single-chapter book (like Obadiah) — has a direct __P link
            a = li.find("a", href=True)
            if a and "__P" in a.get("href", ""):
                name = a.get_text(strip=True)
                if not is_nav_item(name):
                    href = a["href"].lstrip("/")
                    if not href.startswith("http"):
                        href = f"{BASE_URL}/{href}"
                    books.setdefault(name, [])
                    if href not in books[name]:
                        books[name].append(href)
            continue

        # Collect all __P chapter URLs from the nested ul
        chapter_urls = []
        for a in nested_ul.find_all("a", href=True):
            href = a["href"]
            if "__P" in href:
                if not href.startswith("http"):
                    href = f"{BASE_URL}/{href.lstrip('/')}"
                chapter_urls.append(href)

        if not chapter_urls:
            continue

        # Get the book name: clone li, remove nested ul, read remaining text/link
        li_clone = BeautifulSoup(str(li), "html.parser").find("li")
        for sub in li_clone.find_all(["ul", "ol"]):
            sub.decompose()

        # Prefer the first <a> tag's text (if it's not a chapter number)
        first_a = li_clone.find("a")
        if first_a:
            candidate = first_a.get_text(strip=True)
            # Intro pages (Int) or chapter numbers shouldn't be the book name
            if candidate and not candidate.isdigit() and candidate.lower() not in ("int", "for"):
                name = candidate
            else:
                name = li_clone.get_text(separator=" ", strip=True).split()[0] if li_clone.get_text(strip=True) else ""
        else:
            # Plain text book name (e.g. Genesis, Leviticus)
            name = li_clone.get_text(separator=" ", strip=True)
            # Take only the first line / word-group before any digits
            # Split on multiple spaces or newlines only — NOT digits,
            # since book names like "1 Samuel" start with a digit
            name = re.split(r'\s{3,}|\n', name)[0].strip()

        if name and not is_nav_item(name):
            books[name] = chapter_urls

    return books


def fuzzy_match(canonical, scraped_names):
    """Find the best match for a canonical book name among scraped names."""
    c = canonical.lower().strip()
    # Exact
    for s in scraped_names:
        if s.lower().strip() == c:
            return s
    # One contains the other
    for s in scraped_names:
        sl = s.lower().strip()
        if sl in c or c in sl:
            return s
    # Starts-with after removing spaces (handles "1Samuel" vs "1 Samuel", "Psalm" vs "Psalms")
    # Also handles exact singular/plural: "Psalms" <-> "Psalm"
    for s in scraped_names:
        sl = s.lower().replace(" ", "")
        cl = c.replace(" ", "")
        if sl.startswith(cl) or cl.startswith(sl):
            return s
    # Singular/plural match (e.g. "Psalms" vs "Psalm", "Kings" vs "King")
    for s in scraped_names:
        sl = s.lower().strip()
        if sl == c.rstrip("s") or sl.rstrip("s") == c:
            return s
    # Shared significant words (exact set match first)
    stop = {"the", "of", "book"}
    c_words = set(c.split()) - stop
    for s in scraped_names:
        s_words = set(s.lower().split()) - stop
        if c_words and c_words == s_words:
            return s
    # Partial overlap (last resort)
    for s in scraped_names:
        s_words = set(s.lower().split()) - stop
        if c_words & s_words:
            return s
    return None


def extract_chapter_text(html):
    """Extract readable text from a chapter page.
    Removes: navigation, footnote markers, notes section, all hyperlinks.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove scripts, styles, nav
    for tag in soup.find_all(["script", "style", "nav"]):
        tag.decompose()

    # Remove footnote marker links — they link to #NT* anchors or _P*.HTM (concordance)
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("#NT") or href.startswith("#nt") or "_P" in href:
            a.decompose()  # remove entirely, including the marker text like [1] [a]

    # Remove <sup> tags — these are inline footnote reference markers (superscript letters/numbers)
    for sup in soup.find_all("sup"):
        sup.decompose()

    # Remove the notes section at the bottom:
    # It starts after the last <hr> that precedes a list of footnotes.
    # Strategy: find all <hr> tags; everything after the last one is notes.
    hrs = soup.find_all("hr")
    if hrs:
        last_hr = hrs[-1]
        # Remove last_hr and everything after it
        for sibling in list(last_hr.find_next_siblings()):
            sibling.decompose()
        last_hr.decompose()

    # Remove navigation tables (header and breadcrumb) — they are the first two <table> tags
    tables = soup.find_all("table")
    for table in tables[:2]:
        table.decompose()

    # Remove <center> tags that contain navigation links (Previous/Next, concordance)
    for center in soup.find_all("center"):
        center.decompose()

    # Unwrap remaining <a> tags — keep text, discard link
    for a in soup.find_all("a"):
        a.unwrap()

    body = soup.find("body")
    if not body:
        return ""

    paragraphs = []
    for elem in body.find_all(["p", "h1", "h2", "h3", "h4"]):
        t = elem.get_text(separator=" ", strip=True)
        # Skip very short strings (nav remnants, empty paragraphs)
        if t and len(t) > 10:
            paragraphs.append(t)

    return "\n\n".join(paragraphs)


def save_book_as_pdf(book_name, chapters_text, output_path):
    """Save a book's chapters as a single PDF."""
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=inch, leftMargin=inch,
        topMargin=inch, bottomMargin=inch,
    )
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "BibleBody", parent=styles["Normal"],
        fontSize=11, leading=16, alignment=TA_LEFT, spaceAfter=8,
    )
    story = []
    story.append(Paragraph(book_name, styles["Title"]))
    story.append(Paragraph("New American Bible — Vatican Website", styles["Italic"]))
    story.append(Spacer(1, 0.4 * inch))

    for chapter_num, text in enumerate(chapters_text, start=1):
        if not text.strip():
            continue
        story.append(Paragraph(f"Chapter {chapter_num}", styles["Heading2"]))
        story.append(Spacer(1, 0.1 * inch))
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                try:
                    story.append(Paragraph(para, body_style))
                except Exception:
                    story.append(Paragraph(para[:500], body_style))
        story.append(Spacer(1, 0.2 * inch))

    doc.build(story)


def main():
    print("Fetching Bible index...")
    index_html = fetch(INDEX_URL)
    if not index_html:
        print("Failed to fetch index page.")
        return

    print("Parsing book list...")
    books_found = parse_index(index_html)
    print("\n--- All scraped names ---")
    for name in sorted(books_found.keys()):
        print(f"  '{name}'")
    print("---\n")

    print(f"\n--- Scraped {len(books_found)} book entries ---")
    for name in sorted(books_found.keys()):
        print(f"  '{name}': {len(books_found[name])} chapters")
    print("---\n")

    # Match scraped names to canonical list
    matched = {}
    for canonical in BOOK_ORDER:
        best = fuzzy_match(canonical, list(books_found.keys()))
        if best:
            matched[canonical] = books_found[best]

    print(f"Matched {len(matched)} of 73 books.")
    missing = [b for b in BOOK_ORDER if b not in matched]
    if missing:
        print(f"Still missing: {missing}\n")

    for i, book_name in enumerate(BOOK_ORDER, start=1):
        if book_name not in matched:
            print(f"[{i:02d}/73] SKIPPING (not found): {book_name}")
            continue

        safe_name = re.sub(r'[^\w\s-]', '', book_name).strip().replace(' ', '_')
        output_path = OUTPUT_DIR / f"{i:02d}_{safe_name}.pdf"

        if output_path.exists():
            print(f"[{i:02d}/73] Already exists, skipping: {book_name}")
            continue

        chapter_urls = matched[book_name]
        print(f"[{i:02d}/73] Downloading {book_name} ({len(chapter_urls)} pages)...", end=" ", flush=True)

        chapters_text = []
        for url in chapter_urls:
            html = fetch(url, delay=0.8)
            if html:
                chapters_text.append(extract_chapter_text(html))
            time.sleep(0.5)

        if chapters_text:
            save_book_as_pdf(book_name, chapters_text, output_path)
            print(f"saved → {output_path.name}")
        else:
            print("no content retrieved")

    print("\nDone!")
    print(f"Total PDFs: {len(list(OUTPUT_DIR.glob('*.pdf')))}")


if __name__ == "__main__":
    main()