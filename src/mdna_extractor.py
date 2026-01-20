import os
import re
import string
from bs4 import BeautifulSoup

from sec_edgar_downloader import Downloader


def extract_mdna_from_file(filepath, max_scan_siblings=1500):
    """Extract the MD&A section from a 10-Q filing text file.

    Args:
        filepath (str): Path to full-submission.txt file of the SEC filing.
        max_scan_siblings (int): Max # of forward elements to scan (performance guard).

    Returns:
        Extracted MD&A text content if found, otherwise None.
    """

    # Read file with encoding fallback
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin1", errors="ignore") as f:
            content = f.read()

    # Helpers
    PRINTABLE = set(string.printable)

    def printable_ratio(s: str) -> float:
        """How 'text-like' a payload is; uuencode/binary tends to tank this."""
        if not s:
            return 0.0
        sample = s[:50000]
        non_ws = [c for c in sample if c not in "\r\n\t"]
        if not non_ws:
            return 0.0
        return sum(c in PRINTABLE for c in non_ws) / len(non_ws)

    def normalize(s: str) -> str:
        return " ".join(s.replace("\xa0", " ").split()).strip()

    def looks_like_encoded_blob(s: str) -> bool:
        """Catch uuencode/base64-ish junk that sometimes appears in full-submission."""
        if not s:
            return False
        t = s.strip()
        if len(t) < 120:
            return False

        # very low printable ratio => likely not real HTML/text
        if printable_ratio(t) < 0.85:
            return True

        # base64-ish long runs
        compact = t.replace(" ", "")
        if re.match(r"^[A-Za-z0-9+/=]{250,}$", compact):
            return True

        # common uuencode pattern: lots of lines beginning with 'M'
        lines = t.splitlines()
        if len(lines) >= 5:
            m_lines = sum(1 for ln in lines[:50] if ln.strip().startswith("M"))
            if m_lines >= 10:
                return True

        return False

    def parse_document_blocks(raw: str):
        """Return list of (doc_type, payload_text) from <DOCUMENT> blocks."""
        out = []
        for m in re.finditer(
            r"<DOCUMENT>(.*?)</DOCUMENT>", raw, flags=re.DOTALL | re.IGNORECASE
        ):
            block = m.group(1)

            typ_m = re.search(r"<TYPE>\s*([^\s<]+)", block, flags=re.IGNORECASE)
            doc_type = typ_m.group(1).strip().lower() if typ_m else ""

            text_m = re.search(
                r"<TEXT>(.*?)</TEXT>", block, flags=re.DOTALL | re.IGNORECASE
            )
            payload = text_m.group(1) if text_m else block

            out.append((doc_type, payload))
        return out

    def choose_best_html_10q(raw: str):
        """Pick the most-likely real HTML 10-Q document from a full submission."""
        docs = parse_document_blocks(raw)
        candidates = []

        for doc_type, payload in docs:
            if doc_type not in {"10-q", "10q", "10-q/a", "10q/a"}:
                continue

            low = payload.lower()
            if "<html" not in low and "<body" not in low:
                continue

            if looks_like_encoded_blob(payload):
                continue

            # score by "html richness" so we don't pick a tiny fragment
            tag_score = (
                low.count("<p")
                + low.count("<div")
                + low.count("<table")
                + low.count("<tr")
                + low.count("<td")
                + low.count("<span")
                + low.count("<br")
            )
            candidates.append((tag_score, len(payload), payload))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        return candidates[0][2]

    # ----------------------------
    # Select correct HTML 10-Q document
    # ----------------------------
    html_doc = choose_best_html_10q(content)
    if not html_doc:
        print("No usable 10-Q HTML Document Found")
        return None

    soup = BeautifulSoup(html_doc, "html.parser")

    # remove non-content
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Patterns: start at Item 2 (MD&A), stop at Item 3/4/etc.
    item2_pattern = re.compile(
        r"""
        \bitem\s*2\b
        [\.\-:]{0,3}
        \s*
        (?:management[’'‘`′‛]?\s*s?)?
        .*?
        \bdiscussion\b
        .*?
        \banalysis\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    item2_simple_pattern = re.compile(r"\bitem\s*2\b", re.IGNORECASE)

    stop_pattern = re.compile(
        r"""
        \bitem\s*3\b|
        \bitem\s*4\b|
        \bpart\s*ii\b|
        \bcontrols\s+and\s+procedures\b|
        \bquantitative\s+and\s+qualitative\s+disclosures\b|
        \bsignatures\b|
        \bexhibits?\b
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    # Prefer TOC anchor-to-anchor (Item 2 href -> Item 3/4 href)
    def find_toc_anchor(item_num: str):
        """Find the first TOC-ish anchor for a given item number."""
        # We only trust anchors that look like intra-doc navigation (#something)
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if not href or not href.startswith("#"):
                continue
            txt = normalize(a.get_text(" ", strip=True)).lower()
            if txt.startswith(f"item {item_num}") or f"item {item_num}" in txt:
                return a
        return None

    def resolve_anchor_target(a_tag):
        """Resolve #id or #name targets."""
        if not a_tag:
            return None
        target_id = a_tag.get("href", "").lstrip("#")
        if not target_id:
            return None
        return (
            soup.find(id=target_id)
            or soup.find("a", attrs={"name": target_id})
            or soup.find(attrs={"name": target_id})
        )

    # Extraction core
    BLOCK_TAGS = [
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "div",
        "td",
        "th",
        "b",
        "strong",
        "font",
        "span",
        "li",
    ]

    def is_good_text_block(text: str) -> bool:
        """Filter obvious junk but keep normal filing prose."""
        if not text:
            return False

        t = normalize(text)
        if len(t) < 25:
            return False

        if looks_like_encoded_blob(t):
            return False

        # Drop blocks that are mostly non-letters (tables/numbers/page junk)
        letters = sum(c.isalpha() for c in t)
        if letters / max(len(t), 1) < 0.15:
            return False

        return True

    def extract_between(start_tag, end_tag=None):
        """Extract block text walking from start_tag until end_tag or stop_pattern."""
        mdna = []
        seen = set()

        for idx, el in enumerate(start_tag.find_all_next(BLOCK_TAGS)):
            if idx >= max_scan_siblings:
                break

            if end_tag is not None and el is end_tag:
                break

            txt = normalize(el.get_text(" ", strip=True))
            if not txt:
                continue

            # stop on explicit headers
            if stop_pattern.search(txt):
                break

            if not is_good_text_block(txt):
                continue

            # Dedup repeated blocks caused by nested tags / repeated table cells
            key = txt[:200]
            if key in seen:
                continue
            seen.add(key)

            mdna.append(txt)

        out = "\n\n".join(mdna).strip()
        return out if out else None

    # Try anchor-to-anchor first
    item2_a = find_toc_anchor("2")
    if item2_a:
        item2_start = resolve_anchor_target(item2_a) or item2_a.parent

        # Prefer stopping at Item 3 anchor target (or Item 4 if 3 missing)
        item3_a = find_toc_anchor("3")
        item4_a = find_toc_anchor("4")

        item3_target = resolve_anchor_target(item3_a) if item3_a else None
        item4_target = resolve_anchor_target(item4_a) if item4_a else None
        end_target = item3_target or item4_target

        mdna_text = extract_between(item2_start, end_target)
        if mdna_text:
            return mdna_text

    # ----------------------------
    # Fallback: find first "Item 2 ... Discussion ... Analysis" heading in block tags
    # ----------------------------
    start_tag = None
    for tag_name in BLOCK_TAGS + ["a"]:
        for tag in soup.find_all(tag_name):
            txt = normalize(tag.get_text(" ", strip=True))
            if not txt:
                continue
            if item2_pattern.search(txt) or item2_simple_pattern.search(txt):
                start_tag = tag
                break
        if start_tag:
            break

    if not start_tag:
        print("Couldn't Locate 'Item 2' Section")
        return None

    mdna_text = extract_between(start_tag)
    return mdna_text


def download_and_extract_mdna(
    tickers, filing_type="10-Q", after="2013-01-01", base_dir="data"
):
    """Download SEC filings for ticker list after a specified date, extract MD&A,
    and save it as text files.

    Args:
        tickers: List of stock ticker symbols to download filings for.
        filing_type: Type of SEC filing to download (default "10-Q").
        after: Only download filings submitted after this date (YYYY-MM-DD).
        base_dir: Base directory to store filings and MD&A text files.

    Returns:
        Saves MD&A text files on disk; prints status messages.
    """

    # For each ticker, download filings
    dl = Downloader("Aditya Dhanraj", "adityadhanraj@utexas.edu", base_dir)
    for ticker in tickers:
        print(f"\n==> Starting ticker: {ticker}")
        try:
            dl.get(filing_type, ticker, after=after)
        except Exception as error:
            print(f"Failed to Download Filings for {ticker}: {error}")
            continue

        filing_dir = os.path.join(base_dir, "sec-edgar-filings", ticker, filing_type)
        if not os.path.exists(filing_dir):
            print(f"No Filings Found for {ticker}")
            continue

        filings = sorted(os.listdir(filing_dir))
        if not filings:
            print(f"No Filings Downloaded for {ticker}")
            continue

        for filing_folder in filings:
            filing_path = os.path.join(filing_dir, filing_folder)
            full_submission_file = os.path.join(filing_path, "full-submission.txt")
            if not os.path.isfile(full_submission_file):
                continue

            # extract date
            filing_date = "unknown_date"
            try:
                with open(full_submission_file, "r", encoding="utf-8") as f:
                    for _ in range(50):
                        line = f.readline()
                        if "FILED AS OF DATE" in line:
                            m = re.search(r"(\d{8})", line)
                            if m:
                                year = m.group(1)[:4]
                                month = m.group(1)[4:6]
                                day = m.group(1)[6:]
                                filing_date = f"{year}-{month}-{day}"
                            break
            except Exception as error:
                print(f"Could Not Extract Filing Date: {error}")
            print(f"[{ticker}] Processing filing: {filing_folder} ({filing_date})...")

            # Extract MD&A from text file
            mdna_text = extract_mdna_from_file(full_submission_file)
            if mdna_text:
                out_dir = os.path.join("data", "mdna", ticker)
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{filing_date}.txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(mdna_text)
                print(f"[{ticker}] MD&A extracted: {filing_date}.txt")
            else:
                print(f"[{ticker}] No MD&A found in {full_submission_file}")
