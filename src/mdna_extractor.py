import os
from sec_edgar_downloader import Downloader
import re
from bs4 import BeautifulSoup

def extract_mdna_from_file(filepath):
    """
    Extract the Management Discussion and Analysis (MD&A) section from a 10-Q filing text file.

    This function reads the raw full-submission.txt SEC filing file, searches for the
    10-Q HTML document, and attempts to locate the MD&A section (typically Item 2).
    It extracts and returns the text content of the MD&A section up to Item 3.
    If the MD&A section cannot be found, returns None.

    Parameters:
    -----------
    filepath : str
        Path to the full-submission.txt file of the SEC filing.

    Returns:
    --------
    str or None
        Extracted MD&A text content if found, otherwise None.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # extract documents from the filing text
    docs = re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", content, re.DOTALL | re.IGNORECASE)
    html_doc = None
    for doc in docs:
        # find the 10-Q document that contains HTML content
        if "<type>10-q" in doc.lower() and "<html" in doc.lower():
            html_start = doc.lower().find("<html")
            html_doc = doc[html_start:]
            break

    if not html_doc:
        print("No 10-Q HTML Document Found")
        return None

    soup = BeautifulSoup(html_doc, "html.parser")

    # step 1: try to find item 2 anchor link in toc
    item2_anchor = None
    # regex to match "item 2 management's discussion & analysis" or similar
    item2_link_re = re.compile(
        r"item\s*2[^a-zA-Z0-9]{0,5}management[’'‘`′‛]?s?[\s\-:]*discussion\s*(and|&)?\s*analysis",
        re.IGNORECASE,
    )

    for a in soup.find_all('a', href=True):
        link_text = a.get_text(" ", strip=True).replace('\xa0', ' ')
        if item2_link_re.search(link_text):
            item2_anchor = a
            break

    if not item2_anchor:
        # fallback: any link with "item 2"
        for a in soup.find_all('a', href=True):
            if "item 2" in a.get_text(" ", strip=True).lower():
                item2_anchor = a
                break

    # step 2: if anchor found, find the target section
    if item2_anchor:
        target_id = item2_anchor['href'].lstrip('#')
        target_tag = soup.find(id=target_id) or soup.find(name=target_id)
        if not target_tag:
            # fallback: search for <a name="..."> tags
            target_tag = soup.find('a', attrs={'name': target_id})

        if target_tag:
            mdna_text = []
            item3_pattern = re.compile(r"item\s*3[^a-zA-Z0-9]{0,5}", re.IGNORECASE)
            for sibling in target_tag.find_all_next():
                text = sibling.get_text(" ", strip=True)
                if not text:
                    continue
                if item3_pattern.search(text):
                    break
                mdna_text.append(text)
            result = "\n\n".join(mdna_text).strip()
            if result:
                return result

        # fallback: scan forward from anchor parent for item 2 heading
        container = item2_anchor.parent
        for idx, sibling in enumerate(container.find_all_next()):
            if idx > 50: # limit to avoid infinite scanning
                break
            text = sibling.get_text(" ", strip=True).lower()
            if "item 2" in text and ("management" in text or "discussion" in text):
                mdna_text = []
                item3_pattern = re.compile(r"item\s*3[^a-zA-Z0-9]{0,5}", re.IGNORECASE)
                for sib in sibling.find_all_next():
                    txt = sib.get_text(" ", strip=True)
                    if not txt:
                        continue
                    if item3_pattern.search(txt):
                        break
                    mdna_text.append(txt)
                result = "\n\n".join(mdna_text).strip()
                if result:
                    return result

    # step 3: ultimate fallback: search for visible item 2 headings anywhere
    item2_pattern = re.compile(
        r"item\s*2[^a-zA-Z0-9]{0,5}.*?management[’'‘`′‛]?s?[\s\-:]*discussion\s*(and|&)analysis",
        re.IGNORECASE | re.DOTALL
    )
    item2_simple_pattern = re.compile(r"item\s*2\b", re.IGNORECASE)
    heading_tags = ["b", "strong", "font", "p", "div", "span", "td", "th", "h1", "h2", "h3", "h4", "a"]

    candidate_tags = []
    for tag_name in heading_tags:
        for tag in soup.find_all(tag_name):
            full_text = tag.get_text(separator=" ", strip=True)
            if not full_text:
                continue
            if item2_pattern.search(full_text) or item2_simple_pattern.search(full_text):
                candidate_tags.append(tag)

    if not candidate_tags:
        print("Couldn't Locate 'Item 2' Section")
        return None

    item_2_tag = candidate_tags[0]
    if not hasattr(item_2_tag, "find_all_next"):
        item_2_tag = item_2_tag.parent

    mdna_text = []
    item3_pattern = re.compile(r"item\s*3[^a-zA-Z0-9]{0,5}", re.IGNORECASE)

    for sibling in item_2_tag.find_all_next():
        text = sibling.get_text(" ", strip=True)
        if not text:
            continue
        if item3_pattern.search(text):
            break
        mdna_text.append(text)

    result = "\n\n".join(mdna_text).strip()
    return result if result else None

def download_and_extract_mdna(tickers, filing_type="10-Q", after="2013-01-01", base_dir="data"):
    """
    Download SEC filings for a list of tickers after a specified date, extract the MD&A section,
    and save it as text files.

    This function uses the `sec_edgar_downloader` package to download filings (e.g., 10-Q),
    then parses each filing’s raw full-submission.txt to extract the MD&A section text.
    Extracted texts are saved under `data/mdna/<ticker>/<filing_date>.txt`.
    Progress, warnings, and errors are printed during the process.

    Parameters:
    -----------
    tickers : list of str
        List of stock ticker symbols to download filings for.
    filing_type : str, optional
        Type of SEC filing to download (default "10-Q").
    after : str, optional
        Only download filings submitted after this date (YYYY-MM-DD format).
    base_dir : str, optional
        Base directory to store downloaded filings and extracted MD&A text files.

    Returns:
    --------
    None
        Saves MD&A text files on disk; prints status messages.
    """
    dl = Downloader("Aditya Dhanraj", "adityadhanraj@utexas.edu", base_dir)

    for ticker in tickers:
        try:
            dl.get(filing_type, ticker, after=after)
        except Exception as error:
            print(f"Failed to Download Filings for {ticker}: {error}")
            continue

        filing_dir = os.path.join(base_dir, "sec-edgar-filings", ticker, filing_type)
        if not os.path.exists(filing_dir):
            print(f"No Filings Found for {ticker} in {filing_dir}")
            continue

        filings = sorted(os.listdir(filing_dir))
        if not filings:
            print(f"No Filings Downloaded for {ticker}")
            continue

        for filing_folder in filings:
            filing_path = os.path.join(filing_dir, filing_folder)
            if not os.path.isdir(filing_path):
                continue

            full_submission_file = os.path.join(filing_path, "full-submission.txt")
            if not os.path.isfile(full_submission_file):
                print(f"No Raw HTML .txt Found in {filing_path}")
                continue

            filing_date = "unknown_date"
            try:
                with open(full_submission_file, "r", encoding="utf-8") as f:
                    for _ in range(50): # only check first 50 lines for date
                        line = f.readline()
                        if not line:
                            break
                        if "FILED AS OF DATE" in line:
                            raw_date = re.search(r"(\d{8})", line)
                            if raw_date:
                                date_str = raw_date.group(1)
                                filing_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                            break
            except Exception as error:
                print(f"Could Not Extract Filing Date: {error}")

            mdna_text = extract_mdna_from_file(full_submission_file)

            if mdna_text:
                out_dir = os.path.join("data", "mdna", ticker)
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{filing_date}.txt")
                with open(out_file, "w", encoding="utf-8") as f:
                    f.write(mdna_text)
            else:
                print(f"No MD&A Found in {full_submission_file}")