"""
CNKI MCP Server - CNKI academic paper search via MCP.

Tools:
- search_cnki: Search papers (with optional journal filter)
- get_paper_detail: Get full paper metadata
- get_paper_bibtex: Get BibTeX citation entry for a paper
- download_paper_pdf: Download paper PDF (requires institutional IP)
- find_best_match: Find closest title match

Usage:
    pip install -e .
    cnki-mcp
"""

from fastmcp import FastMCP, Context
from fastmcp.dependencies import Depends, CurrentContext
from typing import List, Optional, Annotated
from pydantic import Field
from playwright.async_api import async_playwright, Browser, Page, Playwright
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
import time
import random
import json
import os
import re

# =================== Search type mappings ===================

SEARCH_TYPES = {
    "主题": "SU", "篇关摘": "TKA", "关键词": "KY", "篇名": "TI",
    "全文": "FT", "作者": "AU", "第一作者": "FI", "通讯作者": "RP",
    "作者单位": "AF", "基金": "FU", "摘要": "AB", "参考文献": "RF",
    "分类号": "CLC", "文献来源": "LY", "DOI": "DOI",
}

SEARCH_TYPE_VALUES = {
    "主题": "SU$%=|", "篇关摘": "TKA$%=|", "关键词": "KY$=|",
    "篇名": "TI$%=|", "全文": "FT$%=|", "作者": "AU$=|",
    "第一作者": "FI$=|", "通讯作者": "RP$%=|", "作者单位": "AF$%",
    "基金": "FU$%|", "摘要": "AB$%=|", "参考文献": "RF$%=|",
    "分类号": "CLC$=|??", "文献来源": "LY$%=|", "DOI": "DOI$=|?",
}

SEARCH_TYPE_ALIASES = {
    "subject": "主题", "theme": "主题", "keyword": "关键词",
    "keywords": "关键词", "title": "篇名", "author": "作者",
    "first_author": "第一作者", "corresponding_author": "通讯作者",
    "affiliation": "作者单位", "institution": "作者单位",
    "fund": "基金", "abstract": "摘要", "fulltext": "全文",
    "reference": "参考文献", "source": "文献来源", "doi": "DOI",
}

SORT_TYPES = {
    "相关度": "FFD", "发表时间": "PT", "被引": "CF",
    "下载": "DFR", "综合": "ZH",
}

SORT_TYPE_ALIASES = {
    "relevance": "相关度", "date": "发表时间", "publish_time": "发表时间",
    "time": "发表时间", "cited": "被引", "citation": "被引",
    "citations": "被引", "download": "下载", "downloads": "下载",
    "composite": "综合", "general": "综合",
}

PROFESSIONAL_SEARCH_FIELDS = {
    "主题": "SU", "关键词": "KY", "篇名": "TI", "全文": "FT",
    "作者": "AU", "第一作者": "FI", "通讯作者": "RP",
    "作者单位": "AF", "摘要": "AB", "DOI": "DOI",
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
]


# =================== Paper Registry ===================

class PaperRegistry:
    """Maps short labels to CNKI URLs. Labels are returned to agents instead of URLs."""

    def __init__(self):
        self._labels: dict[str, str] = {}  # label → URL
        self._counter: int = 0

    def register(self, url: str, first_author: str = "", year: str = "", title: str = "") -> str:
        """Register a paper URL and return a short label."""
        self._counter += 1
        author_part = first_author[:6] if first_author else ""
        year_part = year[:4] if year else ""
        title_part = title[:10] if title else ""
        label = f"[{self._counter}] {author_part}{year_part}-{title_part}"
        self._labels[label] = url
        return label

    def resolve(self, label: str) -> str:
        """Resolve a label to a URL. Raises KeyError if not found."""
        if label not in self._labels:
            raise KeyError(f"Unknown paper label: '{label}'. Use search_cnki first to get valid labels.")
        return self._labels[label]


paper_registry = PaperRegistry()


# =================== BrowserPool ===================

class BrowserPool:
    """Manages a singleton Playwright browser with idle timeout."""

    IDLE_TIMEOUT = 600  # 10 min

    def __init__(self):
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._last_used: float = 0
        self._lock = asyncio.Lock()

    async def _create_browser(self) -> Browser:
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        browser = await self._playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-gpu",
            ],
        )
        return browser

    async def _is_browser_alive(self) -> bool:
        if self._browser is None:
            return False
        try:
            return self._browser.is_connected()
        except Exception:
            return False

    async def get_page(self) -> Page:
        """Get a new page from the browser (caller must close it)."""
        async with self._lock:
            now = time.time()
            if self._browser is not None:
                if now - self._last_used > self.IDLE_TIMEOUT:
                    await self._close_internal()
                elif not await self._is_browser_alive():
                    self._browser = None
            if self._browser is None:
                self._browser = await self._create_browser()
            self._last_used = now

        page = await self._browser.new_page(
            user_agent=random.choice(USER_AGENTS),
        )
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        return page

    async def _close_internal(self):
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None

    async def close(self):
        async with self._lock:
            await self._close_internal()
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass
            self._playwright = None


# =================== Helpers ===================

def resolve_search_type(search_type: str) -> str:
    if not search_type:
        return "主题"
    s = search_type.lower().strip()
    if s in SEARCH_TYPE_ALIASES:
        return SEARCH_TYPE_ALIASES[s]
    if search_type in SEARCH_TYPES:
        return search_type
    return "主题"


def resolve_sort_type(sort_type: str) -> str:
    if not sort_type:
        return "相关度"
    s = sort_type.lower().strip()
    if s in SORT_TYPE_ALIASES:
        return SORT_TYPE_ALIASES[s]
    if sort_type in SORT_TYPES:
        return sort_type
    return "相关度"


def find_closest_title(title: str, result_titles: List[str]) -> int:
    max_similar = 0
    best_index = 0
    for i, t in enumerate(result_titles):
        common_chars = sum(c in t for c in title)
        if common_chars > max_similar:
            max_similar = common_chars
            best_index = i
    return best_index


async def random_delay(lo: float = 1.0, hi: float = 2.5):
    await asyncio.sleep(random.uniform(lo, hi))


async def type_slowly(page: Page, selector: str, text: str):
    """Type text character by character to mimic human input."""
    locator = page.locator(selector)
    await locator.clear()
    for char in text:
        await locator.press_sequentially(char, delay=random.uniform(30, 80))


# =================== Search implementations ===================

async def _parse_paper_row(row) -> dict:
    """Parse a search result table row into a paper dict."""
    paper = {}
    try:
        title_el = await row.query_selector('a.fz14')
        paper["title"] = (await title_el.inner_text()).strip() if title_el else ""
        paper["url"] = (await title_el.get_attribute("href")) if title_el else ""
    except Exception:
        paper["title"] = ""
        paper["url"] = ""
    try:
        authors = await row.query_selector_all('td.author a')
        paper["authors"] = [(await a.inner_text()).strip() for a in authors if (await a.inner_text()).strip()]
    except Exception:
        paper["authors"] = []
    try:
        source_el = await row.query_selector('td.source a')
        paper["source"] = (await source_el.inner_text()).strip() if source_el else ""
    except Exception:
        paper["source"] = ""
    try:
        date_el = await row.query_selector('td.date')
        paper["date"] = (await date_el.inner_text()).strip() if date_el else ""
    except Exception:
        paper["date"] = ""
    try:
        cite_el = await row.query_selector('td.quote a')
        paper["cited_count"] = (await cite_el.inner_text()).strip() if cite_el else "0"
    except Exception:
        paper["cited_count"] = "0"
    try:
        dl_el = await row.query_selector('td.download a')
        paper["download_count"] = (await dl_el.inner_text()).strip() if dl_el else "0"
    except Exception:
        paper["download_count"] = "0"
    return paper


async def _collect_results(page: Page, pages: int) -> list:
    """Collect paper results across multiple pages."""
    all_papers = []
    for page_num in range(1, pages + 1):
        try:
            rows = await page.query_selector_all('table.result-table-list tbody tr')
            if not rows:
                await page.wait_for_selector('table.result-table-list tbody tr', timeout=15000)
                rows = await page.query_selector_all('table.result-table-list tbody tr')
            for row in rows:
                paper = await _parse_paper_row(row)
                if paper["title"]:
                    paper["page"] = page_num
                    # Register paper and replace URL with label
                    first_author = paper["authors"][0] if paper.get("authors") else ""
                    year = paper.get("date", "")[:4]
                    label = paper_registry.register(paper["url"], first_author, year, paper["title"])
                    paper["label"] = label
                    del paper["url"]
                    all_papers.append(paper)
        except Exception:
            pass

        if page_num < pages:
            try:
                next_btn = await page.query_selector("#PageNext")
                if next_btn and await next_btn.is_enabled():
                    await next_btn.click()
                    await random_delay(1.5, 2.5)
                else:
                    break
            except Exception:
                break
    return all_papers


async def _simple_search(page: Page, query: str, search_type: str, sort: str, pages: int) -> dict:
    """Search via CNKI homepage (no journal filter)."""
    resolved_type = resolve_search_type(search_type)
    resolved_sort = resolve_sort_type(sort)

    await page.goto("https://www.cnki.net/")
    await random_delay(1, 2)

    if resolved_type != "主题":
        value = SEARCH_TYPE_VALUES.get(resolved_type)
        if value:
            await page.click("#DBFieldBox")
            await asyncio.sleep(0.8)
            await page.click(f'#DBFieldList a[value="{value}"]')
            await asyncio.sleep(0.5)

    await type_slowly(page, "#txt_SearchText", query)
    await page.click(".search-btn")
    await random_delay(2, 3)

    if resolved_sort != "相关度":
        sort_id = SORT_TYPES.get(resolved_sort)
        if sort_id:
            try:
                await page.click(f"#{sort_id}", timeout=10000)
                await random_delay(1.5, 2.5)
                await page.wait_for_selector('table.result-table-list tbody tr', timeout=15000)
            except Exception:
                pass

    all_papers = await _collect_results(page, pages)

    return {
        "query": query, "search_type": resolved_type, "sort": resolved_sort,
        "total_pages": pages, "total_papers": len(all_papers), "papers": all_papers,
    }


def _build_field_expr(field_code: str, query: str) -> str:
    """Build a CNKI professional search field expression.

    CNKI professional search syntax:
      * = AND, + = OR, - = NOT
      SU='term1' * 'term2'  → subject contains term1 AND term2
      SU=('t1' + 't2') * 't3'  → (t1 OR t2) AND t3

    Single term uses fuzzy match: SU%'经济增长'
    Multiple space-separated terms uses AND: SU='北京' * '奥运'
    """
    terms = query.split()
    if len(terms) <= 1:
        # Single term: use fuzzy match
        return f"{field_code}%'{query}'"
    else:
        # Multiple terms: join with * (AND) using exact match
        parts = " * ".join(f"'{t}'" for t in terms)
        return f"{field_code}={parts}"


async def _professional_search(page: Page, query: str, search_type: str, journal: Optional[str], sort: str, pages: int, author: Optional[str] = None) -> dict:
    """Search via CNKI Professional Search (with journal/author filter).

    Journal names use exact match (LY=), topics use fuzzy match (SU%),
    authors use exact match (AU=).
    Multiple terms in query are joined with * (AND) per CNKI syntax.
    Multiple journals can be separated by '+', e.g. '经济研究+管理世界'.
    """
    resolved_type = resolve_search_type(search_type)
    resolved_sort = resolve_sort_type(sort)
    field_code = PROFESSIONAL_SEARCH_FIELDS.get(resolved_type, "SU")

    # Build expression: start with the main query field
    expr = _build_field_expr(field_code, query)

    # Add author filter if provided
    if author:
        expr += f" AND AU='{author}'"

    # Add journal filter if provided
    # Multiple journals: (LY='j1' OR LY='j2')
    if journal:
        journals = [j.strip() for j in journal.split("+") if j.strip()]
        if len(journals) == 1:
            journal_expr = f"LY='{journals[0]}'"
        else:
            journal_expr = "(" + " OR ".join(f"LY='{j}'" for j in journals) + ")"
        expr += f" AND {journal_expr}"

    # Visit main site first for session cookies
    await page.goto("https://www.cnki.net/")
    await random_delay(1, 2)

    await page.goto("https://kns.cnki.net/kns8s/AdvSearch")
    await random_delay(1, 2)

    # Click Professional Search tab
    await page.click('li[name="majorSearch"]')
    await random_delay(0.5, 1)

    # Enter expression
    await page.locator("textarea.majorSearch").fill(expr)
    await random_delay(0.3, 0.5)

    # Click search
    await page.click("input.btn-search")
    await random_delay(2, 3)

    if resolved_sort != "相关度":
        sort_id = SORT_TYPES.get(resolved_sort)
        if sort_id:
            try:
                await page.click(f"#{sort_id}", timeout=10000)
                await random_delay(1.5, 2.5)
                await page.wait_for_selector('table.result-table-list tbody tr', timeout=15000)
            except Exception:
                pass

    all_papers = await _collect_results(page, pages)

    result = {
        "query": query, "search_type": resolved_type,
        "sort": resolved_sort, "expression": expr,
        "total_pages": pages, "total_papers": len(all_papers), "papers": all_papers,
    }
    if author:
        result["author"] = author
    if journal:
        result["journal"] = journal
    return result


# =================== Paper detail ===================

async def _get_paper_detail(page: Page, url: str) -> dict:
    """Navigate to a CNKI paper detail page and extract metadata."""
    paper = {
        "title": "", "title_en": "", "authors": [],
        "institutions": [], "abstract": "", "abstract_en": "",
        "keywords": [], "keywords_en": [], "source": "", "year": "",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "cited_count": "", "download_count": "", "fund": "", "classification": "",
    }

    # Establish session and set referer to avoid captcha
    await page.goto("https://www.cnki.net/")
    await random_delay(1, 2)
    await page.set_extra_http_headers({"Referer": "https://kns.cnki.net/kns8s/AdvSearch"})
    await page.goto(url)
    await random_delay(1.5, 2.5)

    async def text(selector: str, default: str = "") -> str:
        el = await page.query_selector(selector)
        return (await el.inner_text()).strip() if el else default

    async def texts(selector: str) -> List[str]:
        els = await page.query_selector_all(selector)
        result = []
        for el in els:
            t = (await el.inner_text()).strip()
            if t:
                result.append(t)
        return result

    paper["title"] = await text('div.wx-tit h1') or await text('h1')
    paper["title_en"] = await text('div.wx-tit h2')

    # Authors: modern papers have <a> links in h3#authorpart; older papers have plain text
    author_links = await page.query_selector_all('h3#authorpart a')
    if author_links:
        for link in author_links:
            name = await link.evaluate(
                '(el) => { el.querySelectorAll("sup").forEach(s => s.remove()); return el.textContent.trim(); }'
            )
            if name and not re.match(r'^\d*\.', name):
                paper["authors"].append(name)
    else:
        # Older papers: plain text, comma-separated in h3#authorpart span
        author_text = await text('h3#authorpart')
        if author_text:
            paper["authors"] = [a.strip() for a in re.split(r'[,，;；]', author_text) if a.strip()]

    # Institutions: modern papers use h3.author:not(#authorpart) with <a> links
    # Older papers use the second h3.author as plain comma-separated text
    inst_links = await page.query_selector_all('h3.author:not(#authorpart) a')
    if not inst_links:
        inst_links = await page.query_selector_all('h3.orgn span a')
    if inst_links:
        for link in inst_links:
            t = (await link.inner_text()).strip()
            if t:
                t = re.sub(r'^\d+\.', '', t).strip()
                if t:
                    paper["institutions"].append(t)
    else:
        # Older papers: plain text institutions
        inst_h3s = await page.query_selector_all('h3.author:not(#authorpart)')
        for h3 in inst_h3s:
            t = (await h3.inner_text()).strip()
            if t:
                # Split by comma, strip postal codes (6-digit numbers)
                insts = re.split(r'[,，;；]', t)
                for inst in insts:
                    inst = re.sub(r'^\d+\.', '', inst).strip()
                    inst = re.sub(r'\s*\d{6}\s*$', '', inst).strip()
                    if inst and inst not in paper["institutions"]:
                        paper["institutions"].append(inst)

    paper["abstract"] = await text('#ChDivSummary')
    paper["abstract_en"] = await text('#EnChDivSummary')

    kw_els = await page.query_selector_all('p.keywords a')
    paper["keywords"] = [(await k.inner_text()).strip().rstrip(';；') for k in kw_els
                         if (await k.inner_text()).strip()]

    # Source (journal name): first link in top-tip pointing to navi.cnki.net
    paper["source"] = await text('div.top-tip a[href*="navi.cnki.net"]')
    paper["source"] = paper["source"].rstrip(' .')

    # Year/Volume/Issue: parse from top-tip links
    # Format examples: "2026 (02)", "2024,40(05):1-15", "2024,40(05)"
    top_tip_links = await page.query_selector_all('div.top-tip a')
    for link in top_tip_links:
        link_text = (await link.inner_text()).strip()
        # Look for patterns like "2024,40(05):1-15" or "2026 (02)" or "2024 ,40 (05) :1-15"
        # Normalize whitespace
        normalized = re.sub(r'\s+', '', link_text)
        # Pattern: YYYY or YYYY,VOL(ISSUE):PAGES
        m = re.match(r'^(\d{4})(?:,(\d+))?\((\d+)\)(?::(.+))?$', normalized)
        if m:
            paper["year"] = m.group(1)
            if m.group(2):
                paper["volume"] = m.group(2)
            paper["issue"] = m.group(3)
            if m.group(4):
                paper["pages"] = m.group(4)
            break

    # Pages: also check for "页码：X-Y" spans if not found above
    if not paper["pages"]:
        all_spans = await page.query_selector_all('.doc span')
        for span in all_spans:
            t = (await span.inner_text()).strip()
            if t.startswith('页码：'):
                paper["pages"] = t.replace('页码：', '').strip()
                break

    paper["doi"] = await text('li.top-space:has-text("DOI") p')
    paper["cited_count"] = await text('#refs a') or await text('div.total-inform span:has-text("被引") + em')
    paper["download_count"] = await text('#DownLoadParts a') or await text('div.total-inform span:has-text("下载") + em')
    paper["fund"] = await text('li.top-space:has-text("基金") p') or await text('p.funds span')
    paper["classification"] = await text('li:has-text("分类号") p')

    return paper


# =================== BibTeX via CNKI official export ===================

async def _get_cnki_bibtex(page: Page, url: str) -> dict:
    """Get official BibTeX from CNKI export page.

    Flow: detail page → click 引用 → extract export URL → navigate to export
    page → click BibTex → extract content.
    """
    # Establish session
    await page.goto("https://www.cnki.net/")
    await random_delay(1, 2)
    await page.set_extra_http_headers({"Referer": "https://kns.cnki.net/kns8s/AdvSearch"})
    await page.goto(url)
    await random_delay(1.5, 2.5)

    # Click 引用 button to open citation popup
    cite_btn = await page.query_selector('li.btn-quote a')
    if not cite_btn:
        return {"isError": True, "error": "引用按钮未找到"}
    await cite_btn.click()
    await random_delay(1.5, 2.5)

    # Extract export page URL from the popup
    export_link = await page.query_selector('.quote-pop a:has-text("更多引用格式")')
    if not export_link:
        return {"isError": True, "error": "未找到'更多引用格式'链接"}
    export_url = await export_link.get_attribute('href')
    if not export_url:
        return {"isError": True, "error": "导出链接为空"}

    # Navigate the same page to export URL
    await page.goto(export_url)
    await page.wait_for_load_state('networkidle')
    await random_delay(1.5, 2.5)

    # Click BibTex format option
    bibtex_link = await page.query_selector('a:has-text("BibTex")')
    if not bibtex_link:
        return {"isError": True, "error": "导出页面未找到 BibTex 选项"}
    await bibtex_link.click()
    await random_delay(1.5, 2.5)

    # Extract BibTeX content from the literature list
    content_el = await page.query_selector('ul.literature-list')
    if not content_el:
        return {"isError": True, "error": "未找到 BibTeX 内容"}
    bibtex_raw = (await content_el.inner_text()).strip()

    return {"bibtex": bibtex_raw}


def _enrich_bibtex(bibtex_raw: str, paper: dict) -> str:
    """Enrich official CNKI BibTeX with additional metadata (DOI, abstract, keywords)."""
    # Find the closing brace
    if '}' not in bibtex_raw:
        return bibtex_raw

    lines = bibtex_raw.rstrip().rstrip('}').rstrip()

    # Add fields that CNKI export typically omits
    extra_fields = []
    if paper.get("doi") and "doi" not in bibtex_raw.lower():
        extra_fields.append(f"  doi = {{{paper['doi']}}}")
    if paper.get("volume") and "volume" not in bibtex_raw.lower():
        extra_fields.append(f"  volume = {{{paper['volume']}}}")
    if paper.get("abstract") and "abstract" not in bibtex_raw.lower():
        extra_fields.append(f"  abstract = {{{paper['abstract']}}}")
    if paper.get("keywords") and "keywords" not in bibtex_raw.lower():
        kw = ", ".join(paper["keywords"])
        extra_fields.append(f"  keywords = {{{kw}}}")

    if extra_fields:
        # Ensure last existing line ends with comma
        if not lines.rstrip().endswith(','):
            lines = lines.rstrip() + ','
        lines += '\n' + ',\n'.join(extra_fields) + ','
    return lines.rstrip(',') + '\n}'


# =================== PDF download ===================

async def _download_paper_pdf(page: Page, url: str, save_dir: str) -> dict:
    """Navigate to a CNKI paper detail page and download the PDF."""
    # Establish session and navigate to paper page
    await page.goto("https://www.cnki.net/")
    await random_delay(1, 2)
    await page.set_extra_http_headers({"Referer": "https://kns.cnki.net/kns8s/AdvSearch"})
    await page.goto(url)
    await random_delay(1.5, 2.5)

    # Find PDF download button
    pdf_btn = await page.query_selector("a#pdfDown")
    if not pdf_btn:
        return {"isError": True, "error": "PDF下载按钮未找到，可能需要机构IP访问权限或该论文不支持PDF下载"}

    os.makedirs(save_dir, exist_ok=True)

    # Click and wait for download
    async with page.expect_download(timeout=60000) as download_info:
        await pdf_btn.click()
    download = await download_info.value

    suggested_name = download.suggested_filename
    save_path = os.path.join(save_dir, suggested_name)
    await download.save_as(save_path)

    return {
        "file_path": save_path,
        "file_name": suggested_name,
        "file_size": os.path.getsize(save_path),
    }


# =================== MCP server wiring, tools, resources, entry point ===================

@dataclass
class AppContext:
    browser_pool: BrowserPool


@asynccontextmanager
async def lifespan(server: FastMCP):
    pool = BrowserPool()
    try:
        yield AppContext(browser_pool=pool)
    finally:
        await pool.close()


mcp = FastMCP(
    "CNKI 论文检索服务",
    lifespan=lifespan,
    instructions="""
    CNKI (中国知网) 论文检索 MCP 服务器。

    ## 核心概念：论文标签

    搜索结果中每篇论文会返回一个短标签（如 `[1] 张三2024-经济增长与数字`），
    后续操作（获取详情、BibTeX、下载PDF）都使用这个标签，无需传递 URL。

    ## 工具

    ### search_cnki
    搜索 CNKI 论文，返回带标签的论文列表。参数:
    - query: 搜索关键词（必填）。只放主题/关键词/篇名，不要把作者名混入 query。
    - search_type: 搜索类型（主题/关键词/篇名/DOI等）
    - author: 按作者筛选（可选）
    - journal: 限定期刊名称（可选）
    - pages: 页数（1-10）
    - sort: 排序（相关度/发表时间/被引/下载/综合）

    ### get_paper_detail
    获取论文详情。参数: paper（论文标签）

    ### get_paper_bibtex
    获取论文 BibTeX 引用。参数: paper（论文标签）

    ### download_paper_pdf
    下载论文 PDF。参数: paper（论文标签）, save_dir（保存路径）

    ### find_best_match
    查找最匹配的论文标题，返回标签。参数: query（论文标题）

    ## 使用流程
    1. 用 search_cnki 搜索，获取论文标签列表
    2. 用标签调用 get_paper_detail 获取详情
    3. 用标签调用 get_paper_bibtex 获取 BibTeX
    4. 用标签调用 download_paper_pdf 下载 PDF
    """,
)


def get_browser_pool(ctx: Context = CurrentContext()) -> BrowserPool:
    return ctx.request_context.lifespan_context.browser_pool


@mcp.tool()
async def search_cnki(
    query: Annotated[str, Field(description="搜索关键词（主题/篇名等，不要把作者名放在这里，请用 author 参数）。多个关键词用空格分隔，会自动用 AND 连接，如'北京 奥运'→SU='北京' * '奥运'", min_length=1)],
    ctx: Context,
    search_type: Annotated[str, Field(
        description="搜索类型: 主题/关键词/篇名/DOI (英文: subject/keyword/title/doi)。注意：按作者筛选请用 author 参数而非设置 search_type='作者'"
    )] = "主题",
    author: Annotated[Optional[str], Field(
        description="按作者筛选（可与 query 组合使用）。例如搜索某作者关于某主题的论文：query='经济增长', author='张三'。设置后自动使用专业检索。"
    )] = None,
    journal: Annotated[Optional[str], Field(
        description="限定期刊名称（精确匹配），如'经济研究'。多个期刊用+分隔，如'经济研究+管理世界'。设置后使用专业检索。"
    )] = None,
    pages: Annotated[int, Field(description="搜索页数", ge=1, le=10)] = 1,
    sort: Annotated[str, Field(
        description="排序: 相关度/发表时间/被引/下载/综合 (英文: relevance/date/cited/download/composite)"
    )] = "相关度",
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """搜索 CNKI 论文，返回论文列表。支持通过 author 和 journal 参数分别筛选作者和期刊。

    重要：query 参数只放主题/关键词/篇名，不要把作者名混入 query。
    如需按作者搜索，请使用 author 参数。author 和 query 可组合使用。
    示例：搜索张三关于经济增长的论文 → query='经济增长', author='张三'
    """
    await ctx.info(f"搜索 CNKI: query='{query}', author={author}, journal={journal}")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        if journal or author:
            result = await _professional_search(page, query, search_type, journal, sort, pages, author=author)
        else:
            result = await _simple_search(page, query, search_type, sort, pages)
    except Exception as e:
        result = {"isError": True, "error": str(e), "papers": []}
    finally:
        await page.close()

    await ctx.report_progress(progress=100, total=100)
    if result.get("isError"):
        await ctx.error(f"搜索失败: {result.get('error')}")
    else:
        await ctx.info(f"找到 {result.get('total_papers', 0)} 篇论文")
    return result


@mcp.tool()
async def get_paper_detail(
    paper: Annotated[str, Field(description="论文标签（从 search_cnki 返回的 label 字段）")],
    ctx: Context,
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """获取 CNKI 论文详情页的完整信息。"""
    try:
        url = paper_registry.resolve(paper)
    except KeyError as e:
        return {"isError": True, "error": str(e)}

    await ctx.info(f"获取论文详情: {paper}")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        result = await _get_paper_detail(page, url)
    except Exception as e:
        result = {"isError": True, "error": str(e), "paper": paper}
    finally:
        await page.close()

    result["paper"] = paper
    await ctx.report_progress(progress=100, total=100)
    return result


@mcp.tool()
async def download_paper_pdf(
    paper: Annotated[str, Field(description="论文标签（从 search_cnki 返回的 label 字段）")],
    save_dir: Annotated[str, Field(description="PDF 保存目录的绝对路径")],
    ctx: Context,
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """下载 CNKI 论文 PDF 文件到指定目录。需要机构IP访问权限。"""
    try:
        url = paper_registry.resolve(paper)
    except KeyError as e:
        return {"isError": True, "error": str(e)}

    await ctx.info(f"下载论文 PDF: {paper}")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        result = await _download_paper_pdf(page, url, save_dir)
    except Exception as e:
        result = {"isError": True, "error": str(e), "paper": paper}
    finally:
        await page.close()

    result.pop("url", None)
    result["paper"] = paper
    await ctx.report_progress(progress=100, total=100)
    if result.get("isError"):
        await ctx.error(f"下载失败: {result.get('error')}")
    else:
        await ctx.info(f"PDF 已保存: {result.get('file_path')} ({result.get('file_size', 0)} bytes)")
    return result


@mcp.tool()
async def get_paper_bibtex(
    paper: Annotated[str, Field(description="论文标签（从 search_cnki 返回的 label 字段）")],
    ctx: Context,
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """获取 CNKI 论文的 BibTeX 引用条目（来自 CNKI 官方导出，并补充 DOI、摘要、关键词），可直接复制到 .bib 文件中使用。"""
    try:
        url = paper_registry.resolve(paper)
    except KeyError as e:
        return {"isError": True, "error": str(e)}

    await ctx.info(f"获取论文 BibTeX: {paper}")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        paper_detail = await _get_paper_detail(page, url)
        await ctx.report_progress(progress=40, total=100)

        bib_result = await _get_cnki_bibtex(page, url)
        await ctx.report_progress(progress=80, total=100)
    except Exception as e:
        return {"isError": True, "error": str(e), "paper": paper}
    finally:
        await page.close()

    if bib_result.get("isError"):
        await ctx.error(f"官方导出失败: {bib_result.get('error')}")
        return bib_result

    bibtex = _enrich_bibtex(bib_result["bibtex"], paper_detail)
    await ctx.report_progress(progress=100, total=100)
    await ctx.info("BibTeX 已生成（CNKI 官方导出 + 补充字段）")
    return {"paper": paper, "bibtex": bibtex}


@mcp.tool()
async def find_best_match(
    query: Annotated[str, Field(description="论文标题", min_length=1)],
    ctx: Context,
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """快速查找与输入标题最匹配的 CNKI 论文。"""
    await ctx.info(f"查找匹配: '{query[:50]}'")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        await page.goto("https://www.cnki.net/")
        await random_delay(1, 2)
        await type_slowly(page, "#txt_SearchText", query)
        await page.click(".search-btn")
        await random_delay(2, 3)

        titles, urls = [], []
        try:
            await page.wait_for_selector('a.fz14', timeout=15000)
            links = await page.query_selector_all('#gridTable a.fz14')
            for link in links:
                t = (await link.inner_text()).strip()
                u = await link.get_attribute("href")
                if t:
                    titles.append(t)
                    urls.append(u or "")
        except Exception:
            pass

        if not titles:
            result = {"query": query, "best_match": None, "message": "未找到结果"}
        else:
            idx = find_closest_title(query, titles)
            label = paper_registry.register(urls[idx], "", "", titles[idx])
            result = {
                "query": query,
                "best_match": {"title": titles[idx], "label": label},
                "total_results": len(titles),
            }
    except Exception as e:
        result = {"isError": True, "error": str(e)}
    finally:
        await page.close()

    await ctx.report_progress(progress=100, total=100)
    return result


@mcp.resource("cnki://search-types")
async def get_search_types(ctx: Context) -> str:
    return json.dumps({
        "chinese_types": list(SEARCH_TYPES.keys()),
        "english_aliases": list(SEARCH_TYPE_ALIASES.keys()),
        "default": "主题",
    }, ensure_ascii=False, indent=2)


@mcp.resource("cnki://status")
async def get_server_status(ctx: Context) -> str:
    return json.dumps({
        "server_name": "CNKI 论文检索服务",
        "version": "0.1.0",
        "backend": "Playwright (async)",
        "tools": ["search_cnki", "get_paper_detail", "get_paper_bibtex", "download_paper_pdf", "find_best_match"],
        "features": ["journal_filter_via_professional_search", "bibtex_export", "pdf_download", "browser_pool", "idle_timeout"],
    }, ensure_ascii=False, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
