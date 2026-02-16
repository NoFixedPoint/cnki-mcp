"""
CNKI MCP Server - CNKI academic paper search via MCP.

Tools:
- search_cnki: Search papers (with optional journal filter)
- get_paper_detail: Get full paper metadata
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


async def _professional_search(page: Page, query: str, search_type: str, journal: str, sort: str, pages: int) -> dict:
    """Search via CNKI Professional Search (with journal filter)."""
    resolved_type = resolve_search_type(search_type)
    resolved_sort = resolve_sort_type(sort)
    field_code = PROFESSIONAL_SEARCH_FIELDS.get(resolved_type, "SU")

    expr = f"({field_code}='{query}') AND (JN='{journal}')"

    await page.goto("https://kns.cnki.net/kns8s/AdvSearch")
    await random_delay(1, 2)

    # Click Professional Search tab
    try:
        await page.click("li.professional-search", timeout=5000)
    except Exception:
        await page.get_by_text("专业检索").click()
    await random_delay(0.5, 1)

    # Enter expression
    textarea = page.locator("textarea.professional-input, textarea#ExpertValue, textarea.search-input")
    await textarea.fill(expr)
    await random_delay(0.3, 0.5)

    # Click search
    search_btn = page.locator("div.professional-search input.btn-search, input.btn-search")
    await search_btn.click()
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
        "query": query, "search_type": resolved_type, "journal": journal,
        "sort": resolved_sort, "expression": expr,
        "total_pages": pages, "total_papers": len(all_papers), "papers": all_papers,
    }


# =================== Paper detail ===================

async def _get_paper_detail(page: Page, url: str) -> dict:
    """Navigate to a CNKI paper detail page and extract metadata."""
    paper = {
        "url": url, "title": "", "title_en": "", "authors": [],
        "institutions": [], "abstract": "", "abstract_en": "",
        "keywords": [], "keywords_en": [], "source": "", "year": "",
        "volume": "", "issue": "", "pages": "", "doi": "",
        "cited_count": "", "download_count": "", "fund": "", "classification": "",
    }

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
    paper["authors"] = await texts('h3.author span a')
    paper["institutions"] = await texts('h3.orgn span a')
    paper["abstract"] = await text('#ChDivSummary')
    paper["abstract_en"] = await text('#EnChDivSummary')

    kw_els = await page.query_selector_all('p.keywords a')
    paper["keywords"] = [(await k.inner_text()).strip().rstrip(';；') for k in kw_els
                         if (await k.inner_text()).strip()]

    paper["source"] = await text('div.top-tip a[href*="navi.cnki.net"]')
    paper["source"] = paper["source"].rstrip(' .')

    info_text = await text('div.top-tip span')
    if ',' in info_text:
        parts = info_text.split(',')
        paper["year"] = parts[0].strip()
        if len(parts) > 1:
            rest = parts[1]
            if '(' in rest and ')' in rest:
                paper["volume"] = rest.split('(')[0].strip()
                paper["issue"] = rest.split('(')[1].split(')')[0].strip()
            if ':' in rest:
                paper["pages"] = rest.split(':')[-1].strip()

    paper["doi"] = await text('li.top-space:has-text("DOI") p')
    paper["cited_count"] = await text('#refs a') or await text('div.total-inform span:has-text("被引") + em')
    paper["download_count"] = await text('#DownLoadParts a') or await text('div.total-inform span:has-text("下载") + em')
    paper["fund"] = await text('li:has-text("基金") p') or await text('p.funds span')
    paper["classification"] = await text('li:has-text("分类号") p')

    return paper


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

    ## 工具

    ### search_cnki
    搜索 CNKI 论文。参数:
    - query: 搜索关键词（必填）
    - search_type: 搜索类型（主题/关键词/作者/篇名/DOI等）
    - journal: 限定期刊名称（可选，设置后使用专业检索）
    - pages: 页数（1-10）
    - sort: 排序（相关度/发表时间/被引/下载/综合）

    ### get_paper_detail
    获取论文详情。参数: url（CNKI 论文详情页 URL）

    ### find_best_match
    查找最匹配的论文标题。参数: query（论文标题）

    ## 使用建议
    1. 先用 search_cnki 搜索
    2. 用 get_paper_detail 获取详情
    3. 用 journal 参数限定期刊范围
    """,
)


def get_browser_pool(ctx: Context = CurrentContext()) -> BrowserPool:
    return ctx.request_context.lifespan_context.browser_pool


@mcp.tool()
async def search_cnki(
    query: Annotated[str, Field(description="搜索关键词", min_length=1)],
    ctx: Context,
    search_type: Annotated[str, Field(
        description="搜索类型: 主题/关键词/作者/篇名/DOI (英文: subject/keyword/author/title/doi)"
    )] = "主题",
    journal: Annotated[Optional[str], Field(
        description="限定期刊名称，如'经济研究'。设置后使用专业检索。"
    )] = None,
    pages: Annotated[int, Field(description="搜索页数", ge=1, le=10)] = 1,
    sort: Annotated[str, Field(
        description="排序: 相关度/发表时间/被引/下载/综合 (英文: relevance/date/cited/download/composite)"
    )] = "相关度",
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """搜索 CNKI 论文，返回论文列表。支持通过 journal 参数限定期刊。"""
    await ctx.info(f"搜索 CNKI: query='{query}', journal={journal}")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        if journal:
            result = await _professional_search(page, query, search_type, journal, sort, pages)
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
    url: Annotated[str, Field(description="CNKI 论文详情页 URL")],
    ctx: Context,
    browser_pool: BrowserPool = Depends(get_browser_pool),
) -> dict:
    """获取 CNKI 论文详情页的完整信息。"""
    if not url or "cnki" not in url.lower():
        return {"isError": True, "error": "URL 必须是 CNKI 链接"}

    await ctx.info(f"获取论文详情: {url[:80]}...")
    await ctx.report_progress(progress=0, total=100)

    page = await browser_pool.get_page()
    try:
        result = await _get_paper_detail(page, url)
    except Exception as e:
        result = {"isError": True, "error": str(e), "url": url}
    finally:
        await page.close()

    await ctx.report_progress(progress=100, total=100)
    return result


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
            result = {
                "query": query,
                "best_match": {"title": titles[idx], "url": urls[idx]},
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
        "tools": ["search_cnki", "get_paper_detail", "find_best_match"],
        "features": ["journal_filter_via_professional_search", "browser_pool", "idle_timeout"],
    }, ensure_ascii=False, indent=2)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
