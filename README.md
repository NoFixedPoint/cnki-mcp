# cnki-mcp

MCP server for searching academic papers on [CNKI](https://www.cnki.net/) (China National Knowledge Infrastructure). Built with [FastMCP](https://github.com/jlowin/fastmcp) and [Playwright](https://playwright.dev/). Supports journal filtering via CNKI professional search.

## Install

```bash
pip install git+https://github.com/NoFixedPoint/cnki-mcp.git
playwright install chromium
```

## Configure

**Claude Code** — add to `.mcp.json` or run `claude mcp add cnki cnki-mcp`:

```json
{
  "mcpServers": {
    "cnki": { "command": "cnki-mcp" }
  }
}
```

**Cursor** — add to `~/.cursor/mcp.json` with the same format.

## Tools

| Tool | Description |
|------|-------------|
| `search_cnki` | Search papers by keyword, author, title, DOI, etc. Optional `journal` param restricts to a specific journal via CNKI professional search. |
| `get_paper_detail` | Get full metadata (abstract, authors, keywords, DOI, citations, ...) for a paper URL. |
| `find_best_match` | Find the paper whose title best matches the input. |

## Examples

- "Search CNKI for papers about 人民币国际化"
- "Search CNKI for 人民币国际化 papers in 经济研究"
- "Search CNKI for 人民币国际化 papers in 管理世界"
- "Get details for this CNKI paper: \<url\>"

## Requirements

- Python >= 3.10
- Chrome/Chromium (installed via `playwright install chromium`)

## License

MIT
