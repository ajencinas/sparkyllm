"""Tools available to the local sparkyllm agent.

Each tool is a Callable[[str], str] — takes a single string argument and returns
a string. Add a new tool by writing the function and adding it to TOOLS.
"""
from __future__ import annotations

import ast
import html
import json
import operator
import os
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Callable, Dict


# ---- Lightweight .env loader ----
# Reads BRAVE_API_KEY (and any other KEY=value lines) from a .env file in
# the project root if it exists. Silently skips if the file isn't there.
# Colab users don't need .env — they set os.environ from userdata instead.

def _load_dotenv() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, ".env"),                           # local_agent/.env
        os.path.join(here, "..", ".env"),                     # repo root/.env
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_dotenv()


# ---- Safe arithmetic evaluator ----
# No eval(), no __import__, no attribute access, no name lookups.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UN_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_safe_eval(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


# ---- Tools ----

def calculator(expression: str) -> str:
    """Evaluate a numeric expression. Supports + - * / // % ** and parentheses."""
    expr = (expression or "").strip()
    # Strip a leading "calculator(" / trailing ")" if the model wrote it that way
    if expr.lower().startswith("calculator(") and expr.endswith(")"):
        expr = expr[len("calculator("):-1].strip()
    if not expr:
        return "ERROR: empty expression"
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
    except SyntaxError as e:
        return f"ERROR: bad syntax: {e.msg}"
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except Exception as e:
        return f"ERROR: {e}"
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    if isinstance(result, float):
        # Trim very long floats
        return f"{result:.6g}"
    return str(result)


def time_tool(_args: str = "") -> str:
    """Return the current local date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def no_tool(_args: str = "") -> str:
    """Sentinel: model has decided no tool is needed. Returns empty string."""
    return ""


# ---- Web search ----

_USER_AGENT = "sparkyllm-local-agent/0.1 (https://github.com/ajencinas/sparkyllm)"
_HTTP_TIMEOUT = 6  # seconds


def _http_get_json(url: str, headers: Dict[str, str] = None) -> dict:
    merged = {"User-Agent": _USER_AGENT}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, headers=merged)
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _truncate(text: str, n: int = 400) -> str:
    text = " ".join((text or "").split())  # collapse whitespace
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


def _clean_snippet(text: str) -> str:
    """Unescape HTML entities and strip tag markers from a search snippet."""
    if not text:
        return ""
    # Brave sometimes returns <strong>...</strong> around query matches
    text = text.replace("<strong>", "").replace("</strong>", "")
    text = text.replace("<b>", "").replace("</b>", "")
    return html.unescape(text)


def _brave_search(query: str) -> str:
    """Brave Search API — requires BRAVE_API_KEY. Returns snippet or empty."""
    key = os.environ.get("BRAVE_API_KEY", "").strip()
    if not key:
        return ""
    url = "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({
        "q": query, "count": "3",
        "search_lang": "en", "country": "us",  # prefer English results
    })
    try:
        data = _http_get_json(url, headers={
            "X-Subscription-Token": key,
            "Accept": "application/json",
        })
    except Exception:
        return ""
    results = (data.get("web", {}) or {}).get("results", []) or []
    if not results:
        return ""
    top = results[0]
    # Prefer description (2-3 sentence factual snippet); fall back to title
    text = _clean_snippet(top.get("description") or top.get("title") or "")
    return text


def _wikipedia_summary(query: str) -> str:
    """Search Wikipedia for the query, return the summary of the top hit."""
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "format": "json",
        "srsearch": query, "srlimit": "1",
    })
    sd = _http_get_json(search_url)
    hits = sd.get("query", {}).get("search", [])
    if not hits:
        return ""
    title = hits[0].get("title", "")
    if not title:
        return ""
    sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    summary = _http_get_json(sum_url)
    return summary.get("extract", "") or ""


def _ddg_instant_answer_strict(query: str) -> str:
    """DuckDuckGo Instant Answer — authoritative fields only.

    Drops RelatedTopics because it surfaces noisy regional associations
    (e.g. "Encinasola, Huelva" → Jamón). Use as last-resort fallback only.
    """
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    try:
        data = _http_get_json(url)
    except Exception:
        return ""
    for key in ("AbstractText", "Answer", "Definition"):
        val = data.get(key)
        if val:
            return str(val)
    return ""


def web_search(query: str) -> str:
    """Search the web. Order: Brave (if keyed) → Wikipedia → strict DDG."""
    q = (query or "").strip()
    # Strip a leading "web_search(" / trailing ")" if the model wrote it that way
    if q.lower().startswith("web_search(") and q.endswith(")"):
        q = q[len("web_search("):-1].strip()
    if not q:
        return "ERROR: empty query"

    # 1) Brave (only if API key is configured — real web relevance)
    text = _brave_search(q)
    if text:
        return _truncate(text)

    # 2) Wikipedia — reliable for people, places, concepts
    try:
        text = _wikipedia_summary(q)
    except Exception:
        text = ""
    if text:
        return _truncate(text)

    # 3) DuckDuckGo Instant Answer — authoritative fields only
    text = _ddg_instant_answer_strict(q)
    if text:
        return _truncate(text)

    return "no results"


TOOLS: Dict[str, Callable[[str], str]] = {
    "calculator": calculator,
    "time": time_tool,
    "web_search": web_search,
    "none": no_tool,
}


def tool_descriptions() -> str:
    """Short description block for the agent prompt."""
    return (
        "- calculator — does arithmetic. Input is a math expression like 2+3*4.\n"
        "- time — returns the current date and time. Input is empty.\n"
        "- web_search — looks up a topic on the web. Input is a short search query.\n"
        "- none — answer directly without a tool. Input is empty."
    )
