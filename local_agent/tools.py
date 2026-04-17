"""Tools available to the local sparkyllm agent.

Each tool is a Callable[[str], str] — takes a single string argument and returns
a string. Add a new tool by writing the function and adding it to TOOLS.
"""
from __future__ import annotations

import ast
import json
import operator
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Callable, Dict, Optional


# ---- Structured Tool Response ----

@dataclass
class ToolResult:
    """Structured response from a tool."""
    success: bool
    value: str
    error: Optional[str] = None
    
    def __str__(self):
        if self.success:
            return self.value
        return f"ERROR: {self.error}"


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


# ---- Input Cleaning ----

def _clean_tool_input(raw_input: str, tool_name: str) -> str:
    """Normalize input for each tool type."""
    inp = (raw_input or "").strip()
    
    # Remove common model hallucinations
    prefixes = [f"{tool_name}(", f"{tool_name}:", f"{tool_name} "]
    for prefix in prefixes:
        if inp.lower().startswith(prefix.lower()):
            inp = inp[len(prefix):].rstrip(")").strip()
    
    return inp


# ---- Tools ----

def calculator(expression: str) -> str:
    """Evaluate a numeric expression. Supports + - * / // % ** and parentheses."""
    expr = _clean_tool_input(expression, "calculator")
    
    if not expr:
        return str(ToolResult(success=False, value="", error="empty expression"))
    
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
    except SyntaxError as e:
        return str(ToolResult(success=False, value="", error=f"bad syntax: {e.msg}"))
    except ZeroDivisionError:
        return str(ToolResult(success=False, value="", error="division by zero"))
    except Exception as e:
        return str(ToolResult(success=False, value="", error=str(e)))
    
    # Format result
    if isinstance(result, float) and result.is_integer():
        result_str = str(int(result))
    elif isinstance(result, float):
        result_str = f"{result:.6g}"
    else:
        result_str = str(result)
    
    return str(ToolResult(success=True, value=result_str))


def time_tool(_args: str = "") -> str:
    """Return the current local date and time."""
    return str(ToolResult(success=True, value=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def no_tool(_args: str = "") -> str:
    """Sentinel: model has decided no tool is needed. Returns empty string."""
    return str(ToolResult(success=True, value=""))


# ---- Web search ----

_USER_AGENT = "sparkyllm-local-agent/0.1 (https://github.com/ajencinas/sparkyllm)"
_HTTP_TIMEOUT = 6  # seconds
_MAX_RETRIES = 2


def _http_get_json(url: str, max_retries: int = _MAX_RETRIES) -> dict:
    """HTTP GET with retry and exponential backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    raise last_error


def _truncate(text: str, n: int = 400) -> str:
    text = " ".join((text or "").split())  # collapse whitespace
    if len(text) <= n:
        return text
    return text[: n - 3].rstrip() + "..."


# ---- TTL Cache for web results ----

class TTLCache:
    """Simple time-to-live cache for web search results."""
    
    def __init__(self, maxsize: int = 256, ttl: int = 300):
        self.cache: Dict[str, tuple] = {}
        self.maxsize = maxsize
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[str]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: str):
        # Evict oldest if at capacity
        if len(self.cache) >= self.maxsize:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        self.cache[key] = (value, time.time())


_web_cache = TTLCache(maxsize=256, ttl=300)  # 5 minute TTL


def _ddg_instant_answer(query: str) -> str:
    """Try DuckDuckGo's Instant Answer API. Returns text or empty string."""
    # Check cache first
    cache_key = f"ddg:{query}"
    cached = _web_cache.get(cache_key)
    if cached is not None:
        return cached
    
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    try:
        data = _http_get_json(url)
    except Exception:
        return ""
    
    result = ""
    for key in ("AbstractText", "Answer", "Definition"):
        val = data.get(key)
        if val:
            result = str(val)
            break
    
    if not result:
        for topic in data.get("RelatedTopics", []) or []:
            if isinstance(topic, dict) and topic.get("Text"):
                result = str(topic["Text"])
                break
    
    if result:
        _web_cache.set(cache_key, result)
    
    return result


def _wikipedia_summary(query: str) -> str:
    """Search Wikipedia for the query, return the summary of the top hit."""
    # Check cache first
    cache_key = f"wiki:{query}"
    cached = _web_cache.get(cache_key)
    if cached is not None:
        return cached
    
    search_url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "list": "search", "format": "json",
        "srsearch": query, "srlimit": "1",
    })
    
    try:
        sd = _http_get_json(search_url)
    except Exception:
        return ""
    
    hits = sd.get("query", {}).get("search", [])
    if not hits:
        return ""
    
    title = hits[0].get("title", "")
    if not title:
        return ""
    
    sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    
    try:
        summary = _http_get_json(sum_url)
    except Exception:
        return ""
    
    result = summary.get("extract", "") or ""
    if result:
        _web_cache.set(cache_key, result)
    
    return result


def web_search(query: str) -> str:
    """Search the web (DuckDuckGo Instant Answer, then Wikipedia)."""
    q = _clean_tool_input(query, "web_search")
    
    if not q:
        return str(ToolResult(success=False, value="", error="empty query"))

    # 1) DuckDuckGo Instant Answer
    text = _ddg_instant_answer(q)
    if text:
        return str(ToolResult(success=True, value=_truncate(text)))

    # 2) Wikipedia fallback
    try:
        text = _wikipedia_summary(q)
    except Exception as e:
        return str(ToolResult(success=False, value="", error=str(e)))
    
    if text:
        return str(ToolResult(success=True, value=_truncate(text)))

    return str(ToolResult(success=False, value="", error="no results"))


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
