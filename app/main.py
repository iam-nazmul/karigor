import os
import sys
import argparse
import subprocess

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS

# https://docs.langchain.com/oss/python/integrations/chat/
from langchain_ollama import ChatOllama

# from langchain_openai import AzureChatOpenAI
# from langchain_openai import ChatOpenAI
# # from langchain_google_vertexai import ChatVertexAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
# from databricks_langchain import ChatDatabricks
# from langchain_groq import ChatGroq
# from langchain_litellm import ChatLiteLLM
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_mistralai import ChatMistralAI
# from langchain_openrouter import ChatOpenRouter
# from langchain_cohere import ChatCohere
# from langchain_xai import ChatXAI
# from langchain_nvidia_ai_endpoints import ChatNVIDIA
# from langchain_deepseek import ChatDeepSeek
# from langchain_together import ChatTogether
# from langchain_amazon_nova import ChatAmazonNova


# from dotenv import load_dotenv
# load_dotenv()

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain.agents import create_agent
from langchain.tools import tool


# SYSTEM_PROMPT = (
#     "You are a helpful assistant. Answer clearly and concisely. "
#     "Use the read_file tool whenever you need the contents of a file on disk; "
#     "never guess what a file contains."
# )

SYSTEM_PROMPT =(
    "user"
)

MAX_LINE_LENGTH = 20000
DEFAULT_LIMIT = 20000
MAX_OUTPUT_CHARS = 30000
DEFAULT_TIMEOUT = 60

# Web search backing search_docs. No API key — DuckDuckGo via ddgs.
SEARCH_RESULTS = 5
FETCH_PAGES = 2
MAX_PAGE_CHARS = 6000
MIN_PAGE_CHARS = 200
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) karigor/0.1"
CONTENT_SELECTORS = ("main", "article", "[role=main]", "#content", ".markdown")
LLM_MODEL="gemma4:e2b"
# LLM_MODEL="qwen3:8b"
# Ollama defaults to a 4096-token context, which a single tool result fills —
# the model then has no room left to answer and returns empty text.
NUM_CTX = 32768


@tool
def read_file(path: str, offset: int = 1, limit: int = DEFAULT_LIMIT) -> str:
    """Read a text file from the local filesystem.

    Returns the file contents with line numbers, in `cat -n` format.

    Args:
        path: Absolute or relative path to the file to read.
        offset: 1-indexed line number to start reading from.
        limit: Maximum number of lines to return.
    """
    target = os.path.abspath(os.path.expanduser(path))

    if not os.path.exists(target):
        return f"Error: file not found: {target}"
    if os.path.isdir(target):
        return f"Error: {target} is a directory, not a file"

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Error: could not read {target}: {e}"

    if not lines:
        return f"{target} is empty"

    start = max(offset, 1) - 1
    if start >= len(lines):
        return f"Error: offset {offset} is past the end of {target} ({len(lines)} lines)"
    end = min(start + max(limit, 1), len(lines))

    out = []
    for i in range(start, end):
        text = lines[i].rstrip("\n")
        if len(text) > MAX_LINE_LENGTH:
            text = text[:MAX_LINE_LENGTH] + "... [line truncated]"
        out.append(f"{i + 1}\t{text}")

    if end < len(lines):
        out.append(f"... [truncated, {len(lines) - end} more lines]")

    return "\n".join(out)


@tool
def write_file(path: str, content: str) -> str:
    """Write text to a file on the local filesystem.

    Creates the file if it does not exist and overwrites it if it does.
    Missing parent directories are created. Prefer reading a file first
    before overwriting it.

    Args:
        path: Absolute or relative path to the file to write.
        content: The full text to write to the file.
    """
    target = os.path.abspath(os.path.expanduser(path))

    if os.path.isdir(target):
        return f"Error: {target} is a directory, not a file"

    existed = os.path.exists(target)

    parent = os.path.dirname(target)
    if parent and not os.path.isdir(parent):
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"Error: could not create directory {parent}: {e}"

    try:
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: could not write {target}: {e}"

    verb = "Updated" if existed else "Created"
    return f"{verb} {target} ({len(content.splitlines())} lines, {len(content)} chars)"


@tool
def bash(command: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    """Run a shell command and return its output.

    The command runs through bash in the current working directory.
    stdout and stderr are captured together. Use this for things the
    other tools cannot do: listing directories, running tests, git.

    Args:
        command: The shell command to run.
        timeout: Seconds to wait before killing the command.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=max(timeout, 1),
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s: {command}"
    except OSError as e:
        return f"Error: could not run command: {e}"

    output = (proc.stdout or "") + (proc.stderr or "")
    output = output.strip()

    if len(output) > MAX_OUTPUT_CHARS:
        dropped = len(output) - MAX_OUTPUT_CHARS
        output = output[:MAX_OUTPUT_CHARS] + f"\n... [truncated, {dropped} more chars]"

    if not output:
        return f"(no output, exit code {proc.returncode})"
    if proc.returncode != 0:
        return f"exit code {proc.returncode}\n{output}"
    return output


def _fetch_text(url: str) -> str:
    """Fetch a page and strip it down to readable text, or return an error string."""
    try:
        r = httpx.get(
            url,
            follow_redirects=True,
            timeout=20.0,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: {e.response.status_code} fetching {url}"
    except httpx.HTTPError as e:
        return f"Error: could not fetch {url}: {e}"

    soup = BeautifulSoup(r.text, "html.parser")
    # Nav chrome and scripts are pure noise once the page is flattened to text.
    for junk in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        junk.decompose()

    # Prefer the page's main content region. Sidebars are rarely marked up as
    # <nav>, so flattening the whole body buries the docs under link lists.
    candidates = [soup.select_one(sel) for sel in CONTENT_SELECTORS]
    candidates.append(soup.body or soup)
    best = max((c for c in candidates if c), key=lambda c: len(c.get_text()))

    # Walk block elements rather than flattening everything: get_text() on a
    # whole page puts every inline span on its own line, which shreds both
    # prose and code. <pre> keeps its newlines; everything else gets reflowed.
    blocks = []
    for el in best.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre"]):
        if el.name == "pre":
            code = el.get_text().strip()
            if code:
                blocks.append(f"```\n{code}\n```")
            continue

        text = " ".join(el.get_text(" ", strip=True).split())
        # Sidebar entries survive as short <li>s ("Vector stores"); real list
        # content in docs is a sentence or close to it.
        if not text or (el.name == "li" and len(text.split()) < 5):
            continue
        if not blocks or blocks[-1] != text:
            blocks.append(text)

    return "\n\n".join(blocks)


@tool
def search_docs(library: str, query: str, version: str = "") -> str:
    """Look up current documentation and code examples for a library or framework.

    Searches the web and reads the top documentation pages, so the answer
    comes from the docs as they are published today rather than from model
    memory. Use this for any question about a library, framework, SDK, or
    CLI tool — API syntax, configuration, setup, or migration.

    Args:
        library: Library name, e.g. "LangChain", "Next.js", "Prisma".
        query: What to look up, as a phrase — "how to define a custom tool",
            not a single word. Keep it to one concept per call.
        version: Optional version to bias the search, e.g. "v14". Search
            results are whatever the web serves, so this is a hint, not a pin.
    """
    terms = " ".join(t for t in [library, version, query, "documentation"] if t)

    try:
        results = DDGS().text(terms, max_results=SEARCH_RESULTS)
    except Exception as e:  # ddgs raises its own error types for rate limits
        return f"Error: search failed for {terms!r}: {e}"

    if not results:
        return f"No search results for {terms!r}"

    out = [f"Search: {terms}"]
    for i, hit in enumerate(results, start=1):
        title = hit.get("title", "")
        url = hit.get("href", "")
        out.append(f"\n{i}. {title}\n   {url}")

        # Only the top few pages are worth the round trip; the rest stay as
        # the search snippet so the model can ask for one by name.
        snippet = hit.get("body", "")
        if i <= FETCH_PAGES:
            body = _fetch_text(url)
            # Client-rendered doc sites return a shell with no prose in it;
            # the search snippet is then the more useful of the two.
            if body.startswith("Error:") or len(body) < MIN_PAGE_CHARS:
                body = snippet or body
        else:
            body = snippet

        if len(body) > MAX_PAGE_CHARS:
            body = body[:MAX_PAGE_CHARS] + "\n... [truncated]"
        out.append(body)

    return "\n".join(out)


TOOLS = {t.name: t for t in [read_file, write_file, bash, search_docs]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    # gemma3 has no tool-calling support in Ollama; use a tool-capable model.
    # p.add_argument("-m", "--model", default=LLM_MODEL)
    args = p.parse_args()

    # llm = ChatOllama(model=args.model)
    llm = ChatOllama(model=LLM_MODEL, num_ctx=NUM_CTX)

    # Advertise: the tool schema goes to the model with every request.
    llm_with_tools = llm.bind_tools(list(TOOLS.values()))

    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(args.p),
    ]

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        # The model answered directly; nothing to execute.
        print(ai_msg.text)
        return

    # Execute: the model only *asks* for a call — we run it and hand back the result.
    for call in ai_msg.tool_calls:
        print(f"[{call['name']}({call['args']})]", flush=True)
        tool = TOOLS.get(call["name"])
        if tool is None:
            messages.append(
                ToolMessage(
                    content=f"Error: unknown tool {call['name']}",
                    tool_call_id=call["id"],
                )
            )
            continue
        # .invoke() on the whole tool_call returns a ToolMessage with the id set.
        messages.append(tool.invoke(call))

    # One more turn so the model can answer from the tool output.
    for chunk in llm_with_tools.stream(messages):
        print(chunk.text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
