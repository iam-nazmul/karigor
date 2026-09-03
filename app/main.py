import os
import sys
import argparse
import logging
import subprocess
import warnings
from pathlib import Path

# Unstructured and its dependencies log INFO chatter on first use — font
# managers, thread counts, a one-off spaCy model download — which would
# interleave with the model's streamed answer. Importing them also turns on
# root logging, which is what makes httpx narrate every Ollama call, so those
# loggers are quieted here too. Set before importing them.
for _noisy in (
    "unstructured", "unstructured_client", "numexpr", "matplotlib",
    "pikepdf", "pdfminer", "PIL", "httpx", "httpcore",
):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from pydantic import ValidationError

# Any file type -> text: PDF, Office documents, email, epub, images. Local
# partitioning, no API key; the format-specific extras live in pyproject.
from langchain_unstructured import UnstructuredLoader

# langchain-community warns on import that it is being sunset. It is still the
# only home for DirectoryLoader, and the notice on every CLI run is noise.
warnings.filterwarnings("ignore", message=".*langchain-community.*sunset.*")
from langchain_community.document_loaders import DirectoryLoader

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


SYSTEM_PROMPT = (
    "You are a coding assistant working on the user's machine. Answer clearly "
    "and concisely.\n\n"
    "Do the work with the tools rather than describing it. Never say a file was "
    "created, changed or run unless the matching tool call returned a result "
    "saying so — a description of a write is not a write. Never guess what a "
    "file contains: read it.\n\n"
    "Creating or changing a file means calling write_file with both arguments, "
    "file_path and content, where content is the complete text of the file. "
    "If a tool returns an error, read it and try again — a bad argument can be "
    "corrected on the next call."
)

MAX_LINE_LENGTH = 20000
DEFAULT_LIMIT = 20000
MAX_OUTPUT_CHARS = 30000
DEFAULT_TIMEOUT = 60

# Formats that are not text on disk, so read_file routes them through
# Unstructured instead of decoding the bytes. Anything else that turns out to
# be binary is caught by sniffing, so this list only has to cover the common
# document types by name.
DOCUMENT_EXTS = {
    ".pdf", ".docx", ".doc", ".odt", ".rtf",
    ".pptx", ".ppt", ".odp",
    ".xlsx", ".xls", ".ods",
    ".epub", ".msg", ".eml", ".heic",
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp",
}

# Checked when the extension says nothing: PDF, the zip container behind every
# OOXML document, and the OLE2 container behind the older .doc/.xls/.ppt.
DOCUMENT_MAGIC = (b"%PDF", b"PK\x03\x04", b"\xd0\xcf\x11\xe0")

# read_directory reads whole trees, so it needs tighter caps than read_file:
# one runaway call otherwise fills the context window with a virtualenv.
MAX_DIR_FILES = 50
MAX_FILE_CHARS = 4000
MAX_DIR_CHARS = 40000
DIR_GLOB = "**/*"
# Path.match compares from the right, so directory patterns like "**/.git/**"
# do not work here — these are file patterns only. Trees too big to read are
# caught by MAX_DIR_FILES instead.
DIR_EXCLUDE = ("*.lock", "*.pyc", "*.so", "*.bin", "*.pack", "*.woff2")

# Web search backing search_docs. No API key — DuckDuckGo via ddgs.
SEARCH_RESULTS = 5
FETCH_PAGES = 2
MAX_PAGE_CHARS = 6000
MIN_PAGE_CHARS = 200
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) karigor/0.1"
CONTENT_SELECTORS = ("main", "article", "[role=main]", "#content", ".markdown")

# browse fetches one URL the model names, so it gets its own budget: a page read
# on purpose is worth more room than one of several search hits.
MAX_BROWSE_CHARS = 30000
BROWSE_TIMEOUT = 30.0
TEXTUAL_TYPES = ("text/", "json", "xml", "javascript", "csv", "yaml", "x-sh")
LLM_MODEL="gemma4:e2b"
# LLM_MODEL="qwen3:8b"
# Ollama defaults to a 4096-token context, which a single tool result fills —
# the model then has no room left to answer and returns empty text.
NUM_CTX = 32768
MAX_TURNS = 25
LLM_RETRIES = 3


def _is_document(path: str) -> bool:
    """True if the file should go through Unstructured rather than be decoded.

    read_file decodes with errors="replace", so without this a PDF comes back as
    pages of mojibake. Extension first, then the file's own bytes — a document
    saved under the wrong name is still a document.
    """
    if os.path.splitext(path)[1].lower() in DOCUMENT_EXTS:
        return True

    try:
        with open(path, "rb") as f:
            head = f.read(4096)
    except OSError:
        return False

    if head.startswith(DOCUMENT_MAGIC):
        return True
    if b"\x00" in head:
        return True
    try:
        # Drop the last few bytes: a multi-byte character split by the 4096-byte
        # cut is not evidence of a binary file.
        head[:-3].decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _unstructured_text(path: str) -> tuple[str, str | None]:
    """Convert any supported document to text. Returns (text, error)."""
    try:
        # One Document per element (title, paragraph, table, list item);
        # joining them back in order reads as the document did.
        docs = UnstructuredLoader(file_path=path).load()
    except ImportError as e:
        return "", (
            f"Error: {os.path.basename(path)} needs an Unstructured extra that is "
            f"not installed: {e}"
        )
    except Exception as e:
        # Unstructured raises format-specific errors from a dozen underlying
        # libraries; the model can act on the message, a traceback ends the run.
        return "", f"Error: could not extract text from {path}: {type(e).__name__}: {e}"

    return "\n".join(d.page_content.strip() for d in docs if d.page_content.strip()), None


@tool
def read_file(file_path: str, offset: int = 1, limit: int = DEFAULT_LIMIT) -> str:
    """Read a file of any type from the local filesystem.

    Text files are read as they are. PDFs, Word/PowerPoint/Excel documents,
    email, epub and images are converted to text with Unstructured first, so
    this tool works on any file — no need to check the format beforehand.

    Returns the contents with line numbers, in `cat -n` format.

    Args:
        file_path: Absolute or relative path to the file to read.
        offset: 1-indexed line number to start reading from.
        limit: Maximum number of lines to return.
    """
    target = os.path.abspath(os.path.expanduser(file_path))

    if not os.path.exists(target):
        return f"Error: file not found: {target}"
    if os.path.isdir(target):
        return f"Error: {target} is a directory, not a file. Use read_directory."

    note = ""
    ext = os.path.splitext(target)[1].lower()

    if _is_document(target):
        text, err = _unstructured_text(target)
        if err:
            return err
        if not text:
            return f"{target} converted to no text ({ext or 'no extension'})"
        # Line numbers are over the extracted text, not the file on disk; say so
        # rather than letting the model cite "line 12 of the PDF".
        note = f"[{ext or 'binary'} converted to text by Unstructured]\n"
        lines = text.splitlines(keepends=True)
    else:
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

    return note + "\n".join(out)


@tool
def write_file(file_path: str, content: str) -> str:
    """Write text to a file on the local filesystem.

    Creates the file if it does not exist and overwrites it if it does.
    Missing parent directories are created. Prefer reading a file first
    before overwriting it.

    Args:
        file_path: Absolute or relative path to the file to write.
        content: The full text to write to the file.
    """
    target = os.path.abspath(os.path.expanduser(file_path))

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
def read_directory(
    dir_path: str,
    glob: str = DIR_GLOB,
    max_files: int = MAX_DIR_FILES,
) -> str:
    """Read every file in a directory, of any type, in one call.

    Loads the whole tree — text, PDFs, Office documents, email — and returns
    each file's text under its path. Use this to get the lay of a folder you
    have not seen; use read_file when you know which file you want, since this
    truncates each file to a preview rather than returning it in full.

    Args:
        dir_path: Directory to read. Searched recursively.
        glob: Which files to load, e.g. "**/*.pdf" for just the PDFs or
            "*.md" for Markdown in the top level only. Defaults to everything.
        max_files: Refuse to load more than this many files. Narrow the glob
            or point at a subdirectory if a tree comes back over the cap.
    """
    target = os.path.abspath(os.path.expanduser(dir_path))

    if not os.path.exists(target):
        return f"Error: directory not found: {target}"
    if not os.path.isdir(target):
        return f"Error: {target} is a file, not a directory. Use read_file."

    # Count first, with DirectoryLoader's own matching rules. Loading a tree is
    # the expensive half — a virtualenv or a .git directory would be thousands
    # of files parsed one by one before anything came back.
    matches = [
        p for p in Path(target).glob(glob)
        if p.is_file() and not any(p.match(x) for x in DIR_EXCLUDE)
    ]
    if not matches:
        return f"No files match {glob!r} in {target}"
    if len(matches) > max_files:
        return (
            f"Error: {glob!r} matches {len(matches)} files in {target}, over the "
            f"{max_files}-file cap. Narrow the glob (e.g. '*.md', '**/*.pdf') or "
            "point at a subdirectory."
        )

    loader = DirectoryLoader(
        target,
        glob=glob,
        loader_cls=UnstructuredLoader,
        exclude=DIR_EXCLUDE,
        # One unreadable file in a folder should not fail the whole read.
        silent_errors=True,
        use_multithreading=True,
    )

    try:
        docs = loader.load()
    except Exception as e:
        return f"Error: could not read {target}: {type(e).__name__}: {e}"

    # One Document per element, so group them back into one text per file.
    by_file: dict[str, list[str]] = {}
    for d in docs:
        source = d.metadata.get("source", "unknown")
        if d.page_content.strip():
            by_file.setdefault(source, []).append(d.page_content.strip())

    if not by_file:
        return f"Loaded {len(matches)} files from {target}, but none yielded text"

    out = [f"{target} — {len(by_file)} of {len(matches)} files yielded text ({glob})"]
    total = 0
    for source in sorted(by_file):
        body = "\n".join(by_file[source])
        if len(body) > MAX_FILE_CHARS:
            body = body[:MAX_FILE_CHARS] + "\n... [truncated, read_file for the rest]"

        rel = os.path.relpath(source, target)
        out.append(f"\n=== {rel} ({len(body)} chars) ===\n{body}")

        total += len(body)
        if total > MAX_DIR_CHARS:
            out.append(f"\n... [stopped at {MAX_DIR_CHARS} chars, {len(by_file)} files total]")
            break

    skipped = sorted(set(os.path.abspath(str(p)) for p in matches) - set(by_file))
    if skipped:
        names = ", ".join(os.path.relpath(s, target) for s in skipped[:10])
        more = f" (+{len(skipped) - 10} more)" if len(skipped) > 10 else ""
        out.append(f"\nNo text extracted from: {names}{more}")

    return "\n".join(out)


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


def _get(url: str, timeout: float):
    """GET a URL. Returns (response, None) or (None, error string)."""
    try:
        r = httpx.get(
            url,
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return None, f"Error: {e.response.status_code} fetching {url}"
    except httpx.HTTPError as e:
        return None, f"Error: could not fetch {url}: {e}"
    return r, None


def _extract_text(html: str) -> str:
    """Strip an HTML document down to its readable text."""
    soup = BeautifulSoup(html, "html.parser")
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


def _fetch_text(url: str) -> str:
    """Fetch a page and strip it down to readable text, or return an error string."""
    r, err = _get(url, timeout=20.0)
    if err:
        return err
    return _extract_text(r.text)


@tool
def browse(url: str, offset: int = 0, limit: int = MAX_BROWSE_CHARS) -> str:
    """Fetch a URL over the internet and return its content as text.

    Use this to read a page whose address you already have: release notes, a
    GitHub issue or raw file, a JSON API response, a link that came back from
    search_docs. Use search_docs instead when you still need to find the page.

    HTML is stripped to its main content — headings, paragraphs, list items and
    code blocks. JSON, plain text and other textual bodies are returned as sent.
    Long pages are truncated; call again with a larger offset to read on.

    Args:
        url: The URL to fetch. http and https only; a bare domain is read as https.
        offset: Character offset to start from, for paging through a long page.
        limit: Maximum number of characters to return.
    """
    url = url.strip()
    if not url:
        return "Error: no url given"
    if "://" not in url:
        url = "https://" + url

    scheme = url.split("://", 1)[0].lower()
    # Anything else — file://, data://, ftp:// — is either a local read the
    # read_file tool already does or something this tool has no business doing.
    if scheme not in ("http", "https"):
        return f"Error: unsupported scheme {scheme!r}; browse handles http and https"

    r, err = _get(url, timeout=BROWSE_TIMEOUT)
    if err:
        return err

    content_type = r.headers.get("content-type", "").split(";")[0].strip().lower()

    if "html" in content_type or not content_type:
        body = _extract_text(r.text)
        # A client-rendered page is a shell with no prose in it. Saying so beats
        # returning nothing, which reads as "the page is empty".
        if not body:
            return (
                f"{url}\n({content_type or 'unknown type'}, no readable text — the page "
                "likely renders its content with JavaScript)"
            )
    elif any(t in content_type for t in TEXTUAL_TYPES):
        body = r.text
    else:
        return (
            f"Error: {url} is {content_type} ({len(r.content)} bytes), not text. "
            "Use bash with curl if you need to download it."
        )

    header = f"{url}\n({content_type or 'unknown type'}, {len(body)} chars)"

    start = max(offset, 0)
    if start >= len(body):
        return f"{header}\nError: offset {offset} is past the end of the content"
    end = min(start + max(limit, 1), len(body))

    out = body[start:end]
    if end < len(body):
        out += f"\n... [truncated, {len(body) - end} more chars; continue with offset={end}]"

    return f"{header}\n\n{out}"


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


TOOLS = {
    t.name: t
    for t in [read_file, read_directory, write_file, bash, search_docs, browse]
}


def _signature(tool) -> str:
    """The tool's arguments as a one-line signature, for error messages."""
    schema = tool.tool_call_schema.model_json_schema()
    required = set(schema.get("required", []))

    parts = []
    for name, spec in schema.get("properties", {}).items():
        kind = spec.get("type", "any")
        parts.append(f"{name}: {kind}" + ("" if name in required else " (optional)"))
    return ", ".join(parts)


def _format_validation_error(e: ValidationError) -> str:
    """Pydantic's multi-line report, flattened to what the model has to fix."""
    problems = []
    for err in e.errors():
        field = ".".join(str(p) for p in err["loc"]) or "argument"
        problems.append(f"{field}: {err['msg']}")
    return "; ".join(problems)


def _stream_turn(llm, messages):
    """Run one assistant turn, printing text as it arrives.

    Returns the accumulated message. Summing the streamed chunks rebuilds
    the tool calls too, so the loop can stream and still see what to run.
    """
    # Ollama drops the connection when it is loading or evicting a model — a
    # 7 GB model on CPU takes long enough that the first turn of a run is the
    # one that gets cut off. Unretried, the whole run ends in a traceback with
    # the work half done and nothing written.
    for attempt in range(1, LLM_RETRIES + 1):
        full = None
        try:
            for chunk in llm.stream(messages):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                full = chunk if full is None else full + chunk
        except httpx.HTTPError as e:
            if attempt == LLM_RETRIES:
                raise
            print(f"\n[llm connection lost ({e}); retrying turn {attempt}/{LLM_RETRIES}]",
                  file=sys.stderr, flush=True)
            continue

        if full is not None and full.text:
            print()
        return full


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

    # The agent loop: keep going until the model answers without asking for a
    # tool. One pass only ever gets one step done — "read this file and fix the
    # bug" needs the model to see the read before it can write.
    for _ in range(MAX_TURNS):
        try:
            ai_msg = _stream_turn(llm_with_tools, messages)
        except httpx.HTTPError as e:
            print(f"\nError: lost the connection to Ollama: {e}", file=sys.stderr)
            print("Is `ollama serve` running, and does the model fit in memory?",
                  file=sys.stderr)
            sys.exit(1)

        if ai_msg is None:
            print("\nError: the model returned nothing. A context window too "
                  "small for the tool results is the usual cause; NUM_CTX is "
                  f"{NUM_CTX}.", file=sys.stderr)
            sys.exit(1)

        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # No call requested, so this was the final answer.
            return

        # Execute: the model only *asks* for a call — we run it and hand back
        # the result, then loop so it can act on what it just learned.
        for call in ai_msg.tool_calls:
            print(f"[{call['name']}({call['args']})]", flush=True)
            tool = TOOLS.get(call["name"])
            if tool is None:
                known = ", ".join(sorted(TOOLS))
                messages.append(
                    ToolMessage(
                        content=f"Error: unknown tool {call['name']}. Available: {known}",
                        tool_call_id=call["id"],
                    )
                )
                continue

            try:
                # .invoke() on the whole tool_call returns a ToolMessage with the id set.
                messages.append(tool.invoke(call))
            except ValidationError as e:
                # The model picked the right tool and got its arguments wrong —
                # a missing `content`, the old `path` name, a number where a
                # string belongs. Uncaught, that ends the run with a traceback
                # and nothing written. Handed back as a message with the schema
                # in it, the model can fix the call on the next turn.
                messages.append(
                    ToolMessage(
                        content=(
                            f"Error: bad arguments for {call['name']}: "
                            f"{_format_validation_error(e)}\n"
                            f"{call['name']} takes: {_signature(tool)}"
                        ),
                        tool_call_id=call["id"],
                    )
                )
            except Exception as e:
                # A tool that raises anything else should not take the run with
                # it either; the model can try another approach.
                messages.append(
                    ToolMessage(
                        content=f"Error: {call['name']} failed: {type(e).__name__}: {e}",
                        tool_call_id=call["id"],
                    )
                )

    # A model that keeps calling tools without concluding would otherwise spin
    # here forever, burning tokens and touching the filesystem each time.
    print(f"[stopped after {MAX_TURNS} turns]", file=sys.stderr)


if __name__ == "__main__":
    main()
