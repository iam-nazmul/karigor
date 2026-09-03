import os
import sys
import argparse
import subprocess

import httpx

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


from dotenv import load_dotenv

load_dotenv()

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

# https://context7.com — up-to-date docs pulled from each library's own source.
CONTEXT7_BASE_URL = "https://context7.com/api/v1"
DEFAULT_DOC_TOKENS = 3000
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


def _slug(name: str) -> str:
    """Lowercase, alphanumerics only — so "Next.js" and "nextjs" compare equal."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _context7_get(path: str, params: dict, accept_text: bool = False):
    """GET against the Context7 API. Returns parsed JSON, text, or an error string."""
    key = os.environ.get("CONTEXT7_API_KEY")
    if not key:
        return "Error: CONTEXT7_API_KEY is not set (see .env.example)"

    try:
        r = httpx.get(
            f"{CONTEXT7_BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {key}"},
            timeout=30.0,
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        return f"Error: Context7 returned {e.response.status_code} for {path}"
    except httpx.HTTPError as e:
        return f"Error: could not reach Context7: {e}"

    return r.text if accept_text else r.json()


@tool
def search_docs(library: str, query: str, version: str = "") -> str:
    """Look up current documentation and code examples for a library or framework.

    Fetches docs straight from the library's own source, so the answer
    reflects the released version rather than model memory. Use this for
    any question about a library, framework, SDK, or CLI tool — API syntax,
    configuration, setup, or migration.

    Args:
        library: Library name, e.g. "LangChain", "Next.js", "Prisma".
        query: What to look up, as a phrase — "how to define a custom tool",
            not a single word. Keep it to one concept per call.
        version: Optional version to pin, e.g. "v14.3.0". Omit for latest.
    """
    found = _context7_get("/search", {"query": library})
    if isinstance(found, str):
        return found

    results = found.get("results") or []
    if not results:
        return f"No library named {library!r} found on Context7"

    # Search is fuzzy and always answers, so a nonsense name still comes back
    # with something unrelated. Keep only titles that actually match the name
    # asked for, then let quality decide between them — the raw relevance
    # score is not comparable across queries.
    wanted = _slug(library)
    titles = [(r, _slug(r.get("title", ""))) for r in results]
    # Exact title first: a prefix match alone pulls in "LangChain NVIDIA" for
    # "LangChain", and those sub-projects often outrank the real docs.
    matches = [r for r, t in titles if t == wanted]
    if not matches:
        matches = [r for r, t in titles if t.startswith(wanted)]
    if not matches:
        close = ", ".join(f"{r.get('title')} ({r['id']})" for r in results[:3])
        return f"No confident match for {library!r} on Context7. Closest: {close}"

    best = max(
        matches,
        key=lambda r: (r.get("benchmarkScore") or 0, r.get("trustScore") or 0),
    )
    library_id = best["id"]

    if version:
        available = [v for v in best.get("versions") or [] if version in v]
        if not available:
            return (
                f"Version {version!r} is not available for {library_id}. "
                f"Known versions: {', '.join(best.get('versions') or []) or 'none'}"
            )
        library_id = f"{library_id}/{available[0]}"

    docs = _context7_get(
        f"/{library_id.lstrip('/')}",
        {"type": "txt", "topic": query, "tokens": DEFAULT_DOC_TOKENS},
        accept_text=True,
    )
    if docs.startswith("Error:"):
        return docs
    if not docs.strip():
        return f"No documentation found in {library_id} for {query!r}"

    return f"Docs for {best.get('title', library)} ({library_id}):\n\n{docs.strip()}"


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
