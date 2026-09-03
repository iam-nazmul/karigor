# karigor

 `karigor` is a terminal-based AI coding assistant that uses Large Language Models (LLMs) to understand code and complete programming tasks. In this challenge, you'll build  `karigor` as your own Claude Code that is capable of editing files, running commands, and iterating until the task is done.

Along the way, you'll learn about LLM APIs, tool calling, agent loops, and how to integrate multiple tools into an AI assistant.

## 1. Local Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com)
- `ollama serve` running (listens on `http://localhost:11434`)
- A tool-calling model pulled:

  ```bash
  ollama pull gemma4:e2b
  #ollama pull qwen3:8b
  ```

### Install

```bash
uv sync
```

### Run

```bash
uv run karigor -p "read app/main.py and summarize it"
```

`karigor` is the console script declared under `[project.scripts]` in `pyproject.toml`
(→ `app.main:main`). Equivalent invocations:

```bash
uv run python -m app.main -p "..."
.venv/bin/karigor -p "..."
```

### Flags

| flag | required | default | notes |
| --- | --- | --- | --- |
| `-p` | yes | — | the prompt |

### Choosing a model

The model is the `LLM_MODEL` constant in `app/main.py`, not a flag. It must be
one Ollama lists `tools` for under `ollama show <model>` — a model without tool
support runs fine and simply never calls anything:

```bash
ollama show gemma4:e2b | grep -A6 Capabilities
```

`NUM_CTX` sits next to it. Ollama defaults to a 4096-token context, which a
single tool result fills entirely; the model then hits its length limit and
returns empty text instead of an answer. 32768 leaves room to work.

### Credentials

None. Everything runs against local Ollama, and the docs tool uses DuckDuckGo
without a key. The values in `.env.example` apply only if you swap `ChatOllama`
for one of the commented-out cloud chat models at the top of `app/main.py`.

## 2. Communicate with the LLM

`ChatOllama` talks to the local Ollama server. A prompt goes in as a list of
messages — a `SystemMessage` setting the assistant's behaviour, then the user's
`HumanMessage` — and the reply streams back token by token:

```python
llm = ChatOllama(model=LLM_MODEL, num_ctx=NUM_CTX)
for chunk in llm.stream(messages):
    print(chunk.text, end="", flush=True)
```

Swapping providers means swapping this one line. The commented-out imports at the
top of `app/main.py` list the alternatives; the rest of the program is unchanged.

## 3. Advertise the read tool

Advertising is telling the model what it *may* call. The `@tool` decorator turns
a plain function into a schema, and `bind_tools` attaches that schema to every
request:

```python
@tool
def read_file(file_path: str, offset: int = 1, limit: int = DEFAULT_LIMIT) -> str:
    """Read a text file from the local filesystem.

    Returns the file contents with line numbers, in `cat -n` format.
    ...
    """

llm_with_tools = llm.bind_tools(list(TOOLS.values()))
```

The docstring becomes the tool's description and the type hints become its
parameters, so what the model is told and what the code does cannot drift apart.
To see exactly what is sent:

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
convert_to_openai_tool(TOOLS["read_file"])
# {'name': 'read_file', 'description': 'Read a text file...',
#  'parameters': {'properties': {'file_path': ..., 'offset': ..., 'limit': ...}}}
```

`read_file` handles any file type — see [step 10](#10-read-any-file-type-and-whole-directories)
for the document conversion layered on top of the text path described here. It
returns lines numbered `cat -n` style so the model can refer to a
line by number, truncates very long lines, and reports missing files, directories
and past-the-end offsets as plain strings — an error the model can read and
recover from beats an exception that kills the program.

## 4. Execute the read tool

The model never runs anything. It *asks*, by returning an assistant message with
`tool_calls` instead of text, and the program does the work:

```python
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

for call in ai_msg.tool_calls:          # {'name', 'args', 'id'}
    tool = TOOLS.get(call["name"])
    messages.append(tool.invoke(call))  # -> ToolMessage
```

Two details matter. Passing the **whole call** to `tool.invoke()` — not just
`call["args"]` — returns a `ToolMessage` with `tool_call_id` already set to match
the request, which is how the model pairs a result with the call it made when
several run at once. (A hand-built dict also needs `"type": "tool_call"` in it,
or LangChain reads the whole dict as the arguments.) And an unknown tool name
gets an error `ToolMessage` rather than a crash, since the model chooses these
names and can get one wrong.

**Wrong arguments need the same treatment as a wrong name.** The model picks the
right tool and then omits `content`, or uses the old `path` spelling, or sends a
number where a string belongs. `tool.invoke()` raises `ValidationError` on each,
and uncaught that ends the run with a traceback and nothing written — which looks
exactly like "the write tool doesn't create files". Caught, it becomes a message
the model can act on, with the schema attached so the retry has something to aim
at:

```python
except ValidationError as e:
    messages.append(ToolMessage(
        content=(f"Error: bad arguments for {call['name']}: {_format_validation_error(e)}\n"
                 f"{call['name']} takes: {_signature(tool)}"),
        tool_call_id=call["id"],
    ))
```

```
[write_file({'file_path': '/tmp/out.txt'})]
Error: bad arguments for write_file: content: Field required
write_file takes: file_path: string, content: string
[write_file({'file_path': '/tmp/out.txt', 'content': 'hi\n'})]
Created /tmp/out.txt (1 lines, 3 chars)
```

The turn after an error is where the work actually gets done, so the loop has to
still be running to reach it.

```
$ uv run karigor -p "Use the read_file tool on README.md and quote its first line."
[read_file({'file_path': 'README.md', 'limit': 1, 'offset': 1})]
The first line of `README.md` is:
`1\t# karigor`
```

## 5. Implement the agent loop

A single pass completes exactly one step. "Read this file and fix the bug" needs
the read to come back *before* the write can be written, so the exchange repeats
until the model answers without asking for anything:

```python
for _ in range(MAX_TURNS):
    ai_msg = _stream_turn(llm_with_tools, messages)
    messages.append(ai_msg)

    if not ai_msg.tool_calls:
        return                      # no call requested: this is the answer

    for call in ai_msg.tool_calls:
        messages.append(TOOLS[call["name"]].invoke(call))
```

Streaming and looping appear to conflict — you need the finished message to see
its `tool_calls`, but you want text as it arrives. `_stream_turn` does both: it
prints each chunk on arrival while summing chunks back into one message, and the
accumulated chunk still carries fully-formed `tool_calls`.

`MAX_TURNS` caps the loop. Without it a model that keeps calling tools without
ever concluding spins forever, burning tokens and touching the filesystem every
iteration.

The other way a run ends with nothing done is the connection dropping. Ollama
disconnects while it loads or evicts a model — a 7 GB model on CPU takes long
enough that the *first* turn is the one that gets cut off — and
`httpx.RemoteProtocolError` mid-stream would otherwise surface as a traceback
after the model had already announced what it was about to write. `_stream_turn`
retries the turn `LLM_RETRIES` times, and a server that is genuinely unreachable
exits with a message rather than a stack trace:

```
[llm connection lost (Server disconnected without sending a response.); retrying turn 1/3]
...
Error: lost the connection to Ollama: [Errno 111] Connection refused
Is `ollama serve` running, and does the model fit in memory?
```

The result is a program that chains steps on its own:

```
$ uv run karigor -p "Read buggy.py, find the bug, fix it, then run it to prove the fix works."
[read_file({'file_path': 'buggy.py'})]
[write_file({'content': 'def add(a, b):\n    return a + b\n\nprint(add(2, 3))', 'file_path': 'buggy.py'})]
[bash({'command': 'python buggy.py'})]
The bug was that `add` subtracted instead of adding. Running it now yields 5.
```

## 6. Implement the write tool

Same two halves as Read — advertise, then execute — with the filesystem effects
on the write side:

```python
@tool
def write_file(file_path: str, content: str) -> str:
    """Write text to a file on the local filesystem. ..."""
```

It creates the file when missing, overwrites when present, and creates missing
parent directories along the way. It returns `Created /abs/path (2 lines, 18
chars)` or `Updated ...` rather than nothing, so the model gets a confirmation it
can reason about instead of guessing whether the write landed.

Adding it to the program is one line — `TOOLS` is a name→tool registry, and the
dispatch loop from step 4 picks up anything in it unchanged:

```python
TOOLS = {
    t.name: t
    for t in [read_file, read_directory, write_file, bash, search_docs, browse]
}
```

## 7. Implement the bash tool

Shell access, which is what turns a file editor into an assistant that can verify
its own work:

```python
proc = subprocess.run(command, shell=True, executable="/bin/bash",
                      capture_output=True, text=True, timeout=timeout)
```

stdout and stderr come back merged, prefixed with `exit code N` when the command
fails so the model can tell a failure from a silent success. A command producing
no output returns `(no output, exit code 0)` rather than an empty string — an
empty tool result is ambiguous, and models tend to retry or invent output when
handed nothing. Timeouts (60s default) and spawn failures return strings too.

> **This runs arbitrary shell commands with your full user permissions.** There
> is no allowlist and no confirmation step: if the model emits
> `rm -rf something`, it runs. Inside the agent loop the model also chains
> commands it composed itself. Add a confirmation prompt before pointing this at
> a directory you care about.

## 8. Implement the search tool

Beyond the challenge, and the reason the assistant can work with libraries whose
APIs changed after the model's training cutoff:

```python
@tool
def search_docs(library: str, query: str, version: str = "") -> str:
    """Look up current documentation and code examples for a library. ..."""
```

It runs a DuckDuckGo search (`ddgs`, no API key), fetches the top `FETCH_PAGES`
results with `httpx`, and extracts text with BeautifulSoup. Extraction walks
block elements — `h1`-`h4`, `p`, `li`, `pre` — instead of flattening the page:
`<pre>` keeps its newlines so code examples survive, and preferring
`main`/`article`/`#content` over `<body>` avoids burying the content in sidebar
link lists, which are rarely marked up as `<nav>`. Pages that render client-side
yield no prose, so their search snippet is used instead.

Two limits worth knowing: `version` biases the search but cannot pin it, and web
search always answers — a misspelled library name returns something confident and
unrelated rather than an error.

## 9. Implement the browse tool

`search_docs` finds pages; `browse` reads one you already have the address for —
a release-note page, a GitHub issue, a raw file, a JSON endpoint, or a link that
came back from a search:

```python
@tool
def browse(url: str, offset: int = 0, limit: int = MAX_BROWSE_CHARS) -> str:
    """Fetch a URL over the internet and return its content as text. ..."""
```

The HTML extraction is the same code `search_docs` uses — `_fetch_text` was split
into `_get` (fetch, with HTTP errors returned as strings) and `_extract_text`
(soup to blocks), and both tools compose them. What `browse` adds is the handling
a single named URL needs:

- **Content types.** HTML goes through the extractor; JSON, plain text, XML, CSV
  and friends are returned verbatim, since running a JSON body through an HTML
  parser yields nothing. Anything else — an image, a tarball — reports its type
  and size instead of dumping bytes into the context window.
- **Paging.** A page over `MAX_BROWSE_CHARS` is cut with
  `continue with offset=N`, so a long document is readable across calls rather
  than silently ending mid-sentence.
- **Scheme check.** `http`/`https` only. A bare domain gets `https://` prepended;
  `file://` is refused, because that is `read_file`'s job.
- **Empty pages.** A client-rendered site extracts to nothing, which reads as "the
  page is empty" — it says the content is JavaScript-rendered instead.

```
$ uv run karigor -p "Browse https://example.com and tell me what that domain is for."
[browse({'url': 'https://example.com'})]
It's a reserved domain for use in documentation examples, no permission needed.
```

`browse` fetches whatever URL the model asks for, including addresses it read out
of a file or a web page. Anything reachable from this machine is in scope — a
`localhost` admin endpoint or a cloud metadata IP as much as a public docs site.

## 10. Read any file type, and whole directories

`read_file` originally decoded bytes as UTF-8, which is fine for source code and
useless for a PDF — `errors="replace"` turns one into pages of mojibake. Two
loaders fix that:

```python
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import DirectoryLoader
```

`UnstructuredLoader` converts PDFs, Word/PowerPoint/Excel documents, email,
epub and images to text locally — no API key, no `partition_via_api`. It returns
one `Document` per element (title, paragraph, table, list item); joining them in
order reproduces the document's flow.

**`read_file` routes by content, not by name.** A file is sent through
Unstructured when its extension is in `DOCUMENT_EXTS`, when its first bytes match
`DOCUMENT_MAGIC` (`%PDF`, the `PK\x03\x04` zip container behind every OOXML file,
the OLE2 container behind old `.doc`/`.xls`), or when it simply does not decode as
UTF-8. So a `.docx` saved as `report.txt` still comes back as prose, and the model
never has to check a format before reading:

```
$ uv run karigor -p "Read report.docx and tell me the revenue growth figure."
[read_file({'file_path': '.../report.docx'})]
The report states that revenue grew **12% in Q3**, driven by the pilot program.
```

Converted output is still line-numbered, but prefixed with
`[.pdf converted to text by Unstructured]` — the numbering is over the extracted
text, not the file on disk, and the model should not cite "line 12 of the PDF"
as if it were a real line.

**`read_directory` reads a whole tree in one call**, via `DirectoryLoader` with
`UnstructuredLoader` as its `loader_cls` and `silent_errors=True`, so one
unreadable file does not sink the folder. Results are grouped back per source
file, since the loader returns a flat list of elements from every file at once.

The cap matters more than the loading. `**/*` inside this repo matches ~62,000
files once `.venv` and `.git` are counted, and `Path.match` compares from the
right, so directory patterns like `**/.git/**` cannot be excluded. So the tool
counts matches first and refuses over `MAX_DIR_FILES`, before any parsing
happens:

```
Error: '**/*' matches 62752 files in /home/nazmul/Desktop/karigor, over the
50-file cap. Narrow the glob (e.g. '*.md', '**/*.pdf') or point at a subdirectory.
```

Per-file output is capped at `MAX_FILE_CHARS` and the whole call at
`MAX_DIR_CHARS`; `read_directory` is for surveying a folder, `read_file` for
reading one thing in full.

Two operational notes. The format extras are pinned in `pyproject.toml`
(`unstructured[docx,pptx,xlsx,md,pdf]`) — a format outside them returns
`needs an Unstructured extra that is not installed` rather than crashing, and
OCR for scanned PDFs and images additionally needs the `tesseract` binary.
Unstructured also downloads a spaCy model on first use and its dependencies
enable root logging, which is why `app/main.py` quiets a list of loggers
(including `httpx`, which otherwise narrates every Ollama call) before importing
them.

## Where things live

| | |
| --- | --- |
| tools | `read_file`, `read_directory`, `write_file`, `bash`, `search_docs`, `browse` in `app/main.py` |
| registry | `TOOLS` — add a tool here and the loop picks it up |
| loop | `main()`, one turn at a time via `_stream_turn()` |
| knobs | `LLM_MODEL`, `NUM_CTX`, `MAX_TURNS`, `SYSTEM_PROMPT` near the top |

`SYSTEM_PROMPT` is worth attention. It was a placeholder — the literal string
`"user"` — and a prompt that never tells the model to prefer a tool call over a
description makes tool use noticeably less reliable: a small model will happily
answer "I've created the file" without ever calling `write_file`. It now says to
do the work with the tools, never to claim a file was written unless the call
returned saying so, and to correct and retry a call the loop rejected.
