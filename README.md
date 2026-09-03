# karigor

 `karigor` is a terminal-based AI coding assistant that uses Large Language Models (LLMs) to understand code and complete programming tasks. In this challenge, you'll build  `karigor` as your own Claude Code that is capable of editing files, running commands, and iterating until the task is done.

Along the way, you'll learn about LLM APIs, tool calling, agent loops, and how to integrate multiple tools into an AI assistant.

## 1. Local Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com)
- `ollama serve` running (listens on `http://localhost:11434`)
- A tool-calling model pulled:

  ```bash
  ollama pull qwen3:8b
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
<!--| `-m` / `--model` | no | `qwen3:8b` | any Ollama model **with tool support**; `gemma3:4b` runs but never calls `read_file` |-->

### Credentials

The Ollama path needs none. The keys in `.env.example` apply only if you swap
`ChatOllama` for one of the commented-out cloud chat models at the top of `app/main.py`.

## 2. Communicate with the LLM
## 3. Advertise the read tool
## 4. Execute the read tool
## 5. Implement the agent loop
## 6. Implement the write tool
## 7. Implement the bash tool