import os
import sys
import argparse

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

from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_agent
from langchain.tools import tool


SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and concisely. "
    "Use the read_file tool whenever you need the contents of a file on disk; "
    "never guess what a file contains."
)

MAX_LINE_LENGTH = 2000
DEFAULT_LIMIT = 2000


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    # gemma3 has no tool-calling support in Ollama; use a tool-capable model.
    p.add_argument("-m", "--model", default="qwen3:8b")
    args = p.parse_args()

    llm = ChatOllama(model=args.model)

    agent = create_agent(
        model=llm,
        tools=[read_file],
        system_prompt=SYSTEM_PROMPT,
    )

    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": args.p}]},
        version="v3",
    )
    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            for token in item.text:
                print(token, end="", flush=True)
        elif kind == "tool_calls":
            print(f"\n[{item.tool_name}({item.input})]", flush=True)
    print()


if __name__ == "__main__":
    main()
