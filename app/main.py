import os
import sys
import argparse

# https://docs.langchain.com/oss/python/integrations/chat/

from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI
# from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from databricks_langchain import ChatDatabricks
from langchain_groq import ChatGroq
from langchain_litellm import ChatLiteLLM
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_mistralai import ChatMistralAI
from langchain_openrouter import ChatOpenRouter
from langchain_cohere import ChatCohere
from langchain_xai import ChatXAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_deepseek import ChatDeepSeek
from langchain_together import ChatTogether
from langchain_amazon_nova import ChatAmazonNova


# from dotenv import load_dotenv
# load_dotenv()




def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    


if __name__ == "__main__":
    main()
