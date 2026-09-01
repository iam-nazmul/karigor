import os
import sys
import argparse
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")




def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()
   

    if not API_KEY:
        print("Error: OPENROUTER_API_KEY is not set in the environment variables.")

    


if __name__ == "__main__":
    main()
