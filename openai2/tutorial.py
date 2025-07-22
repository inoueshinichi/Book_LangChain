import sys
import os
from pathlib import Path
import shutil
from pprint import pprint

import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] (Line:%(lineno)d) at %(name)s : %(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)
# logger.setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from openai import OpenAI

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


def case1():
    # Hello World
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "こんにちは",
            }
        ],
        model="gpt-3.5-turbo"
    )

    generated_text = response.choices[0].message.content
    logger.info(generated_text)


def case2():
    # リクエストの詳細
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "あなたはアシスタントです."},
            {"role": "user", "content": "こんにちは"},
            {"role": "assistant", "content": "初めまして"},
        ],
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        presence_penalty=0,
        frequency_penalty=0,
    )

    # logger.info(completion)
    pprint(completion)


def case3():
    # レスポンスの詳細

    

if __name__ == "__main__":
    # case1()
    case2()