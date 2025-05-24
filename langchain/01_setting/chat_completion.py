""" OpenAI Chat Completions Model
"""

import sys
import os
from pathlib import Path
import shutil
from textwrap import indent
from typing import (
    Optional,
    Callable,
    Union,
    Tuple,
    List,
    Set,
    Sequence,
    Dict,
)
import json
import inspect

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

""" Environment Validable
"""
# logger.debug(f"[CHECK] Environments variable")
# for i, (k, v) in enumerate(os.environ.items()):
#     # logger.debug(f"{i}th {k}: {v}")
#     print(f"[{k}] {v}")

def chat_completions_case1():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')
    
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    try:

        response = client.chat.completions.create(
            model = "gpt-4",
            messages = [
                {
                    "role": "user",
                    "content": 'iphone 8 plusのリリース日を教えて! ',
                }
            ]
        )

        print(response.json(indent=2))

        logger.info(f"id: {response.id}")
        logger.info(f"model: {response.model}")
        logger.info(f"object: {response.object}")
        logger.info(f"prompt_tokens: {response.usage.prompt_tokens}")
        logger.info(f"complettion_tokens: {response.usage.completion_tokens}")
        logger.info(f"total_tokens: {response.usage.total_tokens}")
        print('---------[response message]-----------')
        logger.info(response.choices[0].message.content)
       

    except Exception as e:
        print(e)

def chat_completions_case2():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')
    
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    try:
        num_content = 2

        response = client.chat.completions.create(
            model = "gpt-4",
            messages = [
                {
                    "role": "user",
                    "content": "そばの原材料を教えて",
                }
            ],
            max_tokens=100, # 出力トークン
            temperature=1, 
            n=num_content,
        )

        print(response.json(indent=2))

        logger.info(f"id: {response.id}")
        logger.info(f"model: {response.model}")
        logger.info(f"object: {response.object}")
        logger.info(f"prompt_tokens: {response.usage.prompt_tokens}")
        logger.info(f"complettion_tokens: {response.usage.completion_tokens}")
        logger.info(f"total_tokens: {response.usage.total_tokens}")
        print('---------[response message]-----------')

        for n in range(num_content):
            logger.info(response.choices[n].message.content)
            logger.info(f"終了原因: {response.choices[n].finish_reason}")

       
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # chat_completions_case1()
    chat_completions_case2()

