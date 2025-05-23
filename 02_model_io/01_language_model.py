""" Model I/O Language
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
# from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI  #← モジュールをインポート
from langchain.schema import HumanMessage 

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

""" Environment Validable
"""
# logger.debug(f"[CHECK] Environments variable")
# for i, (k, v) in enumerate(os.environ.items()):
#     # logger.debug(f"{i}th {k}: {v}")
#     print(f"[{k}] {v}")

def language_case1():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')
    
    try:
        num_content = 1

        chat = ChatOpenAI(
            api_key=OPENAI_SECRET_KEY,
            verbose=True,
            model="gpt-4",
            temperature=0.7,
            # max_tokens=100,
            n=num_content,
        )

        result = chat(messages=[
            HumanMessage(content="Next.jsの利点について教えて"),
        ])

        logger.info(result)
        print('---------[response message]-----------')
        logger.info(result.content)

    except Exception as e:
        logger.error(e)

    
    
if __name__ == "__main__":
    language_case1()