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
from langchain import OpenAI
from langchain_community.chat_models import ChatOpenAI  #← モジュールをインポート
from langchain_community.llms import gpt4all
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain.prompts import (
    PromptTemplate,
    load_prompt,
)
from langchain.output_parsers import (
    CommaSeparatedListOutputParser,
)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))


def complete_model_case1():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')

    try:
        llm = OpenAI(model='gpt-3.5-turbo-instruct', api_key=OPENAI_SECRET_KEY)
        result = llm("拝啓、")

        logger.info(result)

        print('---------[response message]-----------')
        logger.info(result)

    except Exception as e:
        logger.error(e)


def complete_model_case2():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')

    try:
        llm = gpt4all.GPT4All() # GPUのみ対応
        result = llm("拝啓、")

        logger.info(result)

        print('---------[response message]-----------')
        logger.info(result)

    except Exception as e:
        logger.error(e)

if __name__ == "__main__":
    # complete_model_case1()
    # complete_model_case2()
    pass
