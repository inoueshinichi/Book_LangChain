""" OpenAI Completion Model
"""
import sys
import os
from pathlib import Path
import shutil
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


def completion_case1():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')
    
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    try:
        num_content = 2

        prompt_msg = "今日は天気が良いです。自分はバイクを持っていて、熊本県の阿蘇方面に"

        response = client.completions.create(
            model="gpt-3.5-turbo-instruct", 
            prompt=prompt_msg,  #←promptを指定
            stop="。",  #←文字が出現したら文章を終了する
            max_tokens=100,  #←最大のトークン数
            n=num_content,  #←生成する文章の数
            temperature=0.5  #←多様性を表すパラメータ
        )

        print(response.json(indent=2, ensure_ascii=False))

        logger.info(f"id: {response.id}")
        logger.info(f"model: {response.model}")
        logger.info(f"object: {response.object}")
        logger.info(f"prompt_tokens: {response.usage.prompt_tokens}")
        logger.info(f"complettion_tokens: {response.usage.completion_tokens}")
        logger.info(f"total_tokens: {response.usage.total_tokens}")
        print('---------[response message]-----------')
        
        logger.info(f"[Prompt] {prompt_msg}")
        for n in range(num_content):
            logger.info(response.choices[n].text)
            logger.info(f"終了原因: {response.choices[n].finish_reason}")

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    completion_case1()