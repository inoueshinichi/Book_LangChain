
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
# from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI  #← モジュールをインポート
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain.prompts import (
    PromptTemplate,
)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))


def prompt_case1():
    prompt = PromptTemplate(
        template="{shop}はどこにありますか？",
        input_variables = [
            "shop",
        ]
    )

    print(prompt.format(shop="株式会社篠栗モーター"))
    print(prompt.format(shop="Ducati"))


if __name__ == "__main__":
    prompt_case1()