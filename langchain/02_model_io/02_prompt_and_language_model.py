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


def prompt_and_lang_case1():
    logger.info(f"[Execute] {inspect.currentframe().f_code.co_name}")

    OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
    if OPENAI_SECRET_KEY is None:
        raise Exception('OpenAIのシークレットキーがNoneです')

    prompt = PromptTemplate(
        template="{shop}はどこにありますか？",
        input_variables = [
            "shop",
        ]
    )

    print(prompt.format(shop="株式会社篠栗モーター"))
    print(prompt.format(shop="Ducati"))


    try:
        # AIMessageを使って言語モデルからの返答を表す
        num_content = 1

        chat = ChatOpenAI(
            api_key=OPENAI_SECRET_KEY,
            verbose=True,
            model="gpt-4",
            temperature=0.7,
            # max_tokens=100,
            n=num_content,
        )

        filepath = Path(__file__).parent / Path('prompt.json')
        prompt_json = prompt.save(filepath)
        loaded_prompt = load_prompt(filepath)

        output_parser = CommaSeparatedListOutputParser()


        result = chat(messages=[
            # HumanMessage(content=prompt.format(shop="株式会社篠栗モーター")),
            # HumanMessage(content=loaded_prompt.format(shop="Aprillia")),
            # HumanMessage(content=loaded_prompt.format(shop="BMW")),
            HumanMessage(content="世界で代表的なバイクメーカーを教えてください"),
            HumanMessage(content=output_parser.get_format_instructions()), # 区切り指示
        ])

        # logger.info(result)

        output = output_parser.parse(text=result.content)

        for item in output:
            print(item)


        
        print('---------[response message]-----------')
        logger.info(result)

        

    except Exception as e:
        logger.error(e)


if __name__ == "__main__":
    prompt_and_lang_case1()