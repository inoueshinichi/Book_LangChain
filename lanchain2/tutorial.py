import sys
import os
from pathlib import Path
import shutil
from pprint import pprint
import json

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
import tiktoken

# from langchain.llms import openai # 非推奨
from langchain_community.llms import OpenAI
from langchain.cache import (
    InMemoryCache,
    SQLiteCache,
    RedisCache,
    SQLAlchemyCache,
    GPTCache,
    MomentoCache,
)
from langchain.globals import (
    set_llm_cache,
    set_debug,
    set_verbose,
)

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


def llm():
    # Modelを実行
    llm = OpenAI(openai_api_key=OPENAI_SECRET_KEY)
    response = llm.invoke("こんにちは")
    pprint(response)

def generate():
    llm = OpenAI(openai_api_key=OPENAI_SECRET_KEY)
    response = llm.generate([
       "日本の首都はどこですか",
       "インドネシアの首都はどこですか",
    ])
    pprint(response.generations)
    print(response.generations[0][0].text)
    print(response.generations[1][0].text)

    pprint(response.llm_output)

    # 入力文字列のトークン数を計算
    print("tokens of `日本の首都はどこですか` ", llm.get_num_tokens("日本の首都はどこですか"))

def cache():
    """キャッシュの格納先
        InMemory     : PCのメモリ内キャッシュ
        SQLite       : SQLiteを使用
        Redis        : Redisを使用
        SQAlchemy    : SQLAlchemyを通じて様々なデータベースに接続
        GTPCache     : GPTCacheライブラリを使用して類似性に基づきキャッシュ
        MomentoCache : Momentoキャッシュサービスを使用
    """
    set_verbose(False)
    set_debug(False)

    # set_llm_cache(InMemoryCache())
    set_llm_cache(SQLiteCache(database_path=".langchain.db"))

    llm = OpenAI(openai_api_key=OPENAI_SECRET_KEY, max_tokens=300, cache=True)
    response = llm.generate(["福岡県の特徴について教えてください！"])
    pprint(response)
    print(response.generations[0][0].text)
    print(response.llm_output)

def chat1():
    chat = ChatOpenAI(max_tokens=300, api_key=OPENAI_SECRET_KEY)
    response = chat.invoke([
        SystemMessage("文章を英語に翻訳してください"),
        HumanMessage("私は35歳独身です。これからどのように生きていけば良いでしょうか。好きなことは技術です。")
    ])
    pprint(response.content)

def chat2():
    chat = ChatOpenAI(max_tokens=300, api_key=OPENAI_SECRET_KEY)
    response = chat.generate([
        [
            SystemMessage("文章を英語に翻訳してください"),
            HumanMessage("先日外国籍のお客様が来店されて、車のバンパー修理にかかる費用について尋ねました。")
        ],
        [
            SystemMessage("文章を英語に翻訳してください"),
            HumanMessage("私は彼の英語を通訳して社員に日本語で伝えました。")
        ],
    ])
    pprint(response.generations[0][0].text)
    pprint(chat.get_num_tokens_from_messages([
        SystemMessage("文章を英語に翻訳してください"),
        HumanMessage("先日外国籍のお客様が来店されて、車のバンパー修理にかかる費用について尋ねました。")
    ]))
    print()
    pprint(response.generations[1][0].text)
    pprint(chat.get_num_tokens_from_messages([
        SystemMessage("文章を英語に翻訳してください"),
            HumanMessage("私は彼の英語を通訳して社員に日本語で伝えました。")
    ]))

    pprint(response.llm_output)

def invoke():
    llm = OpenAI(openai_api_key=OPENAI_SECRET_KEY)
    chat = ChatOpenAI(api_key=OPENAI_SECRET_KEY)

    # predict は invoke　に変わった
    # print(f"こんにちは -(llm)->{llm.predict("こんにちは")}") # 削除された。代わりにinvokeを使う
    print(f"こんにちは -(llm)->{llm.invoke("こんにちは")}")
    print(f"こんにちは -(chat)->{chat.invoke("こんにちは")}")

if __name__ == "__main__":
    # llm()
    # generate()
    # cache()
    # chat1()
    # chat2()
    invoke()