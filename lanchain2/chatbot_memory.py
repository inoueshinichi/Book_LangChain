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
# from langchain_community.llms import OpenAI
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

# Memory
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory,
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


def memory():
    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message("こんにちは")
    memory.chat_memory.add_ai_message("どうしましたか？")
    memory.chat_memory.add_user_message("太陽系の惑星は何個ありますか？")
    memory.chat_memory.add_ai_message("8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。")

    print(memory.load_memory_variables({}))
    print("-----")
    print(memory.load_memory_variables({})["history"])
    print()

    # リスト形式でレスポンスを取得する
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_user_message("こんにちは")
    memory.chat_memory.add_ai_message("どうしましたか？")
    pprint(memory.load_memory_variables({})['history'])
    print()

    # systemメッセージを入力するにはadd_messageを使う
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_message(SystemMessage(content="メッセージを要約してください"))
    memory.chat_memory.add_message(HumanMessage(content="太陽系の惑星は何個ありますか？"))
    memory.chat_memory.add_message(AIMessage(content="8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。"))
    pprint(memory.load_memory_variables({})['history'])
    print()

    # save_contextでHummanMessageとAIMessageの対会話を保存する
    memory = ConversationBufferMemory(return_messages=True)
    memory.save_context(
        {"input": "太陽系の惑星は何個ありますか？"},
        {"output": "8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。"}
    )
    pprint(memory.load_memory_variables({})['history'])
    print()


def window_memory():
    """最新のk回分の会話のみ保存する"""
    memory = ConversationBufferWindowMemory(return_messages=True, k=1)
    memory.chat_memory.add_user_message("こんにちは")
    memory.chat_memory.add_ai_message("どうしましたか？")
    memory.chat_memory.add_user_message("太陽系の惑星は何個ありますか？")
    memory.chat_memory.add_ai_message("8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。")
    print(memory.load_memory_variables({}))
    print("-----")
    print(memory.load_memory_variables({})["history"])
    print()

    # k=1だと古いMessageが取り出されないので注意
    memory = ConversationBufferWindowMemory(return_messages=True, k=1)
    memory.chat_memory.add_message(SystemMessage(content="メッセージを要約してください"))
    memory.chat_memory.add_message(HumanMessage(content="太陽系の惑星は何個ありますか？"))
    memory.chat_memory.add_message(AIMessage(content="8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。"))
    pprint(memory.load_memory_variables({})['history'])
    print()


def chatbot_memory():
    chat_model = ChatOpenAI(max_tokens=300, api_key=OPENAI_SECRET_KEY)
    memory = ConversationBufferWindowMemory(return_messages=True, k=1)
    memory.chat_memory.add_user_message("こんにちは")
    memory.chat_memory.add_ai_message("どうしましたか？")
    memory.chat_memory.add_user_message("太陽系の惑星は何個ありますか？")
    memory.chat_memory.add_ai_message("8個あります。水星、金星、地球、火星、木星、土星、天王星、海王星です。")
    response = chat_model.generate([
        memory.load_memory_variables({})['history']
    ])
    pprint(response)
    

def token_memory():
    chat_model = ChatOpenAI(max_tokens=300, api_key=OPENAI_SECRET_KEY)

    # Memoryの初期化
    memory = ConversationTokenBufferMemory(llm=chat_model, max_token_limit=30)
    memory.save_context({"input": "こんにちは"}, {"output": "どうも"}) # トークン数のチェック
    memory.save_context({"input": "元気ですか？"}, {"output": "まあまあですね。"}) # トークン数のチェック
    print(memory.load_memory_variables({}))

if __name__ == "__main__":
    # memory()
    # window_memory()
    # chatbot_memory()
    token_memory()
    pass