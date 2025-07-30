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


def chatbot():
    chat_llm = ChatOpenAI(max_tokens=500, api_key=OPENAI_SECRET_KEY)
    summarize_llm = ChatOpenAI(max_tokens=200, api_key=OPENAI_SECRET_KEY)

    # 会話の履歴を保存するリスト
    conversation_history = []
    max_allowed_tokens = 300
    question_count = 0

    # ユーザーの質問が10回になるまでループ
    while question_count < 10:
        user_input = input("質問内容を入力してください：")
        human_message = HumanMessage(content=user_input)

        # ユーザー入力のトークン数を取得
        user_input_tokens = chat_llm.get_num_tokens_from_messages([human_message])

        # トークン数が制限を超えていれば警告を出力
        if user_input_tokens > max_allowed_tokens:
            print(f"入力文字列が長いdす。{max_allowed_tokens}トークン以下にしてください。")
            continue

        # ユーザー入力を会話履歴に追加
        conversation_history.append(human_message)

        # チャットぼっとからの返答
        chatbot_response = chat_llm.invoke(conversation_history)

        conversation_history.append(chatbot_response)
        # pprint(f"conversation_history: {conversation_history}")

        # トークン数が制限を越えていれば会話を要約
        total_tokens = chat_llm.get_num_tokens_from_messages(conversation_history)
        if total_tokens > max_allowed_tokens:
            conversation_history.append(
                SystemMessage(content="これまでの会話を全て要約してください。")
            )
            summary = summarize_llm.invoke(conversation_history)
            summary_message = summary.content

            # 会話履歴を要約だけにする
            conversation_history = [
                SystemMessage(content=f"過去の要約: {summary_message}")
            ]

        print(conversation_history) # 会話履歴の表示
        question_count += 1


if __name__ == "__main__":
    chatbot()