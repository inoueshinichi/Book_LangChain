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


def summary_memory():
    chat_model = ChatOpenAI(max_tokens=300, api_key=OPENAI_SECRET_KEY)

    # Messageを格納する際に要約して履歴として残す
    memory = ConversationSummaryMemory(llm=chat_model)
    memory.save_context({"input": "こんにちは"}, {"output": "どうも"})
    print(memory.load_memory_variables({})["history"])

    memory.save_context({"input": "今日の天気はどうなりますか?"}, 
                        {"output": "申し訳ございませんが、私はリアルタイムの天気情報を持っていません"})
    
    print("---")

    print(memory.load_memory_variables({})["history"])


def summary_token_memory():
    chat_model = ChatOpenAI(max_tokens=200, api_key=OPENAI_SECRET_KEY)

    # 指定トークン数を超えると自動で要約した内容を履歴として残す
    memory = ConversationSummaryBufferMemory(llm=chat_model, max_token_limit=60)
    memory.save_context({"input": "今日の天気はどうなりますか?"}, 
                        {"output": "申し訳ございませんが、私はリアルタイムの天気情報を持っていません"})
    
    print(memory.load_memory_variables({})["history"])

    print("---")

    memory.save_context({"input": "明日の天気はどうなりますか？"},
                        {"output": "申し訳ございませんが、私はリアルタイムの天気情報を持っていません"})
    
    print(memory.load_memory_variables({})["history"])


from langchain.schema.messages import get_buffer_string
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.summary import SummarizerMixin
from langchain_core.prompts import BasePromptTemplate

CUSTOM_DEFAULT_SUMMARIZER_TEMPLATE = """
履歴と最新の会話を要約してください。

# 履歴
{summary}

# 最新の会話
Human: {new_lines}

出力：
"""

CUSTOM_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"],
    template=CUSTOM_DEFAULT_SUMMARIZER_TEMPLATE
)

class CustomConversationSummaryBufferMemory(ConversationSummaryBufferMemory):
    prompt: BasePromptTemplate = CUSTOM_SUMMARY_PROMPT # langchainの基底クラスのフィールドのオーバーライドはannotation必須.



def custom_summary_token_memory():
    chat_model = ChatOpenAI(max_tokens=200, api_key=OPENAI_SECRET_KEY)

    # custom memory with buffer
    memory = CustomConversationSummaryBufferMemory(llm=chat_model, max_token_limit=60)
    memory.save_context({"input": "今日の天気はどうなりますか?"}, 
                        {"output": "申し訳ございませんが、私はリアルタイムの天気情報を持っていません"})
    
    print(memory.load_memory_variables({})["history"])

    print("---")

    memory.save_context({"input": "明日の天気はどうなりますか？"},
                        {"output": "申し訳ございませんが、私はリアルタイムの天気情報を持っていません"})
    
    print(memory.load_memory_variables({})["history"])



def chatbot_summary_history():
    chat_llm = ChatOpenAI(max_tokens=500, api_key=OPENAI_SECRET_KEY)

    # 会話履歴
    conversation_memory = ConversationBufferWindowMemory(return_messages=True, k=2)

    question_count = 0

    # ユーザーの質問が10回になるまでループ
    while question_count < 10:
        user_input = input("質問内容を入力してください：")
        human_message = HumanMessage(content=user_input)

        conversation_memory.chat_memory.add_message(human_message)
        print(type(conversation_memory.load_memory_variables({})["history"]))
        print(conversation_memory.load_memory_variables({})["history"])

        # チャットぼっとからの返答 (この時Memoryから過去の2回の対話を取り出す)
        chatbot_response = chat_llm.predict_messages(
            conversation_memory.load_memory_variables({})["history"]
        )
        conversation_memory.chat_memory.add_message(chatbot_response)

        print(conversation_memory.load_memory_variables({}))
        question_count += 1

if __name__ == "__main__":
    # memory()
    # window_memory()
    # chatbot_memory()
    # token_memory()
    # summary_memory()
    # summary_token_memory()
    # custom_summary_token_memory()
    chatbot_summary_history()
    pass