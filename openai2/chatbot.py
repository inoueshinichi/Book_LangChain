import sys
import os
from pathlib import Path
import shutil
from pprint import pprint

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

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


def chatbot_once():
    
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    message = input("質問内容を入力してください：")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message}
        ],
        temperature=0.7,
        max_tokens=100,
        
    )
    print(completion.choices[0].message.content)

    """ 
        1) 一つ目のユーザーの入力
        2) 一つ目の入力に対するAIからの応答
        3) 二つ目のユーザーの入力
        から次の会話結果を出力する。
        この処理方法で、ChatCompletionモデルとユーザーの会話が成立する.
    """
    response_message = completion.choices[0].message.content
    new_message = input("再度、質問内容を入力してください：")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_message},
            {"role": "user", "content": new_message}
        ],
        presence_penalty=0,
        frequency_penalty=0,
        temperature=0.7,
        max_tokens=1000,
    )

    print(completion.choices[0].message.content)


def chatbot_loop():

    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    conversation_history = [] # GPT APIへのリクエストとなる会話履歴

    iteration_count = 0

    while iteration_count < 10:
        user_input = input("質問内容を入力してください：")
        conversation_history.append({
            # ユーザーの入力メッセージを追加
            "role": "user",
            "content": user_input
        })
        chatbot_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=1000,
            presence_penalty=0,
            frequency_penalty=0,
        )
        assistant_response = chatbot_response.choices[0].message.content
        used_tokens = chatbot_response.usage.total_tokens
        conversation_history.append({
            # AIからのレスポンスメッセージ
            "role": "assistant",
            "content": assistant_response
        })
        print(assistant_response)
        print(f"Used tokens: {used_tokens}")
        iteration_count += 1

def analyze_tokens():
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = tokenizer.encode("ChatGPTを勉強しています")
    print(tokens)
    print(tokenizer.decode(tokens))


def chatbot_restrict_loop():
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    conversation_history = []
    MAX_ALLOWED_TOKENS = 100
    total_tokens = 0
    iteration_count = 0

    while iteration_count < 10:
        user_input = input("質問内容を入力してください：")
        user_input_tokens = len(tokenizer.encode(user_input)) # 入力トークンを数える
        if user_input_tokens > MAX_ALLOWED_TOKENS:
            logger.warning(f"入力文字列が長いです。{MAX_ALLOWED_TOKENS}トークン以下にしてください。")
            continue

        conversation_history.append({"role": "user", "content": user_input})
        total_tokens += user_input_tokens

        chatbot_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            temperature=0.7,
            max_tokens=100,
            presence_penalty=0,
            frequency_penalty=0,
        )

        assistant_response = chatbot_response.choices[0].message.content
        assistant_tokens = len(tokenizer.encode(assistant_response))
        conversation_history.append({"role": "assistant", "content": assistant_response})
        total_tokens += assistant_tokens

        # トークンの合計が制限を超えた場合に一部を削除
        while total_tokens > MAX_ALLOWED_TOKENS:
            removed_message = conversation_history.pop(0)
            removed_tokens =len(tokenizer.encode(removed_message['content']))
            total_tokens -= removed_tokens
        print(assistant_response)
        iteration_count += 1

        if total_tokens > MAX_ALLOWED_TOKENS:
            # conversation_historyの内容を要約
            conversation_history.append({"role": "system", "content": "これまでの会話をすべて要約してください。"})
            summary = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation_history,
                temperature=0.7,
                max_tokens=100 # 出力の最大トークン数
            )
            summary_message = summary.choices[0].message.content
            # conversation_historyに要約結果をsystemメッセージとして設定して初期化
            conversation_history = [
                {"role": "system", "content": f"過去の要約: {summary_message}"}
            ]
            # total_tokensのリセット
            total_tokens = len(tokenizer.encode(summary_message))


def chatbot_option_1():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "面白い話をしてください："},
            ],
            temperature=1.6,
            max_tokens=200,
            presence_penalty=0,
            frequency_penalty=0,
        )

    print(completion.choices[0].message.content)


def chatbot_option_2():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "ITで注目する技術の話をしてください"},
            ],
            temperature=1.4,
            max_tokens=200,
            # presence_penalty=-2.0,
            presence_penalty=2.0,
            frequency_penalty=0,
        )

    print(completion.choices[0].message.content)


def chatbot_option_3():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    
    completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "ITで注目する技術の話をしてください"},
            ],
            temperature=1.4,
            max_tokens=200,
            # presence_penalty=-2.0,
            # presence_penalty=2.0,
            # frequency_penalty=-2.0,
            frequency_penalty=2.0,
        )

    print(completion.choices[0].message.content)

if __name__ == "__main__":
    # chatbot_once()
    # chatbot_loop()
    # analyze_tokens()
    # chatbot_restrict_loop()
    # chatbot_option_1()
    # chatbot_option_2()
    chatbot_option_3()