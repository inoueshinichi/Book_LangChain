# Flaskバックエンド：要約アプリ
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
from openai import OpenAI

from flask import Flask, render_template, request
import tiktoken

app = Flask(__name__)

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


client = OpenAI(api_key=OPENAI_SECRET_KEY)
MODEL_NAME = "gpt-3.5-turbo"
MAX_INPUT_TOKENS = 1000
ENCODING = tiktoken.encoding_for_model(MODEL_NAME)

def summarize_text(input_text):
    # APIを呼び出して文章を要約する関数
    num_input_tokens = len(ENCODING.encode(input_text))
    if num_input_tokens > MAX_INPUT_TOKENS:
        return "文字数が多すぎます。"
    
    # API
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "userが入力した文書を箇条書きで改行してまとめてください"},
            {"role": "user", "content": input_text}
        ],
        max_tokens=1000,
        temperature=1.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    summarized_text = completion.choices[0].message.content
    return summarized_text


@app.route("/", methods=["GET", "POST"])
def render_summary_page():
    input_text = ""
    summarized_text = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        summarized_text = summarize_text(input_text)
        summarized_text = summarized_text.split("\n")
        logger.info(summarized_text)
    return render_template("summary.html", input_text=input_text, summary=summarized_text)


# function calling
@app.route("/send_email", methods=["GET", "POST"])
def render_send_email_page(): # メール送信画面
    input_text = ""
    email_body = ""
    to_address = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        model_response = prepare_email_summary(input_text) # 外部API呼び出し
        if model_response.finish_reason == "function_call":
            function_call = model_response.message.function_call
            if function_call.name == "send_email":
                arguments = json.loads(function_call.arguments)
                to_address = arguments.get("to_address", "")
                email_body = arguments.get("email_body", "")
                send_email(to_address, email_body) # メール送信(モック)

    return render_template("send_email.html", 
                           input_text=input_text, 
                           email_body=email_body, 
                           to_address=to_address)

# prepare_email_summaryの定義
def prepare_email_summary(input_text):
    num_input_tokens = len(ENCODING.encode(input_text))
    if num_input_tokens > MAX_INPUT_TOKENS:
        return "文字数が多すぎます"
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":  "system", "content": "メールの宛先を抽出してto_addressに、内容のまとめをemail_bodyに設定してください。"},
            {"role": "user", "content": input_text},
        ],
        functions=[
            {
                "name": "send_email",
                "description": "メールを送る処理",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "to_address": {
                            "type": "string",
                            "description": "メールの宛先"
                        },
                        "email_body": {
                            "type": "string",
                            "description": "メールの内容",
                        },
                    },
                    "required": ["to_address", "email_body"]
                }
            }
        ],
        function_call="auto",
        max_tokens=1000
    )

    return completion.choices[0]

def send_email(to_address, email_body):
    # メール送信処理をここに書く。下記はモック
    print(f"{to_address}宛にメールを送信しました。")
    print(email_body)

if __name__ == "__main__":
    app.run(debug=True)

