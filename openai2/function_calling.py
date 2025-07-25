"""Function Calling"""
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

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")

def main():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    def get_current_weather(location, unit="fahrenheit"):
        "指定した場所の現在の天気を取得(°C or F)"
        # ただし下記はダミー関数。本来は外部APIなどを呼び出して可変的にリアルタイムレスポンスを取得する
        weather_info = {
            "location": location,
            "temperature": 72,
            "unit": unit,
            "forecast": "sunny",
        }
        return weather_info

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
             { "role": "user", "content": "東京の現在の天気を教えてください。"}
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "指定した場所の現在の天気を取得",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "都市名と国の名前、例：東京、日本"
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                }
            },
        ],
        function_call="auto",
    )

    print(">>> response")
    pprint(response)

    func = response.choices[0]
    # 該当するfunction callingか否かチェック
    if func.finish_reason == "function_call":
        function_call = func.message.function_call
        if function_call.name == "get_current_weather":
            arguments = function_call.arguments
            arguments = json.loads(arguments) # JSON Str -> dict
            location = arguments.get("location", "")
            unit = arguments.get('unit', "fahrenheit")
            # 外部API呼び出し
            current_weather = get_current_weather(location, unit)

            pprint(current_weather)




if __name__ == "__main__":
    main()