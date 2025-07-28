
"""Quetion and Answering through Image by GPT-4 lo"""
import sys
import os
from pathlib import Path
import shutil
from pprint import pprint
import json
import datetime
import base64

import logging
logging.basicConfig(format="%(asctime)s [%(levelname)s] (Line:%(lineno)d) at %(name)s : %(message)s", datefmt="[%X]")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
# logger.setLevel(logging.WARNING)
# logger.setLevel(logging.ERROR)
# logger.setLevel(logging.CRITICAL)


from dotenv import load_dotenv
import requests
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

# image binary -> base64 str
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def gpt4_with_vision():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)

    img_path = os.path.join(Path(__file__).resolve().parent, "qa_target.jpg")
    print("img_path", img_path)

    base64_img = encode_image(img_path)

    prompt = "この画像には何が写っていますか？ また、この画像を撮影した場所を推測してください。"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_SECRET_KEY}"
    }

    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
    }

    # リクエストを送信する
    response = requests.post("https://api.openai.com/v1/chat/completions", 
                             headers=headers, 
                             json=payload)
    pprint(response.json())


if __name__ == "__main__":
    gpt4_with_vision()