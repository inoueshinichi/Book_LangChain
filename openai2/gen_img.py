"""Generate Image by DALL-E"""
import sys
import os
from pathlib import Path
import shutil
from pprint import pprint
import json
import datetime

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


def dall_e_2():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    response = client.images.generate(
        model="dall-e-2",
        prompt="アニメのような白猫の絵",
        size="256x256",
        n=3, # 生成する画像の枚数
    )

    pprint(response.data) # // List[Image]

    download_path: str = os.path.join(Path(__file__).resolve().parent, "download_images", "dall-e-2")

    for i, data in enumerate(response.data):
        img_url = data.url
        img_base64_json = data.b64_json
        pprint(f"[{i}] {img_url}")
        pprint(f"[{i}] {img_base64_json}")

        now = datetime.datetime.now()
        formatted_string = now.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(download_path, f"img_{formatted_string}.jpg")
        response = requests.get(img_url)
        with open(file_path, "wb") as f:
            f.write(response.content)


def dall_e_3():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    response = client.images.generate(
        model="dall-e-3",
        prompt="アニメのような白猫の絵",
        size="1024x1024",
        # "standard", "hd"
        # quality="standard",
        quality="hd",
        # "vivid", "natural" 
        # style="natural", 
        style="vivid",
    )

    pprint(response.data) # // List[Image]

    download_path: str = os.path.join(Path(__file__).resolve().parent, "download_images", "dall-e-3")

    for i, data in enumerate(response.data):
        img_url = data.url
        img_base64_json = data.b64_json
        pprint(f"[{i}] {img_url}")
        pprint(f"[{i}] {img_base64_json}")

        now = datetime.datetime.now()
        formatted_string = now.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(download_path, f"img_{formatted_string}.jpg")
        response = requests.get(img_url)
        with open(file_path, "wb") as f:
            f.write(response.content)

if __name__ == "__main__":
    # dall_e_2()
    dall_e_3()