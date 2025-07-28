"""Variation Image by DALL-E"""
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

    img_path = os.path.join(Path(__file__).resolve().parent, "nekosan_rgba.png")
    print("img_path", img_path)

    response = client.images.create_variation(
        model="dall-e-2",
        image=open(img_path, "rb"),
        n=1,
        size="256x256",
    )

    img_url = response.data[0].url
    print(img_url)

    download_path: str = os.path.join(Path(__file__).resolve().parent, "download_images", "variation")
    now = datetime.datetime.now()
    formatted_string = now.strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(download_path, f"img_{formatted_string}.png")
    response = requests.get(img_url)
    with open(file_path, "wb") as f:
        f.write(response.content)


if __name__ == "__main__":
    dall_e_2()