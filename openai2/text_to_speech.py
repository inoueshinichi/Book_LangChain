"""Text to Speech"""
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


def tts_1():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    voices = [
        "alloy",
        "ash",
        # "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        # "verse",
    ]

    prompt = "私はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。"
    model="tts-1"
    for voice in voices:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=prompt,
        )

        download_path: str = os.path.join(Path(__file__).resolve().parent, "download_audios", "text_to_speech")
        now = datetime.datetime.now()
        formatted_string = now.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(download_path, f"{model}_{voice}_{formatted_string}.mp3")

        response.write_to_file(file_path)


def tts_1_hd():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    voices = [
        "alloy",
        "ash",
        # "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        # "verse",
    ]

    prompt = "私はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。"
    model="tts-1-hd"
    for voice in voices:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=prompt,
        )

        download_path: str = os.path.join(Path(__file__).resolve().parent, "download_audios", "text_to_speech")
        now = datetime.datetime.now()
        formatted_string = now.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(download_path, f"{model}_{voice}_{formatted_string}.mp3")

        response.write_to_file(file_path)

def gpt_4o_mini_tts():
    client = OpenAI(api_key=OPENAI_SECRET_KEY)
    voices = [
        "alloy",
        "ash",
        # "ballad",
        "coral",
        "echo",
        "fable",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        # "verse",
    ]

    prompt = "私はその人を常に先生と呼んでいた。だからここでもただ先生と書くだけで本名は打ち明けない。"
    model="gpt-4o-mini-tts"
    for voice in voices:
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=prompt,
        )

        download_path: str = os.path.join(Path(__file__).resolve().parent, "download_audios", "text_to_speech")
        now = datetime.datetime.now()
        formatted_string = now.strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(download_path, f"{model}_{voice}_{formatted_string}.mp3")

        response.write_to_file(file_path)


if __name__ == "__main__":
    # tts_1()
    # tts_1_hd()
    gpt_4o_mini_tts()