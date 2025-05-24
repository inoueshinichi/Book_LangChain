""" OpenAI Chat Completions Model with Flask
"""

import sys
import os
from pathlib import Path
import shutil
from textwrap import indent
from typing import (
    Optional,
    Callable,
    Union,
    Tuple,
    List,
    Set,
    Sequence,
    Dict,
)
import json
import inspect

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

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))


app = Flask(__name__)

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")

client = OpenAI(api_key=OPENAI_SECRET_KEY)

@app.route('/')
def index():
    return render_template('index.html', question=None, result=None)

@app.route('/', methods=['POST'])
def submit():
    prompt = request.form['prompt']
    result = access_openai(prompt)
    return render_template('index.html', question=prompt, result=result)

def access_openai(prompt_value):
    response = client.completions.create(
            model="gpt-3.5-turbo-instruct", 
            prompt=prompt_value,  #←promptを指定
            stop="。",  #←文字が出現したら文章を終了する
            max_tokens=100,  #←最大のトークン数
            temperature=0.5  #←多様性を表すパラメータ
        )
    
    return response.choices[0].text.strip()


if __name__ == "__main__":
    app.run(host='localhost',
            port=8001,
            debug=True,
            )