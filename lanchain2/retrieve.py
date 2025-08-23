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

# Retrieve
from langchain.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    DirectoryLoader, 
    PythonLoader,
    WebBaseLoader,
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


def document_loader():
    # テキスト
    loader = TextLoader(os.path.join(BASE_DIR, "data", "index.txt"))
    documents = loader.load()
    print(documents)

    print("---")

    # PDF
    loader = PyPDFLoader(os.path.join(BASE_DIR, "data", "itpassport_syllabus.pdf"))
    pages = loader.load_and_split()
    print(pages)
    print("len(pages)", len(pages))
    print(pages[0])

    # Files
    loader = DirectoryLoader(os.path.join(BASE_DIR, "data"),
                             glob="*.txt",
                             recursive=True,
                             show_progress=True,
                             use_multithreading=False,
                             loader_cls=TextLoader,
                             loader_kwargs={"autodetect_encoding": True})
    docs = loader.load()
    print(docs)

    # Python files
    loader = DirectoryLoader(BASE_DIR, glob="*.py", recursive=True, loader_cls=PythonLoader)
    docs = loader.load()
    print(docs)

    # Web page
    loader = WebBaseLoader(["https://ja.wikipedia.org/wiki/", "https://google.com"])
    docs = loader.load()
    print(docs)


if __name__ == "__main__":
    document_loader()