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

from langchain.text_splitter import (
    CharacterTextSplitter, 
    TokenTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
    PythonCodeTextSplitter,
)

from langchain.document_transformers import (
    Html2TextTransformer,
)

from langchain_core.documents import Document
# from langchain.embeddings import OpenAIEmbeddings # deplicated
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter

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


def document_transformer():
    # CharacterTextSplitter
    loader = TextLoader(os.path.join(BASE_DIR, "data", "kokoro.txt"), autodetect_encoding=True)
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        add_start_index=True,
        separator="。",
    )
    splitted_documents = text_splitter.transform_documents(documents)

    # page_contentのみ表示
    pprint([d.page_content for d in splitted_documents])
    print("---")
    print(splitted_documents[0])

    # TokenTextSplitter
    text_splitter = TokenTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        encoding_name="cl100k_base",
        add_start_index=True,
    )
    splitted_documents = text_splitter.transform_documents(documents)
    pprint([d.page_content for d in splitted_documents])
    print("---")
    print(splitted_documents[0])


    # RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=15,
        chunk_overlap=10,
        add_start_index=True,
        separators=["、", "。"],
    )
    splitted_documents = text_splitter.transform_documents(documents)
    pprint([d.page_content for d in splitted_documents])
    print("---")
    print(splitted_documents[0])

    # for HTML
    html_loader = TextLoader(os.path.join(BASE_DIR, "data", "summary.html"), autodetect_encoding=True)
    html_documents = html_loader.load_and_split()
    html_text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=100,
        chunk_overlap=10,
        add_start_index=True,
    )
    html_splitted_documents = html_text_splitter.transform_documents(html_documents)
    pprint([d.page_content for d in html_splitted_documents])
    print("---")
    print(html_splitted_documents[0])

    # for Python
    python_loader = TextLoader(os.path.join(BASE_DIR, "data", "app.py"), autodetect_encoding=True)
    python_documents = python_loader.load_and_split()
    python_text_splitter = PythonCodeTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    python_splitted_documents = python_text_splitter.transform_documents(python_documents)
    pprint([d.page_content for d in python_splitted_documents])
    print("---")
    print(python_splitted_documents[0])

    # HTML Transform
    html_loader = WebBaseLoader("https://ja.wikipedia.org/wiki/")
    html_docs = html_loader.load()
    pprint(html_docs)

    print("---")

    html2text = Html2TextTransformer()
    transformed_html_docs = html2text.transform_documents(html_docs)
    pprint([d.page_content for d in transformed_html_docs])

    # 変換したhtmlドキュメントをさらに分割
    html_text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        add_start_index=True,
        separator='¥n',
    )
    html_splitted_data = html_text_splitter.transform_documents(transformed_html_docs)
    pprint("---")
    pprint([d.page_content for d in html_splitted_data])


def document_filter():
    loader = TextLoader(os.path.join(BASE_DIR, "data", "embedding_list.txt"), autodetect_encoding=True)
    loaded_documents = loader.load()
    print("loaded_documents", loaded_documents)

    # 区切り文字で分割してリストにする
    content_list = loaded_documents[0].page_content.split()
    print("content_list", content_list)
    meta_data = loaded_documents[0].metadata
    print("meta_data", meta_data)

    # 分割した文字列をDocumentに変換
    documents = [Document(page_content=document, metadata=meta_data) for document in content_list]

    # Embeddingモジュールのダウンロード
    embeddings_model = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY)
    embedding_filter = EmbeddingsRedundantFilter(
        embeddings=embeddings_model,
        similarity_threshold=0.95,
    )

    # 類似性の高いドキュメントを統合して冗長性を排除するフィルタリング
    filtered_documents = embedding_filter.transform_documents(documents)

    pprint([d.page_content for d in filtered_documents])




if __name__ == "__main__":
    # document_loader()
    # document_transformer()
    document_filter()