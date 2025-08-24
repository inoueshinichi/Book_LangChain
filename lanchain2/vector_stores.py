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

# Chroma DB
from langchain.vectorstores import Chroma

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


def chromadb_store():
    loader = TextLoader(os.path.join(BASE_DIR, "data", "programming.txt"), autodetect_encoding=True)
    documents = loader.load_and_split()
    text_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        encoding_name="cl100k_base",
        add_start_index=True,
    )
    pprint(documents)
    splitted_documents = text_splitter.transform_documents(documents)
    print("---")
    pprint(splitted_documents)

    # ChromaにInMemoryで格納する
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY)
    db = Chroma.from_documents(splitted_documents, embedding=embedding_model)
    print("db", db)

    # 類似度検索
    query = "オブジェクト思考とは何ですか？"
    docs = db.similarity_search(query, k=1) # default: k=4
    print("[search]")
    pprint(docs)

    # Chromaに保存するembeddingを永続化する
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)
    db = Chroma.from_documents(splitted_documents,
                               persist_directory=CHROMA_DB_DIR, # フォルダ指定
                               embedding=embedding_model)
    query = "オブジェクト思考とは何ですか？"
    docs = db.similarity_search(query, k=1) # default: k=4
    print("[search1]")
    pprint(docs)

    # 永続化したembeddingをChromaに格納して類似検索
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY))
    query = "オブジェクト思考とは何ですか？"
    docs = db.similarity_search(query, k=1) # default: k=4
    print("[search2]")
    pprint(docs)


def chromadb_collection():
    loader = TextLoader(os.path.join(BASE_DIR, "data", "programming.txt"), autodetect_encoding=True)
    documents = loader.load_and_split()
    text_splitter = TokenTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        encoding_name="cl100k_base",
        add_start_index=True,
    )

    splitted_documents = text_splitter.transform_documents(documents)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY)

    # CollectionはDocumentをまとめるグループ
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)
    mydb = Chroma.from_documents(splitted_documents,
                               persist_directory=CHROMA_DB_DIR, # フォルダ指定
                               embedding=embedding_model,
                               collection_name="my_collection", # chromadbの中の特定のcollectionを指定
                               )
    print("mydb1", mydb)

    query = "オブジェクト思考とは何ですか？"
    docs = mydb.similarity_search(query, k=1) # default: k=4
    print("[search]")
    pprint(docs)

    mydb.delete_collection()

    print("mydb2", mydb)


    # Collection内へのDocumentの追加
    texts = ["apple", "grape", "banana"]
    metadatas = [
        {'source': "apple.txt"},
        {'source': "grape.txt"},
        {'source': "banana.txt"},
    ]
    documents = [Document(page_content=text, 
                          metadata=metadaeta) 
                          for text, metadaeta in zip(texts, metadatas)]
    
    db = Chroma(embedding_function=OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY))
    db.add_documents(documents) # リスト形式を追加

    print(db.similarity_search("ぶどう", k=1))

    pprint(db.get(where={"source": "banana.txt"}))

    # idsでDocumentの操作(更新や削除)
    response = db.get(where={"source": "banana.txt"})
    id = response['ids'][0]
    new_document = Document(page_content="バナナ", metadata={"source": "banana.txt"}) # Documentの再作成
    print("new_document", new_document)
    db.update_document(id, new_document)
    print(db.similarity_search("バナナ", k=1))
    db.delete(ids=id)


    # 実データをChroma DBに格納して読み込む
    html_loader = WebBaseLoader("https://en.wikipedia.org/wiki/LangChain")
    html_data = html_loader.load()
    print("html_data", html_data)
    html2text = Html2TextTransformer()
    html_transformed_data = html2text.transform_documents(html_data)
    html_text_splitter = CharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=10,
        add_start_index=True,
        separator="¥n",
        keep_separator=True,
    )
    html_splitted_data = html_text_splitter.transform_documents(html_transformed_data)
    print("html_splitted_data", html_splitted_data)

    # wikipediaから読み込んだデータをEmbeddingとともに、ChromaDBに格納して、類似検索
    db = Chroma.from_documents(html_splitted_data, persist_directory=CHROMA_DB_DIR,
                               collection_name="wikipedia",
                               embedding=OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY))
    print("---")
    pprint(db.similarity_search("What is Langchain?"))

    response = db.get(where={"source": "https://en.wikipedia.org/wiki/LangChain"})
    db.delete(ids=response['ids'][0])



if __name__ == "__main__":
    # chromadb_store()
    chromadb_collection()
