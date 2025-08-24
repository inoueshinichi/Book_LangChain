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
from langchain_openai import OpenAI

# from langchain.cache import ( # 非推奨
#     InMemoryCache,
#     SQLiteCache,
#     RedisCache,
#     SQLAlchemyCache,
#     GPTCache,
#     MomentoCache,
# )
from langchain_community.cache import (
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

# from langchain_community.chat_models import ChatOpenAI # 非推奨
from langchain_openai import ChatOpenAI
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
# from langchain.document_loaders import ( # 非推奨
#     TextLoader, 
#     PyPDFLoader, 
#     DirectoryLoader, 
#     PythonLoader,
#     WebBaseLoader,
# )
from langchain_community.document_loaders import (
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

# from langchain.document_transformers import ( # 非推奨
#     Html2TextTransformer,
# )
from langchain_community.document_transformers import (
    Html2TextTransformer,
)

from langchain_core.documents import Document
# from langchain.embeddings import OpenAIEmbeddings # deplicated
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter

# Chroma DB
# from langchain.vectorstores import Chroma # 非推奨
from langchain_chroma import Chroma

# Retriever
# from langchain_community.retrievers import WikipediaRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Template
from langchain_core.prompts import ChatPromptTemplate

from langchain.indexes.vectorstore import VectorstoreIndexCreator

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


def retriever():
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)

    embedding_model = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY)
    db = Chroma(persist_directory=CHROMA_DB_DIR,
                collection_name="wikipedia",
                embedding_function=embedding_model)
    
    # Retrieverの作成
    retriever = db.as_retriever()

    # 作成したRetrieverをChainに設定
    # llm_model = OpenAI(api_key=OPENAI_SECRET_KEY)
    # qa_chain = retrieval_qa.from_llm(llm=llm_model, retriver=retriever)

    # QAチェーンを作成
    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:{context} Question: {input}""")
    document_chain = create_stuff_documents_chain(ChatOpenAI(api_key=OPENAI_SECRET_KEY), prompt_template)

    # リトリーバーとQAチェーンの組み合わせ
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # 実行
    response = retrieval_chain.invoke({"input": "What is Langchain?"})
    print("[answer] with retriver about chromadb of wikipedia.")
    print(response["answer"])


    # retriver(for chroma)を通じてchromaに直接documentを格納
    texts = ["apple", "google", "meta"]
    metadatas = [
        {"source": "apple.txt"},
        {"source": "google.txt"},
        {"source": "meta.txt"},
    ]
    documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]
    print("---")
    print("documents", documents)
    retriever.vectorstore.add_documents(documents)


def vector_index_creator():
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)

    index_creator = VectorstoreIndexCreator(
        embedding = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY),
        text_splitter=CharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=10,
            add_start_index=True,
            separator="¥n",
            keep_separator=True,
        ),
        vectorstore_kwargs={
            "persist_directory": CHROMA_DB_DIR,
            "collection_name": "wikipedia",
        },
    )

    texts = ["iOS", "android"]
    metadatas = [
        {"source": "iOS.txt"},
        {"source": "android.txt"},
    ]
    documents = [Document(page_content=text, metadata=metadata) for text, metadata in zip(texts, metadatas)]

    index_creator.from_documents(documents)


    ##
    loader = WebBaseLoader("https://jp.wikipedia.org/wiki/LangChain")
    index_creator.from_loaders([loader])

if __name__ == "__main__":
    # retriever()
    vector_index_creator()