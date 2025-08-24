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

# Template
from langchain_core.prompts import ChatPromptTemplate

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
    UnstructuredMarkdownLoader, 
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
from langchain.indexes.vectorstore import VectorstoreIndexCreator

# Agent
from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chains import LLMMathChain, retrieval_qa

from langchain.utilities import SerpAPIWrapper # google search api
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonREPLTool



# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
logger.debug(f"BASE_DIR: {BASE_DIR}")
load_dotenv(os.path.join(BASE_DIR, ".env"))

OPENAI_SECRET_KEY = os.getenv('OPENAI_API_TOKEN')
if OPENAI_SECRET_KEY is None:
    raise Exception('OpenAIのシークレットキーがNoneです')
else:
    logger.debug(f"OpenAI API Key: {OPENAI_SECRET_KEY}")


SERP_SECRET_KEY = os.getenv("SERP_API_TOKEN")
if SERP_SECRET_KEY is None:
    raise Exception("SerpAPIWrapperのシークレットキーがNoneです")
else:
    logger.debug(f"SerpAPIWrapper API Key: {SERP_SECRET_KEY}")

def fetch_weather(location):
    return "Sunny"


def agent():
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)

    # set_debug(True)

    llm = OpenAI(api_key=OPENAI_SECRET_KEY)

    weather_tool = Tool(
        name="天気",
        func=fetch_weather,
        description="場所の情報をもとに天気の情報を取得する",
    )
    tools = [weather_tool]
    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
    )
    # 実行 (思考→行動→観察)
    agent_executor.run("福岡の今日の天気を教えて")


    # 計算
    llm_math = LLMMathChain(llm=llm)
    math_tool = Tool(
        name="計算機",
        func=llm_math,
        description="数学の計算をするのに使います。",
    )
    tools = [math_tool]
    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iteration=3,
    )
    agent_executor.run("4.5 * 2.3 / 3.5 = ?")

    # 一般的な質問
    llm_tool = Tool(
        name="一般的な質問",
        func=llm.predict,
        description="一般的な質問に対して答えを得られます。",
    )
    tools += [llm_tool]
    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iteration=3,
    )
    agent_executor.run("福岡の県庁所在地はどこですか？")


def agent_with_web_serch():
    # Googleサーチエンジンを使う
    search = SerpAPIWrapper(serpapi_api_key=SERP_SECRET_KEY)
    print(search.run("最近の福岡のニュースを教えて"))

    llm = OpenAI(api_key=OPENAI_SECRET_KEY)
    memory = ConversationBufferMemory()
    search_tool = Tool(
        name="検索",
        func=search.run,
        description="最新の情報を得たい場合に用います。",
    )
    llm_math = LLMMathChain(llm=llm)
    math_tool = Tool(
        name="計算機",
        func=llm_math,
        description="数値演算の内容を文字列で与えて正確な数値演算を行います。",
    )

    tools = [search_tool, math_tool]

    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        memory=memory,
        max_iteration=4,
    )

    agent_executor.run("GPT-3.5とGPT-4の最大入力トークン数の差を計算してください。")

    print(memory)


def agent_gen_python_code():
    llm = OpenAI(api_key=OPENAI_SECRET_KEY)

    agent_executor = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    agent_executor.run("15のフィボナッチ数を計算してください")


def agent_find_bug_code():
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(CHROMA_DB_DIR): os.mkdir(CHROMA_DB_DIR)

    embedding_model = OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY)

    # Markdownを読み込む
    DOCS_DIR = os.path.join(BASE_DIR, "source_code", "docs")
    md_loader = DirectoryLoader(DOCS_DIR, glob="*.md", recursive=True, 
                             loader_cls=UnstructuredMarkdownLoader,
                             loader_kwargs={'autodetect_encoding': True})
    md_documents = md_loader.load()
    md_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        add_start_index=True,
        keep_separator=True,
        separators=["#", "¥n"],
    )
    md_splitted_documents = md_text_splitter.transform_documents(md_documents)
    Chroma.from_documents(md_splitted_documents,
                          persist_directory=CHROMA_DB_DIR,
                          collection_name="docs",
                          embedding=embedding_model)
    
    # Pythonを読み込む
    CODE_DIR = os.path.join(BASE_DIR, "source_code")
    py_loader = DirectoryLoader(CODE_DIR, glob="*.py",
                             recursive=True,
                             loader_cls=PythonLoader)
    py_documents = py_loader.load()
    py_text_splitter = PythonCodeTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        add_start_index=True,
    )
    python_documents = py_text_splitter.transform_documents(py_documents)

    # HTMLを読み込む
    html_loader = DirectoryLoader(CODE_DIR, glob="*.html",
                                  recursive=True, loader_cls=TextLoader,
                                  loader_kwargs={"autodetect_encoding": True})
    html_documents = html_loader.load()
    html_text_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.HTML,
        chunk_size=500,
        chunk_overlap=20,
        add_start_index=True,
    )
    html_documents = html_text_splitter.transform_documents(html_documents)
    source_code_documents = python_documents + html_documents

    # collection_name='code'のChromaに保存
    Chroma.from_documents(source_code_documents,
                          persist_directory=CHROMA_DB_DIR,
                          collection_name="code",
                          embedding=embedding_model)
    
    # Chromaから読み出してRetrieverを作成
    docs_db = Chroma(persist_directory=CHROMA_DB_DIR, collection_name="docs",
                     embedding_function=OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY))
    code_db = Chroma(persist_directory=CHROMA_DB_DIR, collection_name="code",
                     embedding_function=OpenAIEmbeddings(api_key=OPENAI_SECRET_KEY))
    docs_retriever = docs_db.as_retriever()
    code_retriever = code_db.as_retriever()

    llm = OpenAI(api_key=OPENAI_SECRET_KEY)
    
    # 検索用のChainを作成
    # docs_chain = retrieval_qa(llm=llm, retriever=docs_retriever)
    # code_chain = retrieval_qa(llm=llm, retriever=code_retriever)

    prompt_template = ChatPromptTemplate.from_template(
        """Answer the question based only on the following context:{context} Question: {input}""")
    
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    docs_chain = create_retrieval_chain(docs_retriever, document_chain)
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    code_chain = create_retrieval_chain(code_retriever, document_chain)

    # docs_chainとcode_chainでAgentを作成
    docs_tool = Tool(
        name="ドキュメント",
        func=docs_chain.invoke, # ここ動かない
        description="設計書など°キュメント情報を得たい場合に用います",
    )
    code_tool = Tool(
        name="コード",
        func=code_chain.invoke, # ここ動かない
        description="実際に記述したソースコードを参考にしたい場合に用います"
    )

    tools = [docs_tool, code_tool]

    memory = ConversationBufferMemory()
    agent_executor = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        verbose=True,
        memory=memory,
        max_iteration=4,
    )

    response = agent_executor.run("レストラン詳細画面はどんな画面ですか？")
    print(response)

    response = agent_executor.run("restraurantsテーブルのモデルのソースコードを全て教えてください")
    print(response)


if __name__ == "__main__":
    # agent()
    # agent_with_web_serch()
    # agent_gen_python_code()
    agent_find_bug_code()


