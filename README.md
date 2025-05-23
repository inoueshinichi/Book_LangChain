# Book_LangChain
LangChain 完全入門
+ RAG(Retrieval-Augumented Generation) : 言語モデルが知らない情報に対して答えさせる技術
+ ReACT(Reasoning and Acting) : 推論と行動を言語モデル自身に判断させることでネット検索やファイルへの保存を自律的に実行させる技術

## Model I/O
+ モデルを扱いやすくする
1. プロンプトの準備
2. 言語モデルの呼び出し
3. 結果の受け取り

## Retrieval
+ 未知のデータを扱えるようにする
+ RAGの機能

## Memory
+ 過去の対話を短期・長期で記憶する
+ 会話履歴の保存と読み込み

## Chains
+ 複数の処理をまとめる
  
## Agents
+ ReAct
+ OpenAI Function Calling
+ 言語モデルの呼び出しだけでは対応できないタスクをサポートする
+ ファイル保存など

## Callbacks
+ 様々なイベント発生時の処理を行う
+ 他のモジュールと組み合わせる前提
+ ログ出力や外部ライブラリとの連携

