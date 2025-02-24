import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv


# OpenAI APIキーの設定（StreamlitのSecretsを使う）
api_key = st.secrets["OPENAI_API_KEY"]
if not api_key:
    st.error("APIキーが設定されていません。StreamlitのSecretsを確認してください。")
    st.stop()

# Streamlitのページ設定
st.title("🐾 対話式小噺作成アシスタント")
st.write("アシスタントと対話しながら、オリジナルの小噺を作っていきましょう！")

# セッション状態の初期化
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()
if "chain" not in st.session_state:
    llm = OpenAI(api_key=api_key, temperature=0.7)
    st.session_state.chain = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        prompt=PromptTemplate(
            input_variables=["history", "input"],
            template="""あなたは経験豊富な落語家のアシスタントです。
ユーザーと対話しながら、面白い小噺を一緒に作っていきましょう。

これまでの会話：
{history}

ユーザー: {input}
アシスタント: """
        )
    )

# チャット履歴の表示
for message in st.session_state.memory.chat_memory.messages:
    with st.chat_message("assistant" if message.type == "ai" else "user"):
        st.write(message.content)

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください"):
    # ユーザーのメッセージを表示
    with st.chat_message("user"):
        st.write(prompt)

    # アシスタントの応答を生成
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            response = st.session_state.chain.predict(input=prompt)
            st.write(response)

# 初回メッセージ
if len(st.session_state.memory.chat_memory.messages) == 0:
    with st.chat_message("assistant"):
        initial_message = """小噺を一緒に作っていきましょう！
まずは、どんな動物を主人公にしたいですか？"""
        st.write(initial_message)
        # 初回メッセージを直接メモリに追加
        st.session_state.memory.chat_memory.add_ai_message(initial_message)

# リセットボタン
if st.button("会話をリセット"):
    st.session_state.memory.clear()
    st.rerun()
