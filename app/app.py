import streamlit as st
from src.utils import *
import os
from dotenv import load_dotenv

from langchain_openai import (
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    ChatOpenAI
)
from langchain_core.messages import (
    HumanMessage, 
    AIMessage,
)

USER_NAME = "user"
ASSISTANT_NAME = "assistant"

def main():
    # streamlit hellow world app
    st.title("Azure Chat Demo")

    with st.sidebar:
        if st.button('Clear chat'):
            st.session_state.chat_log = []
        # チャットログの履歴を保持する数
        message_num = st.slider('会話履歴数',min_value=5, max_value=50, value=10)

    # embeddingsのModelを取得
    embeddings = None
    if os.getenv('AZURE_OPENAI_API_KEY') != "":
        # Azureの場合
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            openai_api_version="2023-05-15",
        )
    else:
        print("APIKeyの設定を確認してください")

    # chatのモデルを取得
    model = None
    if os.getenv('AZURE_OPENAI_API_KEY') != "":
        # Azureの場合
        model = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            openai_api_version="2024-12-01-preview"
        )
    else:
        print("APIKeyの設定を確認してください")
    
    # chainの取得
    contextualize_chain = get_contextualize_prompt_chain(model)
    chain = get_chain(model)

    # FAISSからRetrieverを取得
    retriever = pull_from_faiss(embeddings)

    # チャットログを保存したセッション情報を初期化
    if "Chat_log" not in st.session_state:
        st.session_state.chat_log = []

    # ユーザーのメッセージ入力
    user_msg = st.chat_input("ここにメッセージ入力")

    if user_msg: 

        # 以前のチャットログを表示
        for chat in st.session_state.chat_log:
            if isinstance(chat, AIMessage):
                with st.chat_message(ASSISTANT_NAME):
                    st.write(chat.content)
            else:
                with st.chat_message(USER_NAME):
                    st.write(chat.content)

        # ユーザーのメッセージを表示
        with st.chat_message("USER_NAME"):
            st.write(user_msg)

        # 質問を修正する
        if st.session_state.chat_log:
            new_msg = contextualize_chain.invoke
            ({"chat_history": st.session_state.chat_log, "input": user_msg})
        else:
            new_msg = user_msg
        print(user_msg, "=>", new_msg)

        # 類似ドキュメントを取得
        relavant_docs = retriever.invoke(new_msg, k=3)

        # 質問の解答を表示
        # response = chain.invoke({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": new_msg})
        # response = response.content
        # with st.chat_message(ASSISTANT_NAME):
        #     st.write(response)
        # ストリーム
        response = ""
        with st.chat_message(ASSISTANT_NAME):
            msg_placeholder = st.empty()

            for r in chain.stream({"chat_history": st.session_state.chat_log, "context": relavant_docs, "input": user_msg}):
                response += r.content
                msg_placeholder.markdown(response  + "■")
            msg_placeholder.markdown(response)

        # セッションにチャットログを追加
        st.session_state.chat_log.extend([
            HumanMessage(content=user_msg),
            AIMessage(content=response)
        ])

        # チャットログを保持する数を超えた場合、古いログを削除
        if len(st.session_state.chat_log) > message_num: 
            st.session_state.chat_log = st.session_state.chat_log[-message_num:]
        print(st.session_state.chat_log, len(st.session_state.chat_log))

if __name__ == '__main__':
    main()