import os
import streamlit as st
from src.utils import *
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

def main():
    web_url = st.text_input('URL',
                            '',
                            key="web_url")
    load_button = st.button("Load data to FAISS", kei="load_button")

    # embeddingsのModelを習得
        # emmbeddingsのモデルを取得
    embeddings = None
    if os.getenv('AZURE_OPENAI_API_KEY') != "":
        # Azureの場合
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            openai_api_version="2023-05-15"
        )
    else:
        st.error("APIKeyの設定を確認してください")

    # FAISSの初期化
    faiss_db = None
    st.write("0. FAISSの初期化")

    # load_buttonクリックでウェブデータをFAISSに格納
    if load_button:    
        # 1. ウェブデータ取得
        with st.spineer(f'ウェブデータ取得 処理中...'):
            site_data = get_website_data(web_url)
            st.write("1.ウェブデータ取得")

        # 2. データをチャンクに小分けにする
        docs_chunks = split_data(site_data)
        st.write("2. データをチャンクに小分けにする")
        # 3. FAISSにベクトル化して格納
        with st.spinner(f'ベクトルデータ 処理中...'):
            faiss_db = add_to_faiss(
                faiss_db=faiss_db,               
                docs=docs_chunks,
                embeddings=embeddings
            )
                
            if faiss_db is not None:
                faiss_db.save_local("vector_store")
                st.write("3. ベクトルデータの保存")
                st.success("完了！")

if __name__ == '__main__':
    main()