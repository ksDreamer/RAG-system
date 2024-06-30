# RAG system
# Using LLM and RAG to build a Q-A system based on specific knowledge. GUI is implemented by Streamlit.
# Author: Kevin Stark, Jiarui Feng, Bowen Wang
# Date: 2024/06/30
# Version: 1.3
# You must have these Python packages: streamlit, PyPDF2, sentence_transformers, faiss, requests, json, langchain
# Put the files to be read under `./documents` directory manually, or use the upload button in GUI.
# Run it by `streamlit run app.py`

import streamlit as st
import os
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 定义 API 密钥
## 请在自己电脑上配置 OpenAI API KEY 环境变量
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
API_URL = os.environ.get('API_URL') # OpenAI或者你选择的服务商与 HTTP Post 相关的 URL，例如 "https://api2.aigcbest.top/v1/chat/completions"


# 定义路径
## 文件路径
DOCUMENTS_DIR = './documents'
## 缓存路径
CACHE_DIR = './cache'
DOCUMENTS_CACHE_DIR = os.path.join(CACHE_DIR, 'cache_of_documents')
DOCUMENTS_CACHE_FILE = os.path.join(DOCUMENTS_CACHE_DIR, 'cache_of_documents.json')
EMBEDDINGS_CACHE_DIR = os.path.join(CACHE_DIR, 'cache_of_embeddings')
EMBEDDINGS_BATCH_PREFIX = 'cache_of_embeddings_batch'
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_CACHE_DIR, 'cache_of_embeddings.npy')


# 用 PyPDF2 加载PDF文件提取文本
@st.cache_resource(hash_funcs={dict: id})
def load_documents(documents_dir=DOCUMENTS_DIR, cache_file_path=DOCUMENTS_CACHE_FILE):
    # 检查缓存文件夹和缓存文件是否存在
    if not os.path.exists(DOCUMENTS_CACHE_DIR):
        os.makedirs(DOCUMENTS_CACHE_DIR)
    if os.path.exists(cache_file_path):
        # 读取和检查缓存文件的修改时间
        cache_mtime = os.path.getctime(cache_file_path)
        for filename in os.listdir(documents_dir):
            if filename.endswith('.pdf'):
                filepath = os.path.join(documents_dir, filename)
                # 如果文件修改时间晚于缓存文件的修改时间，则删除缓存文件
                if os.path.getctime(filepath) > cache_mtime:
                    os.remove(cache_file_path)
                    break

    # 如果缓存文件不存在 重新加载并缓存
    if not os.path.exists(cache_file_path):
        documents = []
        for filename in os.listdir(documents_dir):
            if filename.endswith('.pdf'):
                filepath = os.path.join(documents_dir, filename)
                reader = PdfReader(open(filepath, 'rb'))
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    documents.append({
                        'document_name': filename,
                        'page_number': page_num + 1,
                        'text': text
                    })
        
        # 保存到缓存文件
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=4)
    else: # 如果缓存文件存在 从缓存文件加载
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            documents = json.load(f)
    
    # 列出加载的文件
    with st.sidebar.expander("已加载文件:"):
        items = os.listdir(DOCUMENTS_DIR)
        file_names = [item for item in items if not item.startswith('.')]
        files_string = "\n ".join(file_names)
        st.write(f"\n```\n{files_string}\n```")

    return documents


# 用 langchain 的 TextSplitter 做文本分割
@st.cache_resource
def split_text_chunks(documents):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    for doc in documents:
        texts = text_splitter.split_text(doc['text'])
        for text in texts:
            chunks.append({
                'document_name': doc['document_name'],
                'page_number': doc['page_number'],
                'text': text
            })
    print("已顺利进行文本分割")
    return chunks

# 用 sentence-transformers 做嵌入向量生成
@st.cache_resource
def generate_embeddings(chunks, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', batch_size=32):
    if not chunks:
        st.write("现在没有文件，请您上传文件")
        return np.empty((0, 384))
    # 如果嵌入文件已经存在，加载并返回
    ## 目前版本如果文件有变动，缓存嵌入向量需要手动删除，重新生成，考虑在后继版本实现增量更新。
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        print("已成功读取缓存嵌入文件")
        return embeddings
    if not os.path.exists(EMBEDDINGS_CACHE_DIR):
        os.makedirs(EMBEDDINGS_CACHE_DIR)

    # 初始化 SentenceTransformer 模型
    model = SentenceTransformer(model_name)

    # 分批处理文本块以减少内存占用
    embeddings = []
    total_chunks = len(chunks)
    batch_num = total_chunks // batch_size + (1 if total_chunks % batch_size != 0 else 0)

    for i in range(batch_num):
        try:
            # 获取当前批次的文本块
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_chunks)
            batch_chunks = [chunk['text'] for chunk in chunks[start_idx:end_idx]]

            # 生成嵌入
            batch_embeddings = model.encode(batch_chunks, convert_to_tensor=True)
            batch_embeddings = batch_embeddings.cpu().numpy()

            # 保存当前批次的嵌入到指定文件夹
            batch_file = f'{EMBEDDINGS_BATCH_PREFIX}_{i}.npy'
            np.save(os.path.join(EMBEDDINGS_CACHE_DIR, batch_file), batch_embeddings)

            # 将当前批次的嵌入添加到列表中
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f'Error processing batch {i}: {e}')
            # 如果出错，会清理当前批次的临时文件
            batch_file = f'{EMBEDDINGS_BATCH_PREFIX}_{i}.npy'
            if os.path.exists(os.path.join(EMBEDDINGS_CACHE_DIR, batch_file)):
                os.remove(os.path.join(EMBEDDINGS_CACHE_DIR, batch_file))
            break

    # 合并所有嵌入向量成 Numpy 数组 保存
    embeddings = np.vstack(embeddings)
    np.save(EMBEDDINGS_FILE, embeddings)
    print("已成功生成嵌入向量")
    return embeddings


# 用 faiss 存储嵌入到向量数据库
@st.cache_resource
def create_vector_database(embeddings, chunks):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    print("已成功创建向量数据库")
    return index, chunks


# 根据用户输入检索相似文本块
def retrieve(query, index, chunks, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', top_k=5):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding = query_embedding.cpu().numpy()
    distances, indices = index.search(query_embedding, top_k)
    results = [chunks[idx] for idx in indices[0]]
    return results


# 用 LLM 生成答案
def generate_answer(query, context):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. You must seek answers within the provided context first and output answers based on the context. When you are unable to find the answer within the context, you must use your own knowledge base to answer the question. You are not allowed to refuse to answer. If you are forced to answer without being able to find the answer, you need to indicate at the end of your response, in parentheses, that this answer was not found in the context. For questions with a definitive answer, provide the key answer directly without lengthy explanations. The output should be in Chinese."
            },
            {
                "role": "user",
                "content": f"问题: {query} 上下文: {context}"
            }
        ]
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(data)) # 从官方 API 文档获得更多信息
    response_json = response.json()
    answer = response_json['choices'][0]['message']['content']
    return answer

# 以下为用 streamlit 实现的 GUI
st.title("RAG-system: 知识库问答系统")
st.write("Author: Mengyang Gao, Jiarui Feng, Bowen Wang")

# 文件上传功能
with st.sidebar.expander("上传新文件"):
    uploaded_files = st.file_uploader("在此处上传新的文件", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        st.success("文件上传成功！")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 加载和预处理文档
documents = load_documents(DOCUMENTS_DIR)
chunks = split_text_chunks(documents)
embeddings = generate_embeddings(chunks)
index, chunks = create_vector_database(embeddings, chunks)

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入
if prompt := st.chat_input("请输入您的问题..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 获取答案
    with st.chat_message("assistant"):

        supporting_passages = retrieve(prompt, index, chunks)
        context = ' '.join([passage['text'] for passage in supporting_passages])

        answer = generate_answer(prompt, context)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})