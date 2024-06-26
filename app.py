# RAG system
# Using LLM and RAG to build a Q-A system based on specific knowledge. GUI is implemented by Streamlit.
# Author: Kevin Stark, Jiarui Feng
# Date: 2024/06/26
# Version: 1.1
# You must have these Python packages: streamlit, PyPDF2, sentence_transformers, faiss, requests, json, langchain
# Put the files to be read under `./documents` directory 
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
API_URL = os.environ.get('API_URL') # OpenAI或者你选择的服务商与 HTTP Post 相关的 URL

# 定义路径
## 文件路径
DOCUMENTS_DIR = './documents'
## 加载文件缓存路径
CACHE_FILE_PATH = './documents_cache.json'
## 嵌入向量缓存路径
EMBEDDINGS_DIR = 'embeddings'
EMBEDDINGS_FILE_PREFIX = 'embeddings_batch'
EMBEDDINGS_FINAL_FILE = 'embeddings.npy'

# 用 PyPDF2 加载PDF文件提取文本
@st.cache_resource(hash_funcs={dict: id})
def load_documents(documents_dir, cache_file_path=CACHE_FILE_PATH):
    # 检查缓存文件是否存在
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
    items = os.listdir(DOCUMENTS_DIR)
    file_names = [item for item in items if not item.startswith('.')]
    files_string = "\n ".join(file_names)
    st.write(f"已成功加载这些文件：\n```\n{files_string}\n```")
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
    st.write("已顺利进行文本分割")
    return chunks

# 用 sentence-transformers 做嵌入向量生成
@st.cache_resource
def generate_embeddings(chunks, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', batch_size=32):
    # 如果嵌入文件已经存在，加载并返回
    if os.path.exists(os.path.join(EMBEDDINGS_DIR, EMBEDDINGS_FINAL_FILE)):
        embeddings = np.load(os.path.join(EMBEDDINGS_DIR, EMBEDDINGS_FINAL_FILE))
        return embeddings
    # 如果嵌入文件夹不存在，创建它
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

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
            batch_file = f'{EMBEDDINGS_FILE_PREFIX}_{i}.npy'
            np.save(os.path.join(EMBEDDINGS_DIR, batch_file), batch_embeddings)

            # 将当前批次的嵌入添加到列表中
            embeddings.append(batch_embeddings)
        except Exception as e:
            print(f'Error processing batch {i}: {e}')

            # 清理当前批次的临时文件
            batch_file = f'{EMBEDDINGS_FILE_PREFIX}_{i}.npy'
            os.remove(os.path.join(EMBEDDINGS_DIR, batch_file))
            break

    # 合并所有批次的嵌入
    embeddings = np.vstack(embeddings)

    # 保存最终的嵌入到磁盘
    final_file_path = os.path.join(EMBEDDINGS_DIR, EMBEDDINGS_FINAL_FILE)
    np.save(final_file_path, embeddings)
    st.write("已成功生成嵌入向量")
    return embeddings


# 用 faiss 存储嵌入到向量数据库
@st.cache_resource
def create_vector_database(embeddings, chunks):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    st.write("已成功创建向量数据库")
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
st.title("LLM RAG 知识库问答系统")

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