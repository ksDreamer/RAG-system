**RAG system**

Using LLM and RAG to build a Q-A system based on specific knowledge. GUI is implemented by Streamlit.

Author: Kevin Stark, Jiarui Feng

Date: 2024/06/26

Version: 1.1

# Getting Started

```
conda create -n rag python=3.10
conda activate rag
pip install streamlit, PyPDF2, sentence_transformers, faiss, requests, langchain
```

Put the files to be read under `./documents` directory

Run it by `streamlit run app.py`

# File Structure

```
├── app.py
├── documents # 存放 PDF 文档的文件夹
│   └── etc.
├── documents_cache.json # 载入文档的缓存，第一次运行时生成
├── embeddings # 向量嵌入的缓存，第一次运行时生成
│   ├── embeddings.npy
│   ├── embeddings_batch_0.npy
│   └── etc.
└── README.md
```

# Developing Log
v1.1: 添加多个注释方便阅读；优化函数preprocess_documents名改为split_text_chunks；在GUI里增加各步骤执行成功的反馈；考虑添加local llm；Streamlit里对'''多行注释'''好像有些独特用法，st.markdown (TD)  
v1.0: 增加文档载入和向量嵌入的缓存机制；添加向量嵌入的批处理操作避免内存占用过大；暂时没有添加“上传文档”按钮，没有做文件更新缓存变动检查。
