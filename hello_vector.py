#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pip install chromadb numpy modelscope sentence-transformers torch

import os
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from sentence_transformers import SentenceTransformer

def print_vector_preview(vector, preview_length=8):
    """打印向量的预览，只显示前几个维度和最后一个维度"""
    if len(vector) <= preview_length * 2:
        return f"[{', '.join(f'{x:.4f}' for x in vector)}]"
    
    front = ', '.join(f'{x:.4f}' for x in vector[:preview_length])
    back = ', '.join(f'{x:.4f}' for x in vector[-1:])
    return f"[{front}, ..., {back}] (维度: {len(vector)})"

def main():
    """最简化的向量数据库读写演示"""
    print("初始化向量数据库...")
    
    # 创建数据库目录
    db_path = "./minimal_db"
    os.makedirs(db_path, exist_ok=True)
    
    # 初始化ChromaDB客户端
    client = chromadb.PersistentClient(path=db_path)
    
    # 使用ModelScope下载模型
    model_dir = snapshot_download("BAAI/bge-large-zh-v1.5")
    model = SentenceTransformer(model_dir)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_dir  # 使用下载好的模型路径
    )
    
    # 创建或获取集合
    collection = client.get_or_create_collection(
        name="minimal_collection",
        embedding_function=embedding_function
    )
    
    # 准备示例文档
    documents = [
        "向量数据库是一种专门设计用于存储和检索向量数据的数据库系统。",
        "向量数据库在语义搜索、推荐系统和图像检索等领域有广泛应用。",
        "向量数据库使用近似最近邻算法来查找相似的向量。",
        "HNSW是一种常用的向量索引算法，可以提高搜索效率。",
        "猫爱吃鱼，狗爱吃肉，奥特曼爱吃小怪兽" ,
        "奥特曼是正义的化身，小怪兽是邪恶的象征",
        "奥特曼和小怪兽是好朋友，他们一起打败了怪兽",
        "RAG是一种基于向量数据库的检索技术，可以提高搜索效率。",
        "橘子比苹果好吃",
        "奖学金很多钱",
        ""
    ]
    
    # 为每个文档添加ID和元数据
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadatas = [{"source": "示例数据", "index": i} for i in range(len(documents))]
    
    # 生成并显示每个文档的向量表示
    print("\n=== 文档及其向量表示 ===")
    embeddings = embedding_function(documents)
    for doc, embedding in zip(documents, embeddings):
        print(f"\n文档: {doc}")
        print(f"向量: {print_vector_preview(embedding)}")
        print(f"向量范数: {np.linalg.norm(embedding):.4f}")  # 向量的长度
    
    # 写入操作：将文档添加到集合中
    print("\n向向量数据库添加文档...")
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    # 获取集合中的文档数量
    print(f"集合中的文档数量: {collection.count()}")
    
    # 读取操作：执行查询
    queries = [
        "向量数据库是什么？",

    ]
    
    for query in queries:
        print(f"\n执行查询: '{query}'")
        # 显示查询文本的向量表示
        query_embedding = embedding_function([query])[0]
        print(f"查询向量: {print_vector_preview(query_embedding)}")
        
        # 执行查询
        results = collection.query(
            query_texts=[query],
            n_results=10  # 返回最相关的5个结果
        )
        
        # 打印查询结果
        print("\n查询结果:")
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0])):
            print(f"\n结果 {i+1}:")
            print(f"  文档: {doc}")
            print(f"  元数据: {metadata}")
            print(f"  相似度距离: {distance:.4f}")
            # 获取结果文档的向量表示
            result_embedding = embedding_function([doc])[0]
            print(f"  文档向量: {print_vector_preview(result_embedding)}")
            # 计算与查询向量的余弦相似度
            cosine_sim = np.dot(query_embedding, result_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
            )
            print(f"  余弦相似度: {cosine_sim:.4f}")

if __name__ == "__main__":
    main() 
