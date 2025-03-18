#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pip install chromadb numpy modelscope sentence-transformers torch flagembedding openai python-dotenv

import os
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from modelscope.hub.snapshot_download import snapshot_download
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker
from openai import OpenAI
from dotenv import load_dotenv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def print_vector_preview(vector, preview_length=8):
    """打印向量的预览，只显示前几个维度和最后一个维度"""
    if len(vector) <= preview_length * 2:
        return f"[{', '.join(f'{x:.4f}' for x in vector)}]"
    
    front = ', '.join(f'{x:.4f}' for x in vector[:preview_length])
    back = ', '.join(f'{x:.4f}' for x in vector[-1:])
    return f"[{front}, ..., {back}] (维度: {len(vector)})"

class RAGQASystem:
    def __init__(self, api_key: str):
        """初始化RAG问答系统
        
        Args:
            api_key: Qwen-Long API密钥
        """
        # 加载环境变量
        self.api_key = api_key
        
        # 初始化LLM客户端
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        # 创建数据库目录
        self.db_path = "./minimal_db"
        os.makedirs(self.db_path, exist_ok=True)
        
        # 初始化向量数据库
        logging.info("初始化向量数据库...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 下载和初始化向量嵌入模型
        logging.info("加载嵌入模型...")
        self.model_dir = snapshot_download("BAAI/bge-large-zh-v1.5")
        self.model = SentenceTransformer(self.model_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_dir
        )
        
        # 下载和初始化重排序模型
        logging.info("加载重排序模型...")
        self.reranker_dir = snapshot_download("BAAI/bge-reranker-v2-m3")
        self.reranker = FlagReranker(model_name_or_path=self.reranker_dir, use_fp16=True)
    
    def initialize_collection(self, documents: List[str], collection_name: str = "minimal_collection"):
        """初始化或更新向量数据库集合
        
        Args:
            documents: 要存储的文档列表
            collection_name: 集合名称
        """
        # 检查集合是否存在，如果存在则删除
        try:
            self.client.delete_collection(collection_name)
            logging.info(f"已删除现有集合: {collection_name}")
        except:
            logging.info(f"创建新集合: {collection_name}")
        
        # 创建集合
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # 为每个文档添加ID和元数据
        ids = [f"doc_{i}" for i in range(len(documents))]
        metadatas = [{"source": "示例数据", "index": i} for i in range(len(documents))]
        
        # 将文档添加到集合中
        logging.info("向向量数据库添加文档...")
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )
        
        logging.info(f"集合中的文档数量: {collection.count()}")
        return collection
    
    def get_collection(self, collection_name: str = "minimal_collection"):
        """获取已存在的集合
        
        Args:
            collection_name: 集合名称
            
        Returns:
            集合对象
        """
        return self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def retrieve(self, query: str, collection_name: str = "minimal_collection", n_results: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """检索相关文档
        
        Args:
            query: 查询文本
            collection_name: 集合名称
            n_results: 返回结果数量
            
        Returns:
            tuple: (原始检索结果, 重排序后的结果)
        """
        logging.info(f"执行查询: '{query}'")
        
        try:
            collection = self.get_collection(collection_name)
        except:
            logging.error(f"集合 {collection_name} 不存在")
            return [], []
        
        # 执行查询
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # 提取检索结果
        original_results = []
        for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0])):
            original_results.append({
                "document": doc,
                "metadata": metadata,
                "distance": distance
            })
        
        # 使用重排序器进行重排序
        logging.info("使用重排序器进行结果重排序...")
        pairs = [[query, doc] for doc in results['documents'][0]]
        rerank_scores = self.reranker.compute_score(pairs)
        
        # 将原始结果与重排序分数结合
        reranked_data = list(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
            rerank_scores
        ))
        
        # 按重排序分数排序（分数越高越相关）
        reranked_data.sort(key=lambda x: x[3], reverse=True)
        
        # 格式化重排序结果
        reranked_results = []
        for doc, metadata, distance, rerank_score in reranked_data:
            reranked_results.append({
                "document": doc,
                "metadata": metadata,
                "original_distance": distance,
                "rerank_score": rerank_score
            })
        
        return original_results, reranked_results
    
    def answer_question(self, query: str, retrieved_contexts: List[Dict[str, Any]], 
                        temperature: float = 0.3) -> str:
        """使用Qwen-Long模型回答问题
        
        Args:
            query: 用户问题
            retrieved_contexts: 检索到的相关文档
            temperature: 温度参数，控制生成的随机性
            
        Returns:
            str: 生成的回答
        """
        # 构建提示词
        contexts = "\n\n".join([f"文档{i+1}：{ctx['document']}" 
                              for i, ctx in enumerate(retrieved_contexts)])
        
        prompt = f"""您是一个专业的AI助手。请基于以下提供的参考信息回答用户的问题。
如果提供的参考信息不足以回答问题，请诚实地告知您无法回答，不要编造信息。

参考信息：
{contexts}

用户问题：{query}

请提供详细且准确的回答："""

        print("=================")
        print(prompt)
        print("=================")
        # 调用LLM
        try:
            response = self.llm_client.chat.completions.create(
                model="qwen-long",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"调用LLM时出错: {str(e)}")
            return f"抱歉，生成回答时出现错误: {str(e)}"

def parse_args():
    parser = argparse.ArgumentParser(description='RAG问答系统')
    parser.add_argument('--query', type=str, default="奥特曼有什么特点？", 
                       help='用户查询 (默认: "向量数据库是什么？")')
    return parser.parse_args()

def main():
    args = parse_args()
    load_dotenv()
    
    # 从环境变量获取API密钥
    api_key = os.getenv("API_KEY")
    if not api_key:
        logging.error("未设置 API_KEY 环境变量")
        return
    
    # 初始化系统
    rag_system = RAGQASystem(api_key)
    
    # 示例文档
    documents = [
        "向量数据库是一种专门设计用于存储和检索向量数据的数据库系统。",
        "向量数据库在语义搜索、推荐系统和图像检索等领域有广泛应用。",
        "向量数据库使用近似最近邻算法来查找相似的向量。",
        "HNSW是一种常用的向量索引算法，可以提高搜索效率。",
        "猫爱吃鱼，狗爱吃肉，奥特曼爱吃小怪兽",
        "奥特曼是正义的化身，小怪兽是邪恶的象征",
        "奥特曼和小怪兽是好朋友，他们一起打败了怪兽",
        "RAG是一种基于向量数据库的检索技术，可以提高搜索效率。",
        "橘子比苹果好吃",
        "奖学金很多钱",
        "向量数据库的工作原理是将文本或其他数据转换为高维向量，然后通过计算向量之间的距离或相似度来找到最相似的内容。"
    ]
    
    # 初始化集合
    try:
        collection = rag_system.initialize_collection(documents)
    except Exception as e:
        logging.error(f"初始化集合失败: {str(e)}")
        return
    
    # 执行查询
    query = args.query
    original_results, reranked_results = rag_system.retrieve(query)
    
    # 打印检索结果
    logging.info("\n原始检索结果:")
    for i, result in enumerate(original_results):
        logging.info(f"\n结果 {i+1}:")
        logging.info(f"  文档: {result['document']}")
        logging.info(f"  元数据: {result['metadata']}")
        logging.info(f"  相似度距离: {result['distance']:.4f}")
    
    logging.info("\n重排序后的结果:")
    for i, result in enumerate(reranked_results):
        logging.info(f"\n重排序结果 {i+1}:")
        logging.info(f"  文档: {result['document']}")
        logging.info(f"  元数据: {result['metadata']}")
        logging.info(f"  原向量距离: {result['original_distance']:.4f}")
        logging.info(f"  重排序分数: {result['rerank_score']:.4f}")
    
    # 使用重排序结果回答问题
    answer = rag_system.answer_question(query, reranked_results)
    
    # 打印回答
    print("\n" + "="*50)
    print(f"问题: {query}")
    print("="*50)
    print(f"回答:\n{answer}")
    print("="*50)

if __name__ == "__main__":
    main() 
