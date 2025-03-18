#!/usr/bin/env python
# -*- coding: utf-8 -*-
# pip install chromadb numpy modelscope sentence-transformers torch flagembedding openai python-dotenv python-docx

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
import docx  # Import for processing DOCX files

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
        self.db_path = "./docx_vector_db"
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
    
    def process_docx(self, docx_path: str, collection_name: str = "docx_collection", chunk_size: int = 512, context_size: int = 128):
        """处理DOCX文件并存储到向量数据库中
        
        Args:
            docx_path: DOCX文件路径
            collection_name: 集合名称
            chunk_size: 每个切片的字符数
            context_size: 上下文窗口大小（前后各多少字符）
            
        Returns:
            collection: 向量数据库集合对象
        """
        logging.info(f"处理DOCX文件: {docx_path}")
        
        # 尝试获取现有集合，如果存在则删除重新创建
        try:
            self.client.delete_collection(collection_name)
            logging.info(f"已删除现有集合: {collection_name}")
        except:
            logging.info(f"创建新集合: {collection_name}")
        
        # 创建新集合
        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
        # 读取DOCX文件内容
        try:
            doc = docx.Document(docx_path)
            full_text = ""
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    full_text += text + "\n"
            
            logging.info(f"成功读取DOCX文件，共 {len(full_text)} 字符")
        except Exception as e:
            logging.error(f"读取DOCX文件时出错: {str(e)}")
            return None
        
        # 分割成固定大小的切片，包含重叠
        chunks = []
        chunk_positions = []
        
        # 从头开始，每次向前移动 chunk_size 个字符
        start_pos = 0
        while start_pos < len(full_text):
            # 计算当前块的开始和结束位置
            chunk_end = min(start_pos + chunk_size, len(full_text))
            
            # 计算上下文窗口
            context_start = max(0, start_pos - context_size)
            context_end = min(len(full_text), chunk_end + context_size)
            
            # 提取上下文
            prefix = full_text[context_start:start_pos] if start_pos > context_start else ""
            current_chunk = full_text[start_pos:chunk_end]
            suffix = full_text[chunk_end:context_end] if chunk_end < context_end else ""
            
            # 构建带有上下文的文本块
            if prefix:
                context_text = f"前文: {prefix}\n\n"
            else:
                context_text = ""
                
            context_text += f"正文: {current_chunk}\n\n"
            
            if suffix:
                context_text += f"后文: {suffix}"
            
            chunks.append(context_text)
            chunk_positions.append((start_pos, chunk_end))
            
            # 移动到下一个块的起始位置
            start_pos += chunk_size
        
        logging.info(f"将文档分割为 {len(chunks)} 个重叠块")
        
        # 为每个块生成embedding并存储到向量数据库
        for i, (chunk, (start, end)) in enumerate(zip(chunks, chunk_positions)):
            try:
                # 构建唯一ID
                doc_id = f"doc_{os.path.basename(docx_path)}_{i}"
                
                # 添加到向量数据库
                collection.add(
                    documents=[chunk],
                    ids=[doc_id],
                    metadatas=[{
                        "source": os.path.basename(docx_path),
                        "chunk_index": i,
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_size": chunk_size,
                        "context_size": context_size
                    }]
                )
            except Exception as e:
                logging.error(f"处理第 {i} 个块时出错: {str(e)}")
        
        logging.info(f"成功将 {len(chunks)} 个文本块添加到向量数据库")
        return collection
    
    def get_collection(self, collection_name: str):
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
    
    def retrieve(self, query: str, collection_name: str, n_results: int = 3) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
    parser = argparse.ArgumentParser(description='基于DOCX文件的RAG问答系统')
    parser.add_argument('--query', type=str, default="我被留校察看了，我怎么样才可以申请学位？", 
                       help='用户查询 (默认: "哪种竞赛的学分最多？")')
    parser.add_argument('--docx', type=str, default="1.docx",
                       help='DOCX文件路径 (默认: "1.docx")')
    parser.add_argument('--collection', type=str, default="docx_collection",
                       help='集合名称 (默认: "docx_collection")')
    parser.add_argument('--process_docx', action='store_true', default=True,
                       help='是否处理DOCX文件 (默认: True)')
    parser.add_argument('--n_results', type=int, default=3,
                       help='检索结果数量 (默认: 3)')
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
    
    # 处理DOCX文件
    if args.process_docx:
        if os.path.exists(args.docx):
            collection = rag_system.process_docx(
                docx_path=args.docx,
                collection_name=args.collection,
                chunk_size=512,
                context_size=128
            )
            if collection is None:
                logging.error("处理DOCX文件失败")
                return
        else:
            logging.error(f"文件不存在: {args.docx}")
            return
    
    # 执行查询
    query = args.query
    original_results, reranked_results = rag_system.retrieve(
        query=query, 
        collection_name=args.collection,
        n_results=args.n_results
    )
    
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
