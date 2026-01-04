"""Retrieval engine for RAG system"""
import logging
from typing import List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from core.embedding_engine import EmbeddingEngine
from core.qwen_llm import QwenChat
from config.settings import Settings


class RetrievalEngine:
    """Handles the retrieval part of RAG - finding relevant documents for a query"""

    def __init__(self, embedding_engine: EmbeddingEngine):
        """
        Initialize retrieval engine

        Args:
            embedding_engine: Initialized EmbeddingEngine instance
        """
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        self.embedding_engine = embedding_engine

        # Initialize LLM based on configuration
        if self.settings.OPENAI_API_KEY:
            # Check if this is a Qwen model and handle accordingly
            if "Qwen" in self.settings.MODEL_NAME:
                # For Qwen models, use the custom QwenChat class that handles enable_thinking parameter
                self.llm = QwenChat(
                    model_name=self.settings.MODEL_NAME,
                    api_key=self.settings.OPENAI_API_KEY,
                    base_url=self.settings.BASE_URL,
                    temperature=0.1
                )
            else:
                # For non-Qwen models, use standard configuration
                self.llm = ChatOpenAI(
                    api_key=self.settings.OPENAI_API_KEY,
                    model=self.settings.MODEL_NAME,
                    base_url=self.settings.BASE_URL,
                    temperature=0.1
                )
        else:
            # Use local model as fallback
            self.llm = ChatOllama(model="llama2", temperature=0.1)

    def create_qa_chain(self, k: int = 4, search_type: str = "auto"):
        """
        Create a question-answering chain

        Args:
            k: Number of documents to retrieve
            search_type: Type of search ('similarity', 'mmr', 'hybrid', 'auto')

        Returns:
            Runnable for question answering
        """
        # Define a custom prompt template for enterprise knowledge base
        template = """
        你是一个企业知识库助手。请根据以下上下文信息回答问题。
        如果上下文信息不足，请告知用户无法根据现有信息回答。
        请确保回答准确、简洁且专业。

        上下文: {context}

        问题: {input}

        回答:"""

        prompt = ChatPromptTemplate.from_template(template)

        # Create a chain that formats documents
        def format_docs(docs):
            # Add debugging and protection against non-string page_content
            formatted_parts = []
            for doc in docs:
                # Check if doc is a Document object or a string
                if isinstance(doc, str):
                    # If doc is a string, use it directly
                    content = doc
                elif hasattr(doc, 'page_content'):
                    # If doc is a Document object, get its page_content
                    content = doc.page_content
                elif isinstance(doc, dict) and 'page_content' in doc:
                    # If doc is a dictionary with page_content, use that
                    content = doc['page_content']
                else:
                    # If doc is neither a string nor a Document object, log and skip
                    self.logger.warning(f"Unexpected document type: {type(doc)}, value: {doc}")
                    continue

                # Ensure content is a string
                if isinstance(content, dict):
                    # Log the issue for debugging
                    self.logger.warning(f"Document page_content is dict instead of string: {content}")
                    # Convert dict to string representation
                    content = str(content)
                elif not isinstance(content, str):
                    # Handle other unexpected types
                    self.logger.warning(f"Document page_content is unexpected type {type(content)}: {content}")
                    content = str(content)
                # Additional protection: ensure content is valid for embedding models
                if content is None:
                    content = ""
                formatted_parts.append(content)
            return "\n\n".join(formatted_parts)

        # Create the RAG chain
        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | self.llm
            | StrOutputParser()
        )

        from langchain_core.runnables import RunnableLambda
        # Create a custom chain that retrieves documents directly to avoid potential issues with retriever
        def retrieve_and_answer(inputs):
            query = inputs["input"]

            # Check if vector store is ready before attempting to retrieve
            if not self.embedding_engine.is_vector_store_ready():
                # Return empty context when vector store is not ready
                empty_context = []
                # Create the final input for the LLM with empty context
                llm_input = {
                    "context": format_docs(empty_context),
                    "input": query
                }
                # Get the answer from the LLM
                answer = rag_chain_from_docs.invoke(llm_input)
                return {
                    "context": empty_context,  # Return empty context
                    "answer": "抱歉，知识库尚未准备好，请稍后再试。"
                }

            # Retrieve documents directly using the enhanced search
            docs = self.embedding_engine.enhanced_similarity_search(query, k=k, search_type=search_type)

            # 验证并过滤 docs，确保它们是 Document 对象
            validated_docs = []
            for doc in docs:
                if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                    # 这是一个 Document 对象
                    if not isinstance(doc.page_content, str):
                        self.logger.warning(f"Retrieved document has non-string page_content: {type(doc.page_content)}")
                        doc.page_content = str(doc.page_content) if doc.page_content is not None else ""
                    validated_docs.append(doc)
                else:
                    # 如果不是 Document 对象，跳过
                    self.logger.warning(f"Non-document object in retrieved docs: {type(doc)}")
                    continue

            # Format the validated docs
            formatted_context = format_docs(validated_docs)
            # Create the final input for the LLM
            llm_input = {
                "context": formatted_context,
                "input": query
            }
            # Get the answer from the LLM
            answer = rag_chain_from_docs.invoke(llm_input)
            return {
                "context": validated_docs,  # Return the validated documents
                "answer": answer
            }

        return RunnableLambda(retrieve_and_answer)

    def retrieve_documents(self, query: str, k: int = 4, search_type: str = "auto") -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User query
            k: Number of documents to retrieve
            search_type: Type of search ('similarity', 'mmr', 'hybrid', 'auto')

        Returns:
            List of relevant documents
        """
        # Check if vector store is ready
        if not self.embedding_engine.is_vector_store_ready():
            self.logger.warning("Vector store is not ready. No documents available for retrieval.")
            return []

        docs = self.embedding_engine.enhanced_similarity_search(query, k=k, search_type=search_type)

        # 验证返回的文档类型
        for i, doc in enumerate(docs):
            if not hasattr(doc, 'page_content'):
                self.logger.error(f"Document {i} does not have page_content attribute: {type(doc)}")
            if isinstance(doc, str):
                self.logger.error(f"Document {i} is string: {doc}")

        return docs

    def answer_query(self, query: str, k: int = 4, search_type: str = "auto") -> Dict[str, Any]:
        """
        Answer a user query using RAG

        Args:
            query: User query
            k: Number of documents to retrieve
            search_type: Type of search ('similarity', 'mmr', 'hybrid', 'auto')

        Returns:
            Dictionary containing answer and source documents
        """
        # Check if vector store is ready and has documents
        if not self.embedding_engine.is_vector_store_ready():
            self.logger.warning("Vector store is not ready. Returning default response.")
            return {
                "query": query,
                "answer": "抱歉，知识库尚未准备好，请先上传文档。",
                "source_documents": []
            }

        # Check if there are actually documents in the vector store
        try:
            # Try to get a count of documents or perform a minimal search
            test_search = self.embedding_engine.similarity_search("test", k=1)
            if not test_search:
                self.logger.warning("Vector store is ready but appears to be empty.")
                return {
                    "query": query,
                    "answer": "抱歉，知识库中没有文档内容，请先上传文档。",
                    "source_documents": []
                }
        except:
            # If test search fails, vector store might not be properly initialized
            self.logger.warning("Vector store test search failed.")
            return {
                "query": query,
                "answer": "抱歉，知识库尚未准备好，请稍后再试。",
                "source_documents": []
            }

        # Create QA chain with specified search type
        qa_chain = self.create_qa_chain(k=k, search_type=search_type)

        # Get response
        response = qa_chain.invoke({"input": query})

        # Ensure source documents have proper format
        source_docs = response.get("context", [])

        # 验证并过滤 source_docs，确保它们是 Document 对象
        validated_source_docs = []
        for doc in source_docs:
            if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # 这是一个 Document 对象
                if not isinstance(doc.page_content, str):
                    self.logger.warning(f"Source document has non-string page_content: {type(doc.page_content)}")
                    doc.page_content = str(doc.page_content) if doc.page_content is not None else ""
                validated_source_docs.append(doc)
            else:
                # 如果不是 Document 对象，跳过或创建一个模拟的 Document
                self.logger.warning(f"Non-document object in source_docs: {type(doc)}")
                continue

        result = {
            "query": query,
            "answer": response.get("answer", "抱歉，无法生成答案"),
            "source_documents": validated_source_docs
        }

        self.logger.info(f"Processed query: {query[:50]}...")
        return result
