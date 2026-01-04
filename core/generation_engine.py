"""Generation engine for RAG system"""
import logging
from typing import List, Dict, Any

from langchain_classic.chains.llm import LLMChain
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from core.qwen_llm import QwenChat
from config.settings import Settings


class GenerationEngine:
    """Handles the generation part of RAG - creating responses based on retrieved context"""
    
    def __init__(self):
        """Initialize generation engine with LLM"""
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()
        
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
    
    def generate_response(self, query: str, context_docs: List[Document]) -> str:
        """
        Generate a response based on query and retrieved context documents

        Args:
            query: User query
            context_docs: List of retrieved context documents

        Returns:
            Generated response string
        """
        # Combine context documents into a single context string
        # Handle both Document objects and strings
        context_parts = []
        for doc in context_docs:
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
                self.logger.warning(f"Unexpected document type in context_docs: {type(doc)}, value: {doc}")
                continue

            # Ensure content is a string
            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)

            context_parts.append(content)

        context_str = "\n\n".join(context_parts)

        # Define prompt template
        template = """
        你是一个企业知识库助手。请根据以下上下文信息回答问题。
        如果上下文信息不足，请告知用户无法根据现有信息回答。
        请确保回答准确、简洁且专业。

        上下文: {context}

        问题: {input}

        回答:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "input"]
        )

        # Create a chain using the LLM and prompt
        chain = prompt | self.llm

        # Generate response
        response = chain.invoke({
            "context": context_str,
            "input": query
        })

        # Extract the content from the response, if it's a message object
        if hasattr(response, 'content'):
            return response.content.strip()
        else:
            return str(response).strip()
    
    def generate_condensed_question(self, chat_history: List[str], question: str) -> str:
        """
        Generate a standalone question from conversation history and current question
        
        Args:
            chat_history: List of previous questions/responses
            question: Current question that may reference previous context
            
        Returns:
            Standalone question
        """
        if not chat_history:
            return question
        
        # Combine chat history into a single string
        history_str = "\n".join(chat_history)
        
        template = """
        给定以下对话历史和后续问题，请将其转换为一个独立的问题。
        独立问题应该包含所有必要的上下文信息，使其不依赖于对话历史。
        
        对话历史:
        {chat_history}
        
        后续问题: {question}
        
        独立问题:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_history", "question"]
        )
        
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        
        standalone_question = llm_chain.run({
            "chat_history": history_str,
            "question": question
        })
        
        return standalone_question.strip()