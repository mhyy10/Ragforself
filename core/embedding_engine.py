"""Embedding engine for RAG system"""
import os
import logging
import re
from typing import List, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from config.settings import Settings


class EmbeddingEngine:
    """Handles text embedding and vector storage operations"""
    
    def __init__(self):
        """Initialize embedding engine with appropriate model"""
        self.logger = logging.getLogger(__name__)
        self.settings = Settings()


        # Initialize embedding model based on configuration
        logging.info("Initializing embedding engine")
        logging.info(self.settings.OPENAI_API_KEY is not None)
        if self.settings.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(
                api_key=self.settings.OPENAI_API_KEY,
                model="Qwen/Qwen3-Embedding-8B",
                base_url=self.settings.BASE_URL
            )
        else:
            # Use open-source embedding model as fallback
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        # Initialize vector store
        self.vector_store = None
        self.collection_name = "rag_documents"

        # Auto-load vector store if it exists
        self._auto_load_vector_store()

    def _auto_load_vector_store(self):
        """Automatically load vector store if it exists"""
        import os
        if os.path.exists(self.settings.VECTOR_DB_PATH) and os.listdir(self.settings.VECTOR_DB_PATH):
            try:
                self.load_vector_store()
                self.logger.info("Auto-loaded existing vector store")
            except Exception as e:
                self.logger.warning(f"Could not auto-load vector store: {str(e)}")
    
    def is_vector_store_ready(self) -> bool:
        """
        Check if the vector store is ready for queries

        Returns:
            True if vector store is ready, False otherwise
        """
        if self.vector_store is None:
            return False

        # Try a simple operation to verify the vector store is functional
        try:
            # Perform a minimal similarity search to test
            test_results = self.vector_store.similarity_search("test", k=1)
            return len(test_results) >= 0  # If no exception, it's ready
        except:
            return False

    def create_vector_store(self, documents: List[Document], persist_directory: Optional[str] = None) -> Chroma:
        """
        Create a vector store from documents

        Args:
            documents: List of documents to index
            persist_directory: Directory to persist the vector store

        Returns:
            Initialized Chroma vector store
        """
        if persist_directory is None:
            persist_directory = self.settings.VECTOR_DB_PATH

        # Debug: Check document format before creating vector store
        for i, doc in enumerate(documents):
            if not isinstance(doc.page_content, str):
                self.logger.warning(f"Document {i} has non-string page_content: {type(doc.page_content)}, value: {doc.page_content}")
                # Convert to string if it's not already
                doc.page_content = str(doc.page_content)

        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=self.collection_name
        )

        self.logger.info(f"Created vector store with {len(documents)} documents")
        return self.vector_store
    
    def load_vector_store(self, persist_directory: Optional[str] = None) -> Chroma:
        """
        Load an existing vector store

        Args:
            persist_directory: Directory where vector store is persisted

        Returns:
            Loaded Chroma vector store
        """
        if persist_directory is None:
            persist_directory = self.settings.VECTOR_DB_PATH

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=self.collection_name,
            embedding_function=self.embeddings
        )

        self.logger.info(f"Loaded vector store from {persist_directory}")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the existing vector store

        Args:
            documents: List of documents to add

        Returns:
            List of IDs of added documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")

        # Debug: Check document format before adding to vector store
        for i, doc in enumerate(documents):
            if not isinstance(doc.page_content, str):
                self.logger.warning(f"Document {i} has non-string page_content: {type(doc.page_content)}, value: {doc.page_content}")
                # Convert to string if it's not already
                doc.page_content = str(doc.page_content)

        ids = self.vector_store.add_documents(documents)
        self.logger.info(f"Added {len(documents)} documents to vector store")
        return ids

    def detect_query_intent(self, query: str) -> str:
        """
        Detect the intent of the query to choose the best search strategy

        Args:
            query: Original query string

        Returns:
            Search strategy type ('similarity', 'mmr', 'hybrid')
        """
        # Convert to lowercase for consistent matching
        query_lower = query.lower()

        # Keywords that suggest specific search strategies
        similarity_keywords = [
            'who', 'what', 'when', 'where', 'define', 'explain',
            'describe', 'tell me about', 'information about'
        ]

        diversity_keywords = [
            'compare', 'different', 'versus', 'vs', 'alternatives',
            'options', 'types', 'kinds', 'various', 'multiple'
        ]

        # Check for diversity keywords (suggesting MMR for diverse results)
        for keyword in diversity_keywords:
            if keyword in query_lower:
                return "mmr"

        # Default to hybrid for most queries
        return "hybrid"

    def expand_query(self, query: str) -> List[str]:
        """
        Expand the query with synonyms and related terms to improve search results

        Args:
            query: Original query string

        Returns:
            List of expanded query terms
        """
        # For now, we'll implement a simple expansion based on common synonyms
        # In a production system, you might use a thesaurus API or NLP models
        query_lower = query.lower()

        # Define some common expansions
        expansions = []

        # Common political/societal terms expansion
        if 'president' in query_lower:
            expansions.extend(['president', 'leader', 'head of state', 'head of government', 'commander'])
        if 'captured' in query_lower or 'arrested' in query_lower:
            expansions.extend(['captured', 'arrested', 'detained', 'imprisoned', 'apprehended'])
        if 'impact' in query_lower or 'effect' in query_lower or 'affect' in query_lower:
            expansions.extend(['impact', 'effect', 'affect', 'influence', 'consequence', 'result'])
        if 'world' in query_lower or 'global' in query_lower:
            expansions.extend(['world', 'global', 'international', 'worldwide', 'earth'])
        if 'economy' in query_lower:
            expansions.extend(['economy', 'economic', 'financial', 'monetary', 'market'])
        if 'politics' in query_lower or 'political' in query_lower:
            expansions.extend(['politics', 'political', 'government', 'policy', 'diplomatic'])

        # Remove duplicates while preserving order
        unique_expansions = []
        for exp in expansions:
            if exp not in unique_expansions and exp not in query_lower:
                unique_expansions.append(exp)

        # Return original query with expansions
        all_terms = [query] + unique_expansions
        return all_terms

    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query to improve search accuracy

        Args:
            query: Original query string

        Returns:
            Preprocessed query string
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())

        # Convert to lowercase for consistent matching
        query = query.lower()

        # You can add more preprocessing steps here like:
        # - Removing common stop words
        # - Query expansion (now implemented separately)

        return query

    def enhanced_similarity_search(self, query: str, k: int = 4, search_type: str = "auto") -> List[Document]:
        """
        Perform enhanced similarity search with multiple strategies

        Args:
            query: Query string
            k: Number of results to return
            search_type: Type of search ('similarity', 'mmr', 'hybrid', 'auto')

        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")

        # If search_type is 'auto', detect the intent from the query
        if search_type == "auto":
            search_type = self.detect_query_intent(query)

        # Preprocess the query
        processed_query = self.preprocess_query(query)

        # Different search strategies based on type
        if search_type == "mmr":  # Maximum Marginal Relevance
            results = self.vector_store.max_marginal_relevance_search(
                processed_query, k=k, fetch_k=min(k*2, 20)
            )
        elif search_type == "hybrid":
            # For hybrid search, we'll combine multiple strategies
            # First, get similarity search results
            similarity_results = self.vector_store.similarity_search(processed_query, k=k)

            # Then get MMR results for diversity
            mmr_results = self.vector_store.max_marginal_relevance_search(
                processed_query, k=k, fetch_k=min(k*2, 20)
            )

            # Combine and deduplicate results, prioritizing by relevance
            combined_results = []
            seen_content = set()

            for doc in similarity_results + mmr_results:
                content_key = doc.page_content[:100]  # Use first 100 chars as identifier
                if content_key not in seen_content:
                    combined_results.append(doc)
                    seen_content.add(content_key)

            # Return top k results
            results = combined_results[:k]
        else:  # Default to similarity search
            # Try query expansion for better matching
            expanded_queries = self.expand_query(query)

            # Perform search with original query and expanded terms
            all_results = []
            seen_content = set()

            for q in expanded_queries:
                try:
                    # Preprocess each expanded query
                    processed_q = self.preprocess_query(q)
                    query_results = self.vector_store.similarity_search(processed_q, k=max(k, 2))

                    for doc in query_results:
                        content_key = doc.page_content[:100]  # Use first 100 chars as identifier
                        if content_key not in seen_content:
                            all_results.append(doc)
                            seen_content.add(content_key)
                except Exception as e:
                    # If expanded query fails, continue with original
                    self.logger.warning(f"Error with expanded query '{q}': {str(e)}")
                    continue

            # Return top k results from all searches
            results = all_results[:k] if all_results else self.vector_store.similarity_search(processed_query, k=k)

        # 确保返回的都是 Document 对象，并且具有正确的属性
        validated_results = []
        for i, item in enumerate(results):
            if isinstance(item, str):
                # 如果是字符串，记录错误并跳过
                self.logger.error(f"Vector store returned string instead of Document at index {i}: {item}")
                continue
            elif hasattr(item, 'page_content') and hasattr(item, 'metadata'):
                # 这是一个 Document 对象，验证 page_content 是字符串
                if not isinstance(item.page_content, str):
                    self.logger.warning(f"Document {i} has non-string page_content: {type(item.page_content)}")
                    item.page_content = str(item.page_content) if item.page_content is not None else ""
                validated_results.append(item)
            else:
                # 不是预期的类型，记录错误
                self.logger.error(f"Vector store returned unexpected type at index {i}: {type(item)}, value: {item}")
                continue

        self.logger.info(f"Found {len(validated_results)} valid documents for query: {query[:50]}...")
        return validated_results

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search in the vector store

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of similar documents
        """
        # Use the enhanced search as the default with auto-detection
        return self.enhanced_similarity_search(query, k, search_type="auto")

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple[Document, float]]:
        """
        Perform similarity search with scores

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of tuples (document, score)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store or load_vector_store first.")

        # Use the enhanced search approach with scores
        # First, get results using the enhanced search method
        expanded_queries = self.expand_query(query)

        all_results = []
        seen_content = set()

        for q in expanded_queries:
            try:
                # Preprocess each expanded query
                processed_q = self.preprocess_query(q)
                query_results = self.vector_store.similarity_search_with_score(processed_q, k=max(k, 2))

                for doc, score in query_results:
                    content_key = doc.page_content[:100]  # Use first 100 chars as identifier
                    if content_key not in seen_content:
                        all_results.append((doc, score))
                        seen_content.add(content_key)
            except Exception as e:
                # If expanded query fails, continue with original
                self.logger.warning(f"Error with expanded query '{q}': {str(e)}")
                continue

        # If no results from expanded queries, fall back to original query
        if not all_results:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        else:
            # Sort by score (ascending, as lower scores mean higher similarity in some implementations)
            results = sorted(all_results, key=lambda x: x[1])[:k]

        # Debug: Check if any retrieved documents have non-string page_content
        for i, (doc, score) in enumerate(results):
            if not isinstance(doc.page_content, str):
                self.logger.warning(f"Retrieved document {i} has non-string page_content: {type(doc.page_content)}, value: {doc.page_content}")
                # Convert to string if it's not already
                doc.page_content = str(doc.page_content)

        self.logger.info(f"Found {len(results)} similar documents with scores for query: {query[:50]}...")
        return results