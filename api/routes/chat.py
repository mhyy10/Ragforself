"""Chat and query routes for RAG system API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from core.services import get_service_manager


class QueryRequest(BaseModel):
    query: str
    k: int = 4  # Number of documents to retrieve
    search_type: str = "hybrid"  # Type of search ('similarity', 'mmr', 'hybrid')


class QueryResponse(BaseModel):
    query: str
    answer: str
    source_documents: list


# Initialize router
router = APIRouter()

# Get services from the global service manager
service_manager = get_service_manager()
retrieval_engine = service_manager.get_retrieval_engine()


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base using RAG
    """
    try:
        # Log the incoming request for debugging
        print(f"Received query: {request.query}, k: {request.k}, search_type: {request.search_type}")

        # Answer the query using RAG with specified search type
        result = retrieval_engine.answer_query(request.query, k=request.k, search_type=request.search_type)

        # Extract source document information
        # Extract source document information
        source_docs_info = []
        for doc in result.get("source_documents", []):
            # 添加调试信息
            print(f"Processing doc type: {type(doc)}, value: {doc}")

            # 检查是否为字符串
            if isinstance(doc, str):
                content = doc
                metadata = {}
            elif hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                # 这是一个 Document 对象
                content = doc.page_content
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'page_content' in doc:
                # 这是一个字典，包含 page_content
                content = doc['page_content']
                metadata = doc.get('metadata', {})
            else:
                # 未知格式，跳过
                print(f"Skipping unknown doc type: {type(doc)}")
                continue

            # 确保 content 是字符串
            if isinstance(content, dict):
                content = str(content)
            elif not isinstance(content, str):
                content = str(content)

            source_info = {
                "content": content[:200] + "..." if len(content) > 200 else content,  # Truncate for response
                "metadata": metadata
            }
            source_docs_info.append(source_info)

        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            source_documents=source_docs_info
        )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.get("/health")
async def chat_health():
    """
    Health check for chat service
    """
    return {"status": "healthy", "service": "chat"}