import json
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_async_session
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    MessageResponse,
)
from app.schemas.common import SuccessResponse
from app.services.rag_service import RAGService
from app.services.conversation_service import ConversationService

router = APIRouter(tags=["Chat"])


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    db: AsyncSession = Depends(get_async_session),
):
    conv_service = ConversationService(db)
    conversation = await conv_service.create_conversation()
    await db.commit()
    return ConversationResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    db: AsyncSession = Depends(get_async_session),
):
    conv_service = ConversationService(db)
    conversations = await conv_service.list_conversations()
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_messages(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    conv_service = ConversationService(db)
    messages = await conv_service.get_messages(conversation_id)
    return [MessageResponse.model_validate(m) for m in messages]


@router.delete("/conversations/{conversation_id}", response_model=SuccessResponse)
async def delete_conversation(
    conversation_id: uuid.UUID,
    db: AsyncSession = Depends(get_async_session),
):
    conv_service = ConversationService(db)
    await conv_service.delete_conversation(conversation_id)
    await db.commit()
    return SuccessResponse(message="Conversation deleted")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_session),
):
    rag_service = RAGService(db)
    result = await rag_service.generate_answer(
        query=request.query,
        conversation_id=request.conversation_id,
        filters=request.filters,
        rerank=request.rerank,
    )
    await db.commit()
    return result


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_async_session),
):
    rag_service = RAGService(db)

    async def event_generator():
        try:
            # Step 1: Status update
            yield f"data: {json.dumps({'type': 'status', 'content': 'Searching documents...'})}\n\n"

            # Step 2: Retrieve relevant chunks
            search_results = await rag_service.retrieve(
                query=request.query,
                filters=request.filters,
                rerank=request.rerank,
            )

            sources = [
                {
                    "document_id": str(r.document_id),
                    "document_name": r.document_name,
                    "page_number": r.page_number,
                    "chunk_id": str(r.chunk_id),
                    "relevance_score": r.score,
                    "snippet": r.content[:200],
                }
                for r in search_results
            ]
            yield f"data: {json.dumps({'type': 'sources', 'content': sources})}\n\n"

            # Step 3: Get conversation history
            history = []
            if request.conversation_id:
                conv_service = ConversationService(db)
                history = await conv_service.get_recent_history(request.conversation_id)

            # Step 4: Stream generation
            yield f"data: {json.dumps({'type': 'status', 'content': 'Generating answer...'})}\n\n"

            full_answer = ""
            async for token in rag_service.generate_stream(
                query=request.query,
                search_results=search_results,
                history=history,
            ):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            # Step 5: Save to conversation
            conversation_id = request.conversation_id
            if not conversation_id:
                conv_service = ConversationService(db)
                conv = await conv_service.create_conversation(title=request.query[:100])
                conversation_id = conv.id

            conv_service = ConversationService(db)
            await conv_service.add_message(conversation_id, "user", request.query)
            msg = await conv_service.add_message(
                conversation_id,
                "assistant",
                full_answer,
                sources=sources,
            )
            await db.commit()

            yield f"data: {json.dumps({'type': 'done', 'content': {'conversation_id': str(conversation_id), 'message_id': str(msg.id)}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
