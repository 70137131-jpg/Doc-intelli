import uuid
from typing import Optional

from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation, Message
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConversationService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_conversation(self, title: str | None = None) -> Conversation:
        conversation = Conversation(title=title)
        self.db.add(conversation)
        await self.db.flush()
        return conversation

    async def get_conversation(self, conversation_id: uuid.UUID) -> Conversation | None:
        result = await self.db.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        return result.scalar_one_or_none()

    async def list_conversations(self, limit: int = 50) -> list[Conversation]:
        result = await self.db.execute(
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_conversation(self, conversation_id: uuid.UUID) -> None:
        await self.db.execute(
            delete(Conversation).where(Conversation.id == conversation_id)
        )
        await self.db.flush()

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        confidence: str | None = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=sources,
            confidence=confidence,
        )
        self.db.add(message)
        await self.db.flush()

        # Update conversation title if first user message
        if role == "user":
            conversation = await self.get_conversation(conversation_id)
            if conversation and not conversation.title:
                conversation.title = content[:100]
                await self.db.flush()

        return message

    async def get_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> list[Message]:
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_recent_history(
        self,
        conversation_id: uuid.UUID,
        max_messages: int = 10,
    ) -> list[dict]:
        """Get recent messages formatted for LLM context."""
        messages = await self.get_messages(conversation_id, limit=max_messages)
        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
