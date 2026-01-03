import asyncio
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

from sqlmodel import SQLModel, Field, create_engine, Session, Relationship

logger = logging.getLogger("StoreModels")


class Draft(SQLModel, table=True):
    __tablename__ = "drafts"

    id: Optional[int] = Field(default=None, primary_key=True)
    draft_id: str = Field(index=True, unique=True)  # UUID for draft lookup
    timestamp: float  # Original draft timestamp
    full_text: str
    created_at: datetime = Field(default_factory=datetime.now)

class IBranch(SQLModel, table=True):
    __tablename__ = "i_branches"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    created_at: datetime = Field(default_factory=datetime.now)


    
class DraftStore:

    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # Set up database
        db_path = self._store_dir / "drafts.db"
        self._db_url = f"sqlite:///{db_path}"
        self._engine = create_engine(self._db_url, echo=False)

        # Create tables
        SQLModel.metadata.create_all(self._engine)

    async def add_draft(self, draft: Draft):
        with Session(self._engine) as session:
            from sqlmodel import select

            # Check if draft_id already exists
            existing = session.exec(
                select(Draft).where(Draft.draft_id == str(draft.draft_id))
            ).first()
            if existing:
                raise ValueError(
                    f"Draft with draft_id '{draft.draft_id}' already exists in database (record id: {existing.id})"
                )

            # Create draft record
            draft_record = Draft(
                draft_id=str(draft.draft_id),
                timestamp=draft.timestamp,
                full_text=draft.full_text,
            )
            session.add(draft_record)
            session.commit()
            logger.info(f"Saved draft {draft_record.id} to database")
        
    def get_all_drafts(self) -> list[Draft]:
        """Query all drafts from database"""
        with Session(self._engine) as session:
            from sqlmodel import select
            statement = select(Draft)
            return list(session.exec(statement).all())

    def get_draft_by_id(self, draft_id: int) -> Optional[Draft]:
        """Get a specific draft by database ID"""
        with Session(self._engine) as session:
            return session.get(Draft, draft_id)

    def get_draft_by_uuid(self, draft_uuid: str) -> Optional[Draft]:
        """Get a specific draft by its draft_id (UUID)"""
        with Session(self._engine) as session:
            from sqlmodel import select
            statement = select(Draft).where(Draft.draft_id == draft_uuid)
            return session.exec(statement).first()

