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


class ContextEvent(SQLModel, table=True):
    """Log of intent matching events - tracks context over time."""
    __tablename__ = "context_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    intent_name: str  # IBranch name (loose coupling - no FK)
    draft_id: Optional[str]  # Draft UUID that triggered this, if any
    timestamp: float
    stack_name: str = Field(default="default")  # Support multiple context stacks
    match_result: Optional[str] = None  # Text representation of match result for context
    created_at: datetime = Field(default_factory=datetime.now)

class IBranch(SQLModel, table=True):
    __tablename__ = "i_branches"

    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    parent_id: Optional[int] = Field(default=None, foreign_key="i_branches.id")
    version: Optional[int] = None  # Only set on roots - children inherit from parent
    is_active: bool = True  # Mark which version to use for matching
    created_at: datetime = Field(default_factory=datetime.now)

    # Relationships - SQLModel uses Relationship for ORM relationships
    parent: Optional["IBranch"] = Relationship(
        back_populates="children",
        sa_relationship_kwargs={"remote_side": "IBranch.id"}
    )
    children: list["IBranch"] = Relationship(back_populates="parent")

    # Instance methods
    def get_root(self) -> "IBranch":
        """Traverse up the tree to find the root node."""
        node = self
        while node.parent is not None:
            node = node.parent
        return node

    def is_root(self) -> bool:
        """Check if this node is a root (has no parent)."""
        return self.parent_id is None

    def get_children(self) -> list["IBranch"]:
        """Get direct children of this branch (wrapper for .children relationship)."""
        return self.children

    # Class methods (Django ModelManager style)
    @classmethod
    def get_roots(cls, session: Session, active_only: bool = True) -> list["IBranch"]:
        """Get all root branches (branches with no parent).

        Args:
            active_only: If True, only return active roots. If False, return all roots.
        """
        from sqlmodel import select
        statement = select(cls).where(cls.parent_id == None)
        if active_only:
            statement = statement.where(cls.is_active == True)
        return list(session.exec(statement).all())

    @classmethod
    def get_all(cls, session: Session) -> list["IBranch"]:
        """Get all branches."""
        from sqlmodel import select
        statement = select(cls)
        return list(session.exec(statement).all())

    @classmethod
    def get_by_name(cls, session: Session, name: str) -> Optional["IBranch"]:
        """Get a branch by name (returns first match)."""
        from sqlmodel import select
        statement = select(cls).where(cls.name == name)
        return session.exec(statement).first()

    @classmethod
    def filter_by_name(cls, session: Session, name: str) -> list["IBranch"]:
        """Get all branches with the given name."""
        from sqlmodel import select
        statement = select(cls).where(cls.name == name)
        return list(session.exec(statement).all())


    
class IBranchStore:
    """Separate store for intent branches - keeps them isolated from drafts."""

    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # Set up separate database for IBranches
        db_path = self._store_dir / "ibranches.db"
        self._db_url = f"sqlite:///{db_path}"
        self._engine = create_engine(self._db_url, echo=False)

        # Create IBranch table only
        IBranch.metadata.create_all(self._engine)

    @property
    def engine(self):
        """Expose engine for IBranch queries."""
        return self._engine

    def session(self):
        """Create a new session context manager for queries."""
        return Session(self._engine)

    async def add_branch(self, branch: IBranch):
        """Add a new branch to the tree."""
        with Session(self._engine) as session:
            session.add(branch)
            session.commit()
            session.refresh(branch)
            logger.info(f"Saved branch '{branch.name}' (id: {branch.id}) to database")
            return branch


class DraftStore:

    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # Set up database for drafts only
        db_path = self._store_dir / "drafts.db"
        self._db_url = f"sqlite:///{db_path}"
        self._engine = create_engine(self._db_url, echo=False)

        # Create Draft table only
        Draft.metadata.create_all(self._engine)

    @property
    def engine(self):
        """Expose engine for draft queries."""
        return self._engine

    def session(self):
        """Create a new session context manager for queries."""
        return Session(self._engine)

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


class ContextStore:
    """Separate store for context events - tracks intent matching history."""

    def __init__(self, store_dir: Path):
        self._store_dir = Path(store_dir)
        self._store_dir.mkdir(parents=True, exist_ok=True)

        # Set up separate database for context
        db_path = self._store_dir / "context.db"
        self._db_url = f"sqlite:///{db_path}"
        self._engine = create_engine(self._db_url, echo=False)

        # Create ContextEvent table only
        ContextEvent.metadata.create_all(self._engine)

    @property
    def engine(self):
        """Expose engine for context queries."""
        return self._engine

    def session(self):
        """Create a new session context manager for queries."""
        return Session(self._engine)

    async def add_event(self, event: ContextEvent):
        """Add a new context event to the log."""
        with Session(self._engine) as session:
            session.add(event)
            session.commit()
            session.refresh(event)
            logger.info(f"Logged context event: intent='{event.intent_name}', stack='{event.stack_name}'")
            return event

    def get_current_context(self, stack_name: str = "default") -> Optional[ContextEvent]:
        """Get the most recent context event for a stack."""
        with Session(self._engine) as session:
            from sqlmodel import select
            statement = (
                select(ContextEvent)
                .where(ContextEvent.stack_name == stack_name)
                .order_by(ContextEvent.timestamp.desc())
            )
            return session.exec(statement).first()

    def get_context_history(self, stack_name: str = "default", limit: int = 10) -> list[ContextEvent]:
        """Get recent context events for a stack."""
        with Session(self._engine) as session:
            from sqlmodel import select
            statement = (
                select(ContextEvent)
                .where(ContextEvent.stack_name == stack_name)
                .order_by(ContextEvent.timestamp.desc())
                .limit(limit)
            )
            return list(session.exec(statement).all())

