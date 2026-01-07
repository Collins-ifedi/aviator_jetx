# db.py
"""
Database Layer – Production Grade
Responsibilities:
- Async SQLAlchemy Session Management
- Atomic Balance Transactions (Crucial for preventing race conditions)
- User Persistence
"""

import os
import logging
from decimal import Decimal
from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    Column, 
    Integer, 
    String, 
    Numeric, 
    DateTime, 
    select, 
    update
)
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase

# =========================
# CONFIGURATION
# =========================

logger = logging.getLogger("aviator_db")

# Use SQLite for local testing, Postgres for production
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./aviator.db")

# Ensure Postgres URL is async-compatible if using Render/Heroku
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)

# Engine Configuration
# echo=False in production to reduce log noise
engine = create_async_engine(
    DATABASE_URL, 
    echo=False, 
    future=True,
    # Connection pool settings for production stability
    pool_size=20,
    max_overflow=10
)

# Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False, # Important: keeps objects accessible after commit
    autoflush=False
)

# =========================
# MODELS
# =========================

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(String, unique=True, index=True, nullable=False)
    
    # numeric(18, 2) is standard for currency
    balance = Column(Numeric(18, 2), default=0.00, nullable=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<User {self.telegram_id} Bal: {self.balance}>"

# =========================
# LIFECYCLE
# =========================

async def init_db():
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized.")

async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency for DB Sessions."""
    async with AsyncSessionLocal() as session:
        yield session

# =========================
# TRANSACTIONAL LOGIC
# =========================

async def get_or_create_user(session: AsyncSession, telegram_id: str) -> User:
    """
    Fetch a user or create one if they don't exist.
    Optimized to minimize locking.
    """
    stmt = select(User).where(User.telegram_id == telegram_id)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        return user

    # Create new user
    # Note: In a high-concurrency cluster, you might need a try/except IntegrityError block here.
    new_user = User(telegram_id=telegram_id, balance=1000.00) # Start with $1000 demo cash
    session.add(new_user)
    await session.commit()
    await session.refresh(new_user)
    return new_user

async def debit(session: AsyncSession, user: User, amount: Decimal, reference: str = "") -> User:
    """
    Atomically decrement user balance.
    Raises ValueError if insufficient funds.
    """
    # Use database-side calculation to prevent race conditions.
    # "UPDATE users SET balance = balance - :amt WHERE id = :id AND balance >= :amt"
    stmt = (
        update(User)
        .where(User.telegram_id == user.telegram_id)
        .where(User.balance >= amount) # Optimistic Lock Check
        .values(balance=User.balance - amount)
        .returning(User.balance) # Return the NEW balance immediately
    )

    result = await session.execute(stmt)
    new_balance = result.scalar()

    if new_balance is None:
        # If None, the WHERE clause failed (insufficient funds)
        await session.rollback()
        raise ValueError("Insufficient funds")

    # Sync the Python object with the new DB state
    user.balance = new_balance
    await session.commit()
    return user

async def credit(session: AsyncSession, user: User, amount: Decimal, reference: str = "") -> User:
    """
    Atomically increment user balance.
    """
    stmt = (
        update(User)
        .where(User.telegram_id == user.telegram_id)
        .values(balance=User.balance + amount)
        .returning(User.balance)
    )

    result = await session.execute(stmt)
    new_balance = result.scalar()

    if new_balance is None:
        # Should rarely happen unless user was deleted mid-game
        await session.rollback()
        raise ValueError("User not found during credit")

    user.balance = new_balance
    await session.commit()
    return user