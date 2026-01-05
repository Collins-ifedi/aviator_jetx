# db.py
"""
Database Layer – Production Grade

Responsibilities:
- Async database engine & session lifecycle
- User account persistence
- Ledger-safe balance management (Decimal arithmetic)
- Transaction history with Round ID linking
- Audit-ready immutability

Alignment with Engine:
- Uses Decimal for all financial values
- Links transactions to engine round_ids
- Handles currency precision strictly
"""

from __future__ import annotations

import os
import enum
from datetime import datetime
from decimal import Decimal
from typing import AsyncGenerator, Optional

from sqlalchemy import (
    String,
    DateTime,
    Integer,
    Enum,
    ForeignKey,
    func,
    Numeric,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.exc import IntegrityError

# =====================================================
# CONFIG
# =====================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite+aiosqlite:///./aviator.db"
)

# Default starting balance for new demo users
STARTING_BALANCE = Decimal(os.getenv("STARTING_BALANCE", "1000.00"))

DB_ECHO = bool(os.getenv("DB_ECHO", False))


# =====================================================
# BASE
# =====================================================

class Base(DeclarativeBase):
    pass


# =====================================================
# ENUMS
# =====================================================

class TransactionType(str, enum.Enum):
    BET = "bet"
    WIN = "win"  # Renamed from CASHOUT for clarity
    REFUND = "refund"
    DEPOSIT = "deposit"
    WITHDRAW = "withdraw"
    ADJUSTMENT = "adjustment"


# =====================================================
# MODELS
# =====================================================

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    telegram_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )

    # PRECISION: 18 digits total, 2 after decimal. 
    # Max: 9,999,999,999,999,999.99
    balance: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
        default=STARTING_BALANCE,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class Transaction(Base):
    """
    Immutable ledger record (append-only).
    Links financial movement to specific Game Rounds.
    """

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    type: Mapped[TransactionType] = mapped_column(
        Enum(TransactionType, name="transaction_type"),
        nullable=False,
    )

    # Signed amount: -10.00 for bet, +20.00 for win
    amount: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    balance_after: Mapped[Decimal] = mapped_column(
        Numeric(18, 2),
        nullable=False,
    )

    # Linking to engine.py Round ID
    round_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # Extra metadata (e.g. "multiplier: 2.50x")
    reference: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    user: Mapped[User] = relationship(back_populates="transactions")


# =====================================================
# ENGINE & SESSION
# =====================================================

engine = create_async_engine(
    DATABASE_URL,
    echo=DB_ECHO,
    future=True,
    # SSL is critical for Postgres in production
    connect_args={"ssl": "require"} if "postgresql" in DATABASE_URL else {},
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# =====================================================
# INIT
# =====================================================

async def init_db() -> None:
    """
    Creates all tables. Safe to run on every startup.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# =====================================================
# REPOSITORY HELPERS
# =====================================================

async def get_or_create_user(
    session: AsyncSession,
    telegram_id: str,
) -> User:
    """
    Fetches a user or creates one with default balance.
    """
    result = await session.execute(
        select(User).where(User.telegram_id == telegram_id)
    )
    user = result.scalar_one_or_none()

    if user:
        return user

    # Create new user
    new_user = User(telegram_id=telegram_id, balance=STARTING_BALANCE)
    session.add(new_user)
    
    try:
        await session.commit()
        await session.refresh(new_user)
        return new_user
    except IntegrityError:
        # Handle race condition where user was created in parallel
        await session.rollback()
        return await get_or_create_user(session, telegram_id)


async def apply_transaction(
    session: AsyncSession,
    user: User,
    amount: Decimal,
    tx_type: TransactionType,
    round_id: str | None = None,
    reference: str | None = None,
) -> User:
    """
    Atomic balance update + immutable ledger entry.
    
    Args:
        amount: Decimal. Positive for credit, negative for debit usually.
                However, for clarity, we usually pass positive values to 
                debit/credit wrappers and they handle the sign.
                Here, 'amount' is the raw change to be added.
    """
    
    # 1. Lock the user row for update to prevent race conditions
    # (Only works on Postgres/MySQL, ignored on SQLite but harmless)
    result = await session.execute(
        select(User).where(User.id == user.id).with_for_update()
    )
    user_locked = result.scalar_one()

    # 2. Calculate new balance
    # Quantize ensures we never drift beyond 2 decimal places
    amount_quantized = amount.quantize(Decimal("0.01"))
    new_balance = user_locked.balance + amount_quantized

    # 3. Validate
    if new_balance < 0:
        raise ValueError("Insufficient balance")

    # 4. Mutate User
    user_locked.balance = new_balance
    user_locked.updated_at = datetime.now()

    # 5. Create Ledger Entry
    tx = Transaction(
        user_id=user_locked.id,
        type=tx_type,
        amount=amount_quantized,
        balance_after=new_balance,
        round_id=round_id,
        reference=reference,
    )

    session.add(tx)
    await session.commit()
    await session.refresh(user_locked)
    
    return user_locked


# =====================================================
# CONVENIENCE WRAPPERS
# =====================================================

async def debit(
    session: AsyncSession,
    user: User,
    amount: Decimal | float,
    round_id: str | None = None,
    reference: str | None = None,
) -> User:
    """
    Deducts money (e.g., placing a bet).
    Converts float to Decimal automatically.
    """
    val = Decimal(str(amount)) if isinstance(amount, float) else amount
    
    return await apply_transaction(
        session,
        user,
        -abs(val), # Ensure negative
        TransactionType.BET,
        round_id,
        reference,
    )


async def credit(
    session: AsyncSession,
    user: User,
    amount: Decimal | float,
    round_id: str | None = None,
    reference: str | None = None,
) -> User:
    """
    Adds money (e.g., cashing out).
    Converts float to Decimal automatically.
    """
    val = Decimal(str(amount)) if isinstance(amount, float) else amount

    return await apply_transaction(
        session,
        user,
        abs(val), # Ensure positive
        TransactionType.WIN,
        round_id,
        reference,
    )