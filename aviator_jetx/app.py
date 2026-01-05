# app.py
"""
Aviator / Crash Game – Production Entry Point

Responsibilities:
- FastAPI HTTP server
- Request Validation (Pydantic)
- Game API orchestration
- Saga Pattern (Debit -> Try Bet -> Rollback if fail)
- Ledger-safe balance updates
- Static HTML serving

Integration:
- Uses engine.py (Decimal, Async, State Machine)
- Uses db.py (Atomic Transactions, Decimal)
"""

from __future__ import annotations

import os
import asyncio
import logging
from pathlib import Path
from decimal import Decimal
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession

# Imports from our modules
from engine import CrashGameEngine, StateError, BetError, GameState
from db import (
    init_db,
    get_session,
    get_or_create_user,
    debit,
    credit,
    User,
)

# =====================================================
# LOGGING & CONFIG
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aviator_app")

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BASE_URL = os.getenv("BASE_URL", "https://your-app.onrender.com")

ROOT_DIR = Path(__file__).parent
WEBAPP_FILE = ROOT_DIR / "webapp.html"

# =====================================================
# DATA MODELS (Pydantic)
# =====================================================

class UserInitRequest(BaseModel):
    user_id: str = Field(..., min_length=1)

class StartRoundRequest(BaseModel):
    client_seed: str = Field(..., min_length=1)

class BetRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)  # Input is float, converted to Decimal internally
    auto_cashout: Optional[float] = Field(None, gt=1.0)

class CashoutRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    multiplier: float = Field(..., gt=1.0)

# =====================================================
# LIFECYCLE
# =====================================================

engine = CrashGameEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events.
    """
    logger.info("Startup: Initializing Database...")
    await init_db()
    
    if BOT_TOKEN:
        logger.info("Startup: Launching Telegram Bot...")
        asyncio.create_task(run_telegram_bot())
        
    yield
    
    logger.info("Shutdown: Cleaning up...")

# =====================================================
# APP INIT
# =====================================================

app = FastAPI(
    title="Aviator Premium API",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Tighten this for real production!
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ERROR HANDLERS
# =====================================================

@app.exception_handler(StateError)
async def state_error_handler(_, exc: StateError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={"error": "Game State Conflict", "detail": str(exc)},
    )

@app.exception_handler(BetError)
async def bet_error_handler(_, exc: BetError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": "Invalid Bet", "detail": str(exc)},
    )

@app.exception_handler(ValueError)
async def value_error_handler(_, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "Value Error", "detail": str(exc)},
    )

# =====================================================
# STATIC ASSETS
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def serve_webapp() -> str:
    if not WEBAPP_FILE.exists():
        raise HTTPException(status_code=500, detail="webapp.html missing")
    return WEBAPP_FILE.read_text(encoding="utf-8")

# =====================================================
# API – USER
# =====================================================

@app.post("/api/init")
async def api_init(
    payload: UserInitRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Initialize user session and fetch balance.
    """
    user = await get_or_create_user(session, payload.user_id)
    return {
        "user_id": user.telegram_id,
        "balance": float(user.balance),  # Convert Decimal to float for JSON
    }

# =====================================================
# API – GAME CONTROL
# =====================================================

@app.post("/api/start-round")
async def api_start_round(payload: StartRoundRequest):
    """
    Manually starts a round (Client driven for this demo).
    In a fully autonomous backend, a background worker would call this.
    """
    try:
        result = await engine.start_new_round(payload.client_seed)
        return result
    except Exception as e:
        # If round is already running, just return current state
        # This prevents errors if frontend double-taps
        logger.warning(f"Start round failed (likely active): {e}")
        return await engine.get_current_state()

@app.get("/api/state")
async def api_state():
    """
    High-frequency polling endpoint for game state.
    """
    return await engine.get_current_state()

# =====================================================
# API – BETTING & CASHOUT
# =====================================================

@app.post("/api/place-bet")
async def api_place_bet(
    payload: BetRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Places a bet with SAGA pattern consistency:
    1. DB Debit (User pays)
    2. Engine Bet (Register bet)
    3. If Engine fails -> DB Refund (Compensating Transaction)
    """
    user = await get_or_create_user(session, payload.user_id)
    bet_amount = Decimal(str(payload.amount))

    # 1. OPTIMISTIC CHECK: Is engine in betting state?
    # This saves a DB write if the round is obviously closed.
    state = await engine.get_current_state()
    if state["status"] not in ["IDLE", "BETTING", "FLYING"]: # 'FLYING' allowed only for grace period handling in engine
         # Let engine throw the strict error, but we can fast-fail here if crashed
         if state["status"] == "CRASHED":
             raise HTTPException(409, "Round already crashed")

    current_round_id = state.get("round_id")

    # 2. DB DEBIT
    # We must lock funds before telling the engine we have a valid bet.
    try:
        await debit(
            session=session,
            user=user,
            amount=bet_amount,
            round_id=current_round_id,
            reference="bet_entry"
        )
    except ValueError as e:
        raise HTTPException(402, "Insufficient funds")

    # 3. ENGINE REGISTRATION
    try:
        await engine.place_bet(
            user_id=payload.user_id,
            amount=payload.amount,
            auto_cashout=payload.auto_cashout
        )
    except (StateError, ValueError) as e:
        # CRITICAL: SAGA ROLLBACK
        # The engine rejected the bet (e.g., round just turned to FLYING/CRASHED).
        # We must refund the user immediately.
        logger.warning(f"Bet rejected by engine. Refunding {payload.user_id}. Reason: {e}")
        
        await credit(
            session=session,
            user=user,
            amount=bet_amount,
            round_id=current_round_id,
            reference="bet_refund_engine_reject"
        )
        
        raise HTTPException(status_code=409, detail=f"Bet rejected: {str(e)}")

    # 4. SUCCESS
    # Reload user to get fresh balance
    await session.refresh(user)
    return {
        "status": "accepted",
        "new_balance": float(user.balance),
        "round_id": current_round_id
    }


@app.post("/api/cashout")
async def api_cashout(
    payload: CashoutRequest,
    session: AsyncSession = Depends(get_session),
):
    """
    Processes cashout.
    Engine is the authority on the payout amount.
    """
    # 1. ENGINE CASHOUT
    # We try engine first because if it fails (crashed), we touch nothing in DB.
    try:
        payout_amount = await engine.cashout(
            user_id=payload.user_id,
            client_observed_multiplier=payload.multiplier
        )
    except StateError as e:
        raise HTTPException(409, detail="Too late, plane crashed")
    except BetError as e:
        raise HTTPException(400, detail="Bet not found or already cashed")

    # 2. DB CREDIT
    # Engine confirmed the win. Now we pay.
    user = await get_or_create_user(session, payload.user_id)
    
    # We get round ID from engine state for audit trail
    state = await engine.get_current_state()
    
    await credit(
        session=session,
        user=user,
        amount=Decimal(str(payout_amount)),
        round_id=state.get("round_id"),
        reference=f"win_x{payload.multiplier}"
    )

    return {
        "status": "cashed_out",
        "payout": payout_amount,
        "balance": float(user.balance),
    }

# =====================================================
# TELEGRAM BOT (OPTIONAL)
# =====================================================

async def run_telegram_bot():
    """
    Runs the Telegram bot in the background.
    """
    from telegram import Update
    from telegram.ext import (
        ApplicationBuilder,
        CommandHandler,
        ContextTypes,
    )

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not update.message:
            return
            
        await update.message.reply_text(
            "🚀 **Aviator Premium**\n\nReady to fly?",
            parse_mode="Markdown",
            reply_markup={
                "keyboard": [[
                    {
                        "text": "▶️ Play Now",
                        "web_app": {"url": BASE_URL},
                    }
                ]],
                "resize_keyboard": True,
            },
        )

    try:
        app_bot = ApplicationBuilder().token(BOT_TOKEN).build()
        app_bot.add_handler(CommandHandler("start", start))

        logger.info("Bot initializing...")
        await app_bot.initialize()
        await app_bot.start()
        
        # In a web server context, we generally don't use updater.start_polling()
        # because it blocks or conflicts with signal handlers.
        # Ideally, use webhooks. For this demo, we use a simple polling task.
        await app_bot.updater.start_polling()
        logger.info("Bot polling started.")
        
    except Exception as e:
        logger.error(f"Telegram Bot failed to start: {e}")