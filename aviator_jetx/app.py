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
- Conflict-proof Telegram Bot Integration

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
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    Request,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Imports from our modules
from engine import CrashGameEngine, StateError, BetError
from db import (
    init_db,
    get_session,
    get_or_create_user,
    debit,
    credit,
)

# =====================================================
# LOGGING & CONFIG
# =====================================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("aviator_app")

# Ensure critical env vars are loaded
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
BASE_URL = os.getenv("BASE_URL", "https://your-app.onrender.com").rstrip('/')

ROOT_DIR = Path(__file__).parent
WEBAPP_FILE = ROOT_DIR / "webapp.html"

# Global reference to manage bot lifecycle
bot_app = None

# =====================================================
# DATA MODELS (Pydantic)
# =====================================================

class UserInitRequest(BaseModel):
    user_id: str = Field(..., min_length=1)

class StartRoundRequest(BaseModel):
    client_seed: str = Field(..., min_length=1)

class BetRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    amount: float = Field(..., gt=0)
    auto_cashout: Optional[float] = Field(None, gt=1.0)

class CashoutRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    multiplier: float = Field(..., gt=1.0)

# =====================================================
# TELEGRAM BOT LOGIC
# =====================================================

async def run_telegram_bot():
    """
    Runs the Telegram bot in background with conflict resolution.
    """
    global bot_app
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
        logger.info("Bot: Initializing...")
        bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
        bot_app.add_handler(CommandHandler("start", start))

        await bot_app.initialize()
        
        # CRITICAL FIX: Delete webhook to clear ghost connections before polling
        # This prevents the "Conflict: terminated by other getUpdates request" error
        logger.info("Bot: Clearing old webhooks/conflicts...")
        await bot_app.bot.delete_webhook(drop_pending_updates=True)

        await bot_app.start()
        
        # Start polling in a non-blocking way
        # drop_pending_updates=True clears the queue to prevent loop crashes
        logger.info("Bot: Starting polling...")
        await bot_app.updater.start_polling(drop_pending_updates=True)
        
        logger.info("Bot: Polling started successfully.")
        
    except Exception as e:
        logger.error(f"Telegram Bot failed to start: {e}")
        # We do NOT re-raise here, so the web server stays alive even if bot fails

# =====================================================
# LIFECYCLE MANAGEMENT
# =====================================================

engine = CrashGameEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events.
    """
    # --- STARTUP ---
    logger.info("Startup: Initializing Database...")
    await init_db()
    
    if BOT_TOKEN:
        logger.info("Startup: Launching Telegram Bot Task...")
        # Run bot as a background task
        asyncio.create_task(run_telegram_bot())
    else:
        logger.warning("Startup: No BOT_TOKEN found. Bot disabled.")

    yield # Application runs here
    
    # --- SHUTDOWN ---
    logger.info("Shutdown: Cleaning up resources...")
    
    # Gracefully stop the bot to release the token
    if bot_app:
        try:
            logger.info("Shutdown: Stopping bot updater...")
            await bot_app.updater.stop()
            logger.info("Shutdown: Stopping bot application...")
            await bot_app.stop()
            await bot_app.shutdown()
            logger.info("Shutdown: Bot stopped successfully.")
        except Exception as e:
            logger.error(f"Shutdown: Error stopping bot: {e}")

# =====================================================
# APP INIT
# =====================================================

app = FastAPI(
    title="Aviator Premium API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS: Allow all for now (Adjust for strict production later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# STATIC ASSETS & HEALTH CHECKS
# =====================================================

@app.get("/health")
@app.head("/health")
async def health_check():
    """
    Dedicated endpoint for Cron Jobs / Uptime Monitors.
    Does NOT touch the DB or Bot, just returns 200 OK.
    Prevents 405 Method Not Allowed errors on HEAD requests.
    """
    # Simple check to ensure engine is loaded
    status = "alive" if engine else "error"
    return {"status": status}

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
async def serve_webapp(request: Request):
    """
    Serves the main Web App UI.
    Supports HEAD requests for Render health checks.
    """
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
    user = await get_or_create_user(session, payload.user_id)
    return {
        "user_id": user.telegram_id,
        "balance": float(user.balance),
    }

# =====================================================
# API – GAME CONTROL
# =====================================================

@app.post("/api/start-round")
async def api_start_round(payload: StartRoundRequest):
    try:
        result = await engine.start_new_round(payload.client_seed)
        return result
    except Exception as e:
        logger.warning(f"Start round failed (likely active): {e}")
        return await engine.get_current_state()

@app.get("/api/state")
async def api_state():
    return await engine.get_current_state()

# =====================================================
# API – BETTING & CASHOUT
# =====================================================

@app.post("/api/place-bet")
async def api_place_bet(
    payload: BetRequest,
    session: AsyncSession = Depends(get_session),
):
    user = await get_or_create_user(session, payload.user_id)
    bet_amount = Decimal(str(payload.amount))
    state = await engine.get_current_state()
    
    if state["status"] == "CRASHED":
         raise HTTPException(409, "Round already crashed")

    current_round_id = state.get("round_id")

    # DB Debit
    try:
        await debit(
            session=session,
            user=user,
            amount=bet_amount,
            round_id=current_round_id,
            reference="bet_entry"
        )
    except ValueError:
        raise HTTPException(402, "Insufficient funds")

    # Engine Registration
    try:
        await engine.place_bet(
            user_id=payload.user_id,
            amount=payload.amount,
            auto_cashout=payload.auto_cashout
        )
    except (StateError, ValueError) as e:
        # SAGA ROLLBACK
        logger.warning(f"Bet rejected by engine. Refunding {payload.user_id}. Reason: {e}")
        await credit(
            session=session,
            user=user,
            amount=bet_amount,
            round_id=current_round_id,
            reference="bet_refund_engine_reject"
        )
        raise HTTPException(status_code=409, detail=f"Bet rejected: {str(e)}")

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
    try:
        payout_amount = await engine.cashout(
            user_id=payload.user_id,
            client_observed_multiplier=payload.multiplier
        )
    except StateError:
        raise HTTPException(409, detail="Too late, plane crashed")
    except BetError:
        raise HTTPException(400, detail="Bet not found or already cashed")

    user = await get_or_create_user(session, payload.user_id)
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