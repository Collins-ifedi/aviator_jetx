# app.py
"""
Aviator / Crash Game – Production Entry Point (Webhook Mode)

Responsibilities:
- FastAPI HTTP server
- Telegram Bot Webhook Handler
- Game API orchestration with strict ACID compliance
- Enforcement of "Stateless Re-entry" logic

Updates for User Requirements:
1. Optimized `api_place_bet` for zero-latency UI switching.
2. `api_cashout` triggers explicit client-side reset.
3. `api_init` strictly ignores past state to ensure fresh sessions.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from decimal import Decimal
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

# Telegram Imports
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup, 
    WebAppInfo
)
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)

# Internal Modules
from engine import CrashGameEngine, StateError, BetError
from db import (
    init_db,
    get_session,
    get_or_create_user,
    debit,
    credit,
)

# =====================================================
# CONFIGURATION
# =====================================================

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("aviator_app")

# Environment Variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
BASE_URL = os.getenv("BASE_URL", "https://your-app.onrender.com").rstrip('/')
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "secret-token-change-me")

if not BOT_TOKEN:
    logger.error("CRITICAL: BOT_TOKEN is missing!")

# File Paths
ROOT_DIR = Path(__file__).parent
WEBAPP_FILE = ROOT_DIR / "webapp.html"

# Global Bot Application & Engine
bot_app: Optional[Application] = None
engine = CrashGameEngine()

# =====================================================
# TELEGRAM HANDLERS
# =====================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /start command. 
    Sends the Web App button.
    """
    if not update.message:
        return

    keyboard = [
        [
            InlineKeyboardButton(
                text="🚀 START FLYING", 
                web_app=WebAppInfo(url=BASE_URL)
            )
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text=(
            "✨ **Aviator Premium - Hardcore Mode**\n\n"
            "The plane is fueled and the stakes are high.\n"
            "⚠️ **Rule:** If you exit the app, your connection resets and any active bet is forfeited.\n\n"
            "👇 **Click below to take off**"
        ),
        parse_mode="Markdown",
        reply_markup=reply_markup,
    )

# =====================================================
# LIFECYCLE (STARTUP & SHUTDOWN)
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages Database Init and Telegram Webhook Setup.
    """
    global bot_app

    # 1. Database Init
    logger.info("Startup: Initializing Database...")
    await init_db()

    # 2. Telegram Bot Setup
    if BOT_TOKEN:
        logger.info("Startup: Building Telegram Bot...")
        bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
        
        bot_app.add_handler(CommandHandler("start", start_command))

        await bot_app.initialize()
        await bot_app.start()

        # Set Webhook
        webhook_url = f"{BASE_URL}/telegram-webhook/{WEBHOOK_SECRET}"
        logger.info(f"Startup: Setting webhook to {webhook_url}")
        
        await bot_app.bot.set_webhook(
            url=webhook_url,
            secret_token=WEBHOOK_SECRET,
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
    else:
        logger.warning("Startup: No BOT_TOKEN. Bot disabled.")

    yield 

    # 3. Shutdown
    logger.info("Shutdown: Cleaning up...")
    if bot_app:
        await bot_app.stop()
        await bot_app.shutdown()

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Aviator Premium API",
    version="2.6.0",
    lifespan=lifespan,
    docs_url=None, 
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# WEBHOOK ENDPOINT
# =====================================================

@app.post(f"/telegram-webhook/{WEBHOOK_SECRET}")
async def telegram_webhook(request: Request):
    if not bot_app:
        return Response(status_code=500)

    try:
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if secret_header != WEBHOOK_SECRET:
            return Response(status_code=403, content="Invalid Secret")

        data = await request.json()
        update = Update.de_json(data, bot_app.bot)
        await bot_app.process_update(update)
        
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        return Response(status_code=200)

# =====================================================
# Pydantic Models
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
# ERROR HANDLERS
# =====================================================

@app.exception_handler(StateError)
async def state_error_handler(_, exc: StateError):
    # 409 Conflict is the signal for "Plane Crashed" or "Too Late"
    return JSONResponse(status_code=409, content={"detail": str(exc)})

@app.exception_handler(BetError)
async def bet_error_handler(_, exc: BetError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/health")
async def health_check():
    return {"status": "active", "mode": "production"}

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    if not WEBAPP_FILE.exists():
        return HTMLResponse("<h1>Error: webapp.html not found</h1>", status_code=500)
    return WEBAPP_FILE.read_text(encoding="utf-8")

# --- REQUIREMENT #3: STATELESS RE-ENTRY ---
@app.post("/api/init")
async def api_init(payload: UserInitRequest, session: AsyncSession = Depends(get_session)):
    """
    Initialize user session.
    
    CRITICAL BEHAVIOR: 
    This only returns the Balance and ID. It deliberately does NOT check for 
    active bets or past round results. This ensures that if a user leaves 
    and comes back, they start 'fresh' (clean slate), fulfilling the requirement 
    to not show past game results.
    """
    user = await get_or_create_user(session, payload.user_id)
    return {
        "user_id": user.telegram_id, 
        "balance": float(user.balance),
        "session_fresh": True 
    }

@app.post("/api/start-round")
async def api_start_round(payload: StartRoundRequest):
    try:
        return await engine.start_new_round(payload.client_seed)
    except Exception:
        # If round active, ignore safely
        return await engine.get_current_state()

@app.get("/api/state")
async def api_state():
    """
    Global Game State (Heartbeat).
    Returns the flying multiplier for all users.
    """
    return await engine.get_current_state()

# --- REQUIREMENT #1: IMMEDIATE CASHOUT READINESS ---
@app.post("/api/place-bet")
async def api_place_bet(payload: BetRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Bet Placement.
    
    BEHAVIOR:
    This endpoint is optimized to be the single 'Gatekeeper'. 
    Once this returns 200 OK, the Frontend is guaranteed that the bet is 
    registered in the Engine. The Frontend should immediately swap the 
    'BET' button to 'CASHOUT' upon receiving this response.
    """
    user = await get_or_create_user(session, payload.user_id)
    bet_amount = Decimal(str(payload.amount))
    
    # 1. Debit DB (ACID Transaction)
    try:
        # 'bet' reference allows for audit trails
        await debit(session, user, bet_amount, reference="bet")
    except ValueError:
        raise HTTPException(402, "Insufficient funds")

    # 2. Register in Engine (In-Memory Game State)
    try:
        await engine.place_bet(payload.user_id, payload.amount, payload.auto_cashout)
    except Exception as e:
        # ROLLBACK: If engine rejects (e.g., plane crashed ms ago), refund user.
        await credit(session, user, bet_amount, reference="refund_error")
        raise HTTPException(409, detail=str(e))

    await session.refresh(user)
    
    return {
        "status": "accepted", 
        "new_balance": float(user.balance),
        "game_active": True # Explicit signal for UI to show Cashout
    }

# --- REQUIREMENT #2: GAME RESET ON CASHOUT ---
@app.post("/api/cashout")
async def api_cashout(payload: CashoutRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Cashout.
    
    BEHAVIOR:
    On success, this returns 'reset_game: True'. 
    This explicitly instructs the Frontend to stop tracking the flight 
    and reset the user's interface to the 'Waiting' state, 
    even if the global plane is still flying.
    """
    # 1. Validate with Engine (Check if crashed)
    try:
        payout = await engine.cashout(payload.user_id, payload.multiplier)
    except StateError:
        # Plane crashed before request arrived
        raise HTTPException(409, "Plane crashed")
    except BetError:
        raise HTTPException(400, "Invalid bet or already cashed out")

    # 2. Credit DB (ACID Transaction)
    user = await get_or_create_user(session, payload.user_id)
    await credit(session, user, payout, reference=f"win_x{payload.multiplier}")
    
    return {
        "status": "cashed_out", 
        "payout": payout, 
        "balance": float(user.balance),
        "reset_game": True # <-- Enforces Requirement #2
    }