# app.py
"""
Aviator / Crash Game – Production Entry Point (Mini App Architecture)

Responsibilities:
- FastAPI HTTP server
- Telegram Bot Webhook Handler
- Game API orchestration with strict ACID compliance
- Serving the Web App
- BACKGROUND GAME LOOP (Drives the engine)

Updates:
- Strict JSON response consistency
- Atomic transaction management (Debit -> Bet -> [Rollback on Fail])
- Resilient background loop
"""

from __future__ import annotations

import os
import logging
import asyncio
from pathlib import Path
from decimal import Decimal
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, Union

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Response,
    status,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
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
from engine import CrashGameEngine, StateError, BetError, GameState
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

# Tuning
# Time (seconds) to hold the CRASHED state before resetting.
CRASH_COOLDOWN = 4.0 

# File Paths
ROOT_DIR = Path(__file__).parent
WEBAPP_FILE = ROOT_DIR / "webapp.html"

# Global Singletons
bot_app: Optional[Application] = None
engine = CrashGameEngine()

# =====================================================
# BACKGROUND GAME LOOP (The "Driver")
# =====================================================

async def run_game_loop():
    """
    Drives the engine state machine.
    
    Logic:
    1. Polls engine state frequently.
    2. If IDLE/OFFLINE: Auto-starts a new round.
    3. If CRASHED: Waits for cooldown, then restarts.
    4. If BETTING: Does nothing (Engine auto-transitions on timeout).
    5. If FLYING: Does nothing (Engine calculates physics internally).
    """
    logger.info("Game Loop: Started")
    
    while True:
        try:
            # We poll frequently to ensure UI responsiveness and state transitions
            state = await engine.get_current_state()
            status_str = state.get("status")

            if status_str == "IDLE" or status_str == "OFFLINE":
                logger.info("Game Loop: System Idle. Starting new round...")
                await engine.start_new_round()

            elif status_str == "CRASHED":
                # Ensure clients see the crash result for a few seconds
                logger.info(f"Game Loop: CRASHED at {state.get('multiplier')}x. Cooling down...")
                await asyncio.sleep(CRASH_COOLDOWN)
                
                # Reset
                logger.info("Game Loop: Resetting round...")
                await engine.start_new_round()

            elif status_str == "BETTING":
                # The engine handles the max duration timeout internally in `get_current_state`
                # We just wait a bit to avoid hot-looping
                await asyncio.sleep(0.5)

            elif status_str == "FLYING":
                # High frequency polling not strictly needed here for logic, 
                # but keeps the loop responsive.
                await asyncio.sleep(0.1)
                
            else:
                # Catch-all for unknown states
                await asyncio.sleep(1)

        except Exception as e:
            # CRITICAL: Do not let the loop die. Log and retry.
            logger.error(f"Game Loop Exception: {e}", exc_info=True)
            await asyncio.sleep(1)

# =====================================================
# LIFECYCLE (STARTUP & SHUTDOWN)
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages Database, Telegram, and the Background Game Loop.
    """
    global bot_app

    # 1. Database Init
    logger.info("Startup: Initializing Database...")
    await init_db()

    # 2. Start Game Engine Loop (Background Task)
    game_task = asyncio.create_task(run_game_loop())

    # 3. Telegram Bot Setup
    if BOT_TOKEN:
        logger.info("Startup: Building Telegram Bot...")
        bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
        
        bot_app.add_handler(CommandHandler("start", start_command))

        await bot_app.initialize()
        await bot_app.start()

        # Set Webhook if in production/accessible environment
        if BASE_URL and "localhost" not in BASE_URL and "127.0.0.1" not in BASE_URL:
            webhook_url = f"{BASE_URL}/telegram-webhook/{WEBHOOK_SECRET}"
            logger.info(f"Startup: Setting webhook to {webhook_url}")
            try:
                await bot_app.bot.set_webhook(
                    url=webhook_url,
                    secret_token=WEBHOOK_SECRET,
                    drop_pending_updates=True,
                    allowed_updates=Update.ALL_TYPES
                )
            except Exception as e:
                logger.error(f"Failed to set webhook: {e}")
    else:
        logger.warning("Startup: No BOT_TOKEN. Bot disabled.")

    yield 

    # 4. Shutdown Cleanup
    logger.info("Shutdown: Cleaning up...")
    
    game_task.cancel()
    try:
        await game_task
    except asyncio.CancelledError:
        pass

    if bot_app:
        await bot_app.stop()
        await bot_app.shutdown()

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Aviator Premium API",
    version="3.2.0",
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
# TELEGRAM COMMANDS & WEBHOOK
# =====================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /start command. 
    Sends the 'Play' button which opens the Mini App.
    """
    if not update.message:
        return

    web_app_info = WebAppInfo(url=f"{BASE_URL}/")

    keyboard = [
        [
            InlineKeyboardButton(text="🚀 PLAY AVIATOR X", web_app=web_app_info)
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text=(
            "<b>✈️ Aviator X - Production Mode</b>\n\n"
            "The plane is taking off. Place your bets!\n"
            "Provably fair mechanics active.\n\n"
            "👇 <b>Tap below to launch:</b>"
        ),
        parse_mode="HTML",
        reply_markup=reply_markup,
    )

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
# PYDANTIC MODELS (VALIDATION)
# =====================================================

class BetRequest(BaseModel):
    user_id: str = Field(..., description="Telegram User ID")
    amount: float = Field(..., gt=0, description="Bet amount must be positive")
    auto_cashout: Optional[float] = Field(None, gt=1.0, description="Auto cashout multiplier")

class CashoutRequest(BaseModel):
    user_id: str
    multiplier: float = Field(..., gt=1.0, description="Observed multiplier")

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    if not WEBAPP_FILE.exists():
        return HTMLResponse("<h1>Error: webapp.html not found</h1>", status_code=500)
    return WEBAPP_FILE.read_text(encoding="utf-8")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/api/init/{user_id}")
async def api_init(user_id: str, session: AsyncSession = Depends(get_session)):
    """
    Initialize User Session.
    Returns user balance and current game state snapshot.
    """
    try:
        user = await get_or_create_user(session, user_id)
        
        # Get engine state safely
        try:
            game_state = await engine.get_current_state()
        except Exception:
            # Fallback if engine is momentarily locked/busy
            game_state = {"status": "OFFLINE", "multiplier": 1.0}

        active_bet = engine.get_user_bet(int(user_id))
        
        return {
            "user_id": user.telegram_id,
            "balance": float(user.balance),
            "game_state": game_state,
            "active_bet": active_bet
        }
    except Exception as e:
        logger.error(f"Init Error: {e}")
        raise HTTPException(status_code=500, detail="Failed to initialize session")

@app.get("/api/state")
async def api_state():
    """
    High-frequency heartbeat.
    Returns: { "round_id": str, "state": str, "multiplier": float, ... }
    """
    try:
        return await engine.get_current_state()
    except Exception as e:
        logger.error(f"State Error: {e}")
        # Return a safe fallback to prevent frontend crash
        return {"status": "OFFLINE", "multiplier": 1.00}

@app.post("/api/place_bet")
async def api_place_bet(payload: BetRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Bet Placement.
    1. Debit DB (Atomic)
    2. Register Bet in Engine
    3. If Engine Fails -> Rollback DB
    """
    user_id = payload.user_id
    amount = Decimal(str(payload.amount))
    
    # 1. DB Transaction (Debit)
    user = await get_or_create_user(session, user_id)
    try:
        user = await debit(session, user, amount, reference="bet")
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED, 
            detail="Insufficient funds"
        )

    # 2. Engine Registration
    try:
        await engine.place_bet(
            int(user_id), 
            payload.amount, 
            payload.auto_cashout
        )
        
        # Trigger flight logic if engine allows it (e.g. restarts idle timer)
        await engine.trigger_flight()

    except (StateError, ValueError) as e:
        # 3. ROLLBACK (Refund)
        # If the bet failed (round started, closed, etc), we must refund.
        logger.warning(f"Bet failed for user {user_id}: {e}. Refunding.")
        await credit(session, user, amount, reference="refund_bet_error")
        
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail=str(e)
        )
    except Exception as e:
        # Unexpected error -> Refund
        logger.error(f"Unexpected bet error: {e}")
        await credit(session, user, amount, reference="refund_system_error")
        raise HTTPException(status_code=500, detail="System error during bet")

    return {
        "status": "accepted", 
        "balance": float(user.balance),
        "game_active": True, 
        "bet_id": f"b_{user_id}_{int(amount)}"
    }

@app.post("/api/cashout")
async def api_cashout(payload: CashoutRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Cashout.
    1. Validate with Engine (returns payout amount)
    2. Credit DB (Atomic)
    """
    user_id = payload.user_id
    
    # 1. Validate with Engine
    try:
        payout_amount = await engine.cashout(int(user_id), payload.multiplier)
    except StateError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, 
            detail="Plane crashed or round ended"
        )
    except BetError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Bet not found or already cashed out"
        )
    except Exception as e:
        logger.error(f"Cashout error: {e}")
        raise HTTPException(status_code=500, detail="Processing error")

    # 2. DB Transaction (Credit)
    try:
        user = await get_or_create_user(session, user_id)
        user = await credit(session, user, Decimal(str(payout_amount)), reference="win")
    except Exception as e:
        # Critical Error: Engine paid out, DB failed. 
        # In production, this needs a reconciliation queue. 
        # For now, we log heavily.
        logger.critical(f"DB CREDIT FAILED for user {user_id} amount {payout_amount}: {e}")
        raise HTTPException(status_code=500, detail="Payout record error. Contact support.")
    
    return {
        "status": "cashed_out",
        "win_amount": float(payout_amount),
        "balance": float(user.balance),
        "reset_ui": True
    }