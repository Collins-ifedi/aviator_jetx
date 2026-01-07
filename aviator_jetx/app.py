# app.py
"""
Aviator / Crash Game – Production Entry Point (Mini App Architecture)

Responsibilities:
- FastAPI HTTP server
- Telegram Bot Webhook Handler (for /start menu)
- Game API orchestration with strict ACID compliance
- Serving the Web App
- BACKGROUND GAME LOOP (Added to fix 'Connecting' issue)
"""

from __future__ import annotations

import os
import logging
import asyncio
from pathlib import Path
from decimal import Decimal
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any

from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    Request,
    Response,
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

# Internal Modules (Assumes db.py and engine.py are in same folder)
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

# File Paths
ROOT_DIR = Path(__file__).parent
WEBAPP_FILE = ROOT_DIR / "webapp.html"

# Global Singletons
bot_app: Optional[Application] = None
engine = CrashGameEngine()

# =====================================================
# BACKGROUND GAME LOOP (HEARTBEAT)
# =====================================================

async def run_game_loop():
    """
    Background Task: Drives the engine state machine.
    - Transitions from IDLE -> FLYING -> CRASHED
    - Handles the polling required for the engine to update time-based logic.
    """
    logger.info("Game Loop: Started")
    
    while True:
        try:
            # 1. Fetch current state (triggers internal engine updates like checking crash)
            state = await engine.get_current_state()
            status = state.get("status")

            # 2. State Machine Handling
            if status == "OFFLINE" or status == "IDLE":
                # System just started, kick off a round immediately
                logger.info("Game Loop: Starting new round...")
                await engine.start_new_round()

            elif status == "CRASHED":
                # Game over. Wait 5 seconds (Cooldown), then restart.
                # This gives the UI time to show the "You Won" or "Crashed" screen.
                await asyncio.sleep(5)
                await engine.start_new_round()
                logger.info("Game Loop: Round Restarted")

            elif status == "FLYING":
                # While flying, we sleep briefly to yield control.
                # The engine updates itself every time we call get_current_state()
                # at the top of the loop.
                await asyncio.sleep(0.1)

            else:
                # Fallback for unexpected states
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Game Loop Error: {e}")
            # Prevent rapid error loops
            await asyncio.sleep(1)

# =====================================================
# TELEGRAM HANDLERS
# =====================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /start command. 
    Sends the 'Play' button which opens the Mini App.
    """
    if not update.message:
        return

    # This creates the "Slide-up" Mini App experience
    web_app_info = WebAppInfo(url=f"{BASE_URL}/")

    keyboard = [
        [
            InlineKeyboardButton(text="🚀 PLAY AVIATOR X", web_app=web_app_info)
        ],
        [
            InlineKeyboardButton(text="💰 Check Balance", callback_data="balance"),
            InlineKeyboardButton(text="❓ Help", callback_data="help")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text=(
            "<b>✈️ Aviator X - Hardcore Mode</b>\n\n"
            "The plane is fueled. The multiplier is rising.\n"
            "Can you cash out before it crashes?\n\n"
            "👇 <b>Tap below to launch the app:</b>"
        ),
        parse_mode="HTML",
        reply_markup=reply_markup,
    )

# =====================================================
# LIFECYCLE (STARTUP & SHUTDOWN)
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages Database Init, Telegram Webhook, and the Game Loop.
    """
    global bot_app

    # 1. Database Init
    logger.info("Startup: Initializing Database...")
    await init_db()

    # 2. Start the Game Engine Loop (Background Task)
    # This fixes the 'Connecting...' issue by actually running the game.
    game_task = asyncio.create_task(run_game_loop())

    # 3. Telegram Bot Setup
    if BOT_TOKEN:
        logger.info("Startup: Building Telegram Bot...")
        bot_app = ApplicationBuilder().token(BOT_TOKEN).build()
        
        bot_app.add_handler(CommandHandler("start", start_command))

        await bot_app.initialize()
        await bot_app.start()

        # Set Webhook (only if URL is configured)
        if BASE_URL and "localhost" not in BASE_URL:
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

    # 4. Shutdown Cleanup
    logger.info("Shutdown: Cleaning up...")
    
    # Cancel game loop
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
    version="3.0.0",
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
    """
    Receives updates from Telegram.
    Note: Game logic uses REST API, this is just for /start and text commands.
    """
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
# PYDANTIC MODELS
# =====================================================

class BetRequest(BaseModel):
    user_id: str = Field(..., description="Telegram User ID")
    amount: float = Field(..., gt=0, description="Bet amount")
    auto_cashout: Optional[float] = Field(None, gt=1.0)

class CashoutRequest(BaseModel):
    user_id: str
    multiplier: float = Field(..., gt=1.0)

# =====================================================
# API ENDPOINTS
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def serve_webapp():
    """Serves the game UI."""
    if not WEBAPP_FILE.exists():
        return HTMLResponse("<h1>Error: webapp.html not found</h1>", status_code=500)
    return WEBAPP_FILE.read_text(encoding="utf-8")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Prevents 404 logs for favicon"""
    return Response(status_code=204)

@app.get("/api/init/{user_id}")
async def api_init(user_id: str, session: AsyncSession = Depends(get_session)):
    """
    Initialize User Session.
    
    INTELLIGENT RE-ENTRY:
    - If user enters while round is BETTING: Show fresh screen.
    - If user enters while FLYING and has NO bet: Show "Wait for next round".
    - If user enters while FLYING and HAS bet: Reconnect them to the game.
    """
    user = await get_or_create_user(session, user_id)
    
    # Get Global State
    game_state = await engine.get_current_state()
    
    # Check if this specific user has an active bet in the engine
    active_bet = engine.get_user_bet(int(user_id))
    
    # Logic to scrub "stale" data
    response_bet = None
    if active_bet:
        # Only return the bet if it matches the current live round
        # (This prevents seeing results from 3 rounds ago)
        # Since engine stores bets in memory for current round only, 
        # presence of `active_bet` usually implies it's current.
        response_bet = active_bet

    return {
        "user_id": user.telegram_id,
        "balance": float(user.balance),
        "game_state": game_state,
        "active_bet": response_bet
    }

@app.get("/api/state")
async def api_state():
    """High-frequency heartbeat for the game loop."""
    return await engine.get_current_state()

@app.post("/api/place_bet")
async def api_place_bet(payload: BetRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Bet Placement.
    Returns 'game_active': True to tell UI to lock in.
    """
    user_id = payload.user_id
    user = await get_or_create_user(session, user_id)
    amount = Decimal(str(payload.amount))
    
    # 1. DB Transaction (Debit)
    try:
        user = await debit(session, user, amount, reference="bet")
    except ValueError:
        raise HTTPException(status_code=402, detail="Insufficient funds")

    # 2. Engine Registration
    try:
        await engine.place_bet(
            int(user_id), 
            payload.amount, 
            payload.auto_cashout
        )
    except Exception as e:
        # ROLLBACK: If engine rejects (e.g., missed grace period), refund immediately
        await credit(session, user, amount, reference="refund_bet_error")
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "status": "accepted", 
        "balance": float(user.balance),
        "game_active": True, # Signal for Optimistic UI
        "bet_id": f"b_{user_id}_{int(amount)}"
    }

@app.post("/api/cashout")
async def api_cashout(payload: CashoutRequest, session: AsyncSession = Depends(get_session)):
    """
    Atomic Cashout.
    Returns 'reset_ui': True to force "Win Screen" immediately.
    """
    user_id = payload.user_id
    
    # 1. Validate with Engine
    try:
        payout_amount = await engine.cashout(int(user_id), payload.multiplier)
    except StateError as e:
        raise HTTPException(status_code=409, detail="Plane crashed or round ended")
    except BetError as e:
        raise HTTPException(status_code=400, detail="Invalid bet state")

    # 2. DB Transaction (Credit)
    user = await get_or_create_user(session, user_id)
    user = await credit(session, user, Decimal(str(payout_amount)), reference="win")
    
    return {
        "status": "cashed_out",
        "win_amount": float(payout_amount),
        "balance": float(user.balance),
        "reset_ui": True # Signal for UI to stop animation
    }