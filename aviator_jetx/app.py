# app.py
"""
Aviator / Crash Game – Production Entry Point (Webhook Mode)

Responsibilities:
- FastAPI HTTP server
- Telegram Bot Webhook Handler (No Polling)
- Game API orchestration
- Ledger-safe balance updates
- Static HTML serving

Configuration:
- Requires 'BOT_TOKEN' and 'BASE_URL' (Render URL) in env vars.
- Automatically sets webhook on startup.
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

# Global Bot Application
bot_app: Optional[Application] = None
engine = CrashGameEngine()

# =====================================================
# TELEGRAM HANDLERS
# =====================================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles /start command. 
    Sends a high-visibility Inline Button to launch the Web App.
    """
    if not update.message:
        return

    # Create the Inline Keyboard with the Web App button
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
            "✨ **Welcome to Aviator Premium!**\n\n"
            "Experience the thrill of the crash. Watch the multiplier rise, "
            "but don't wait too long—the plane could fly away at any second!\n\n"
            "👇 **Click below to enter the cockpit**"
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

        # Initialize Bot
        await bot_app.initialize()
        await bot_app.start()

        # Set Webhook (This replaces Polling)
        webhook_url = f"{BASE_URL}/telegram-webhook/{WEBHOOK_SECRET}"
        logger.info(f"Startup: Setting webhook to {webhook_url}")
        
        # 'drop_pending_updates=True' clears old conflicts/loops on deploy
        await bot_app.bot.set_webhook(
            url=webhook_url,
            secret_token=WEBHOOK_SECRET,
            drop_pending_updates=True,
            allowed_updates=Update.ALL_TYPES
        )
    else:
        logger.warning("Startup: No BOT_TOKEN. Bot disabled.")

    yield # App runs here

    # 3. Shutdown
    logger.info("Shutdown: Cleaning up...")
    if bot_app:
        # We generally do NOT delete the webhook on shutdown in serverless/container envs
        # to avoid losing messages during restarts, but we must stop the app.
        await bot_app.stop()
        await bot_app.shutdown()

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Aviator Premium API",
    version="2.1.0",
    lifespan=lifespan,
    docs_url=None, # Disable Swagger in Prod for stealth
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
    Receives updates from Telegram servers.
    """
    if not bot_app:
        return Response(status_code=500)

    try:
        # Verify Secret Token (Security Header)
        secret_header = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
        if secret_header != WEBHOOK_SECRET:
            return Response(status_code=403, content="Invalid Secret")

        # Process Update
        data = await request.json()
        update = Update.de_json(data, bot_app.bot)
        
        # Feed into python-telegram-bot application
        await bot_app.process_update(update)
        
        return Response(status_code=200)
    except Exception as e:
        logger.error(f"Webhook Error: {e}")
        # Always return 200 to Telegram to prevent them from retrying bad updates
        return Response(status_code=200)

# =====================================================
# Pydantic Models (Validation)
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
    return JSONResponse(status_code=409, content={"detail": str(exc)})

@app.exception_handler(BetError)
async def bet_error_handler(_, exc: BetError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# =====================================================
# HEALTH & STATIC
# =====================================================

@app.get("/health")
@app.head("/health")
async def health_check():
    """Uptime monitoring endpoint."""
    return {"status": "active", "mode": "webhook"}

@app.get("/", response_class=HTMLResponse)
@app.head("/", response_class=HTMLResponse)
async def serve_webapp():
    """Serves the Game UI."""
    if not WEBAPP_FILE.exists():
        return HTMLResponse("<h1>Error: webapp.html not found</h1>", status_code=500)
    return WEBAPP_FILE.read_text(encoding="utf-8")

# =====================================================
# GAME API ROUTES
# =====================================================

@app.post("/api/init")
async def api_init(payload: UserInitRequest, session: AsyncSession = Depends(get_session)):
    user = await get_or_create_user(session, payload.user_id)
    return {"user_id": user.telegram_id, "balance": float(user.balance)}

@app.post("/api/start-round")
async def api_start_round(payload: StartRoundRequest):
    try:
        return await engine.start_new_round(payload.client_seed)
    except Exception:
        return await engine.get_current_state()

@app.get("/api/state")
async def api_state():
    return await engine.get_current_state()

@app.post("/api/place-bet")
async def api_place_bet(payload: BetRequest, session: AsyncSession = Depends(get_session)):
    user = await get_or_create_user(session, payload.user_id)
    bet_amount = Decimal(str(payload.amount))
    
    # 1. Debit DB
    try:
        await debit(session, user, bet_amount, reference="bet")
    except ValueError:
        raise HTTPException(402, "Insufficient funds")

    # 2. Register in Engine
    try:
        await engine.place_bet(payload.user_id, payload.amount, payload.auto_cashout)
    except Exception as e:
        # Rollback
        await credit(session, user, bet_amount, reference="refund_error")
        raise HTTPException(409, detail=str(e))

    await session.refresh(user)
    return {"status": "accepted", "new_balance": float(user.balance)}

@app.post("/api/cashout")
async def api_cashout(payload: CashoutRequest, session: AsyncSession = Depends(get_session)):
    # 1. Validate with Engine
    try:
        payout = await engine.cashout(payload.user_id, payload.multiplier)
    except StateError:
        raise HTTPException(409, "Plane crashed")
    except BetError:
        raise HTTPException(400, "Invalid bet")

    # 2. Credit DB
    user = await get_or_create_user(session, payload.user_id)
    await credit(session, user, payout, reference=f"win_x{payload.multiplier}")
    
    return {
        "status": "cashed_out", 
        "payout": payout, 
        "balance": float(user.balance)
    }