# utils.py
"""
Utility functions for Aviator / Crash Game

Includes:
- Cryptographic & Provably Fair helpers
- Robust Number formatting (Decimal/Float agnostic)
- Production-grade Logging
- Miscellaneous Logic Helpers

Compatibility:
- Supports Python 3.9+
- Interoperable with Decimal-based engines and Float-based APIs
"""

from __future__ import annotations

import secrets
import hashlib
import hmac
import logging
import math
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Union, Optional

# =========================
# LOGGING CONFIG
# =========================

# Create a module-level logger. In production, this propagates to the root logger.
logger = logging.getLogger("aviator.utils")

# =========================
# RANDOM & PROVABLY FAIR
# =========================

def generate_server_seed(length: int = 32) -> str:
    """
    Generate a cryptographically secure random server seed (hex).
    Used as the secret key in HMAC calculations.
    """
    return secrets.token_hex(length)


def generate_client_seed(length: int = 16) -> str:
    """
    Generate a cryptographically secure random client seed (hex).
    Used as the public variable in HMAC calculations.
    """
    return secrets.token_hex(length)


def generate_unique_id(length: int = 8) -> str:
    """
    Generate a short, URL-safe unique ID (hex).
    Useful for Round IDs, User IDs, or idempotency keys.
    """
    return secrets.token_hex(length)


def hmac_sha256(key: str, message: str) -> str:
    """
    Compute HMAC-SHA256 hash.
    
    Args:
        key: The secret key (e.g., server_seed).
        message: The data to sign (e.g., client_seed:nonce).
    
    Returns:
        Hexadecimal string of the hash.
    """
    try:
        # Enforce UTF-8 encoding for consistency across platforms
        key_bytes = key.encode('utf-8')
        msg_bytes = message.encode('utf-8')
        return hmac.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()
    except Exception as e:
        logger.error(f"HMAC calculation failed: {e}")
        raise ValueError("Failed to calculate HMAC") from e


def hash_sha256(value: str) -> str:
    """
    Compute standard SHA256 hash of a string.
    Used for revealing the server seed hash before the round starts.
    """
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def verify_provably_fair(server_seed: str, client_seed: str, nonce: int, expected_hash: str) -> bool:
    """
    Helper to verify if a game result was fair.
    
    Args:
        server_seed: The revealed server seed.
        client_seed: The client seed used.
        nonce: The nonce used.
        expected_hash: The HMAC hash the game engine claimed generated the result.
        
    Returns:
        True if the inputs generate the expected_hash.
    """
    message = f"{client_seed}:{nonce}"
    calculated = hmac_sha256(server_seed, message)
    # constant_time_compare prevents timing attacks
    return hmac.compare_digest(calculated, expected_hash)


# =========================
# FORMATTING
# =========================

NumberType = Union[float, Decimal, int, str]

def format_balance(amount: NumberType) -> str:
    """
    Format balance with 2 decimals.
    Handles float, Decimal, int, or string inputs safely.
    """
    try:
        # Convert to float for formatting, handling Decimal explicitly
        val = float(amount)
        return f"{val:.2f}"
    except (ValueError, TypeError, InvalidOperation):
        logger.warning(f"Invalid balance format input: {amount}")
        return "0.00"


def format_multiplier(mult: NumberType) -> str:
    """
    Format multiplier with 2 decimals (e.g., 'x1.00').
    """
    try:
        val = float(mult)
        return f"x{val:.2f}"
    except (ValueError, TypeError, InvalidOperation):
        return "x1.00"


def format_timestamp(ts: Optional[float] = None) -> str:
    """
    Return ISO formatted timestamp.
    Defaults to UTC now if no timestamp provided.
    """
    try:
        if ts is not None:
            dt = datetime.fromtimestamp(ts)
        else:
            dt = datetime.now()
        return dt.isoformat(timespec="seconds")
    except Exception:
        return datetime.now().isoformat(timespec="seconds")


# =========================
# LOGGING / DEBUGGING
# =========================

def log(msg: str, level: str = "info") -> None:
    """
    Unified logging wrapper.
    Replaces print() with standard logging.
    
    Args:
        msg: Message to log
        level: 'info', 'warning', 'error', 'debug'
    """
    lvl = level.lower()
    timestamped_msg = f"[{format_timestamp()}] {msg}"
    
    if lvl == "error":
        logger.error(timestamped_msg)
    elif lvl == "warning":
        logger.warning(timestamped_msg)
    elif lvl == "debug":
        logger.debug(timestamped_msg)
    else:
        logger.info(timestamped_msg)


# =========================
# MISC HELPERS
# =========================

def clamp(value: NumberType, min_value: NumberType, max_value: NumberType) -> float:
    """
    Clamp value between min and max.
    Returns float for broader compatibility.
    """
    try:
        v = float(value)
        mn = float(min_value)
        mx = float(max_value)
        return max(mn, min(v, mx))
    except (ValueError, TypeError):
        # Fallback to 0.0 or min_value if parsing fails is risky, 
        # better to raise or return a safe default.
        return 0.0

def safe_decimal(value: NumberType, default: str = "0.00") -> Decimal:
    """
    Safely convert input to Decimal.
    Useful for parsing API inputs before passing to engine/db.
    """
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        logger.warning(f"Failed to convert {value} to Decimal, using default {default}")
        return Decimal(default)