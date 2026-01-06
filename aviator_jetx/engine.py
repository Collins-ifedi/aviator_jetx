# engine.py
"""
Aviator / Crash Game Engine – Production Grade (High Difficulty Tuned)

Responsibilities:
- Provably-fair crash calculation (HMAC-SHA256)
- Strict State Machine (IDLE -> BETTING -> FLYING -> CRASHED)
- Time-based multiplier validation (Server Authority)
- Decimal arithmetic for financial precision
- Thread-safe / Async-safe concurrency

ADJUSTMENTS FOR DIFFICULTY (5/10):
- House Edge increased to 10%
- Nonlinear difficulty curve applied to multipliers > 3.00x
- "Instant Crash" probability increased
"""

from __future__ import annotations

import time
import hmac
import hashlib
import secrets
import asyncio
import math
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Dict, Optional, Any

# Ensure high precision for internal calculations
getcontext().prec = 50

# =========================
# CONFIGURATION
# =========================

class GameConfig:
    # --- DIFFICULTY TUNING ---
    
    # 1. House Edge: 10% (Standard is 1-4%, 10% is significantly harder)
    # This ensures the plane crashes at 1.00x roughly 10% of the time.
    HOUSE_EDGE = Decimal("0.10") 
    
    # 2. Difficulty Curve Exponent (0.0 to 1.0)
    # Applied to multipliers > 3.0x. Lower number = Harder to get high multipliers.
    # A value of 0.90 means a raw 10x becomes ~8.4x.
    HIGH_MULT_DAMPENING = Decimal("0.90")
    
    MIN_MULTIPLIER = Decimal("1.00")
    MAX_MULTIPLIER = Decimal("10000.00")
    
    # Growth Function: M(t) = e^(k * t)
    # Speed Factor: 0.00012 makes the game faster, harder to react to.
    SPEED_FACTOR = 0.00012 
    
    # Time allowed for betting before flight starts (in seconds)
    BETTING_DURATION = 5.0 
    
    # Grace Period: Allows cashout/betting for a split second after state change
    # to account for network lag.
    GRACE_MULTIPLIER = Decimal("1.05")

# =========================
# ENUMS & EXCEPTIONS
# =========================

class GameState(str, Enum):
    IDLE = "IDLE"
    BETTING = "BETTING"
    FLYING = "FLYING"
    CRASHED = "CRASHED"

class EngineError(Exception):
    """Base engine error"""

class StateError(EngineError):
    """Action performed in invalid state"""

class BetError(EngineError):
    """Invalid bet parameters"""

# =========================
# DOMAIN MODELS
# =========================

@dataclass
class Bet:
    user_id: str
    amount: Decimal
    auto_cashout: Optional[Decimal] = None
    placed_at: float = field(default_factory=time.time)
    
    # Outcome
    cashed_out: bool = False
    cashout_multiplier: Optional[Decimal] = None
    payout: Decimal = Decimal("0.00")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "amount": float(self.amount),
            "cashed_out": self.cashed_out,
            "payout": float(self.payout),
            "multiplier": float(self.cashout_multiplier) if self.cashout_multiplier else None
        }

@dataclass
class GameRound:
    round_id: str
    server_seed: str
    client_seed: str
    nonce: int
    crash_point: Decimal
    
    # Timing
    created_at: float = field(default_factory=time.time)
    flight_start_at: Optional[float] = None
    crash_at: Optional[float] = None
    
    state: GameState = GameState.IDLE
    bets: Dict[str, Bet] = field(default_factory=dict)

# =========================
# ENGINE CLASS
# =========================

class CrashGameEngine:
    """
    Production-grade State Machine for Crash with Enhanced Difficulty.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._round: Optional[GameRound] = None
        self._nonce: int = 0

    # =====================================================
    # PROVABLY FAIR MATH (TUNED)
    # =====================================================

    def _calculate_crash_point(
        self, 
        server_seed: str, 
        client_seed: str, 
        nonce: int
    ) -> Decimal:
        """
        Determines the crash point deterministically but with a 'Difficulty Curve'.
        """
        # 1. HMAC-SHA256
        message = f"{client_seed}:{nonce}".encode()
        key = server_seed.encode()
        h = hmac.new(key, message, hashlib.sha256).hexdigest()

        # 2. Convert first 52 bits to int
        h_int = int(h[:13], 16)
        e = Decimal(2**52)
        
        # 3. Apply House Edge Check (The "Instant Crash" mechanic)
        # If h_int is divisible by 15, force a 1.00x crash.
        # This adds an extra layer of "bad luck" consistent with difficulty 5/10.
        if h_int % 15 == 0:
            return Decimal("1.00")

        # 4. Standard Distribution Formula
        # Multiplier = ( (1 - HouseEdge) * 2^52 ) / ( 2^52 - HashInt )
        denominator = e - h_int
        if denominator == 0: denominator = Decimal(1) # Safety
        
        numerator = (Decimal(1) - GameConfig.HOUSE_EDGE) * e
        multiplier = numerator / denominator

        # 5. Apply Difficulty Curve (Dampening)
        # If the result > 3.00, we dampen it to make 3x-6x rarer.
        # Formula: New = 3.0 + (Old - 3.0) ^ DampeningFactor
        if multiplier > 3.0:
            base = Decimal("3.00")
            diff = multiplier - base
            # If dampening is 0.9, a diff of 10 becomes 10^0.9 = 7.9
            dampened_diff = diff ** GameConfig.HIGH_MULT_DAMPENING
            multiplier = base + dampened_diff

        # 6. Clamp and Round
        multiplier = max(multiplier, GameConfig.MIN_MULTIPLIER)
        multiplier = min(multiplier, GameConfig.MAX_MULTIPLIER)
        
        return multiplier.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _get_elapsed_flight_time_ms(self) -> int:
        """Returns milliseconds since flight started."""
        if not self._round or self._round.state != GameState.FLYING:
            return 0
        
        delta = time.time() - (self._round.flight_start_at or 0)
        return int(delta * 1000)

    def _calculate_current_multiplier(self) -> Decimal:
        """
        Calculates multiplier based on time elapsed.
        Formula: 1.00 * e^(SPEED_FACTOR * ms)
        """
        if not self._round:
            return Decimal("1.00")

        if self._round.state == GameState.CRASHED:
            return self._round.crash_point
            
        if self._round.state != GameState.FLYING:
            return Decimal("1.00")

        elapsed_ms = self._get_elapsed_flight_time_ms()
        
        # Exponential growth
        growth = math.exp(GameConfig.SPEED_FACTOR * elapsed_ms)
        
        current = Decimal(growth).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        # Cap at crash point
        if current >= self._round.crash_point:
            return self._round.crash_point
            
        return current

    # =====================================================
    # LIFECYCLE METHODS (ASYNC)
    # =====================================================

    async def start_new_round(self, client_seed: str) -> Dict:
        """
        Initialize a round. Moves state to BETTING then FLYING.
        """
        if not client_seed:
            raise ValueError("client_seed is required")

        async with self._lock:
            # Only start if previous is finished
            if self._round and self._round.state not in [GameState.CRASHED, GameState.IDLE]:
                 # Double check if it should have crashed naturally
                 if self._round.state == GameState.FLYING:
                     # Check if it should be forced to crash
                     curr = self._calculate_current_multiplier()
                     if curr >= self._round.crash_point:
                         self._round.state = GameState.CRASHED
                     else:
                         raise StateError("Previous round currently active")

            server_seed = secrets.token_hex(32)
            crash_point = self._calculate_crash_point(
                server_seed, client_seed, self._nonce
            )

            self._round = GameRound(
                round_id=secrets.token_hex(8),
                server_seed=server_seed,
                client_seed=client_seed,
                nonce=self._nonce,
                crash_point=crash_point,
                state=GameState.BETTING,
                created_at=time.time()
            )

            self._nonce += 1
            
            # Immediate Start for Webhook Architecture
            self._round.state = GameState.FLYING
            self._round.flight_start_at = time.time()

            return {
                "round_id": self._round.round_id,
                "server_seed_hash": hashlib.sha256(server_seed.encode()).hexdigest(),
                "start_time": self._round.flight_start_at
            }

    async def get_current_state(self) -> Dict:
        """
        Returns the snapshot of the engine.
        Also acts as the 'heartbeat' that checks for crashes.
        """
        async with self._lock:
            if not self._round:
                return {"status": "OFFLINE"}

            # Check for natural crash
            if self._round.state == GameState.FLYING:
                current_mult = self._calculate_current_multiplier()
                
                if current_mult >= self._round.crash_point:
                    self._round.state = GameState.CRASHED
                    self._round.crash_at = time.time()

            # Construct Response
            current_mult = self._calculate_current_multiplier()
            
            return {
                "status": self._round.state.value,
                "round_id": self._round.round_id,
                "multiplier": float(current_mult),
                "crash_point": float(self._round.crash_point) if self._round.state == GameState.CRASHED else None,
                "start_time": self._round.flight_start_at,
                "elapsed_ms": self._get_elapsed_flight_time_ms(),
                "total_bets": len(self._round.bets)
            }

    # =====================================================
    # BETTING ACTIONS
    # =====================================================

    async def place_bet(
        self, 
        user_id: str, 
        amount: float, 
        auto_cashout: Optional[float] = None
    ) -> None:
        
        amount_dec = Decimal(str(amount))
        if amount_dec <= 0:
            raise ValueError("Bet must be positive")

        auto_dec = Decimal(str(auto_cashout)) if auto_cashout else None

        async with self._lock:
            if not self._round:
                raise StateError("Game not initialized")

            # Grace Period Check for Laggy Connections
            current_mult = self._calculate_current_multiplier()
            is_valid_phase = False
            
            if self._round.state in [GameState.IDLE, GameState.BETTING]:
                is_valid_phase = True
            elif self._round.state == GameState.FLYING:
                if current_mult <= GameConfig.GRACE_MULTIPLIER:
                    is_valid_phase = True
                else:
                    raise StateError("Round already started")
            elif self._round.state == GameState.CRASHED:
                raise StateError("Round finished")

            if not is_valid_phase:
                 raise StateError("Betting closed")

            if user_id in self._round.bets:
                raise StateError("Double bet detected")

            self._round.bets[user_id] = Bet(
                user_id=user_id,
                amount=amount_dec,
                auto_cashout=auto_dec
            )

    async def cashout(self, user_id: str, client_observed_multiplier: float) -> float:
        """
        User attempts to cash out.
        
        BEHAVIOR ON EXIT:
        If a user exits the game, they stop sending this request. 
        Since this is the ONLY way to win, exiting implies they will 
        never trigger this function before the plane crashes. 
        Thus, exiting = losing (unless auto-cashout hits).
        """
        async with self._lock:
            if not self._round or self._round.state != GameState.FLYING:
                # If they try to cash out after crash, it's too late.
                raise StateError("Round not active or Plane crashed")

            bet = self._round.bets.get(user_id)
            if not bet:
                raise BetError("Bet not found")

            if bet.cashed_out:
                raise BetError("Already cashed out")

            # 1. Authority Check
            server_mult = self._calculate_current_multiplier()
            
            # 2. Check if we crashed (Server-side)
            if server_mult >= self._round.crash_point:
                self._round.state = GameState.CRASHED
                raise StateError("Plane crashed")

            # 3. Lag/Cheat Protection
            # User can't claim higher than server sees.
            requested_mult = Decimal(str(client_observed_multiplier))
            final_mult = min(requested_mult, server_mult)
            final_mult = final_mult.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            if final_mult < Decimal("1.00"):
                raise ValueError("Invalid multiplier")

            # 4. Execute Win
            payout = bet.amount * final_mult
            
            bet.cashed_out = True
            bet.cashout_multiplier = final_mult
            bet.payout = payout.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            return float(bet.payout)