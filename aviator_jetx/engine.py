# engine.py
"""
Aviator / Crash Game Engine – Production Grade

Responsibilities:
- Provably-fair crash calculation (HMAC-SHA256)
- Strict State Machine (IDLE -> BETTING -> FLYING -> CRASHED)
- Time-based multiplier validation (Server Authority)
- Decimal arithmetic for financial precision
- Thread-safe / Async-safe concurrency

IMPORTANT:
- Internal Math uses Decimal
- External API accepts/returns float for compatibility
"""

from __future__ import annotations

import time
import hmac
import hashlib
import secrets
import asyncio
import math
from enum import Enum
from dataclasses import dataclass, field, asdict
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Dict, Optional, Any

# Ensure high precision for internal calculations
getcontext().prec = 50

# =========================
# CONFIGURATION
# =========================

class GameConfig:
    # 4% House Edge to ensure sustainable profitability
    HOUSE_EDGE = Decimal("0.04") 
    
    MIN_MULTIPLIER = Decimal("1.00")
    MAX_MULTIPLIER = Decimal("10000.00")
    
    # Growth Function: M(t) = e^(k * t)
    # Increased to 0.00008 for faster game pacing (harder for users to react)
    SPEED_FACTOR = 0.00008 
    
    # Time allowed for betting before flight starts (in seconds)
    BETTING_DURATION = 5.0 
    
    # Network Latency Grace Period
    # We accept bets for a split second after takeoff to prevent "Round Started"
    # errors for users with slight lag.
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
    Production-grade State Machine for Crash.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._round: Optional[GameRound] = None
        self._nonce: int = 0

    # =====================================================
    # PROVABLY FAIR MATH
    # =====================================================

    def _calculate_crash_point(
        self, 
        server_seed: str, 
        client_seed: str, 
        nonce: int
    ) -> Decimal:
        """
        Determines the crash point deterministically using 1/X distribution.
        
        Formula:
          Multiplier = ( (1 - HouseEdge) * 2^52 ) / ( 2^52 - HashInt )
        
        This creates a natural distribution where:
        1. The probability of crashing at 1.00x is exactly equal to the House Edge.
        2. High multipliers become exponentially rarer.
        """
        # 1. HMAC-SHA256
        message = f"{client_seed}:{nonce}".encode()
        key = server_seed.encode()
        h = hmac.new(key, message, hashlib.sha256).hexdigest()

        # 2. Convert first 52 bits (13 hex chars) to int
        # 2^52 = 4,503,599,627,370,496
        h_int = int(h[:13], 16)
        e = Decimal(2**52)
        
        # 3. Apply House Edge Formula
        # If h_int is close to 0, result is approx 1 - HouseEdge (Instant Crash)
        # If h_int is close to 2^52, result approaches infinity.
        
        # Denominator: (2^52 - h_int). 
        # Since h_int is strictly < 2^52, this is never zero.
        denominator = e - h_int
        
        numerator = (Decimal(1) - GameConfig.HOUSE_EDGE) * e
        
        multiplier = numerator / denominator

        # 4. Clamp and Round
        # Any result calculated as < 1.00 becomes 1.00 (Instant loss)
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
            # If crashed, we return the crash point, NOT the time-based value
            return self._round.crash_point
            
        if self._round.state != GameState.FLYING:
            return Decimal("1.00")

        elapsed_ms = self._get_elapsed_flight_time_ms()
        
        # Exponential growth logic
        # Using standard math.exp then converting to Decimal
        growth = math.exp(GameConfig.SPEED_FACTOR * elapsed_ms)
        
        current = Decimal(growth).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        
        # Safety: current multiplier cannot exceed actual crash point (if we lagged)
        if current >= self._round.crash_point:
            return self._round.crash_point
            
        return current

    # =====================================================
    # LIFECYCLE METHODS (ASYNC)
    # =====================================================

    async def start_new_round(self, client_seed: str) -> Dict:
        """
        Initialize a round. Moves state to BETTING.
        """
        if not client_seed:
            raise ValueError("client_seed is required")

        async with self._lock:
            # Enforce sequential rounds
            if self._round and self._round.state not in [GameState.CRASHED, GameState.IDLE]:
                 if not self._round.state == GameState.CRASHED:
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
            
            # Simulate Immediate Flight Start for this architecture
            self._round.state = GameState.FLYING
            self._round.flight_start_at = time.time()

            return {
                "round_id": self._round.round_id,
                "server_seed_hash": hashlib.sha256(server_seed.encode()).hexdigest(),
                "start_time": self._round.flight_start_at
            }

    async def end_round(self) -> Dict:
        """
        Forcefully ends round (Debugging/Safety).
        """
        async with self._lock:
            if not self._round:
                return {}
            
            self._round.state = GameState.CRASHED
            self._round.crash_at = time.time()
            
            return {
                "round_id": self._round.round_id,
                "crash_point": float(self._round.crash_point),
                "server_seed": self._round.server_seed
            }

    async def get_current_state(self) -> Dict:
        """
        Returns the snapshot of the engine.
        Calculates if crash has occurred based on time.
        """
        async with self._lock:
            if not self._round:
                return {"status": "OFFLINE"}

            # Check if we naturally crashed based on time
            if self._round.state == GameState.FLYING:
                current_mult = self._calculate_current_multiplier()
                
                # Has the time-based multiplier exceeded the pre-calculated crash point?
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

            # 1. Grace Period Check
            # We allow betting in FLYING state ONLY if the plane just took off (<= 1.05x).
            # This handles network latency where client saw 1.00x but server is at 1.03x.
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
        SERVER AUTHORITY is strictly enforced here.
        """
        async with self._lock:
            if not self._round or self._round.state != GameState.FLYING:
                raise StateError("Round not active")

            bet = self._round.bets.get(user_id)
            if not bet:
                raise BetError("Bet not found")

            if bet.cashed_out:
                raise BetError("Already cashed out")

            # 1. Authority Check: What is the ACTUAL multiplier right now?
            server_mult = self._calculate_current_multiplier()
            
            # 2. Check if we crashed already
            if server_mult >= self._round.crash_point:
                self._round.state = GameState.CRASHED
                raise StateError("Plane crashed")

            # 3. Lag Protection / Anti-Cheat
            # If client says "I cashed at 10x" but server says "It's only 5x", they are lying (Cheat).
            # If client says "I cashed at 5x" but server says "It's 5.5x", they lagged. We accept the 5x.
            requested_mult = Decimal(str(client_observed_multiplier))
            
            # STRICT: We never pay out more than the server sees.
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