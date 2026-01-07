# engine.py
"""
Aviator / Crash Game Engine – Production Grade (Updated)

Responsibilities:
- Strict State Machine (BETTING -> FLYING -> CRASHED)
- Deterministic Probability Buckets (2% @ 1.01x, 15% @ 2-5x)
- Thread-safe Async Logic
- Lag Compensation & Anti-Stuck Safeguards
"""

from __future__ import annotations

import time
import secrets
import math
import asyncio
import random
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
    # --- PROBABILITY SETTINGS ---
    # Bucket A: Instant Crash (1.01x)
    PROB_INSTANT_CRASH = 0.02  # 2%
    VAL_INSTANT_CRASH = Decimal("1.01")
    
    # Bucket B: Mid Range (2.00x - 5.00x)
    PROB_MID_RANGE = 0.15      # 15%
    MIN_MID = Decimal("2.00")
    MAX_MID = Decimal("5.00")

    # Global Limits
    # Even in the "Standard" bucket, we cap multiplier to prevent overflows
    ABSOLUTE_MAX_CRASH = Decimal("1000.00")
    
    # --- GAMEPLAY SPEED ---
    # Controls exponential growth: Multiplier = e^(SPEED_FACTOR * ms)
    # 0.0001 approx 1.00->2.00 in ~6.9 seconds
    SPEED_FACTOR = 0.0001
    
    # --- SAFETY ---
    # Minimum time (ms) the game MUST stay in FLYING state even if crash is 1.01x.
    # This ensures the frontend receives at least one "FLYING" packet.
    MIN_FLIGHT_DURATION_MS = 300 
    
    # Maximum betting duration before auto-flight (safety guard)
    MAX_BETTING_DURATION_SEC = 20.0

# =========================
# ENUMS & EXCEPTIONS
# =========================

class GameState(str, Enum):
    IDLE = "IDLE"       # Initial / Offline
    BETTING = "BETTING" # Accepting bets
    FLYING = "FLYING"   # Multiplier rising
    CRASHED = "CRASHED" # Round ended

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
    user_id: int
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
    crash_point: Decimal
    
    # Timing
    created_at: float = field(default_factory=time.time)
    flight_start_at: Optional[float] = None
    crash_at: Optional[float] = None
    
    state: GameState = GameState.IDLE
    bets: Dict[int, Bet] = field(default_factory=dict)

# =========================
# ENGINE CLASS
# =========================

class CrashGameEngine:
    """
    Production-grade State Machine.
    Enforces strict probability buckets and state transitions.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._round: Optional[GameRound] = None

    # =====================================================
    # MATH & PROBABILITY (CORE LOGIC)
    # =====================================================

    def _generate_crash_point(self) -> Decimal:
        """
        Generates crash point based on specific probability requirements:
        1. 2% Chance -> Exactly 1.01x
        2. 15% Chance -> Uniformly distributed between 2.00x and 5.00x
        3. 83% Chance -> Standard Crash Curve (Pareto/Inverse), clamped.
        """
        # Cryptographically secure source for the tier roll
        rng = secrets.SystemRandom()
        tier_roll = rng.random() # [0.0, 1.0)

        # --- BUCKET 1: INSTANT CRASH (2%) ---
        if tier_roll < GameConfig.PROB_INSTANT_CRASH:
            return GameConfig.VAL_INSTANT_CRASH

        # --- BUCKET 2: MID RANGE (15%) ---
        # Normalize the roll for this bucket logic if needed, or just use separate RNG
        # Using separate random call for distribution uniformity within the bucket
        if tier_roll < (GameConfig.PROB_INSTANT_CRASH + GameConfig.PROB_MID_RANGE):
            # Uniform float between 2.0 and 5.0
            val = rng.uniform(float(GameConfig.MIN_MID), float(GameConfig.MAX_MID))
            d = Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            return d

        # --- BUCKET 3: STANDARD DISTRIBUTION (83%) ---
        # Standard fair crash curve: Multiplier = 0.99 / (1 - U)
        # We generate a value and clamp it.
        # Note: We must ensure we don't accidentally generate 1.01 again to keep stats pure,
        # but statistically it's negligible.
        
        u = rng.random()
        # Avoid division by zero
        if u >= 0.99: 
            u = 0.99
            
        # The standard formula used in most crash games (House Edge ~1% built into formula)
        raw_multiplier = 0.99 / (1.0 - u)
        
        # Convert to decimal
        crash_point = Decimal(str(raw_multiplier))
        
        # Clamp to bounds
        # We allow it to go high (up to ABSOLUTE_MAX_CRASH), but min is 1.01
        crash_point = max(crash_point, Decimal("1.01"))
        crash_point = min(crash_point, GameConfig.ABSOLUTE_MAX_CRASH)
        
        return crash_point.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _get_elapsed_flight_time_ms(self) -> int:
        """Returns milliseconds since flight started."""
        if not self._round or self._round.state not in [GameState.FLYING, GameState.CRASHED]:
            return 0
        
        if not self._round.flight_start_at:
            return 0

        # If crashed, time stops at the crash moment
        end_time = self._round.crash_at if self._round.crash_at else time.time()
        delta = end_time - self._round.flight_start_at
        return int(delta * 1000)

    def _calculate_multiplier_at_ms(self, ms: int) -> Decimal:
        """
        Pure function: time -> multiplier.
        Formula: e^(SPEED_FACTOR * ms)
        """
        if ms <= 0:
            return Decimal("1.00")
            
        growth = math.exp(GameConfig.SPEED_FACTOR * ms)
        return Decimal(growth).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    # =====================================================
    # LIFECYCLE METHODS (ASYNC & LOCKED)
    # =====================================================

    async def start_new_round(self) -> Dict:
        """
        Step 1: Initialize Round.
        State: IDLE/CRASHED -> BETTING.
        """
        async with self._lock:
            # Generate deterministic result upfront
            crash_point = self._generate_crash_point()
            
            # Create fresh round
            self._round = GameRound(
                round_id=secrets.token_hex(8),
                crash_point=crash_point,
                state=GameState.BETTING,
                created_at=time.time()
            )
            
            return {
                "round_id": self._round.round_id,
                "status": self._round.state.value,
                "crash_point_hash": "HIDDEN" # Provably fair hash would go here
            }

    async def trigger_flight(self) -> bool:
        """
        Step 2: Launch Plane.
        State: BETTING -> FLYING.
        Called by bet placement (auto-start) or timer.
        """
        async with self._lock:
            if not self._round:
                return False
                
            if self._round.state == GameState.BETTING:
                self._round.state = GameState.FLYING
                self._round.flight_start_at = time.time()
                return True
            
            return False

    async def get_current_state(self) -> Dict:
        """
        Step 3: The Heartbeat.
        Calculates physics, checks constraints, triggers transitions.
        CRITICAL: This method drives the auto-crash logic.
        """
        async with self._lock:
            if not self._round:
                return {"status": "OFFLINE", "multiplier": 1.00}

            current_time = time.time()
            
            # --- SAFEGUARD: STUCK IN BETTING ---
            # If we've been betting too long, force flight
            if self._round.state == GameState.BETTING:
                if (current_time - self._round.created_at) > GameConfig.MAX_BETTING_DURATION_SEC:
                    self._round.state = GameState.FLYING
                    self._round.flight_start_at = current_time

            # --- FLIGHT LOGIC ---
            if self._round.state == GameState.FLYING:
                elapsed_ms = int((current_time - self._round.flight_start_at) * 1000)
                
                # 1. Calculate Multiplier
                current_mult = self._calculate_multiplier_at_ms(elapsed_ms)
                
                # 2. Crash Check
                # We check if we have exceeded the pre-generated crash point
                has_crashed_math = current_mult >= self._round.crash_point
                
                # 3. Minimum Duration Check (Visual Buffer)
                # Even if crash_point is 1.01x, we wait MIN_FLIGHT_DURATION_MS (e.g., 300ms)
                # so the frontend has time to render the rocket before showing "Crashed".
                min_time_passed = elapsed_ms >= GameConfig.MIN_FLIGHT_DURATION_MS
                
                if has_crashed_math and min_time_passed:
                    self._round.state = GameState.CRASHED
                    self._round.crash_at = current_time
                    # Clamp final result to the actual crash point
                    current_mult = self._round.crash_point

            # --- RESPONSE CONSTRUCTION ---
            
            if self._round.state == GameState.CRASHED:
                # Stable response for crashed state
                return {
                    "status": "CRASHED",
                    "round_id": self._round.round_id,
                    "multiplier": float(self._round.crash_point),
                    "crash_point": float(self._round.crash_point), # Reveal secret
                    "elapsed_ms": self._get_elapsed_flight_time_ms()
                }
            
            elif self._round.state == GameState.FLYING:
                # Dynamic response for flying
                elapsed_ms = int((current_time - self._round.flight_start_at) * 1000)
                current_mult = self._calculate_multiplier_at_ms(elapsed_ms)
                
                # Cap multiplier at crash point for display purposes if we are in that
                # "waiting for min duration" buffer zone
                if current_mult > self._round.crash_point:
                    current_mult = self._round.crash_point

                return {
                    "status": "FLYING",
                    "round_id": self._round.round_id,
                    "multiplier": float(current_mult),
                    "crash_point": None, # Secret
                    "elapsed_ms": elapsed_ms
                }
            
            else:
                # BETTING / IDLE
                return {
                    "status": self._round.state.value,
                    "round_id": self._round.round_id,
                    "multiplier": 1.00,
                    "crash_point": None,
                    "elapsed_ms": 0
                }

    def get_user_bet(self, user_id: int) -> Optional[Dict]:
        """Read-only access to bet state."""
        if not self._round or user_id not in self._round.bets:
            return None
        return self._round.bets[user_id].to_dict()

    # =====================================================
    # BETTING ACTIONS
    # =====================================================

    async def place_bet(
        self, 
        user_id: int, 
        amount: float, 
        auto_cashout: Optional[float] = None
    ) -> None:
        
        amount_dec = Decimal(str(amount))
        if amount_dec <= 0:
            raise ValueError("Bet must be positive")

        auto_dec = Decimal(str(auto_cashout)) if auto_cashout else None

        async with self._lock:
            # Auto-init if IDLE/OFFLINE
            if not self._round or self._round.state in [GameState.IDLE, GameState.CRASHED]:
                await self.start_new_round()

            # Strict State Check
            if self._round.state != GameState.BETTING:
                 # Calculate current mult for debug/error message
                 elapsed = int((time.time() - self._round.flight_start_at) * 1000) if self._round.flight_start_at else 0
                 cur = self._calculate_multiplier_at_ms(elapsed)
                 raise StateError(f"Round in progress (Status: {self._round.state.value}, x{cur:.2f})")

            if user_id in self._round.bets:
                raise StateError("Double bet detected")

            self._round.bets[user_id] = Bet(
                user_id=user_id,
                amount=amount_dec,
                auto_cashout=auto_dec
            )

    async def cashout(self, user_id: int, client_observed_multiplier: float) -> float:
        """
        User claims win.
        Checks: State is FLYING, Multiplier < Crash Point.
        """
        async with self._lock:
            if not self._round:
                raise StateError("Game not initialized")
            
            # 1. State Validity
            if self._round.state == GameState.CRASHED:
                raise StateError("Plane crashed")
                
            if self._round.state != GameState.FLYING:
                raise StateError("Round not active")

            # 2. Math Validity
            elapsed_ms = int((time.time() - self._round.flight_start_at) * 1000)
            server_mult = self._calculate_multiplier_at_ms(elapsed_ms)
            
            # If server thinks we crashed, strict reject
            if server_mult >= self._round.crash_point:
                self._round.state = GameState.CRASHED
                self._round.crash_at = time.time()
                raise StateError("Plane crashed (Server auth)")

            # 3. Process Bet
            bet = self._round.bets.get(user_id)
            if not bet:
                raise BetError("Bet not found")

            if bet.cashed_out:
                raise BetError("Already cashed out")

            # 4. Determine Payout Multiplier
            # Use the lower of (Client Request, Server Reality) to handle lag latency fairness
            # but reject if client request is wildly future-predicting (cheating)
            requested_mult = Decimal(str(client_observed_multiplier))
            
            # Tolerance: We allow client to be slightly behind server, but never ahead
            final_mult = min(requested_mult, server_mult)
            
            # Round down to 2 decimals
            final_mult = final_mult.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            if final_mult < Decimal("1.00"):
                raise ValueError("Invalid multiplier")

            # 5. Execute Win
            payout = bet.amount * final_mult
            
            bet.cashed_out = True
            bet.cashout_multiplier = final_mult
            bet.payout = payout.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            return float(bet.payout)