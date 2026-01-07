# engine.py
"""
Aviator / Crash Game Engine – Risk Managed Production Grade

Responsibilities:
- Strict Probability Distribution (Risk Control)
- Max Multiplier Cap (5.00x)
- Dynamic Exposure Limiting (Anti-Bankruptcy)
- State Machine (BETTING -> FLYING -> CRASHED)
- Thread-safe Async Logic with Automatic Recovery
"""

from __future__ import annotations

import time
import secrets
import math
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Dict, Optional, Any

# Ensure high precision for internal calculations
getcontext().prec = 50

# =========================
# CONFIGURATION & RISK TUNING
# =========================

class GameConfig:
    # --- RISK & PROBABILITY SETTINGS ---
    # Total must sum to 100% (1.0)
    
    # 1. Instant Crash (House Edge Enforcer)
    # 2% chance to crash exactly at 1.01x
    PROB_INSTANT = 0.02
    
    # 2. Low Range (House Profit Zone)
    # 80% chance to crash between 1.02x and 1.99x
    PROB_LOW = 0.80
    
    # 3. Mid Range (Player Engagement)
    # 8% chance to crash between 2.00x and 3.00x
    PROB_MID = 0.08
    
    # 4. High Range (Excitement)
    # 9% chance to crash between 3.01x and 4.00x
    # (Note: Using contiguous range 3.01-4.00 to prevent probability gaps)
    PROB_HIGH = 0.09
    
    # 5. Jackpot Range (Rare Wins)
    # 1% chance to crash between 4.01x and 5.00x
    PROB_JACKPOT = 0.01

    # Absolute hard cap for safety
    MAX_MULTIPLIER_CAP = Decimal("5.00")
    
    # --- BANKRUPTCY PROTECTION ---
    # If the total potential payout of all active bets exceeds this amount,
    # the engine will force a crash to protect the platform.
    MAX_EXPOSURE_LIMIT = Decimal("10000.00") 
    
    # --- GAMEPLAY SPEED ---
    # Controls exponential growth: Multiplier = e^(SPEED_FACTOR * ms)
    SPEED_FACTOR = 0.0001
    
    # --- TIMING ---
    # Minimum time (ms) the game MUST stay in FLYING state
    MIN_FLIGHT_DURATION_MS = 300 
    
    # Maximum betting duration before auto-flight
    MAX_BETTING_DURATION_SEC = 10.0
    
    # Time to wait after a crash before allowing a new round (visual cooldown)
    POST_CRASH_COOLDOWN_SEC = 3.0

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
    Risk-Managed State Machine.
    Enforces strict probability buckets and prevents platform bankruptcy.
    Thread-safe and auto-recovering.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._round: Optional[GameRound] = None

    # =====================================================
    # MATH & PROBABILITY (RISK CONTROL)
    # =====================================================

    def _generate_weighted_crash_point(self) -> Decimal:
        """
        Generates a crash point based on defined risk buckets.
        Uses `secrets.SystemRandom` for cryptographic strength.
        """
        rng = secrets.SystemRandom()
        roll = rng.random()  # [0.0, 1.0)
        
        # Cumulative probability thresholds
        t_instant = GameConfig.PROB_INSTANT
        t_low = t_instant + GameConfig.PROB_LOW
        t_mid = t_low + GameConfig.PROB_MID
        t_high = t_mid + GameConfig.PROB_HIGH
        
        # --- BUCKET 1: INSTANT CRASH (2%) ---
        if roll < t_instant:
            return Decimal("1.01")
            
        # --- BUCKET 2: LOW RANGE 1.02x - 1.99x (80%) ---
        elif roll < t_low:
            val = rng.uniform(1.02, 1.99)
            return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            
        # --- BUCKET 3: MID RANGE 2.00x - 3.00x (8%) ---
        elif roll < t_mid:
            val = rng.uniform(2.00, 3.00)
            return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            
        # --- BUCKET 4: HIGH RANGE 3.01x - 4.00x (9%) ---
        elif roll < t_high:
            val = rng.uniform(3.01, 4.00)
            return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
            
        # --- BUCKET 5: JACKPOT RANGE 4.01x - 5.00x (1%) ---
        else:
            val = rng.uniform(4.01, float(GameConfig.MAX_MULTIPLIER_CAP))
            return Decimal(str(val)).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    def _check_exposure_risk(self, current_multiplier: Decimal) -> bool:
        """
        Checks if the current multiplier would cause total potential payout
        to exceed the maximum allowed exposure for the platform.
        """
        if not self._round:
            return False

        current_exposure = Decimal("0.00")
        
        for bet in self._round.bets.values():
            if not bet.cashed_out:
                # Calculate liability if they cashed out NOW
                current_exposure += bet.amount * current_multiplier

        return current_exposure > GameConfig.MAX_EXPOSURE_LIMIT

    def _get_elapsed_flight_time_ms(self) -> int:
        """Returns milliseconds since flight started."""
        if not self._round or self._round.state not in [GameState.FLYING, GameState.CRASHED]:
            return 0
        
        if not self._round.flight_start_at:
            return 0

        # If crashed, time stops at the crash moment
        end_time = self._round.crash_at if self._round.crash_at else time.time()
        delta = end_time - self._round.flight_start_at
        return int(max(0, delta * 1000))

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
    # INTERNAL HELPERS (Lock Assumed)
    # =====================================================

    def _internal_start_new_round(self) -> Dict:
        """
        Internal logic to create a round.
        Assumes self._lock is already acquired by the caller.
        """
        # Generate risk-managed result
        crash_point = self._generate_weighted_crash_point()
        
        self._round = GameRound(
            round_id=secrets.token_hex(8),
            crash_point=crash_point,
            state=GameState.BETTING,
            created_at=time.time()
        )
        return {
            "round_id": self._round.round_id,
            "status": self._round.state.value
        }

    # =====================================================
    # PUBLIC ASYNC METHODS (Thread Safe)
    # =====================================================

    async def start_new_round(self) -> Dict:
        """
        Manually trigger a new round.
        """
        async with self._lock:
            return self._internal_start_new_round()

    async def trigger_flight(self) -> bool:
        """
        Transition: BETTING -> FLYING.
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
        The Heartbeat of the game.
        Calculates physics, checks constraints, triggers transitions, and handles auto-recovery.
        """
        async with self._lock:
            # 1. Auto-Boot if system is completely dead
            if not self._round:
                self._internal_start_new_round()

            current_time = time.time()
            
            # 2. AUTO-RECOVERY: Stuck in CRASHED?
            # If the game has been crashed longer than the cooldown, reset automatically.
            if self._round.state == GameState.CRASHED:
                if self._round.crash_at and (current_time - self._round.crash_at) > GameConfig.POST_CRASH_COOLDOWN_SEC:
                    self._internal_start_new_round()
                    # Fall through to return BETTING state immediately

            # 3. Transition: BETTING -> FLYING (Timeout)
            if self._round.state == GameState.BETTING:
                if (current_time - self._round.created_at) > GameConfig.MAX_BETTING_DURATION_SEC:
                    self._round.state = GameState.FLYING
                    self._round.flight_start_at = current_time

            # 4. Physics: FLYING logic
            if self._round.state == GameState.FLYING:
                elapsed_ms = int((current_time - self._round.flight_start_at) * 1000)
                
                # Calculate Multiplier
                current_mult = self._calculate_multiplier_at_ms(elapsed_ms)
                
                # Risk Checks
                force_crash = self._check_exposure_risk(current_mult)
                has_reached_target = current_mult >= self._round.crash_point
                has_reached_cap = current_mult >= GameConfig.MAX_MULTIPLIER_CAP
                
                # Minimum Duration Safeguard (Prevents instant invisible crashes)
                min_time_passed = elapsed_ms >= GameConfig.MIN_FLIGHT_DURATION_MS
                
                if (has_reached_target or has_reached_cap or force_crash) and min_time_passed:
                    self._round.state = GameState.CRASHED
                    self._round.crash_at = current_time
                    
                    # Update crash point if forced by risk engine
                    if force_crash or has_reached_cap:
                        self._round.crash_point = min(current_mult, GameConfig.MAX_MULTIPLIER_CAP)
                    else:
                        current_mult = self._round.crash_point

            # 5. Build Response
            if self._round.state == GameState.CRASHED:
                return {
                    "status": "CRASHED",
                    "round_id": self._round.round_id,
                    "multiplier": float(self._round.crash_point),
                    "crash_point": float(self._round.crash_point),
                    "elapsed_ms": self._get_elapsed_flight_time_ms()
                }
            
            elif self._round.state == GameState.FLYING:
                elapsed_ms = int((current_time - self._round.flight_start_at) * 1000)
                current_mult = self._calculate_multiplier_at_ms(elapsed_ms)
                
                # Visual clamp so frontend doesn't see > crash_point
                if current_mult > self._round.crash_point:
                    current_mult = self._round.crash_point

                return {
                    "status": "FLYING",
                    "round_id": self._round.round_id,
                    "multiplier": float(current_mult),
                    "crash_point": None,
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
        # Note: No lock needed for read-only dictionary lookup if we accept potential split-second staleness,
        # but to be strictly thread-safe in Python asyncio context (atomic dicts), it's fine.
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
        """
        Places a bet.
        Auto-initializes the round if IDLE.
        Ensures strict state validation.
        """
        amount_dec = Decimal(str(amount))
        if amount_dec <= 0:
            raise ValueError("Bet must be positive")

        auto_dec = Decimal(str(auto_cashout)) if auto_cashout else None

        async with self._lock:
            # 1. AUTO-START: If system is cold or just crashed and stale, start fresh.
            if self._round is None:
                self._internal_start_new_round()
            
            # 2. STATE VALIDATION
            # If we are CRASHED, we check if we should reset.
            if self._round.state == GameState.CRASHED:
                if self._round.crash_at and (time.time() - self._round.crash_at) > GameConfig.POST_CRASH_COOLDOWN_SEC:
                    # Cooldown over, start new round for this bet
                    self._internal_start_new_round()
                else:
                    # Cooldown active, reject bet
                    raise StateError("Round ended. Wait for next round.")

            # If we are IDLE, start (First bet scenario)
            if self._round.state == GameState.IDLE:
                self._internal_start_new_round()

            # Now strictly check if we are in BETTING
            if self._round.state != GameState.BETTING:
                 elapsed = int((time.time() - self._round.flight_start_at) * 1000) if self._round.flight_start_at else 0
                 cur = self._calculate_multiplier_at_ms(elapsed)
                 raise StateError(f"Round in progress (Status: {self._round.state.value}, x{cur:.2f})")

            # 3. RECORD BET
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
            
            if self._round.state == GameState.CRASHED:
                raise StateError("Plane crashed")
                
            if self._round.state != GameState.FLYING:
                raise StateError("Round not active")

            # Validate timing
            elapsed_ms = int((time.time() - self._round.flight_start_at) * 1000)
            server_mult = self._calculate_multiplier_at_ms(elapsed_ms)
            
            # Verify crash status (Strict server-side check)
            if server_mult >= self._round.crash_point:
                self._round.state = GameState.CRASHED
                self._round.crash_at = time.time()
                raise StateError("Plane crashed (Server auth)")

            bet = self._round.bets.get(user_id)
            if not bet:
                raise BetError("Bet not found")

            if bet.cashed_out:
                raise BetError("Already cashed out")

            requested_mult = Decimal(str(client_observed_multiplier))
            
            # Fairness: Allow slight latency tolerance, but never exceed server physics
            final_mult = min(requested_mult, server_mult)
            final_mult = final_mult.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            if final_mult < Decimal("1.00"):
                raise ValueError("Invalid multiplier")

            payout = bet.amount * final_mult
            
            bet.cashed_out = True
            bet.cashout_multiplier = final_mult
            bet.payout = payout.quantize(Decimal("0.01"), rounding=ROUND_DOWN)

            return float(bet.payout)