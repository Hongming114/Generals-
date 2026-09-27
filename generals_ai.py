from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import math
import random
from typing import ClassVar, Iterable, Mapping

from generals_core import (
    AI,
    CITY,
    DIRECTIONS,
    GENERAL,
    HILL,
    HUMAN,
    MAX_PLAYERS,
    MOUNTAIN,
    NEUTRAL,
    PLAIN,
    Board,
    Move,
    Tile,
)


@dataclass(frozen=True, slots=True)
class Difficulty:
    key: str
    label: str
    mistake_chance: float
    exploration_army: int
    coordinate_bonus: int
    consolidation: float
    aggression: float
    search_contact_turns: int
    muster_ratio: float


DIFFICULTIES = {
    "easy": Difficulty("easy", "Easy", 0.30, 2, 5, 0.75, 0.75, 600, 0.50),
    "normal": Difficulty("normal", "Normal", 0.10, 3, 12, 1.0, 1.0, 300, 0.80),
    "hard": Difficulty("hard", "Hard", 0.02, 4, 22, 1.25, 1.30, 180, 1.20),
}


@dataclass(frozen=True, slots=True)
class AIGenome:
    """Evolvable policy weights used by self-play training."""

    aggression: float = 1.0
    defense: float = 1.0
    expansion: float = 1.0
    exploration: float = 1.0
    search: float = 1.0
    coordination: float = 1.0
    home_guard: float = 1.0
    consolidation: float = 1.0
    risk_tolerance: float = 1.0
    all_in: float = 1.0
    development: float = 1.0
    task_commitment: float = 1.0
    local_advantage: float = 1.0

    GENE_NAMES: ClassVar[tuple[str, ...]] = (
        "aggression",
        "defense",
        "expansion",
        "exploration",
        "search",
        "coordination",
        "home_guard",
        "consolidation",
        "risk_tolerance",
        "all_in",
        "development",
        "task_commitment",
        "local_advantage",
    )
    GENE_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "aggression": (0.65, 1.50),
        "defense": (0.65, 1.50),
        "expansion": (0.65, 1.55),
        "exploration": (0.60, 1.60),
        "search": (0.60, 1.65),
        "coordination": (0.60, 1.60),
        "home_guard": (0.75, 1.35),
        "consolidation": (0.65, 1.55),
        "risk_tolerance": (0.60, 1.50),
        "all_in": (0.60, 1.60),
        "development": (0.60, 1.60),
        "task_commitment": (0.65, 1.65),
        "local_advantage": (0.60, 1.60),
    }

    @classmethod
    def baseline(cls) -> "AIGenome":
        return cls()

    @classmethod
    def random(
        cls,
        rng: random.Random,
        *,
        center: Mapping[str, float] | None = None,
        scale: float = 0.16,
    ) -> "AIGenome":
        values: dict[str, float] = {}
        for name in cls.GENE_NAMES:
            center_value = 1.0 if center is None else float(center.get(name, 1.0))
            values[name] = center_value + rng.gauss(0.0, scale)
        return cls.from_mapping(values)

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, object] | None,
    ) -> "AIGenome":
        if values is None:
            return cls.baseline()
        normalized: dict[str, float] = {}
        for name in cls.GENE_NAMES:
            raw = values.get(name, 1.0)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = 1.0
            low, high = cls.GENE_BOUNDS[name]
            normalized[name] = max(low, min(high, value))
        return cls(**normalized)

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.GENE_NAMES}

    def mutate(
        self,
        rng: random.Random,
        *,
        rate: float = 0.18,
        scale: float = 0.11,
        reset_chance: float = 0.015,
    ) -> "AIGenome":
        values = self.to_dict()
        for name in self.GENE_NAMES:
            roll = rng.random()
            if roll < reset_chance:
                values[name] = rng.uniform(0.82, 1.18)
            elif roll < reset_chance + rate:
                values[name] += rng.gauss(0.0, scale)
        return AIGenome.from_mapping(values)

    def crossed(
        self,
        other: "AIGenome",
        rng: random.Random,
    ) -> "AIGenome":
        values: dict[str, float] = {}
        for name in self.GENE_NAMES:
            first = float(getattr(self, name))
            second = float(getattr(other, name))
            blend = rng.uniform(0.30, 0.70)
            child = first * blend + second * (1.0 - blend)
            if rng.random() < 0.22:
                child = first if rng.random() < 0.5 else second
            values[name] = child
        return AIGenome.from_mapping(values)

    def distance(self, other: "AIGenome") -> float:
        return math.sqrt(
            sum(
                (getattr(self, name) - getattr(other, name)) ** 2
                for name in self.GENE_NAMES
            )
        )


DEFEND_HOME_CITY = "defend_home_city"
DEFEND_TERRITORY = "defend_territory"
ATTACK_ENEMY_CITY = "attack_enemy_city"
ATTACK_ENEMY_LAND = "attack_enemy_land"
ATTACK_NEUTRAL_CITY = "attack_neutral_city"
EXPLORE_EXPAND = "explore_expand"
STRATEGY_ATTACK = "attack"
STRATEGY_DEFENSE = "defense"
STRATEGY_DEVELOPMENT = "development"
STRATEGY_EXPLORATION = "exploration"

ARCHETYPE_STANDARD = "standard"
ARCHETYPE_ATTACK = "attack"
ARCHETYPE_FORT = "fort"
ARCHETYPE_TURTLE = "turtle"
ARCHETYPE_LABELS = {
    ARCHETYPE_STANDARD: "标准",
    ARCHETYPE_ATTACK: "进攻",
    ARCHETYPE_FORT: "打堡",
    ARCHETYPE_TURTLE: "龟缩",
}
ARCHETYPE_WEIGHTS = (
    (ARCHETYPE_STANDARD, 0.40),
    (ARCHETYPE_ATTACK, 0.20),
    (ARCHETYPE_FORT, 0.20),
    (ARCHETYPE_TURTLE, 0.20),
)
ARCHETYPE_ATTACK_MISSION_MULTIPLIER = 1.50
ARCHETYPE_ATTACK_ALL_IN_MULTIPLIER = 2.00
ARCHETYPE_FORT_EXPLORE_MULTIPLIER = 1.50
ARCHETYPE_FORT_CITY_MULTIPLIER = 2.00
ARCHETYPE_FORT_WEAK_CITY_ARMY = 16
ARCHETYPE_TURTLE_EARLY_EXPLORE_MULTIPLIER = 0.15
ARCHETYPE_TURTLE_EARLY_ATTACK_MULTIPLIER = 0.30
ARCHETYPE_TURTLE_RELEASED_ATTACK_MULTIPLIER = 1.50
ARCHETYPE_TURTLE_DECAPITATION_MULTIPLIER = 2.00
ARCHETYPE_TURTLE_CAPITAL_TARGET_MIN = 24
ARCHETYPE_TURTLE_CAPITAL_TARGET_MAX = 80
ARCHETYPE_TURTLE_ALL_IN_TURNS = 120

MISSION_SWITCH_MARGIN = {
    "easy": 1_600.0,
    "normal": 2_500.0,
    "hard": 3_400.0,
}
CITY_GATHER_MIN_ARMY = 8
LAND_GATHER_MIN_ARMY = 4
GATHER_ROUTE_MAX_STEPS = 64
GATHER_ROUTE_ORIGIN_LIMIT = 12
GATHER_ROUTE_CASTLE_LIMIT = 4
GATHER_ROUTE_MOVE_BONUS = 22_000.0
GATHER_CASTLE_DISTANCE_RATIO = 2.0
GATHER_CASTLE_ROUTE_BONUS = 24.0
GATHER_IDLE_DISTANCE_DISCOUNTS = (
    (20, 0.00),
    (50, 0.15),
    (100, 0.30),
    (200, 0.45),
    (10**9, 0.60),
)
GATHER_IDLE_ORIGIN_WEIGHT = 8.0
GENERAL_CAMPAIGN_HOLD_TURNS = 180
ENEMY_TARGET_HOLD_TURNS = 70
HOME_GUARD_HARD_RATIO = 0.10
HOME_GUARD_RATIO = 0.20
HOME_GUARD_HARD_SCORE = 1_150_000.0
HOME_THREAT_RADIUS = 1
HOME_THREAT_SCORE = 320_000.0
HOME_AREA_OCCUPY_SCORE = 820_000.0
HOME_SUPPORT_BASE_SCORE = 120_000.0
HOME_SUPPORT_NEED_WEIGHT = 7_500.0
HOME_SUPPORT_THREAT_WEIGHT = 8_500.0
HOME_SUPPORT_EMERGENCY_BONUS = 460_000.0
HOME_SUPPORT_DISTANCE_PENALTY = 20_000.0
HOME_SUPPORT_RATIO_WEIGHT = 25_000.0
HOME_SUPPORT_RATIO_BONUS_CAP = 300_000.0
HOME_SUPPORT_FRONTIER_PENALTY = 58_000.0
HOME_SUPPORT_LOCAL_THREAT_PENALTY = 5_500.0
HOME_SUPPORT_COUNTER_SCORE = 180_000.0
HOME_SUPPORT_FIGHT_VALUE = 19_000.0
HOME_SUPPORT_FIGHT_ADVANTAGE = 6_500.0
HOME_SUPPORT_FIGHT_MARGIN = 32_000.0
CAPITAL_DEPTH_INITIAL_TARGET = 2
CAPITAL_DEPTH_FINAL_TARGET = 5
CAPITAL_DEPTH_STEP_TURNS = 100
CAPITAL_ENEMY_GUARD_MEDIUM_DISTANCE = 8
CAPITAL_ENEMY_GUARD_DEEP_DISTANCE = 15
CAPITAL_GUARD_MEDIUM_RATIO = 0.05
CAPITAL_GUARD_DEEP_RATIO = 0.02
CAPITAL_DEPTH_PRIORITY_SCORE = 240_000.0
CAPITAL_DEPTH_DEFICIT_BONUS = 25_000.0
CAPITAL_DEPTH_DEFICIT_BONUS_CAP = 150_000.0
CAPITAL_REINFORCEMENT_RATIO_THRESHOLD = 8.0
CAPITAL_REINFORCEMENT_RATIO_BONUS = 45_000.0
CAPITAL_REINFORCEMENT_RATIO_SCALE = 12_000.0
CAPITAL_REINFORCEMENT_RATIO_BONUS_CAP = 180_000.0
REGION_REINFORCEMENT_RADIUS = 2
REGION_REINFORCEMENT_RATIO = 3.0
REGION_REINFORCEMENT_BONUS = 18_000.0
GATHER_IDLE_REGION_SIZE = REGION_REINFORCEMENT_RADIUS * 2 + 1
ATTRITION_DEVELOPMENT_MULTIPLIER = 2
ATTRITION_STALE_TURNS = 90
DEVELOPMENT_MIN_HOLD_TURNS = 90
DEVELOPMENT_MAX_HOLD_TURNS = 180
DEVELOPMENT_MIN_DETACHMENT_ARMY = 4
OVERWHELMING_ARMY_RATIO = 1.25
OVERWHELMING_HOLD_TURNS = 140
OVERWHELMING_DEVELOPMENT_FREEZE_TURNS = 180
OVERWHELMING_TRIGGER_CHANCE = {
    "easy": 0.55,
    "normal": 0.72,
    "hard": 0.88,
}
GATHER_STACK_ARMY_CAP = 500
EXPANSION_SHAPE_DIRECTION_BONUS = 90.0
EXPANSION_SHAPE_OPEN_BONUS = 18.0
EXPANSION_DIRECTION_ALIGNMENT_BONUS = 220.0
EXPANSION_TARGET_ALIGNMENT_WEIGHT = 80.0
EXPANSION_TARGET_SHAPE_WEIGHT = 0.75
LOCAL_EXPANSION_PROGRESS_WEIGHT = 200.0
LOCAL_EXPANSION_HOME_WEIGHT = 120.0
LOCAL_EXPANSION_DIRECTION_WEIGHT = 100.0
LOCAL_EXPANSION_SHAPE_MULTIPLIER = 1.40
BORDER_MUSTER_RADIUS = 3
BORDER_MUSTER_MIN_TOTAL_ARMY = 16
BORDER_MUSTER_MIN_ARMY = 8
BORDER_MUSTER_LOCAL_ENEMY_RATIO = 0.5
BORDER_MUSTER_HOLD_TURNS = 110
BORDER_MUSTER_COOLDOWN_TURNS = 60
BORDER_MUSTER_MISSION_MULTIPLIER = 1.85
MAJOR_INVASION_ARMY_RATIO = 0.20
MAJOR_INVASION_RADIUS = 2
MAJOR_INVASION_DEFENSE_MULTIPLIER = 2.0
WAR_TEMPO_START_TURN = 80
WAR_TEMPO_IDLE_TURNS = 90
WAR_TEMPO_ENDGAME_PLAYERS = 4
WAR_TEMPO_ENDGAME_IDLE_TURNS = 30
WAR_TEMPO_PUSH_TURNS = 60
WAR_TEMPO_SIGNIFICANT_BATTLE_RATIO = 0.003
WAR_TEMPO_SIGNIFICANT_BATTLE_MIN = 12
WAR_TEMPO_SIGNIFICANT_ENDGAME_PLAYERS = 8
EXPOSED_GENERAL_ALL_IN_MULTIPLIER = 1.5
LOCAL_ENGULF_RADIUS = 2
LOCAL_ENGULF_MIN_DIFFERENCE = 4
LOCAL_ENGULF_DIFFERENCE_RATIO = 1.2
LOCAL_ENGULF_WEIGHT_MULTIPLIER = 1.5
LOCAL_ENGULF_OFFSETS = tuple(
    (dx, dy)
    for dy in range(-LOCAL_ENGULF_RADIUS, LOCAL_ENGULF_RADIUS + 1)
    for dx in range(-LOCAL_ENGULF_RADIUS, LOCAL_ENGULF_RADIUS + 1)
    if 0 < abs(dx) + abs(dy) <= LOCAL_ENGULF_RADIUS
)
INDEPENDENT_ARMY_LIMIT = 4
INDEPENDENT_ARMY_TROOP_DIVISOR = 8_000
AI_INDEPENDENT_ARMY_TROOP_DIVISOR = 8_000
INDEPENDENT_ARMY_CREATION_COOLDOWN = 10
AI_INDEPENDENT_ARMY_CREATION_COOLDOWN = 35
INDEPENDENT_ARMY_LOW_STRENGTH = 50
INDEPENDENT_ARMY_MIN_ATTACK_STRENGTH = 300
INDEPENDENT_ARMY_MIN_CREATE_STRENGTH = 3
INDEPENDENT_ARMY_AUTO_CREATE_STRENGTH = 3
INDEPENDENT_ARMY_EAGER_CREATE_STRENGTH = 3
INDEPENDENT_ARMY_ATTACK_HOLD_RATIO = 2
INDEPENDENT_ARMY_ATTACK_START_RATIO = 3.0
INDEPENDENT_ARMY_MAJOR_THREAT_MIN = 10
INDEPENDENT_ARMY_MAJOR_THREAT_RATIO = 0.5
INDEPENDENT_ARMY_DEFENSE_FORCE_RATIO = 0.75
INDEPENDENT_ARMY_RESUPPLY_TARGET_LIMIT = 8
INDEPENDENT_ARMY_DAMAGE_PRESSURE = 2
INDEPENDENT_ARMY_SURVIVAL_WEIGHT = 70.0
INDEPENDENT_ARMY_SAFE_BONUS = 40.0
INDEPENDENT_ARMY_CITY_CREATE_BONUS = 65.0
INDEPENDENT_ARMY_EAGER_CREATION_ENABLED = True
INDEPENDENT_ARMY_BOOTSTRAP_MERGE_BONUS = 7_500.0
INDEPENDENT_ARMY_BOOTSTRAP_CITY_BONUS = 2_500.0
AI_ARMY_RAMP_END_TURN = 200
AI_ARMY_CONSERVATION_UTILIZATION = 0.75
AI_ARMY_CONSERVATION_MIN_OPEN_NEIGHBORS = 2
AI_ARMY_CONSERVATION_MOVE_MIN_OPEN_NEIGHBORS = 1
INDEPENDENT_ARMY_DANGER_RATIO = 1.0
INDEPENDENT_ARMY_LOITER_INTERVAL = 30
INDEPENDENT_ARMY_LOITER_SAMPLE_COUNT = 3
INDEPENDENT_ARMY_LOITER_RADIUS = 2
INDEPENDENT_ARMY_LOITER_BREAKOUT_TURNS = 90
INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL = 30
INDEPENDENT_ARMY_DISPLACEMENT_REWARD_DISTANCE = 20
MAIN_ARMY_TAG_IDLE_TURNS = 10
INDEPENDENT_ARMY_MAIN_TAG_RESUPPLY_BONUS = 1_500.0


def independent_army_limit(
    total_army: int,
    troop_divisor: int = INDEPENDENT_ARMY_TROOP_DIVISOR,
) -> int:
    """Army slots are four plus one per full troop-divisor block."""
    return INDEPENDENT_ARMY_LIMIT + (
        max(0, int(total_army)) // max(1, troop_divisor)
    )


def ai_army_creation_limit(turn: int, army_limit: int) -> int:
    """Ramp AI army creation while preserving the full late-game limit."""
    army_limit = max(1, int(army_limit))
    if turn >= AI_ARMY_RAMP_END_TURN:
        return army_limit
    progress = max(0, min(int(turn), AI_ARMY_RAMP_END_TURN))
    return min(
        army_limit,
        1 + ((army_limit - 1) * progress // AI_ARMY_RAMP_END_TURN),
    )


INDEPENDENT_ARMY_DISPLACEMENT_REWARD = 1
ARMY_STATE_ATTACK = "attack"
ARMY_STATE_DEFEND = "defend"
ARMY_STATE_RESUPPLY = "resupply"
ARMY_STATES = frozenset(
    (ARMY_STATE_ATTACK, ARMY_STATE_DEFEND, ARMY_STATE_RESUPPLY)
)
WAR_STATE_PEACE_AFTER_TURNS = 60
WAR_STRATEGY_MULTIPLIER = 1.2
REGION_REINFORCEMENT_OFFSETS = tuple(
    (dx, dy)
    for dy in range(-REGION_REINFORCEMENT_RADIUS, REGION_REINFORCEMENT_RADIUS + 1)
    for dx in range(-REGION_REINFORCEMENT_RADIUS, REGION_REINFORCEMENT_RADIUS + 1)
    if abs(dx) + abs(dy) <= REGION_REINFORCEMENT_RADIUS
)
ENEMY_SEARCH_ATTACK_SCORE = 960_000.0
ENEMY_SEARCH_SUPPORT_SCORE = 180_000.0
KNOWN_GENERAL_CAMPAIGN_SCORE = 1_450_000.0
ENEMY_SEARCH_RING_LIMIT = 4
ENEMY_SEARCH_TARGET_HOLD_TURNS = 45
GLOBAL_SEARCH_TARGET_HOLD_TURNS = 45
ACTIVE_SEARCH_BASE_SCORE = 32_000.0
ACTIVE_SEARCH_URGENCY_RATE = 1_450.0
ACTIVE_SEARCH_STALL_RATE = 4_200.0
ACTIVE_SEARCH_MAX_SCORE = 960_000.0
PEACE_ARMY_SEARCH_TURNS = 30
PEACE_ARMY_SEARCH_MIN_ARMY = 6
MISSION_RISK_BUCKETS = 2_048
MISSION_RISK_CAP = MISSION_RISK_BUCKETS // 2
BOARD_NEIGHBOR_CACHE_LIMIT = 6
BOARD_NEIGHBOR_CACHE: OrderedDict[
    int,
    tuple[Board, list[Tile], tuple[tuple[tuple[int, int], ...], ...]],
] = OrderedDict()
BOARD_OWNER_POSITION_CACHE_LIMIT = 6
BOARD_OWNER_POSITION_CACHE: OrderedDict[
    int,
    tuple[Board, int, tuple[tuple[tuple[int, int], ...], ...]],
] = OrderedDict()
STATE_MISSION_WEIGHTS = {
    STRATEGY_ATTACK: {
        DEFEND_HOME_CITY: 0.90,
        DEFEND_TERRITORY: 0.95,
        ATTACK_ENEMY_CITY: 1.45,
        ATTACK_ENEMY_LAND: 1.40,
        ATTACK_NEUTRAL_CITY: 0.78,
        EXPLORE_EXPAND: 0.82,
    },
    STRATEGY_DEFENSE: {
        DEFEND_HOME_CITY: 1.70,
        DEFEND_TERRITORY: 1.55,
        ATTACK_ENEMY_CITY: 0.78,
        ATTACK_ENEMY_LAND: 0.82,
        ATTACK_NEUTRAL_CITY: 0.68,
        EXPLORE_EXPAND: 0.62,
    },
    STRATEGY_DEVELOPMENT: {
        DEFEND_HOME_CITY: 1.00,
        DEFEND_TERRITORY: 1.00,
        ATTACK_ENEMY_CITY: 0.80,
        ATTACK_ENEMY_LAND: 0.76,
        ATTACK_NEUTRAL_CITY: 1.55,
        EXPLORE_EXPAND: 1.10,
    },
    STRATEGY_EXPLORATION: {
        DEFEND_HOME_CITY: 0.90,
        DEFEND_TERRITORY: 0.90,
        ATTACK_ENEMY_CITY: 0.86,
        ATTACK_ENEMY_LAND: 0.88,
        ATTACK_NEUTRAL_CITY: 1.05,
        EXPLORE_EXPAND: 1.50,
    },
}


@dataclass(slots=True)
class Mission:
    kind: str
    target: tuple[int, int]
    score: float
    army_mode: str = "half"
    reason: str = ""
    age: int = 0


@dataclass(slots=True)
class GatherRoute:
    mode: str
    origin: tuple[int, int]
    rally: tuple[int, int]
    path: tuple[tuple[int, int], ...]
    collected_army: int
    updated_turn: int = 0


@dataclass(slots=True)
class IndependentArmy:
    """A faction-independent army stack with its own action budget."""

    unit_id: int
    player: int
    position: tuple[int, int]
    created_turn: int
    transit_target: tuple[int, int] | None = None
    transit_ready_turn: int = 0
    state: str = ARMY_STATE_RESUPPLY
    target_strength: int = INDEPENDENT_ARMY_LOW_STRENGTH
    target: tuple[int, int] | None = None
    mission: str = "resupply"
    turns: int = 0
    last_review_strength: int = -1
    last_review_position: tuple[int, int] | None = None
    poor_performance_turns: int = 0
    stalled_turns: int = 0
    force_disband: bool = False
    damage_pressure: int = 0
    displacement_last_position: tuple[int, int] | None = None
    displacement_last_turn: int = 0
    displacement_accumulated: int = 0
    displacement_reward: int = 0
    last_move_source: tuple[int, int] | None = None
    last_move_turn: int = 0
    loiter_samples: deque[tuple[int, tuple[int, int]]] = field(
        default_factory=deque
    )
    loiter_center: tuple[int, int] | None = None
    loitering: bool = False
    loiter_breakout_turns: int = 0


@dataclass(slots=True)
class MainArmyTag:
    """Hidden marker for the stack currently directed by a human or main AI."""

    position: tuple[int, int]
    last_operated_turn: int
    transit_target: tuple[int, int] | None = None
    transit_ready_turn: int = 0


class ArmyController:
    """Tracks and commands up to five independent armies for one player."""

    def __init__(self, player: int, rng: random.Random | None = None) -> None:
        self.player = player
        self.rng = rng or random.Random()
        self.armies: list[IndependentArmy] = []
        self.creation_cooldown = 0
        self.next_unit_id = 1
        self.last_activity_turn: dict[tuple[int, int], int] = {}
        self.main_army_tag: MainArmyTag | None = None
        self.conservation_mode = False

    def clear(self) -> None:
        self.armies.clear()
        self.creation_cooldown = 0
        self.next_unit_id = 1
        self.last_activity_turn.clear()
        self.main_army_tag = None
        self.conservation_mode = False

    @property
    def count(self) -> int:
        return len(self.armies)

    @property
    def positions(self) -> set[tuple[int, int]]:
        return {army.position for army in self.armies}

    @property
    def main_army_position(self) -> tuple[int, int] | None:
        if self.main_army_tag is None:
            return None
        return self.main_army_tag.position

    def has_main_army_tag(self, position: tuple[int, int]) -> bool:
        return self.main_army_position == position

    def mark_main_army(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> bool:
        if (
            self.player not in board.active_players
            or not board.in_bounds(*position)
            or position in self.positions
        ):
            return False
        tile = board.tile(*position)
        if (
            tile.owner != self.player
            or tile.terrain == MOUNTAIN
            or tile.army <= 0
        ):
            return False
        self.main_army_tag = MainArmyTag(position, board.turn)
        return True

    def touch_main_army(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> bool:
        return self.mark_main_army(board, position)

    def sync_main_army_tag(self, board: Board) -> None:
        tag = self.main_army_tag
        if tag is None:
            return
        if self.player not in board.active_players:
            self.main_army_tag = None
            return
        if (
            tag.transit_target is not None
            and board.turn >= tag.transit_ready_turn
        ):
            tag.position = tag.transit_target
            tag.transit_target = None
            tag.transit_ready_turn = 0
        position = tag.position
        if not board.in_bounds(*position):
            self.main_army_tag = None
            return
        tile = board.tile(*position)
        if (
            tile.owner != self.player
            or tile.terrain == MOUNTAIN
            or tile.army <= 0
        ):
            self.main_army_tag = None
            return
        if board.turn - tag.last_operated_turn > MAIN_ARMY_TAG_IDLE_TURNS:
            self.main_army_tag = None

    def record_main_army_moves(
        self,
        board: Board,
        moves: Iterable[Move],
    ) -> None:
        tag = self.main_army_tag
        if tag is None:
            return
        for move in moves:
            if move.source != tag.position:
                continue
            if board.movement_turns(
                move.sx,
                move.sy,
                move.tx,
                move.ty,
                self.player,
            ) > 1:
                tag.transit_target = move.target
                tag.transit_ready_turn = board.turn + 1
            else:
                tag.position = move.target
                tag.transit_target = None
                tag.transit_ready_turn = 0
            break

    def release_main_army_tag(self) -> None:
        self.main_army_tag = None

    def occupied_positions(self, board: Board | None = None) -> set[tuple[int, int]]:
        """Include positions an army occupies or is already scheduled to enter."""
        occupied = self.positions
        for army in self.armies:
            if army.transit_target is not None:
                occupied.add(army.transit_target)
            if board is None:
                continue
            for pending in board.pending_moves:
                if (
                    pending.owner == self.player
                    and (pending.sx, pending.sy) == army.position
                ):
                    occupied.add((pending.tx, pending.ty))
        return occupied

    def needs_army_bootstrap(
        self,
        board: Board,
        owned_positions: list[tuple[int, int]] | None = None,
    ) -> bool:
        """Return whether the main AI should form a three-stack for recruitment."""
        if (
            self.player not in board.active_players
            or self.creation_cooldown > 1
        ):
            return False
        positions = (
            owned_positions
            if owned_positions is not None
            else GeneralsAI._positions_by_owner(board)[self.player]
        )
        total_army = sum(board.tile(*position).army for position in positions)
        army_limit = independent_army_limit(
            total_army,
            AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
        )
        return len(self.armies) < ai_army_creation_limit(
            board.turn,
            army_limit,
        )

    def update_conservation_mode(
        self,
        board: Board,
        effective_limit: int,
    ) -> None:
        """Preserve existing armies until the late-game roster is near cap."""
        if board.turn < AI_ARMY_RAMP_END_TURN:
            self.conservation_mode = False
            return
        target = max(
            1,
            math.ceil(effective_limit * AI_ARMY_CONSERVATION_UTILIZATION),
        )
        self.conservation_mode = len(self.armies) < target

    def _has_conservation_neighbor(
        self,
        board: Board,
        position: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> bool:
        """Require enough open cardinals so a new army cannot be boxed in."""
        return (
            self._conservation_open_neighbor_count(
                board,
                position,
                occupied,
            )
            >= AI_ARMY_CONSERVATION_MIN_OPEN_NEIGHBORS
            and not self._has_adjacent_conservation_army(
                board,
                position,
                occupied,
            )
        )

    @staticmethod
    def _has_adjacent_conservation_army(
        board: Board,
        position: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> bool:
        return any(
            board.in_bounds(position[0] + dx, position[1] + dy)
            and (position[0] + dx, position[1] + dy) in occupied
            for dx, dy in DIRECTIONS
        )

    def _conservation_open_neighbor_count(
        self,
        board: Board,
        position: tuple[int, int],
        occupied: set[tuple[int, int]],
    ) -> int:
        open_neighbors = 0
        for dx, dy in DIRECTIONS:
            target = (position[0] + dx, position[1] + dy)
            if not board.in_bounds(*target) or target in occupied:
                continue
            tile = board.tile(*target)
            if (
                tile.owner == self.player
                and tile.terrain not in (MOUNTAIN, GENERAL)
            ):
                open_neighbors += 1
        return open_neighbors

    def begin_turn(
        self,
        board: Board | None = None,
        ai: "GeneralsAI" | None = None,
    ) -> None:
        if board is not None:
            self.sync_main_army_tag(board)
        self.creation_cooldown = max(0, self.creation_cooldown - 1)
        for army in self.armies:
            army.turns += 1
        if (
            board is not None
            and ai is not None
            and board.turn > 1
            and (board.turn - 1) % board.growth_interval == 0
        ):
            self.activate_army_ai_on_growth_round(board, ai)
        if board is not None:
            for army in self.armies:
                self._update_displacement(board, army)
                self._update_loitering(board, army)

    def _update_displacement(
        self,
        board: Board,
        army: IndependentArmy,
    ) -> None:
        """Sample travel distance every 30 turns and reward real movement."""
        if army.displacement_last_position is None:
            army.displacement_last_position = army.position
            army.displacement_last_turn = board.turn
            return
        elapsed = board.turn - army.displacement_last_turn
        army.displacement_accumulated += Board.manhattan(
            army.displacement_last_position,
            army.position,
        )
        army.displacement_last_position = army.position
        if elapsed < INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL:
            return
        if (
            army.displacement_accumulated
            > INDEPENDENT_ARMY_DISPLACEMENT_REWARD_DISTANCE
        ):
            army.displacement_reward += 1
        army.displacement_accumulated = 0
        army.displacement_last_turn = (
            army.displacement_last_turn
            + (
                elapsed
                // INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL
            )
            * INDEPENDENT_ARMY_DISPLACEMENT_INTERVAL
        )

    def _update_loitering(
        self,
        board: Board,
        army: IndependentArmy,
    ) -> None:
        """Detect sustained circling inside a tiny area."""
        if (
            army.loiter_samples
            and board.turn - army.loiter_samples[-1][0]
            < INDEPENDENT_ARMY_LOITER_INTERVAL
        ):
            return
        army.loiter_samples.append((board.turn, army.position))
        while len(army.loiter_samples) > INDEPENDENT_ARMY_LOITER_SAMPLE_COUNT:
            army.loiter_samples.popleft()
        if len(army.loiter_samples) < INDEPENDENT_ARMY_LOITER_SAMPLE_COUNT:
            return

        positions = [position for _turn, position in army.loiter_samples]
        span = max(
            Board.manhattan(first, second)
            for first in positions
            for second in positions
        )
        if span <= INDEPENDENT_ARMY_LOITER_RADIUS:
            if not army.loitering:
                army.loitering = True
                army.loiter_center = positions[0]
            army.loiter_breakout_turns = max(
                army.loiter_breakout_turns,
                INDEPENDENT_ARMY_LOITER_BREAKOUT_TURNS,
            )
            return

        if (
            army.loitering
            and (
                army.loiter_center is None
                or max(
                    Board.manhattan(position, army.loiter_center)
                    for position in positions
                )
                > INDEPENDENT_ARMY_LOITER_RADIUS
            )
        ):
            army.loitering = False
            army.loiter_center = None
            army.loiter_breakout_turns = 0

    def _army_visible_position(
        self,
        board: Board,
        army: IndependentArmy,
    ) -> tuple[int, int]:
        """Return the tile this army occupies after its own pending move."""
        position = army.position
        for pending in board.pending_moves:
            if (
                pending.owner == self.player
                and (pending.sx, pending.sy) == position
            ):
                return (pending.tx, pending.ty)
        return position

    @staticmethod
    def _is_dangerous_enemy_tile(
        board: Board,
        ai: "GeneralsAI",
        position: tuple[int, int],
        moving_army: int,
    ) -> bool:
        """True when standing here could get the whole stack killed."""
        if not board.in_bounds(*position):
            return True
        tile = board.tile(*position)
        if tile.terrain == MOUNTAIN:
            return True
        if ai._is_enemy(tile.owner):
            return moving_army <= tile.army
        danger_threshold = moving_army * INDEPENDENT_ARMY_DANGER_RATIO
        adjacent_enemy = [
            board.tile(position[0] + dx, position[1] + dy).army
            for dx, dy in DIRECTIONS
            if board.in_bounds(position[0] + dx, position[1] + dy)
            and ai._is_enemy(
                board.tile(position[0] + dx, position[1] + dy).owner
            )
        ]
        return bool(adjacent_enemy) and min(adjacent_enemy) >= danger_threshold

    def activate_army_ai_on_growth_round(
        self,
        board: Board,
        ai: "GeneralsAI",
    ) -> None:
        """Reconcile and reactivate each numbered army after a growth round."""
        self.sync(board, ai)
        for army in self.armies:
            if army.state not in ARMY_STATES:
                army.state = ARMY_STATE_RESUPPLY
                army.mission = "resupply"
            army.force_disband = False
            army.stalled_turns = 0
            army.target_strength = self.target_strength(board, army.position)

    def release_at(self, position: tuple[int, int]) -> bool:
        for index, army in enumerate(self.armies):
            if army.position == position:
                self.armies.pop(index)
                return True
        return False

    def review_ai_armies(self, board: Board, ai: "GeneralsAI") -> None:
        """Release only weak underperforming armies; strong armies stay titled."""
        self.sync(board, ai)
        if not self.armies:
            return
        owned_tiles = ai._positions_by_owner(board)[self.player]
        army_limit = independent_army_limit(
            sum(board.tile(*position).army for position in owned_tiles),
            AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
        )
        effective_limit = ai_army_creation_limit(board.turn, army_limit)
        if len(self.armies) > effective_limit:
            self.armies.sort(
                key=lambda army: (
                    board.tile(*army.position).army,
                    -army.created_turn,
                    -army.unit_id,
                ),
                reverse=True,
            )
            self.armies = self.armies[:effective_limit]
            if not self.armies:
                self.update_conservation_mode(board, effective_limit)
                return
        self.update_conservation_mode(board, effective_limit)
        has_free_slot = len(self.armies) < effective_limit
        defense_target = self._defense_target(board, ai)
        surviving: list[IndependentArmy] = []
        for army in self.armies:
            source = board.tile(*army.position)
            strength = source.army
            if army.last_review_strength > strength:
                lost = army.last_review_strength - strength
                army.damage_pressure += lost
            elif strength < army.target_strength:
                army.damage_pressure = max(0, army.damage_pressure - 1)
            else:
                army.damage_pressure = 0
            if strength >= 100:
                army.poor_performance_turns = 0
            elif army.transit_target is not None or defense_target is not None:
                army.poor_performance_turns = 0
            elif army.poor_performance_turns >= 90 and not has_free_slot:
                continue
            else:
                visible = self._visible(board, ai)
                has_attack_intel = bool(ai.known_enemy_generals) or any(
                    ai._is_enemy(board.tile(*position).owner)
                    for position in visible
                )
                progress = (
                    strength - army.last_review_strength
                    if army.last_review_strength >= 0
                    else 0
                )
                moved = (
                    army.last_review_position is not None
                    and army.position != army.last_review_position
                )
                performance = progress * 4.0
                if moved:
                    performance += 12.0
                if has_attack_intel:
                    performance += 8.0
                if army.state == ARMY_STATE_ATTACK:
                    performance += 6.0
                if army.displacement_reward > 0:
                    performance += army.displacement_reward * 10.0
                if performance <= 0.0:
                    army.poor_performance_turns += 1
                else:
                    army.poor_performance_turns = 0
            army.last_review_strength = strength
            army.last_review_position = army.position
            if (
                strength < 100
                and army.poor_performance_turns >= 90
                and not has_free_slot
            ):
                continue
            surviving.append(army)
        self.armies = surviving

    def create_at(
        self,
        board: Board,
        position: tuple[int, int],
        *,
        ignore_cooldown: bool = False,
        enforce_army_limit: bool = False,
        cooldown_turns: int = INDEPENDENT_ARMY_CREATION_COOLDOWN,
    ) -> tuple[bool, str]:
        if enforce_army_limit:
            owned_tiles = GeneralsAI._positions_by_owner(board)[self.player]
            army_limit = independent_army_limit(
                sum(board.tile(*owned).army for owned in owned_tiles)
            )
            if len(self.armies) >= army_limit:
                return (
                    False,
                    "当前兵力最多拥有 "
                    f"{army_limit} 支军队（基础 4 支，每 8000 兵增加 1 支）",
                )
        if self.creation_cooldown > 0 and not ignore_cooldown:
            return False, f"军队创建冷却中，还需 {self.creation_cooldown} 回合"
        if not board.in_bounds(*position):
            return False, "请选择棋盘内的己方地块"
        tile = board.tile(*position)
        if tile.owner != self.player or tile.terrain in (MOUNTAIN, GENERAL):
            return False, "只能在己方非山地、非主城地块创建军队"
        if tile.army < INDEPENDENT_ARMY_MIN_CREATE_STRENGTH:
            return False, "该地块兵力不足，至少需要 3 兵"
        if position in self.positions:
            return False, "该地块已经是一支军队"
        if self.has_main_army_tag(position):
            self.release_main_army_tag()

        target_strength = self.target_strength(board, position)
        army = IndependentArmy(
            self.next_unit_id,
            self.player,
            position,
            board.turn,
            displacement_last_position=position,
            displacement_last_turn=board.turn,
            target_strength=target_strength,
            state=(
                ARMY_STATE_ATTACK
                if tile.army >= target_strength
                else ARMY_STATE_RESUPPLY
            ),
            mission=(
                "attack"
                if tile.army >= target_strength
                else "resupply"
            ),
        )
        board.cancel_pending_moves(self.player, position)
        self.next_unit_id += 1
        self.armies.append(army)
        self.creation_cooldown = cooldown_turns
        return True, f"已在 ({position[0] + 1},{position[1] + 1}) 创建军队"

    def target_strength(
        self,
        board: Board,
        anchor: tuple[int, int],
    ) -> int:
        highest_other = 0
        army_positions = self.positions
        for position in GeneralsAI._positions_by_owner(board)[self.player]:
            if (
                position == anchor
                or position in army_positions
                or self.has_main_army_tag(position)
            ):
                continue
            tile = board.tile(*position)
            if tile.terrain == GENERAL:
                continue
            highest_other = max(
                highest_other,
                tile.army,
            )
        return max(
            INDEPENDENT_ARMY_LOW_STRENGTH,
            highest_other * 2,
        )

    def maybe_create_auto(self, board: Board, ai: "GeneralsAI") -> bool:
        """Let an AI create a new independent army when one is useful."""
        if (
            self.creation_cooldown > 0
            or self.player not in board.active_players
        ):
            return False

        owned_positions = ai._positions_by_owner(board)[self.player]
        army_limit = independent_army_limit(
            sum(board.tile(*position).army for position in owned_positions),
            AI_INDEPENDENT_ARMY_TROOP_DIVISOR,
        )
        effective_limit = ai_army_creation_limit(board.turn, army_limit)
        if len(self.armies) >= effective_limit:
            return False

        general = board.general_position(self.player)
        occupied = self.positions
        candidates: list[tuple[float, tuple[int, int]]] = []
        visible = self._visible(board, ai)
        known_enemy = [
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
        ]
        fronts = [
            position
            for position in visible
            if ai._is_enemy(board.tile(*position).owner)
        ]
        threats = known_enemy + fronts
        non_army_strengths = sorted(
            (
                (board.tile(*position).army, position)
                for position in owned_positions
                if position not in occupied
                and board.tile(*position).terrain != GENERAL
            ),
            reverse=True,
        )
        for position in owned_positions:
            x, y = position
            tile = board.tile(x, y)
            if (
                tile.terrain == MOUNTAIN
                or tile.army < INDEPENDENT_ARMY_AUTO_CREATE_STRENGTH
                or position == general
                or position in occupied
            ):
                continue
            if (
                self.conservation_mode
                and not self._has_conservation_neighbor(
                    board,
                    position,
                    occupied,
                )
            ):
                continue
            if self.conservation_mode and ai._is_frontier(
                board,
                x,
                y,
                visible,
            ):
                continue
            highest_other = next(
                (
                    army
                    for army, other_position in non_army_strengths
                    if other_position != position
                ),
                0,
            )
            target_a = max(
                INDEPENDENT_ARMY_LOW_STRENGTH,
                highest_other * 2,
            )
            eager_ready = (
                tile.army >= INDEPENDENT_ARMY_EAGER_CREATE_STRENGTH
            )
            dominant_ready = tile.army >= target_a
            city_ready = (
                tile.terrain == CITY
                and tile.army >= INDEPENDENT_ARMY_MIN_CREATE_STRENGTH
            )
            eager_ready = (
                eager_ready
                and INDEPENDENT_ARMY_EAGER_CREATION_ENABLED
            )
            city_ready = (
                city_ready
                and INDEPENDENT_ARMY_EAGER_CREATION_ENABLED
            )
            if not (dominant_ready or eager_ready or city_ready):
                continue
            front_distance = (
                min(
                    abs(x - tx) + abs(y - ty)
                    for tx, ty in threats
                )
                if threats
                else 20
            )
            if self.conservation_mode:
                score = (
                    tile.army * 34.0
                    + min(front_distance, 24) * 22.0
                    + ai._friendly_neighbor_count(board, x, y) * 35.0
                )
            else:
                score = tile.army * 34.0 + min(front_distance, 12) * 8.0
                if ai._is_frontier(board, x, y, visible):
                    score += 90.0
            if tile.terrain == CITY:
                score += INDEPENDENT_ARMY_CITY_CREATE_BONUS
            candidates.append((score, position))

        if not candidates:
            return False
        candidates.sort(key=lambda item: (-item[0], item[1]))
        created, _message = self.create_at(
            board,
            candidates[0][1],
            cooldown_turns=AI_INDEPENDENT_ARMY_CREATION_COOLDOWN,
        )
        return created

    def sync(self, board: Board, ai: "GeneralsAI") -> None:
        if self.player not in board.active_players:
            self.clear()
            return
        self.sync_main_army_tag(board)
        main_army_position = self.main_army_position

        surviving: list[IndependentArmy] = []
        occupied: set[tuple[int, int]] = set()
        for army in self.armies:
            position = army.position
            matching_pending = next(
                (
                    pending
                    for pending in board.pending_moves
                    if pending.owner == self.player
                    and (pending.sx, pending.sy) == position
                    and army.transit_target is not None
                    and (pending.tx, pending.ty) == army.transit_target
                ),
                None,
            )
            position_owned = (
                board.in_bounds(*position)
                and board.tile(*position).owner == self.player
            )
            transit_owned = (
                army.transit_target is not None
                and board.in_bounds(*army.transit_target)
                and board.tile(*army.transit_target).owner == self.player
            )
            if matching_pending is not None:
                if not position_owned:
                    continue
            elif transit_owned and board.turn >= army.transit_ready_turn:
                army.position = army.transit_target
                army.transit_target = None
            elif not position_owned:
                # The anchor was destroyed. A delayed move can still survive
                # only when its destination is already friendly.
                continue
            elif army.transit_target is not None:
                army.transit_target = None

            if (
                main_army_position is not None
                and army.position == main_army_position
            ):
                board.cancel_pending_moves(self.player, army.position)
                continue
            if army.position in occupied:
                continue
            occupied.add(army.position)
            final_tile = board.tile(*army.position)
            minimum_troops = (
                1
                if self.conservation_mode
                else 2
            )
            if final_tile.army <= minimum_troops or (
                final_tile.terrain == GENERAL
                and final_tile.owner == self.player
            ):
                board.cancel_pending_moves(self.player, army.position)
                continue
            army.force_disband = False
            army.target_strength = self.target_strength(board, army.position)
            surviving.append(army)
        self.armies = surviving

    def plan_moves(
        self,
        board: Board,
        ai: "GeneralsAI",
        phase: int = 0,
        incoming_moves: tuple[Move, ...] = (),
        observed_moves: tuple[Move, ...] = (),
    ) -> list[Move]:
        self.sync(board, ai)
        if not self.armies:
            return []

        moves: list[Move] = []
        reserved_sources: set[tuple[int, int]] = set()
        reserved_targets = set(self.positions)
        reserved_targets.update(
            army.transit_target
            for army in self.armies
            if army.transit_target is not None
        )
        reserved_targets.update(
            (move.tx, move.ty)
            for move in incoming_moves
        )
        reserved_targets.update(
            (pending.tx, pending.ty)
            for pending in board.pending_moves
            if pending.owner == self.player
            and (pending.sx, pending.sy) in self.positions
        )
        surviving: list[IndependentArmy] = []
        for move in (*incoming_moves, *observed_moves):
            self.last_activity_turn[move.source] = board.turn
            self.last_activity_turn[move.target] = board.turn
        for army in self.armies:
            if board.tile(*army.position).owner != self.player:
                continue
            move = self._plan_army_move(
                board,
                ai,
                army,
                reserved_sources,
                reserved_targets,
                incoming_moves,
                observed_moves,
            )
            if army.force_disband:
                board.cancel_pending_moves(self.player, army.position)
                continue
            if move is None and army.transit_target is None:
                # A friendly stack or another independent army can temporarily
                # occupy every legal target. Keep the army alive and retry next
                # turn instead of silently deleting its title.
                army.stalled_turns += 1
                surviving.append(army)
                continue
            army.stalled_turns = 0
            surviving.append(army)
            if move is None:
                continue
            moves.append(move)
            reserved_sources.add(move.source)
            reserved_targets.add(move.target)
        self.armies = surviving
        return moves

    def _peace_search_target(
        self,
        board: Board,
        ai: "GeneralsAI",
    ) -> tuple[int, int] | None:
        """Choose the best outward objective when no battle is active."""
        own_general = board.general_position(self.player)
        candidates = (
            ai.enemy_hunt_target,
            ai.campaign_target,
            ai.search_waypoint,
            ai.mission.target if ai.mission is not None else None,
        )
        for target in candidates:
            if target is None or not board.in_bounds(*target):
                continue
            if target == own_general:
                continue
            tile = board.tile(*target)
            if tile.owner == self.player and target not in ai.known_enemy_tiles.get(
                tile.owner,
                set(),
            ):
                continue
            return target
        return None

    def _plan_army_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        reserved_sources: set[tuple[int, int]],
        reserved_targets: set[tuple[int, int]],
        incoming_moves: tuple[Move, ...],
        observed_moves: tuple[Move, ...],
    ) -> Move | None:
        if army.transit_target is not None:
            return None
        if not board.in_bounds(*army.position):
            army.force_disband = True
            return None
        source = board.tile(*army.position)
        if source.army < 2 or source.terrain == MOUNTAIN:
            army.force_disband = True
            return None

        own_positions = ai._positions_by_owner(board)[self.player]
        moving_sources = set(reserved_sources)
        moving_sources.update(
            (pending.sx, pending.sy)
            for pending in board.pending_moves
            if pending.owner == self.player
        )
        moving_sources.update(
            move.source
            for move in incoming_moves
                if move.source in own_positions
        )
        moving_sources.update(
            move.source
            for move in observed_moves
            if move.source in own_positions
        )
        blocked_targets = set(moving_sources)
        blocked_targets.update(reserved_targets)

        capture_move = self._winning_general_capture_move(
            board,
            ai,
            army,
            blocked_targets,
        )
        if capture_move is not None:
            army.state = ARMY_STATE_ATTACK
            army.mission = "attack"
            army.target = capture_move.target
            self._record_army_move(army, capture_move.target, board)
            return capture_move

        threats = self._moving_enemy_threats(
            board,
            ai,
            incoming_moves,
            observed_moves,
            army.position,
            source.army,
        )
        defense_target = threats[0][2] if threats and threats[0][0] == 0 else None
        preferred_defense = bool(threats and threats[0][0] == 1)
        minimum_enemy = self._minimum_adjacent_enemy_army(board, ai, army.position)

        army_attack_strength = (
            max(
                army.target_strength,
                INDEPENDENT_ARMY_MIN_ATTACK_STRENGTH,
            )
            if self.conservation_mode
            else army.target_strength
        )
        if self.conservation_mode and source.army < army_attack_strength:
            army.state = ARMY_STATE_RESUPPLY
            army.mission = "resupply"
            move = self._conservation_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if move is not None:
                return move
            move = self._safe_rear_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if move is not None:
                return move
            move = self._retreat_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if move is not None:
                return move
            move = self._resupply_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if move is not None:
                return move
            return self._forced_army_move(
                board,
                ai,
                army,
                army.target,
                blocked_targets=blocked_targets,
            )

        if defense_target is not None:
            army.state = ARMY_STATE_DEFEND
            army.mission = "defend"
            army.target = defense_target
            move = self._move_toward(
                board,
                ai,
                army.position,
                defense_target,
                board.tile(*army.position).army - 1,
                blocked_targets=blocked_targets,
            )
            if move is not None:
                self._record_army_move(army, move.target, board)
                return move
            return self._forced_army_move(
                board,
                ai,
                army,
                defense_target,
                blocked_targets=blocked_targets,
            )

        if preferred_defense:
            army.state = ARMY_STATE_DEFEND
            army.mission = "defend"
            army.target = threats[0][2]
            move = self._move_toward(
                board,
                ai,
                army.position,
                threats[0][2],
                source.army - 1,
                blocked_targets=blocked_targets,
            )
            if move is not None:
                self._record_army_move(army, move.target, board)
                return move
            return self._forced_army_move(
                board,
                ai,
                army,
                threats[0][2],
                blocked_targets=blocked_targets,
            )

        if army.loitering:
            move = self._loiter_breakout_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if move is not None:
                return move

        if (
            ai.war_state == "peace"
            and ai.turns_since_contact >= PEACE_ARMY_SEARCH_TURNS
            and source.army >= PEACE_ARMY_SEARCH_MIN_ARMY
        ):
            peace_search_target = self._peace_search_target(board, ai)
        else:
            peace_search_target = None
        if peace_search_target is not None:
            army.state = ARMY_STATE_ATTACK
            army.mission = "attack"
            army.target = peace_search_target
            move = self._move_toward(
                board,
                ai,
                army.position,
                peace_search_target,
                source.army - 1,
                blocked_targets=blocked_targets,
            )
            if move is not None:
                self._record_army_move(army, move.target, board)
                return move
            return self._forced_army_move(
                board,
                ai,
                army,
                peace_search_target,
                blocked_targets=blocked_targets,
            )

        visible_position = self._army_visible_position(board, army)
        currently_endangered = self._is_dangerous_enemy_tile(
            board,
            ai,
            visible_position,
            source.army - 1,
        )
        if (
            army.damage_pressure >= INDEPENDENT_ARMY_DAMAGE_PRESSURE
            and source.army < army.target_strength
            and currently_endangered
        ):
            army.state = ARMY_STATE_RESUPPLY
            army.mission = "resupply"
            retreat = self._retreat_move(
                board,
                ai,
                army,
                blocked_targets,
            )
            if retreat is not None:
                return retreat

        attack_hold_ok = (
            minimum_enemy is None
            or source.army
            > minimum_enemy * INDEPENDENT_ARMY_ATTACK_HOLD_RATIO
        )
        attack_start_ok = (
            minimum_enemy is None
            or source.army
            > minimum_enemy * INDEPENDENT_ARMY_ATTACK_START_RATIO
        )
        keep_attacking = (
            army.state == ARMY_STATE_ATTACK
            and attack_hold_ok
        )
        if (
            source.army >= army_attack_strength
            and attack_start_ok
        ) or keep_attacking:
            army.state = ARMY_STATE_ATTACK
            army.mission = "attack"
            attack_target = self._attack_target(board, ai, army)
            if attack_target is not None:
                army.target = attack_target
                move = self._move_toward(
                    board,
                    ai,
                    army.position,
                    attack_target,
                    source.army - 1,
                    blocked_targets=blocked_targets,
                )
                if (
                    move is not None
                    and not self._army_step_is_unfavorable(
                        board,
                        ai,
                        move,
                        source.army - 1,
                    )
                ):
                    self._record_army_move(army, move.target, board)
                    return move
            return self._forced_army_move(
                board,
                ai,
                army,
                attack_target,
                blocked_targets=blocked_targets,
            )

        army.state = ARMY_STATE_RESUPPLY
        army.mission = "resupply"
        move = self._resupply_move(
            board,
            ai,
            army,
            blocked_targets,
        )
        if move is not None:
            return move
        if not self._can_reach_friendly_frontier(board, ai, army.position):
            army.force_disband = True
            return None
        return self._forced_army_move(
            board,
            ai,
            army,
            army.target,
            blocked_targets=blocked_targets,
        )

    def _conservation_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        moving_sources: set[tuple[int, int]],
    ) -> Move | None:
        """Keep a low-strength army inside friendly land while it rebuilds."""
        source = board.tile(*army.position)
        amount = source.army - 1
        if amount < 1:
            return None
        threats = [
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
            if board.in_bounds(*position)
        ]
        best: tuple[float, Move] | None = None
        vacated_armies = set(self.positions)
        vacated_armies.discard(army.position)
        for dx, dy in DIRECTIONS:
            target_position = (
                army.position[0] + dx,
                army.position[1] + dy,
            )
            if not board.in_bounds(*target_position):
                continue
            if target_position in moving_sources or target_position in self.positions:
                continue
            target = board.tile(*target_position)
            if target.owner != self.player:
                continue
            if self.has_main_army_tag(target_position):
                continue
            if self._is_immediate_reverse(army, target_position, board):
                continue
            if target.terrain in (MOUNTAIN, GENERAL):
                continue
            if amount <= 2 and target.army < 1:
                continue
            adjacent_armies = sum(
                1
                for other_dx, other_dy in DIRECTIONS
                if board.in_bounds(
                    target_position[0] + other_dx,
                    target_position[1] + other_dy,
                )
                and (
                    target_position[0] + other_dx,
                    target_position[1] + other_dy,
                )
                in self.positions
                and (
                    target_position[0] + other_dx,
                    target_position[1] + other_dy,
                )
                != army.position
            )
            if adjacent_armies:
                continue
            open_neighbors = self._conservation_open_neighbor_count(
                board,
                target_position,
                vacated_armies,
            )
            if open_neighbors < AI_ARMY_CONSERVATION_MOVE_MIN_OPEN_NEIGHBORS:
                continue
            move = Move(
                army.position[0],
                army.position[1],
                target_position[0],
                target_position[1],
                amount,
            )
            if not board.move_legal(move, self.player):
                continue
            danger = self._is_dangerous_enemy_tile(
                board,
                ai,
                target_position,
                amount,
            )
            threat_distance = (
                min(
                    Board.manhattan(target_position, threat)
                    for threat in threats
                )
                if threats
                else 20
            )
            score = (
                target.army * 60.0
                + min(threat_distance, 20) * 24.0
                + open_neighbors * 90.0
                + ai._friendly_neighbor_count(
                    board,
                    target_position[0],
                    target_position[1],
                )
                * 20.0
            )
            if danger:
                score -= INDEPENDENT_ARMY_SURVIVAL_WEIGHT * 4.0
            else:
                score += INDEPENDENT_ARMY_SAFE_BONUS
            if best is None or score > best[0]:
                best = (score, move)
        if best is None:
            return None
        self._record_army_move(army, best[1].target, board)
        return best[1]

    def _safe_rear_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        moving_sources: set[tuple[int, int]],
    ) -> Move | None:
        """Move a rebuilding army inside friendly land without exposing it."""
        source = board.tile(*army.position)
        amount = source.army - 1
        if amount < 1:
            return None
        own_positions = ai._positions_by_owner(board)[self.player]
        general = board.general_position(self.player)
        controlled = self.positions
        threats = [
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
            if board.in_bounds(*position)
        ]
        visible = self._visible(board, ai)
        threats.extend(
            position
            for position in visible
            if ai._is_enemy(board.tile(*position).owner)
        )
        candidates: list[tuple[float, tuple[int, int]]] = []
        for position in own_positions:
            if (
                position == army.position
                or position == general
                or position in controlled
                or position in moving_sources
                or self.has_main_army_tag(position)
                or self._is_immediate_reverse(army, position, board)
            ):
                continue
            tile = board.tile(*position)
            if tile.terrain == MOUNTAIN:
                continue
            if self._has_adjacent_conservation_army(
                board,
                position,
                controlled,
            ):
                continue
            distance = Board.manhattan(army.position, position)
            if distance > 14:
                continue
            threat_distance = (
                min(Board.manhattan(position, threat) for threat in threats)
                if threats
                else 30
            )
            frontier_penalty = 900.0 if ai._is_frontier(
                board,
                position[0],
                position[1],
                visible,
            ) else 0.0
            score = (
                min(threat_distance, 30) * 28.0
                + tile.army * 12.0
                - distance * 4.0
                - frontier_penalty
            )
            if tile.terrain == CITY:
                score += 1_100.0
            candidates.append((score, position))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _score, target in candidates[:4]:
            move = self._move_toward(
                board,
                ai,
                army.position,
                target,
                amount,
                blocked_targets=moving_sources,
            )
            if move is None:
                continue
            landing = board.tile(*move.target)
            if landing.owner != self.player or landing.terrain == MOUNTAIN:
                continue
            if amount <= 2 and landing.army < 1:
                continue
            if self._has_adjacent_conservation_army(
                board,
                move.target,
                controlled,
            ):
                continue
            army.target = target
            self._record_army_move(army, move.target, board)
            return move
        return None

    def _forced_army_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        goal: tuple[int, int] | None = None,
        blocked_targets: set[tuple[int, int]] | None = None,
    ) -> Move | None:
        """Guarantee a legal army step whenever the stack can move at all."""
        source = board.tile(*army.position)
        if source.army < 2 or source.terrain == MOUNTAIN:
            return None

        if goal is None:
            goal = army.target

        if goal is not None and board.in_bounds(*goal):
            move = self._move_toward(
                board,
                ai,
                army.position,
                goal,
                source.army - 1,
                blocked_targets=blocked_targets,
            )
            if (
                move is not None
                and not self._is_dangerous_enemy_tile(
                    board,
                    ai,
                    move.target,
                    source.army - 1,
                )
                and not self._army_step_is_unfavorable(
                    board,
                    ai,
                    move,
                    source.army - 1,
                )
                and not self.has_main_army_tag(move.target)
                and not self._is_immediate_reverse(
                    army,
                    move.target,
                    board,
                )
            ):
                army.target = move.target
                self._record_army_move(army, move.target, board)
                return move

        safe_best: tuple[float, Move] | None = None
        risky_best: tuple[float, Move] | None = None
        own_frontier = self._friendly_frontier_targets(board, ai)
        for dx, dy in DIRECTIONS:
            target_position = (army.position[0] + dx, army.position[1] + dy)
            if not board.in_bounds(*target_position):
                continue
            if target_position in self.positions:
                continue
            if self._is_immediate_reverse(army, target_position, board):
                continue
            if blocked_targets is not None and target_position in blocked_targets:
                continue
            target = board.tile(*target_position)
            if target.terrain == MOUNTAIN or (
                target.terrain == GENERAL
                and target.owner == self.player
                and not self.has_main_army_tag(target_position)
            ):
                continue
            move = Move(
                army.position[0],
                army.position[1],
                target_position[0],
                target_position[1],
                source.army - 1,
            )
            if not board.move_legal(move, self.player):
                continue
            score = 0.0
            if target.owner == self.player:
                score += 28.0
                if target_position in own_frontier:
                    score += 22.0
            elif target.owner == NEUTRAL:
                score += 44.0
            elif ai._is_enemy(target.owner):
                score += 72.0
                if source.army - 1 > target.army:
                    score += 34.0
                if self.conservation_mode:
                    score -= 180.0
            if (
                target.owner == self.player
                and self.has_main_army_tag(target_position)
            ):
                score -= 10_000.0
            if self._is_dangerous_enemy_tile(
                board,
                ai,
                target_position,
                source.army - 1,
            ):
                score -= INDEPENDENT_ARMY_SURVIVAL_WEIGHT
            else:
                score += INDEPENDENT_ARMY_SAFE_BONUS
            if goal is not None:
                score += (
                    Board.manhattan(army.position, goal)
                    - Board.manhattan(target_position, goal)
                ) * 12.0
            score += self.rng.random()
            if self._army_step_is_unfavorable(
                board,
                ai,
                move,
                source.army - 1,
            ):
                if risky_best is None or score > risky_best[0]:
                    risky_best = (score, move)
            elif safe_best is None or score > safe_best[0]:
                safe_best = (score, move)

        best = safe_best or risky_best
        if best is None:
            army.force_disband = True
            return None
        army.target = best[1].target
        self._record_army_move(army, best[1].target, board)
        return best[1]

    @staticmethod
    def _is_immediate_reverse(
        army: IndependentArmy,
        target: tuple[int, int],
        board: Board,
    ) -> bool:
        """Prevent a two-step army action from undoing its first step."""
        return (
            army.last_move_source == target
            and army.last_move_turn == board.turn
        )

    @staticmethod
    def _army_step_is_unfavorable(
        board: Board,
        ai: "GeneralsAI",
        move: Move,
        amount: int,
    ) -> bool:
        target = board.tile(*move.target)
        if target.owner == ai.player:
            return amount <= 2 and target.army < 1
        if target.owner == NEUTRAL:
            return amount <= target.army
        if target.terrain == GENERAL:
            return amount <= target.army
        return (
            ai._is_enemy(target.owner)
            and amount
            <= target.army * INDEPENDENT_ARMY_ATTACK_START_RATIO
        )

    def _moving_enemy_threats(
        self,
        board: Board,
        ai: "GeneralsAI",
        incoming_moves: tuple[Move, ...],
        observed_moves: tuple[Move, ...],
        army_position: tuple[int, int],
        army_strength: int,
    ) -> list[tuple[int, int, tuple[int, int], int]]:
        """Return moving enemy threats that justify army defense.

        Each item is ``(priority, distance, target, strength)``. Priority 0
        forces defense, while priority 1 only takes precedence when the army
        is not materially outnumbered.
        """
        own_positions = set(ai._positions_by_owner(board)[self.player])
        totals: dict[tuple[int, int], int] = {}
        seen: set[tuple[int, tuple[int, int], tuple[int, int]]] = set()

        def add(
            owner: int,
            source: tuple[int, int],
            target: tuple[int, int],
            amount: int,
        ) -> None:
            if (
                owner < 0
                or not ai._is_enemy(owner)
                or target not in own_positions
                or amount <= 0
            ):
                return
            key = (owner, source, target)
            if key in seen:
                return
            seen.add(key)
            totals[target] = totals.get(target, 0) + amount

        for pending in board.pending_moves:
            add(
                pending.owner,
                (pending.sx, pending.sy),
                (pending.tx, pending.ty),
                pending.amount,
            )
        for move in (*incoming_moves, *observed_moves):
            owner = board.tile(move.sx, move.sy).owner
            add(owner, move.source, move.target, move.amount)

        major_threshold = max(
            INDEPENDENT_ARMY_MAJOR_THREAT_MIN,
            math.ceil(army_strength * INDEPENDENT_ARMY_MAJOR_THREAT_RATIO),
        )
        threats: list[tuple[int, int, tuple[int, int], int]] = []
        for target, strength in totals.items():
            if strength < major_threshold:
                continue
            distance = Board.manhattan(army_position, target)
            if distance <= 10:
                threats.append((0, distance, target, strength))
            elif (
                distance <= 20
                and army_strength
                >= strength * INDEPENDENT_ARMY_DEFENSE_FORCE_RATIO
            ):
                threats.append((1, distance, target, strength))
        threats.sort(key=lambda item: (item[0], item[1], item[3], item[2]))
        return threats

    @staticmethod
    def _minimum_adjacent_enemy_army(
        board: Board,
        ai: "GeneralsAI",
        position: tuple[int, int],
    ) -> int | None:
        strengths = [
            board.tile(position[0] + dx, position[1] + dy).army
            for dx, dy in DIRECTIONS
            if board.in_bounds(position[0] + dx, position[1] + dy)
            and ai._is_enemy(
                board.tile(position[0] + dx, position[1] + dy).owner
            )
        ]
        return min(strengths) if strengths else None

    @staticmethod
    def _friendly_frontier_targets(
        board: Board,
        ai: "GeneralsAI",
    ) -> set[tuple[int, int]]:
        frontier: set[tuple[int, int]] = set()
        for position in ai._positions_by_owner(board)[ai.player]:
            x, y = position
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not board.in_bounds(nx, ny):
                    frontier.add(position)
                    break
                neighbor = board.tile(nx, ny)
                if neighbor.owner != ai.player and neighbor.terrain != MOUNTAIN:
                    frontier.add(position)
                    break
        return frontier

    def _can_reach_friendly_frontier(
        self,
        board: Board,
        ai: "GeneralsAI",
        position: tuple[int, int],
    ) -> bool:
        if board.tile(*position).owner == self.player:
            return True
        frontier = self._friendly_frontier_targets(board, ai)
        if not frontier:
            return False
        distances = ai._distance_map(board, list(frontier))
        return distances.get(position) is not None

    def _resupply_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        moving_sources: set[tuple[int, int]],
    ) -> Move | None:
        """Move the army itself toward a local stack, castle, or quiet zone."""
        own_positions = ai._positions_by_owner(board)[self.player]
        general = board.general_position(self.player)
        army_positions = self.positions
        frontier = self._friendly_frontier_targets(board, ai)
        candidates: list[tuple[float, tuple[int, int]]] = []
        for position in own_positions:
            if (
                position == army.position
                or (
                    position == general
                    and not self.has_main_army_tag(position)
                )
                or position in army_positions
                or position in moving_sources
                or self.has_main_army_tag(position)
                or self._is_immediate_reverse(army, position, board)
            ):
                continue
            tile = board.tile(*position)
            if tile.terrain == MOUNTAIN:
                continue
            if tile.army <= 1 and position not in frontier:
                continue
            distance = Board.manhattan(army.position, position)
            last_active = self.last_activity_turn.get(
                position,
                army.created_turn,
            )
            idle_turns = max(0, board.turn - last_active)
            score = (
                tile.army * 34.0
                + min(idle_turns, 80) * 11.0
                - distance * 7.0
            )
            if tile.terrain == CITY:
                score += 900.0
            if position in frontier:
                score += 420.0
            if self.has_main_army_tag(position):
                score += (
                    INDEPENDENT_ARMY_MAIN_TAG_RESUPPLY_BONUS
                    + tile.army * 35.0
                )
            if self._is_dangerous_enemy_tile(
                board,
                ai,
                position,
                board.tile(*army.position).army - 1,
            ):
                score -= INDEPENDENT_ARMY_SURVIVAL_WEIGHT
            else:
                score += INDEPENDENT_ARMY_SAFE_BONUS
            candidates.append((score, position))

        source_army = board.tile(*army.position).army - 1
        known_enemies = {
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
            if board.in_bounds(*position)
            and position not in army_positions
        }
        nearby_enemies = {
            (army.position[0] + dx, army.position[1] + dy)
            for dx, dy in DIRECTIONS
            if board.in_bounds(
                army.position[0] + dx,
                army.position[1] + dy,
            )
            and ai._is_enemy(
                board.tile(
                    army.position[0] + dx,
                    army.position[1] + dy,
                ).owner
            )
        }
        for position in nearby_enemies:
            if self.conservation_mode:
                continue
            if position in moving_sources:
                continue
            tile = board.tile(*position)
            if source_army <= tile.army:
                continue
            candidates.append(
                (
                    1_000.0 - tile.army * 4.0,
                    position,
                )
            )
        for position in sorted(
            known_enemies,
            key=lambda position: (
                Board.manhattan(army.position, position),
                position,
            ),
        )[:INDEPENDENT_ARMY_RESUPPLY_TARGET_LIMIT]:
            if self.conservation_mode:
                continue
            if position in moving_sources:
                continue
            tile = board.tile(*position)
            if source_army <= tile.army:
                continue
            distance = Board.manhattan(army.position, position)
            candidates.append(
                (
                    720.0
                    - distance * 5.0
                    - tile.army * 3.0,
                    position,
                )
            )

        if board.tile(*army.position).owner != self.player:
            for position in frontier:
                distance = Board.manhattan(army.position, position)
                candidates.append(
                    (
                        1_600.0 - distance * 10.0,
                        position,
                    )
                )

        if known_enemies and not candidates:
            nearest_enemies = sorted(
                known_enemies,
                key=lambda position: (
                    Board.manhattan(army.position, position),
                    position,
                ),
            )[:INDEPENDENT_ARMY_RESUPPLY_TARGET_LIMIT]
            for position in nearest_enemies:
                tile = board.tile(*position)
                distance = Board.manhattan(army.position, position)
                score = 500.0 + tile.army * 20.0 - distance * 8.0
                candidates.append((score, position))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        attempts = candidates[: min(3, len(candidates))]
        for _score, target in attempts:
            army.target = target
            move = self._move_toward(
                board,
                ai,
                army.position,
                target,
                board.tile(*army.position).army - 1,
                blocked_targets=moving_sources,
            )
            if (
                move is not None
                and not self._army_step_is_unfavorable(
                    board,
                    ai,
                    move,
                    board.tile(*army.position).army - 1,
                )
                and not self._is_immediate_reverse(
                    army,
                    move.target,
                    board,
                )
            ):
                self._record_army_move(army, move.target, board)
                return move
        return None

    def _retreat_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        moving_sources: set[tuple[int, int]],
    ) -> Move | None:
        """Pull a hurt army back to strength instead of trading it away."""
        own_positions = ai._positions_by_owner(board)[self.player]
        army_positions = self.positions
        general = board.general_position(self.player)
        threats = [
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
            if board.in_bounds(*position)
        ]
        visible = self._visible(board, ai)
        threats.extend(
            position
            for position in visible
            if ai._is_enemy(board.tile(*position).owner)
        )
        source_army = board.tile(*army.position).army
        candidates: list[tuple[float, tuple[int, int]]] = []
        for position in own_positions:
            if (
                position == army.position
                or (
                    position == general
                    and not self.has_main_army_tag(position)
                )
                or position in army_positions
                or position in moving_sources
                or self.has_main_army_tag(position)
                or self._is_immediate_reverse(army, position, board)
            ):
                continue
            tile = board.tile(*position)
            if tile.terrain == MOUNTAIN:
                continue
            if tile.army <= 1 and tile.terrain != CITY:
                continue
            danger = self._is_dangerous_enemy_tile(
                board,
                ai,
                position,
                source_army - 1,
            )
            threat_distance = (
                min(
                    Board.manhattan(position, threat)
                    for threat in threats
                )
                if threats
                else 20
            )
            score = (
                tile.army * 26.0
                + min(threat_distance, 20) * 26.0
                - Board.manhattan(army.position, position) * 9.0
            )
            if tile.terrain == CITY:
                score += 1_200.0
            if danger:
                score -= INDEPENDENT_ARMY_SURVIVAL_WEIGHT * 2.0
            else:
                score += INDEPENDENT_ARMY_SAFE_BONUS
            candidates.append((score, position))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], item[1]))
        for _score, target in candidates[:3]:
            army.target = target
            move = self._move_toward(
                board,
                ai,
                army.position,
                target,
                source_army - 1,
                blocked_targets=moving_sources,
            )
            if (
                move is not None
                and not self._army_step_is_unfavorable(
                    board,
                    ai,
                    move,
                    source_army - 1,
                )
                and not self._is_immediate_reverse(
                    army,
                    move.target,
                    board,
                )
            ):
                self._record_army_move(army, move.target, board)
                return move
        return None

    def _visible(
        self,
        board: Board,
        ai: "GeneralsAI",
    ) -> set[tuple[int, int]]:
        owned = ai._positions_by_owner(board)[self.player]
        return ai._cached_visibility(board, owned)

    def _defense_target(
        self,
        board: Board,
        ai: "GeneralsAI",
        incoming_moves: tuple[Move, ...] = (),
        observed_moves: tuple[Move, ...] = (),
        army_position: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        if not self.armies:
            return None
        if army_position is None:
            army_position = min(self.positions)
        if not board.in_bounds(*army_position):
            return None
        strength = board.tile(*army_position).army
        threats = self._moving_enemy_threats(
            board,
            ai,
            incoming_moves,
            observed_moves,
            army_position,
            strength,
        )
        return threats[0][2] if threats else None

    def _attack_target(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
    ) -> tuple[int, int] | None:
        generals = [
            position
            for position in ai.known_enemy_generals.values()
            if board.in_bounds(*position)
        ]
        if generals:
            return min(
                generals,
                key=lambda position: Board.manhattan(position, army.position),
            )

        remembered = [
            position
            for positions in ai.known_enemy_tiles.values()
            for position in positions
        ]
        visible = self._visible(board, ai)
        enemies = [
            position
            for position in visible
            if ai._is_enemy(board.tile(*position).owner)
        ]
        candidates = remembered + enemies
        if candidates:
            return min(
                candidates,
                key=lambda position: Board.manhattan(position, army.position),
            )
        if ai.mission is not None:
            return ai.mission.target
        return (board.width // 2, board.height // 2)

    def _winning_general_capture_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        blocked_targets: set[tuple[int, int]],
    ) -> Move | None:
        """Capture an adjacent enemy general whenever the army can win."""
        source = board.tile(*army.position)
        amount = source.army - 1
        if amount < 1:
            return None
        visible = self._visible(board, ai)
        for dx, dy in DIRECTIONS:
            target_position = (
                army.position[0] + dx,
                army.position[1] + dy,
            )
            if (
                not board.in_bounds(*target_position)
                or target_position in blocked_targets
            ):
                continue
            target = board.tile(*target_position)
            if (
                target.terrain != GENERAL
                or not ai._is_enemy(target.owner)
                or target.owner not in board.active_players
                or amount <= target.army
            ):
                continue
            if (
                target_position not in visible
                and ai.known_enemy_generals.get(target.owner)
                != target_position
            ):
                continue
            move = Move(
                army.position[0],
                army.position[1],
                target_position[0],
                target_position[1],
                amount,
            )
            if board.move_legal(move, self.player):
                return move
        return None

    def _loiter_breakout_move(
        self,
        board: Board,
        ai: "GeneralsAI",
        army: IndependentArmy,
        blocked_targets: set[tuple[int, int]],
    ) -> Move | None:
        """Force a loitering army away from its repeated small-area anchor."""
        source = board.tile(*army.position)
        amount = source.army - 1
        if amount < 1:
            return None
        center = army.loiter_center or army.position
        current_distance = Board.manhattan(army.position, center)
        own_general = board.general_position(self.player)
        safe_best: tuple[float, Move] | None = None
        risky_best: tuple[float, Move] | None = None
        for dx, dy in DIRECTIONS:
            target_position = (
                army.position[0] + dx,
                army.position[1] + dy,
            )
            if (
                not board.in_bounds(*target_position)
                or target_position in blocked_targets
                or target_position in self.positions
                or self.has_main_army_tag(target_position)
                or self._is_immediate_reverse(
                    army,
                    target_position,
                    board,
                )
            ):
                continue
            target = board.tile(*target_position)
            if target.terrain == MOUNTAIN:
                continue
            if (
                target.terrain == GENERAL
                and target.owner == self.player
                and not self.has_main_army_tag(target_position)
            ):
                continue
            move = Move(
                army.position[0],
                army.position[1],
                target_position[0],
                target_position[1],
                amount,
            )
            if not board.move_legal(move, self.player):
                continue
            distance_gain = (
                Board.manhattan(target_position, center)
                - current_distance
            )
            score = distance_gain * 260.0
            if (
                target.terrain == GENERAL
                and ai._is_enemy(target.owner)
                and amount > target.army
            ):
                score += 2_000_000.0
            elif ai._is_enemy(target.owner):
                score += 420.0 + min(amount, target.army) * 4.0
            elif target.owner == NEUTRAL:
                score += 180.0
            elif target.owner == self.player:
                score += 90.0
            if target_position == own_general:
                score -= 500.0
            if self._is_dangerous_enemy_tile(
                board,
                ai,
                target_position,
                amount,
            ):
                if risky_best is None or score > risky_best[0]:
                    risky_best = (score, move)
            elif safe_best is None or score > safe_best[0]:
                safe_best = (score, move)

        best = safe_best or risky_best
        if best is None:
            return None
        army.target = best[1].target
        army.loiter_breakout_turns = max(
            0,
            army.loiter_breakout_turns - 1,
        )
        if (
            army.loiter_center is not None
            and Board.manhattan(
                best[1].target,
                army.loiter_center,
            )
            > INDEPENDENT_ARMY_LOITER_RADIUS
        ):
            army.loitering = False
            army.loiter_center = None
            army.loiter_breakout_turns = 0
        self._record_army_move(army, best[1].target, board)
        return best[1]

    def _move_toward(
        self,
        board: Board,
        ai: "GeneralsAI",
        source: tuple[int, int],
        target: tuple[int, int],
        amount: int,
        blocked_targets: set[tuple[int, int]] | None = None,
        minimum_distance: int | None = None,
    ) -> Move | None:
        if amount < 1 or source == target:
            return None
        distance = ai._distance_map(board, [target])
        current = distance.get(source)
        if current is None:
            return None
        safe_best: tuple[int, int, int] | None = None
        safe_position: tuple[int, int] | None = None
        risky_best: tuple[int, int, int] | None = None
        risky_position: tuple[int, int] | None = None
        for dx, dy in DIRECTIONS:
            position = (source[0] + dx, source[1] + dy)
            if not board.in_bounds(*position):
                continue
            if blocked_targets is not None and position in blocked_targets:
                continue
            tile = board.tile(*position)
            if tile.terrain == MOUNTAIN or (
                tile.terrain == GENERAL
                and tile.owner == self.player
                and not self.has_main_army_tag(position)
            ):
                continue
            if position in self.positions:
                continue
            next_distance = distance.get(position)
            if next_distance is None:
                continue
            if next_distance >= current:
                if minimum_distance is None or next_distance > minimum_distance:
                    continue
            candidate = (next_distance, -tile.army, position[0] + position[1])
            dangerous = self._is_dangerous_enemy_tile(
                board,
                ai,
                position,
                amount,
            )
            if dangerous:
                if risky_best is None or candidate < risky_best:
                    risky_best = candidate
                    risky_position = position
            elif safe_best is None or candidate < safe_best:
                safe_best = candidate
                safe_position = position
        if safe_position is not None:
            best_position = safe_position
        elif risky_position is not None:
            best_position = risky_position
        else:
            return None
        move = Move(source[0], source[1], best_position[0], best_position[1], amount)
        if not board.move_legal(move, self.player):
            return None
        return move

    def _record_army_move(
        self,
        army: IndependentArmy,
        target: tuple[int, int],
        board: Board,
    ) -> None:
        army.last_move_source = army.position
        army.last_move_turn = board.turn
        self.last_activity_turn[army.position] = board.turn
        self.last_activity_turn[target] = board.turn
        army.transit_target = target
        if board.movement_turns(
            army.position[0],
            army.position[1],
            target[0],
            target[1],
            army.player,
        ) > 1:
            army.transit_ready_turn = board.turn + 1
            return
        # The controller reconciles this prediction after movement resolution.
        army.transit_ready_turn = board.turn


class DistanceField:
    __slots__ = ("width", "height", "values", "infinity", "_sparse")

    def __init__(
        self,
        width: int,
        height: int,
        values: list[int] | dict[int, int],
        infinity: int,
    ) -> None:
        self.width = width
        self.height = height
        self.values = values
        self.infinity = infinity
        self._sparse = isinstance(values, dict)

    def get(
        self,
        position: tuple[int, int],
        default: int | None = None,
    ) -> int | None:
        x, y = position
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return default
        if self._sparse:
            value = self.values.get(y * self.width + x, self.infinity)
        else:
            value = self.values[y * self.width + x]
        return default if value == self.infinity else value

    def __getitem__(self, position: tuple[int, int]) -> int:
        value = self.get(position)
        if value is None:
            raise KeyError(position)
        return value

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DistanceField)
            and self.width == other.width
            and self.height == other.height
            and self.values == other.values
        )


def assign_ai_archetypes(count: int, seed: int | None = None) -> list[str]:
    if count < 0:
        raise ValueError("AI archetype count cannot be negative.")
    if count == 0:
        return []
    quotas = [
        int(count * weight)
        for _name, weight in ARCHETYPE_WEIGHTS
    ]
    assigned = [
        name
        for (name, _weight), quota in zip(ARCHETYPE_WEIGHTS, quotas)
        for _ in range(quota)
    ]
    rng = random.Random(seed)
    while len(assigned) < count:
        assigned.append(
            rng.choices(
                [name for name, _weight in ARCHETYPE_WEIGHTS],
                weights=[weight for _name, weight in ARCHETYPE_WEIGHTS],
                k=1,
            )[0]
        )
    rng.shuffle(assigned)
    return assigned


class GeneralsAI:
    """A fog-respecting tactical AI that treats every other player as hostile."""

    def __init__(
        self,
        player: int = AI,
        difficulty: str = "normal",
        seed: int | None = None,
        genome: AIGenome | Mapping[str, object] | None = None,
        archetype: str | None = None,
    ) -> None:
        if not 0 <= player < MAX_PLAYERS:
            raise ValueError(f"AI player must be between 0 and {MAX_PLAYERS - 1}.")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"Unknown difficulty: {difficulty}")
        self.player = player
        self.difficulty = DIFFICULTIES[difficulty]
        self.genome = (
            genome
            if isinstance(genome, AIGenome)
            else AIGenome.from_mapping(genome)
        )
        self.archetype = archetype or ARCHETYPE_STANDARD
        if self.archetype not in ARCHETYPE_LABELS:
            raise ValueError(f"Unknown AI archetype: {self.archetype}")
        self.archetype_label = ARCHETYPE_LABELS[self.archetype]
        self.archetype_capital_target_army = 0
        self.turtle_accumulating = self.archetype == ARCHETYPE_TURTLE
        self.turtle_released = False
        self.turtle_all_in_turns = 0
        self.archetype_update_turn = -1
        self.rng = random.Random(seed)
        self.army_controller = ArmyController(player, self.rng)
        self.last_move: Move | None = None
        self.exploration_direction: tuple[int, int] | None = None
        self.mission: Mission | None = None
        self.mission_changed = False
        self.strategy_state = STRATEGY_EXPLORATION
        self.strategy_state_reason = "opening exploration"
        self.strategy_state_turns = 0
        self.strategy_state_ready = False
        self.strategy_update_turn = -1
        self.enemy_army_estimate: dict[int, int] = {}
        self.enemy_visible_tiles: dict[int, int] = {}
        self.overwhelming_target: tuple[int, int] | None = None
        self.overwhelming_target_player = -1
        self.overwhelming_streak = 0
        self.overwhelming_last_turn = -1
        self.overwhelming_hold_turns = 0
        self.development_frozen_turns = 0
        self.border_muster_target: tuple[int, int] | None = None
        self.border_muster_player = -1
        self.border_muster_hold_turns = 0
        self.border_muster_cooldown_turns = 0
        self.border_muster_army_target = 0
        self.border_muster_ready = False
        self.last_attrition_turn = -10_000
        self.last_significant_attrition_turn = 0
        self.war_tempo_cooldown_turns = 0
        self.planning_owned_tiles: list[tuple[int, int]] = []
        self.planning_has_visible_enemy = False
        self.planning_max_mobile_army = 1
        self.planning_army_bootstrap_ready = False
        self.visibility_cache_stamp: tuple[int, int, int] | None = None
        self.visibility_cache: set[tuple[int, int]] = set()
        self.gather_mode: str | None = None
        self.gather_army_mode = "half"
        self.gather_route: GatherRoute | None = None
        self.gather_idle_region_last_move: dict[tuple[int, int], int] = {}
        self.known_terrain: dict[tuple[int, int], int] = {}
        self.seen_land: set[tuple[int, int]] = set()
        self.turns_since_contact = 0
        self.search_pressure = 0.0
        self.last_owned_tile_count = 0
        self.last_expansion_turn = -10_000
        self.expansion_stall_turns = 0
        self.aggression_heat = 0
        self.last_seen_enemy: tuple[int, int] | None = None
        self.last_contact_turn = -10_000
        self.search_waypoint: tuple[int, int] | None = None
        self.search_turn = -10_000
        self.global_search_turn = -10_000
        self.search_phase = 0
        self.long_range_search = False
        self.planning_phase = 0
        self.last_planned_phase = (-1, -1)
        self.primary_stack: tuple[int, int] | None = None
        self.primary_transit_target: tuple[int, int] | None = None
        self.primary_reinforce_target: tuple[int, int] | None = None
        self.frontline_stall_turns = 0
        self.home_anchor: tuple[int, int] | None = None
        self.muster_turns = 0
        self.forced_offensive = False
        self.force_mission_reselect = False
        self.offensive_hold_turns = 0
        self.best_search_distance = 10**9
        self.search_stall_turns = 0
        self.campaign_target: tuple[int, int] | None = None
        self.campaign_target_player: int = -1
        self.campaign_target_kind: str | None = None
        self.campaign_hold_turns = 0
        self.home_threat_target: tuple[int, int] | None = None
        self.major_invasion = False
        self.major_invasion_player = -1
        self.major_invasion_army = 0
        self.major_invasion_turn = -1
        self.home_guard_target_army = 0
        self.home_guard_deficit = 0
        self.home_guard_hard_target_army = 0
        self.home_guard_hard_deficit = 0
        self.home_support_need = 0
        self.home_support_last_source: tuple[int, int] | None = None
        self.home_support_last_score = 0.0
        self.capital_depth = 1
        self.capital_enemy_distance: int | None = None
        self.reinforcement_priority: dict[tuple[int, int], float] = {}
        self.passive_defense_turns = 0
        self.all_in_target: tuple[int, int] | None = None
        self.all_in_hold_turns = 0
        self.exposed_general_player = -1
        self.attrition_by_opponent: dict[int, int] = {}
        self.attrition_last_update: dict[int, int] = {}
        self.attrition_ingested_turn = -1
        self.war_state = "peace"
        self.war_state_opponent = -1
        self.war_state_started_turn = -1
        self.war_last_exchange_turn = -10_000
        self.war_last_exchange_opponent = -1
        self.war_last_exchange_amount = 0
        self.war_force_ratio = 1.0
        self.global_army_rank = 0
        self.global_army_rank_fraction = 0.5
        self.global_army_rank_low = False
        self.global_army_rank_high = False
        self.development_target: tuple[int, int] | None = None
        self.development_target_kind: str | None = None
        self.development_hold_turns = 0
        self.development_trigger_opponent = -1
        self.development_stack: tuple[int, int] | None = None
        self.development_transit_target: tuple[int, int] | None = None
        self.development_phase = 0
        self.known_neutral_cities: set[tuple[int, int]] = set()
        self.known_enemy_tiles: dict[int, set[tuple[int, int]]] = {}
        self.known_enemy_generals: dict[int, tuple[int, int]] = {}
        self.enemy_search_visited: set[tuple[int, int]] = set()
        self.enemy_hunt_player = -1
        self.enemy_hunt_target: tuple[int, int] | None = None
        self.enemy_hunt_hold_turns = 0
        self.distance_cache: dict[
            tuple[
                int,
                int,
                tuple[tuple[int, int], ...],
                int | None,
            ],
            DistanceField,
        ] = {}
        self.mission_distance_cache: dict[
            tuple[
                int,
                int,
                int,
                int,
                str,
                tuple[tuple[int, int], ...],
            ],
            DistanceField,
        ] = {}
        self.distance_cache_stamp: tuple[int, int] | None = None
        self.persistent_distance_cache: OrderedDict[
            tuple[int, int, int, tuple[tuple[int, int], ...]],
            DistanceField,
        ] = OrderedDict()
        self.mission_evaluated_turn = -1
        self.planning_engulf_cache: dict[
            tuple[int, int],
            tuple[int, int],
        ] = {}
        self.planning_engulf_stamp: tuple[int, int] | None = None

    def plan_moves(
        self,
        board: Board,
        phase: int = 0,
        observed_moves: tuple[Move, ...] = (),
    ) -> list[Move]:
        """Plan one ordinary AI action plus one independent action per army."""
        if board.winner is not None or self.player not in board.active_players:
            return []
        self.army_controller.sync(board, self)
        main_moves = self._plan_main_move(board, phase)
        army_positions = self.army_controller.positions
        army_exclusion = self.army_controller.occupied_positions(board)
        main_moves = [
            move
            for move in main_moves
            if move.source not in army_exclusion
            and move.target not in army_exclusion
        ]
        if not main_moves:
            main_moves = self._safe_main_fallback(board)
        army_moves = self.army_controller.plan_moves(
            board,
            self,
            phase,
            tuple(main_moves),
            observed_moves=observed_moves,
        )
        return [*main_moves, *army_moves]

    def _plan_main_move(self, board: Board, phase: int = 0) -> list[Move]:
        if board.winner is not None or self.player not in board.active_players:
            return []
        self.planning_phase = phase
        cache_stamp = (board.turn, phase)
        if self.planning_engulf_stamp != cache_stamp:
            self.planning_engulf_cache.clear()
            self.planning_engulf_stamp = cache_stamp
        if self.distance_cache_stamp != cache_stamp:
            if self.distance_cache_stamp is None or self.distance_cache_stamp[0] != board.turn:
                self.distance_cache.clear()
                self.mission_distance_cache.clear()
            self.distance_cache_stamp = cache_stamp

        owned_tiles: list[tuple[int, int]] = []
        movable: list[tuple[int, int]] = []
        land_tiles: list[tuple[int, int]] = []
        own_cities: list[tuple[int, int]] = []
        self.planning_max_mobile_army = 1
        positions = self._positions_by_owner(board)
        army_positions = self.army_controller.positions
        grid = board.grid
        player = self.player
        if self.player < len(positions):
            for position in positions[self.player]:
                tile = grid[position[1]][position[0]]
                owned_tiles.append(position)
                if tile.army > 1 and position not in army_positions:
                    movable.append(position)
                    self.planning_max_mobile_army = max(
                        self.planning_max_mobile_army,
                        tile.army - 1,
                    )
                if tile.terrain in (PLAIN, HILL):
                    land_tiles.append(position)
                if tile.terrain == CITY:
                    own_cities.append(position)
        if not movable:
            return []
        self.planning_army_bootstrap_ready = (
            self.army_controller.needs_army_bootstrap(
                board,
                owned_tiles,
            )
        )
        self.planning_owned_tiles = owned_tiles
        visible = self._cached_visibility(board, owned_tiles)
        self.planning_has_visible_enemy = any(
            (owner := grid[position[1]][position[0]].owner) >= 0
            and owner != player
            for position in visible
        )

        general = board.general_position(self.player)
        if general is not None and self.home_anchor is None:
            self.home_anchor = general
        if general is not None and self.exploration_direction is None:
            horizontal = 1 if general[0] < board.width / 2 else -1
            vertical = 1 if general[1] < board.height / 2 else -1
            self.exploration_direction = (horizontal, vertical)
        self._update_reinforcement_priority(board, owned_tiles)
        for position in visible:
            tile = grid[position[1]][position[0]]
            self.known_terrain[position] = tile.terrain
            if tile.terrain == CITY and tile.owner == NEUTRAL:
                self.known_neutral_cities.add(position)
            elif tile.terrain == CITY:
                self.known_neutral_cities.discard(position)
        self._update_enemy_intel(board, visible)
        self._sync_primary_transit(board)
        self._sync_development_transit(board)
        if (
            self.primary_reinforce_target == self.primary_stack
            and self._primary_stack_usable(board)
            and board.tile(*self.primary_stack).army > 1
        ):
            self.primary_reinforce_target = None
        self._update_attrition_state(board, owned_tiles)
        self._update_strategy_state(board, visible, owned_tiles, general)
        self._update_defense_intent(board, visible, owned_tiles, general)
        self._update_archetype_state(board, general)
        self._update_self_state(board, visible, owned_tiles, general)
        reuse_mission = (
            self.mission is not None
            and self.mission_evaluated_turn == board.turn
            and not self.force_mission_reselect
            and self._mission_target_valid(board, self.mission, visible)
        )
        # Re-evaluate missions every two turns on every map. Forced reselection
        # and invalid targets still bypass this interval.
        mission_interval = 2
        if (
            not reuse_mission
            and self.mission is not None
            and self.mission_evaluated_turn >= 0
            and board.turn - self.mission_evaluated_turn < mission_interval
            and not self.force_mission_reselect
            and self._mission_target_valid(board, self.mission, visible)
        ):
            reuse_mission = True
        if reuse_mission:
            self.mission_changed = False
        else:
            self.mission, self.mission_changed = self._update_mission(
                board,
                visible,
                owned_tiles,
                land_tiles,
                own_cities,
                general,
            )
            self.mission_evaluated_turn = board.turn
        if self.mission is None:
            return self._fallback_move(board, visible, movable)
        self._update_border_muster(
            board,
            visible,
            owned_tiles,
            general,
        )
        self._update_war_tempo(
            board,
            visible,
            owned_tiles,
            general,
        )
        self._update_gather_mode(board, visible, land_tiles, own_cities)
        self._update_development_state(board, visible, owned_tiles)

        decapitation_move = self._decapitation_move(
            board,
            visible,
            movable,
        )
        if (
            decapitation_move is not None
            and decapitation_move.source not in army_positions
            and decapitation_move.target not in army_positions
        ):
            self._record_main_move_activity(board, decapitation_move)
            self.last_move = decapitation_move
            return [decapitation_move]

        if (
            general is not None
            and (
                self.mission.kind == DEFEND_HOME_CITY
                or self.home_threat_target is not None
            )
        ):
            defense_move = self._forced_home_defense_move(
                board,
                visible,
                movable,
                general,
            )
            if (
                defense_move is not None
                and defense_move.source not in army_positions
                and defense_move.target not in army_positions
            ):
                self._record_main_move_activity(board, defense_move)
                self.last_move = defense_move
                return [defense_move]

        if self._should_run_development_phase(board):
            development_move = self._forced_development_move(board)
            if (
                development_move is not None
                and development_move.source not in army_positions
                and development_move.target not in army_positions
            ):
                self._record_main_move_activity(board, development_move)
                self.last_move = development_move
                return [development_move]

        if self.forced_offensive:
            vanguard_move = self._forced_vanguard_move(board, visible)
            if (
                vanguard_move is not None
                and vanguard_move.source not in army_positions
                and vanguard_move.target not in army_positions
            ):
                self._record_main_move_activity(board, vanguard_move)
                self.last_move = vanguard_move
                return [vanguard_move]

        strategic_goal = self.mission.kind != EXPLORE_EXPAND
        local_expansion = (
            self.mission.kind == EXPLORE_EXPAND
            and not self.forced_offensive
            and self.search_waypoint != self.mission.target
        )
        mission_distance = (
            {}
            if local_expansion
            else self._mission_distance_map(
                board,
                visible,
                [self.mission.target],
            )
        )
        home_distance = self._distance_map(board, [general]) if general is not None else {}
        max_distance = board.width + board.height
        candidates: list[tuple[float, Move]] = []

        for sx, sy in movable:
            source = grid[sy][sx]
            source_frontier = self._is_frontier(board, sx, sy, visible)
            source_distance = mission_distance.get((sx, sy), max_distance)
            source_home_distance = home_distance.get((sx, sy), max_distance)

            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                tx, ty = sx + dx, sy + dy
                if not board.in_bounds(tx, ty):
                    continue
                target = grid[ty][tx]
                if target.terrain == MOUNTAIN:
                    continue
                if (tx, ty) in army_positions:
                    continue

                target_visible = (tx, ty) in visible
                target_distance = mission_distance.get((tx, ty), max_distance)
                target_home_distance = home_distance.get((tx, ty), max_distance)
                if local_expansion:
                    progress = (
                        Board.manhattan((sx, sy), self.mission.target)
                        - Board.manhattan((tx, ty), self.mission.target)
                    )
                else:
                    progress = source_distance - target_distance
                home_progress = source_home_distance - target_home_distance
                away_from_home = max(0, target_home_distance - source_home_distance)
                direction = self.exploration_direction or (0, 0)
                direction_alignment = direction[0] * dx + direction[1] * dy
                expansion = self._open_neighbor_count(board, tx, ty)
                expansion_shape = self._expansion_shape_score(
                    board,
                    tx,
                    ty,
                    visible,
                    self.player,
                )
                score = -100_000.0
                amount = 0

                if target_visible and target.owner == self.player:
                    target_is_frontier = self._is_frontier(board, tx, ty, visible)
                    threat = self._nearby_enemy_strength(board, tx, ty, visible)
                    if source.army >= 3 and target_is_frontier and threat > 0:
                        score = (
                            5_000
                            + max(0, progress) * 320
                            + target.army * 24
                            + threat * 18
                        ) * self.difficulty.consolidation
                        amount = source.army - 1
                    elif source.army >= 3 and progress > 0:
                        score = (
                            3_600
                            + progress * 430
                            + target.army * 22
                            + min(source.army, 20) * 120
                            + away_from_home * 80
                        ) * self.difficulty.consolidation
                        amount = source.army - 1
                    elif (
                        source.army >= 3
                        and target_is_frontier
                        and target.army >= source.army
                        and progress >= 0
                    ):
                        score = 2_700 + target.army * 28
                        amount = source.army - 1
                    elif source.army >= 2 and progress > 0:
                        score = (
                            2_400
                            + progress * 360
                            + min(source.army, 10) * 75
                        )
                        amount = 1

                elif target_visible and target.terrain == GENERAL and self._is_enemy(target.owner):
                    if source.army - 1 > target.army:
                        score = 1_000_000 + source.army * 25
                        amount = source.army - 1

                elif target_visible and target.owner == NEUTRAL and target.terrain == CITY:
                    if source.army - 1 > target.army:
                        score = (
                            18_000
                            + max(0, 50 - target.army) * 28
                            + max(0, progress) * 320
                        ) * self.difficulty.aggression
                        amount = source.army - 1

                elif target_visible and target.owner == NEUTRAL:
                    if (
                        (not strategic_goal or progress >= 0)
                        and source.army >= 3
                    ):
                        score = (
                            4_300
                            + max(0, progress)
                            * (
                                LOCAL_EXPANSION_PROGRESS_WEIGHT
                                if local_expansion
                                else 470
                            )
                            + away_from_home
                            * (
                                LOCAL_EXPANSION_HOME_WEIGHT
                                if local_expansion
                                else 240
                            )
                            + expansion * 45
                            + expansion_shape
                            * (
                                LOCAL_EXPANSION_SHAPE_MULTIPLIER
                                if local_expansion
                                else 1.0
                            )
                            + min(source.army, 10) * 24
                            + direction_alignment
                            * (
                                LOCAL_EXPANSION_DIRECTION_WEIGHT
                                if local_expansion
                                else EXPANSION_DIRECTION_ALIGNMENT_BONUS
                            )
                        )
                        if local_expansion:
                            continuing = (
                                self.planning_phase == 1
                                and self.last_move is not None
                                and (sx, sy) == self.last_move.target
                            )
                            amount = (
                                source.army - 1
                                if continuing
                                else max(1, (source.army + 1) // 2)
                            )
                        else:
                            amount = source.army - 1
                    elif (
                        progress >= 0
                        or (not strategic_goal and direction_alignment > 0)
                    ):
                        score = (
                            2_100
                            + max(0, progress)
                            * (
                                LOCAL_EXPANSION_PROGRESS_WEIGHT
                                if local_expansion
                                else 320
                            )
                            + away_from_home
                            * (
                                LOCAL_EXPANSION_HOME_WEIGHT
                                if local_expansion
                                else 220
                            )
                            + expansion * 34
                            + expansion_shape
                            * (
                                LOCAL_EXPANSION_SHAPE_MULTIPLIER
                                if local_expansion
                                else 1.0
                            )
                            + direction_alignment
                            * (
                                LOCAL_EXPANSION_DIRECTION_WEIGHT
                                if local_expansion
                                else EXPANSION_DIRECTION_ALIGNMENT_BONUS
                            )
                        )
                        amount = 1
                    elif source_frontier and expansion >= 2:
                        score = (
                            850
                            + away_from_home * 210
                            + expansion * 25
                            + expansion_shape
                            + direction_alignment
                            * EXPANSION_DIRECTION_ALIGNMENT_BONUS
                        )
                        amount = 1
                    else:
                        score = (
                            180
                            + away_from_home * 180
                            + expansion * 18
                            + expansion_shape
                            + direction_alignment
                            * EXPANSION_DIRECTION_ALIGNMENT_BONUS
                        )
                        amount = 1

                elif target_visible and self._is_enemy(target.owner):
                    safety_margin = 1 if self.difficulty.key != "hard" else 0
                    if source.army - 1 > target.army + safety_margin:
                        value = (
                            9_500
                            + target.army * 36
                            + max(0, progress) * 300
                        ) * self.difficulty.aggression
                        if target.terrain == CITY:
                            value += 5_500
                        if target.terrain == GENERAL:
                            value += 900_000
                        score = value + self._nearby_enemy_strength(
                            board, tx, ty, visible
                        ) * 5
                        amount = source.army - 1

                elif not target_visible:
                    if (
                        self.forced_offensive
                        and self.primary_stack == (sx, sy)
                    ):
                        amount = source.army - 1
                    elif source.army >= 3:
                        amount = min(source.army - 1, self.difficulty.exploration_army)
                    else:
                        amount = 1
                    score = (
                        1_250
                        + max(0, progress) * 260
                        + away_from_home * 210
                        + expansion * 22
                        + expansion_shape * 0.8
                        + direction_alignment
                        * EXPANSION_DIRECTION_ALIGNMENT_BONUS
                    )

                if amount > 0:
                    move = Move(sx, sy, tx, ty, amount)
                    move = self._cap_home_guard_move(board, move)
                    if move is None:
                        continue
                    if board.move_legal(move, self.player):
                        mission_bonus = self._mission_move_bonus(
                            board,
                            move,
                            source,
                            target,
                            visible,
                            progress,
                            source_frontier,
                        )
                        score += mission_bonus
                        score += self._army_bootstrap_move_bonus(
                            board,
                            move,
                            target,
                        )
                        score *= (
                            self._genome_action_multiplier(
                                target.owner,
                                target.terrain,
                                target_visible,
                                progress,
                            )
                            * self._local_engulf_weight_multiplier(
                                board,
                                visible,
                                move,
                                target,
                                target_visible,
                            )
                        )
                        if (
                            self.last_move is not None
                            and move.source == self.last_move.target
                            and move.target == self.last_move.source
                            and not self.mission_changed
                        ):
                            score -= 250_000
                        score -= max(0, home_progress) * 420
                        friendly_support = self._friendly_neighbor_count(board, tx, ty)
                        score += friendly_support * self.difficulty.coordinate_bonus
                        if (
                            local_expansion
                            and target.owner == NEUTRAL
                        ):
                            if friendly_support == 1:
                                score += 620
                            elif friendly_support > 1:
                                score -= (friendly_support - 1) * 190
                            if (
                                self.planning_phase == 1
                                and self.last_move is not None
                                and (sx, sy) == self.last_move.target
                            ):
                                score += 6_000
                        score += self.rng.random() * 4.0
                        candidates.append((score, move))

        if not candidates:
            return self._fallback_move(board, visible, movable)
        candidates.sort(key=lambda item: (-item[0], item[1].source))
        if (
            self.difficulty.mistake_chance
            and len(candidates) > 1
            and self.rng.random() < self.difficulty.mistake_chance
        ):
            pick_window = min(5 if self.difficulty.key == "easy" else 3, len(candidates))
            choice = self.rng.randrange(1, pick_window)
        else:
            pick_window = 3 if self.difficulty.key == "easy" else 1
            choice = self.rng.randrange(min(pick_window, len(candidates)))
        selected = candidates[choice][1]
        if (
            self.forced_offensive
            and selected.source == self.primary_stack
        ):
            self._advance_primary(board, selected)
        self._record_main_move_activity(board, selected)
        self.last_move = selected
        return [selected]

    def _fallback_move(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        movable: list[tuple[int, int]],
    ) -> list[Move]:
        best: tuple[float, Move] | None = None
        target_goal = self.mission.target if self.mission is not None else None
        army_positions = self.army_controller.occupied_positions(board)
        for sx, sy in movable:
            source = board.tile(sx, sy)
            for tx, ty in board.legal_targets(sx, sy, self.player):
                if (tx, ty) in army_positions:
                    continue
                target = board.tile(tx, ty)
                amount = source.army - 1
                move = Move(sx, sy, tx, ty, amount)
                if not board.move_legal(move, self.player):
                    continue
                move = self._cap_home_guard_move(board, move)
                if move is None:
                    continue
                score = source.army * 120.0
                score += self._army_bootstrap_move_bonus(
                    board,
                    move,
                    target,
                )
                if target.owner == NEUTRAL:
                    score += 3_000
                elif target.owner == self.player:
                    score += 800 + target.army * 20
                elif self._is_enemy(target.owner):
                    if source.army - 1 > target.army:
                        score += 2_500 - target.army * 40
                    else:
                        score -= 2_000 + target.army * 50
                if target_goal is not None:
                    before = Board.manhattan((sx, sy), target_goal)
                    after = Board.manhattan((tx, ty), target_goal)
                    score += (before - after) * 520
                if target_goal == (tx, ty):
                    score += 1_200
                score *= self._genome_action_multiplier(
                    target.owner,
                    target.terrain,
                    target.owner >= 0,
                    0,
                )
                if (
                    self.last_move is not None
                    and move.source == self.last_move.target
                    and move.target == self.last_move.source
                    and not self.mission_changed
                ):
                    score -= 2_000
                score += self.rng.random() * 5.0
                if best is None or score > best[0]:
                    best = (score, move)
        if best is None:
            return []
        capped = best[1]
        if (
            self.forced_offensive
            and capped.source == self.primary_stack
        ):
            self._advance_primary(board, capped)
        self._record_main_move_activity(board, capped)
        self.last_move = capped
        return [capped]

    def _safe_main_fallback(self, board: Board) -> list[Move]:
        """Keep the main AI active without touching independent army cells."""
        positions = self._positions_by_owner(board)
        if self.player >= len(positions):
            return []
        army_positions = self.army_controller.occupied_positions(board)
        movable = [
            position
            for position in positions[self.player]
            if position not in army_positions
            and board.tile(*position).army > 1
        ]
        if not movable:
            return []
        return self._fallback_move(board, set(), movable)

    def _army_bootstrap_move_bonus(
        self,
        board: Board,
        move: Move,
        target: object,
    ) -> float:
        if not self.planning_army_bootstrap_ready:
            return 0.0
        if target.owner != self.player:
            return 0.0
        if target.terrain in (MOUNTAIN, GENERAL):
            return 0.0
        if target.army >= INDEPENDENT_ARMY_MIN_CREATE_STRENGTH:
            return 0.0
        if (
            target.army + move.amount
            < INDEPENDENT_ARMY_MIN_CREATE_STRENGTH
        ):
            return 0.0
        bonus = INDEPENDENT_ARMY_BOOTSTRAP_MERGE_BONUS
        if target.terrain == CITY:
            bonus += INDEPENDENT_ARMY_BOOTSTRAP_CITY_BONUS
        return bonus

    def _decapitation_move(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        movable: list[tuple[int, int]],
    ) -> Move | None:
        enemy_generals = [
            position
            for position in visible
            if board.tile(*position).terrain == GENERAL
            and self._is_enemy(board.tile(*position).owner)
        ]
        if not enemy_generals:
            return None

        best: tuple[int, Move] | None = None
        for sx, sy in movable:
            source = board.tile(sx, sy)
            for dx, dy in DIRECTIONS:
                target_position = (sx + dx, sy + dy)
                if target_position not in enemy_generals:
                    continue
                target = board.tile(*target_position)
                amount = source.army - 1
                if amount <= target.army:
                    continue
                move = Move(
                    sx,
                    sy,
                    target_position[0],
                    target_position[1],
                    amount,
                )
                if not board.move_legal(move, self.player):
                    continue
                if best is None or amount > best[0]:
                    best = (amount, move)
        return None if best is None else best[1]

    def _independent_army_source_excluded(
        self,
        position: tuple[int, int],
    ) -> bool:
        return position in self.army_controller.positions

    def _local_engulf_metrics(
        self,
        board: Board,
        target_position: tuple[int, int],
    ) -> tuple[int, int]:
        cached = self.planning_engulf_cache.get(target_position)
        if cached is not None:
            return cached

        target = board.tile(*target_position)
        general = board.general_position(self.player)
        home_guard = (
            0
            if self._all_in_penalties_suspended()
            else self.home_guard_hard_target_army
        )
        reserved = self._waiting_gather_positions()
        army_positions = self.army_controller.positions
        local_army = 0
        tx, ty = target_position
        for dx, dy in LOCAL_ENGULF_OFFSETS:
            sx = tx + dx
            sy = ty + dy
            if not board.in_bounds(sx, sy):
                continue
            source = board.tile(sx, sy)
            if source.owner != self.player or source.army <= 1:
                continue
            if (
                (sx, sy) in reserved
                or (sx, sy) in army_positions
            ):
                continue
            available = source.army - 1
            if (sx, sy) == general:
                available = max(0, available - home_guard)
            if available > 0:
                local_army += available

        difference = local_army - target.army
        required = max(
            LOCAL_ENGULF_MIN_DIFFERENCE,
            math.ceil(target.army * LOCAL_ENGULF_DIFFERENCE_RATIO),
        )
        self.planning_engulf_cache[target_position] = (difference, required)
        return difference, required

    def _local_engulf_weight_multiplier(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        move: Move,
        target: object,
        target_visible: bool,
    ) -> float:
        if (
            not target_visible
            or target.terrain == MOUNTAIN
            or not self._is_enemy(target.owner)
            or self._independent_army_source_excluded(move.source)
        ):
            return 1.0
        difference, required = self._local_engulf_metrics(
            board,
            move.target,
        )
        if difference > required:
            return LOCAL_ENGULF_WEIGHT_MULTIPLIER
        return 1.0

    def _waiting_gather_positions(self) -> set[tuple[int, int]]:
        reserved: set[tuple[int, int]] = set()
        if (
            self.border_muster_hold_turns > 0
            and not self.border_muster_ready
            and self.border_muster_target is not None
        ):
            reserved.add(self.border_muster_target)
        if self.development_stack is not None and self.development_hold_turns > 0:
            reserved.add(self.development_stack)
        if self.gather_mode in ("land", "city"):
            if self.primary_stack is not None:
                reserved.add(self.primary_stack)
        return reserved

    def _update_enemy_intel(
        self,
        board: Board,
        visible: set[tuple[int, int]],
    ) -> None:
        grid = board.grid
        player = self.player
        for opponent, observed in list(self.known_enemy_tiles.items()):
            if opponent in board.active_players:
                continue
            self.enemy_search_visited.difference_update(observed)
            self.known_enemy_tiles.pop(opponent, None)
            self.known_enemy_generals.pop(opponent, None)
            if self.enemy_hunt_player == opponent:
                self.enemy_hunt_player = -1
                self.enemy_hunt_target = None
                self.enemy_hunt_hold_turns = 0
            if self.exposed_general_player == opponent:
                self.exposed_general_player = -1
            if self.campaign_target_player == opponent:
                self._clear_campaign()

        for position in visible:
            tile = grid[position[1]][position[0]]
            if tile.terrain != MOUNTAIN:
                self.seen_land.add(position)

        for position in visible:
            tile = grid[position[1]][position[0]]
            opponent = tile.owner
            if opponent < 0 or opponent == player:
                continue
            self.known_enemy_tiles.setdefault(opponent, set()).add(position)
            self.enemy_search_visited.add(position)
            if tile.terrain == GENERAL:
                self.known_enemy_generals[opponent] = position

        for opponent, position in list(self.known_enemy_generals.items()):
            if opponent not in board.active_players:
                self.known_enemy_generals.pop(opponent, None)
                continue
            if position not in visible:
                continue
            tile = grid[position[1]][position[0]]
            if tile.terrain != GENERAL or tile.owner != opponent:
                self.known_enemy_generals.pop(opponent, None)
                if self.exposed_general_player == opponent:
                    self.exposed_general_player = -1
                if (
                    self.campaign_target == position
                    and self.campaign_target_player == opponent
                ):
                    self._clear_campaign()

        known_generals = [
            (opponent, position)
            for opponent, position in self.known_enemy_generals.items()
            if opponent in board.active_players
            and board.in_bounds(*position)
        ]
        if known_generals:
            current = next(
                (
                    item
                    for item in known_generals
                    if item[0] == self.exposed_general_player
                ),
                None,
            )
            if current is None:
                current = min(
                    known_generals,
                    key=lambda item: (
                        min(
                            (
                                Board.manhattan(item[1], own)
                                for own in self.planning_owned_tiles
                            ),
                            default=0,
                        ),
                        item[1],
                    ),
                )
            opponent, position = current
            if self.exposed_general_player != opponent:
                self.exposed_general_player = opponent
                self.force_mission_reselect = True
            if (
                self.campaign_target != position
                or self.campaign_target_player != opponent
                or self.campaign_target_kind != ATTACK_ENEMY_CITY
            ):
                self.campaign_target = position
                self.campaign_target_player = opponent
                self.campaign_target_kind = ATTACK_ENEMY_CITY
                self.force_mission_reselect = True
            self.campaign_hold_turns = max(
                self.campaign_hold_turns,
                GENERAL_CAMPAIGN_HOLD_TURNS,
            )
            if self.enemy_hunt_player == opponent:
                self.enemy_hunt_player = -1
                self.enemy_hunt_target = None
                self.enemy_hunt_hold_turns = 0
            return

        if not any(self.known_enemy_tiles.values()):
            self.enemy_hunt_player = -1
            self.enemy_hunt_target = None
            self.enemy_hunt_hold_turns = 0
            return

        opponent = self.enemy_hunt_player
        if (
            opponent not in board.active_players
            or not self.known_enemy_tiles.get(opponent)
        ):
            preferred = (
                self.campaign_target_player,
                self.overwhelming_target_player,
                self.border_muster_player,
            )
            opponent = next(
                (
                    player
                    for player in preferred
                    if player in board.active_players
                    and self.known_enemy_tiles.get(player)
                ),
                -1,
            )
            if opponent < 0:
                opponent = min(
                    self.known_enemy_tiles,
                    key=lambda player: (
                        min(
                            (
                                Board.manhattan(position, own)
                                for position in self.known_enemy_tiles[player]
                                for own in self.planning_owned_tiles
                            ),
                            default=10**9,
                        ),
                        player,
                    ),
                )
            self.enemy_hunt_player = opponent
            self.enemy_hunt_target = None
            self.enemy_hunt_hold_turns = 0

        target = self.enemy_hunt_target
        target_is_stale = target is None
        if target is not None and board.in_bounds(*target):
            target_tile = board.tile(*target)
            if target in visible:
                target_is_stale = (
                    target_tile.owner != opponent
                    or target_tile.terrain == GENERAL
                )
        else:
            target_is_stale = True
        if target_is_stale:
            target = self._select_enemy_search_target(
                board,
                visible,
            )
            self.enemy_hunt_target = target
            self.enemy_hunt_hold_turns = 0
            if target is not None:
                self.force_mission_reselect = True
        else:
            self.enemy_hunt_hold_turns += 1
            if self.enemy_hunt_hold_turns >= ENEMY_SEARCH_TARGET_HOLD_TURNS:
                target = self._select_enemy_search_target(
                    board,
                    visible,
                )
                self.enemy_hunt_target = target
                self.enemy_hunt_hold_turns = 0
                if target is not None:
                    self.force_mission_reselect = True

    def _select_enemy_search_target(
        self,
        board: Board,
        visible: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        opponent = self.enemy_hunt_player
        known = self.known_enemy_tiles.get(opponent, set())
        if not known:
            return None

        all_known = set().union(*self.known_enemy_tiles.values())
        ring = set(known)
        scanned = set(ring)
        own_tiles = self.planning_owned_tiles

        for depth in range(1, ENEMY_SEARCH_RING_LIMIT + 1):
            next_ring: set[tuple[int, int]] = set()
            unvisited: list[tuple[int, int]] = []
            visited: list[tuple[int, int]] = []
            for x, y in ring:
                for dx, dy in DIRECTIONS:
                    position = (x + dx, y + dy)
                    if position in scanned or not board.in_bounds(*position):
                        continue
                    scanned.add(position)
                    tile = board.tile(*position)
                    if tile.terrain == MOUNTAIN:
                        continue
                    next_ring.add(position)
                    if (
                        position in all_known
                        or position in self.seen_land
                        or position in visible
                        and tile.owner == self.player
                    ):
                        continue
                    if position in self.enemy_search_visited:
                        visited.append(position)
                    else:
                        unvisited.append(position)
            candidates = unvisited or visited
            if candidates:
                def score(position: tuple[int, int]) -> tuple[int, int, int, int]:
                    adjacency = sum(
                        (position[0] + dx, position[1] + dy) in known
                        for dx, dy in DIRECTIONS
                    )
                    own_distance = min(
                        (
                            Board.manhattan(position, own)
                            for own in own_tiles
                        ),
                        default=10**9,
                    )
                    return (
                        -adjacency,
                        -own_distance,
                        depth,
                        position[0] + position[1],
                    )

                target = min(candidates, key=score)
                self.enemy_search_visited.add(target)
                return target
            ring = next_ring
            if not ring:
                break

        global_target = self._select_global_search_target(
            board,
            own_tiles,
            preferred=known,
        )
        if global_target is not None:
            self.enemy_search_visited.add(global_target)
            return global_target

        target = max(
            known,
            key=lambda position: (
                min(
                    (
                        Board.manhattan(position, own)
                        for own in own_tiles
                    ),
                    default=0,
                ),
                position,
            ),
        )
        return target

    def _select_global_search_target(
        self,
        board: Board,
        owned_tiles: Iterable[tuple[int, int]],
        *,
        preferred: Iterable[tuple[int, int]] = (),
        beacon: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        """Pick unseen land ahead of the current strategic search direction."""
        seen = self.seen_land
        owned = tuple(owned_tiles)
        general = board.general_position(self.player)
        if general is not None:
            origin = general
        elif self.home_anchor is not None:
            origin = self.home_anchor
        elif owned:
            origin = (
                sum(position[0] for position in owned) // len(owned),
                sum(position[1] for position in owned) // len(owned),
            )
        else:
            origin = (board.width // 2, board.height // 2)

        preferred_positions = tuple(
            position
            for position in preferred
            if board.in_bounds(*position)
        )
        preferred_center: tuple[int, int] | None = None
        if preferred_positions:
            preferred_center = (
                sum(position[0] for position in preferred_positions)
                // len(preferred_positions),
                sum(position[1] for position in preferred_positions)
                // len(preferred_positions),
            )

        direction = self.exploration_direction or (0, 0)

        if not preferred_positions and beacon is not None and board.in_bounds(*beacon):
            dx = 0 if beacon[0] == origin[0] else 1 if beacon[0] > origin[0] else -1
            dy = 0 if beacon[1] == origin[1] else 1 if beacon[1] > origin[1] else -1
            if dx or dy:
                if (
                    beacon not in seen
                    and self.known_terrain.get(beacon) != MOUNTAIN
                    and self.search_stall_turns < 12
                ):
                    self.enemy_search_visited.add(beacon)
                    return beacon
                minimum_step = max(6, min(board.width, board.height) // 6)
                for step in range(minimum_step, max(board.width, board.height) + 1):
                    position = (origin[0] + dx * step, origin[1] + dy * step)
                    if not board.in_bounds(*position):
                        break
                    if position in seen:
                        continue
                    if self.known_terrain.get(position) == MOUNTAIN:
                        continue
                    self.enemy_search_visited.add(position)
                    return position

        candidates: set[tuple[int, int]] = set()
        for x, y in seen:
            for dx, dy in DIRECTIONS:
                position = (x + dx, y + dy)
                if not board.in_bounds(*position):
                    continue
                if position in seen:
                    continue
                if self.known_terrain.get(position) == MOUNTAIN:
                    continue
                candidates.add(position)

        if not candidates:
            for x, y in owned:
                for dx, dy in DIRECTIONS:
                    position = (x + dx, y + dy)
                    if not board.in_bounds(*position):
                        continue
                    if position in seen:
                        continue
                    if self.known_terrain.get(position) == MOUNTAIN:
                        continue
                    candidates.add(position)

        if not candidates:
            return None

        unvisited = [
            position
            for position in candidates
            if position not in self.enemy_search_visited
        ]
        if unvisited:
            candidates = unvisited

        def distance_to_preferred(position: tuple[int, int]) -> int:
            if not preferred_positions:
                return 10**9
            if len(preferred_positions) <= 64:
                return min(
                    Board.manhattan(position, anchor)
                    for anchor in preferred_positions
                )
            assert preferred_center is not None
            return Board.manhattan(position, preferred_center)

        def score(position: tuple[int, int]) -> tuple[int, int, int, int, int]:
            own_distance = Board.manhattan(position, origin)
            information_gain = sum(
                (position[0] + dx, position[1] + dy) not in seen
                and self.known_terrain.get(
                    (position[0] + dx, position[1] + dy)
                )
                != MOUNTAIN
                for dx, dy in DIRECTIONS
            )
            if direction != (0, 0):
                dx = position[0] - origin[0]
                dy = position[1] - origin[1]
                alignment = (
                    direction[0] * (1 if dx > 0 else -1 if dx < 0 else 0)
                    + direction[1] * (1 if dy > 0 else -1 if dy < 0 else 0)
                )
            else:
                alignment = 0
            lead = (
                distance_to_preferred(position)
                if preferred_positions
                else -alignment
            )
            return (
                lead,
                -own_distance,
                -information_gain,
                -alignment,
                position[1] * board.width + position[0],
            )

        target = min(candidates, key=score)
        self.enemy_search_visited.add(target)
        return target

    def _update_strategy_state(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        area = board.width * board.height
        active_players = max(1, len(board.active_players))
        self.long_range_search = area >= 5_000 or area / active_players >= 1_200

        enemy_positions = [
            position
            for position in visible
            if self._is_enemy(board.tile(*position).owner)
        ]
        largest_position = (
            max(
                owned_tiles,
                key=lambda position: board.tile(*position).army,
            )
            if owned_tiles
            else None
        )
        largest_army = (
            board.tile(*largest_position).army
            if largest_position is not None
            else 0
        )
        if self.last_planned_phase == (board.turn, self.planning_phase):
            return
        self.last_planned_phase = (board.turn, self.planning_phase)
        time_step = self.planning_phase == 0

        if time_step:
            owned_count = len(owned_tiles)
            if owned_count > self.last_owned_tile_count:
                self.last_expansion_turn = board.turn
                self.expansion_stall_turns = 0
            else:
                self.expansion_stall_turns += 1
            self.last_owned_tile_count = owned_count

        if self.campaign_target is not None:
            target = self.campaign_target
            if (
                self.campaign_target_player not in board.active_players
                or board.tile(*target).terrain == MOUNTAIN
                or target in visible
                and not self._is_enemy(board.tile(*target).owner)
            ):
                self._clear_campaign()

        if enemy_positions:
            self.turns_since_contact = 0
            self.search_pressure = max(0.0, self.search_pressure - 20.0)
            self.aggression_heat = max(0, self.aggression_heat - 8)
            general_contacts = [
                position
                for position in enemy_positions
                if board.tile(*position).terrain == GENERAL
            ]
            if general_contacts:
                nearest_general = min(
                    general_contacts,
                    key=lambda position: (
                        min(
                            Board.manhattan(position, own)
                            for own in owned_tiles
                        )
                        if owned_tiles
                        else 0,
                        position,
                    ),
                )
                target_tile = board.tile(*nearest_general)
                if (
                    self.campaign_target != nearest_general
                    or self.campaign_target_player != target_tile.owner
                ):
                    self.campaign_target = nearest_general
                    self.campaign_target_player = target_tile.owner
                    self.campaign_target_kind = ATTACK_ENEMY_CITY
                    self.force_mission_reselect = True
                self.campaign_hold_turns = GENERAL_CAMPAIGN_HOLD_TURNS
            elif self.campaign_target is None:
                anchor = max(
                    enemy_positions,
                    key=lambda position: (
                        board.tile(*position).army,
                        board.tile(*position).terrain == CITY,
                        -min(
                            (
                                Board.manhattan(position, own)
                                for own in owned_tiles
                            ),
                            default=0,
                        ),
                    ),
                )
                tile = board.tile(*anchor)
                self.campaign_target = anchor
                self.campaign_target_player = tile.owner
                self.campaign_target_kind = (
                    ATTACK_ENEMY_CITY
                    if tile.terrain == CITY
                    else ATTACK_ENEMY_LAND
                )
                self.campaign_hold_turns = ENEMY_TARGET_HOLD_TURNS
                self.force_mission_reselect = True
            elif self.campaign_target in visible:
                self.campaign_hold_turns = max(
                    self.campaign_hold_turns,
                    ENEMY_TARGET_HOLD_TURNS,
                )
            total_x = sum(position[0] for position in enemy_positions)
            total_y = sum(position[1] for position in enemy_positions)
            self.last_seen_enemy = (
                total_x // len(enemy_positions),
                total_y // len(enemy_positions),
            )
            self.last_contact_turn = board.turn
            self.search_waypoint = self.last_seen_enemy
            self.search_turn = board.turn
            if (
                not self.forced_offensive
                or not self._primary_stack_usable(board)
            ):
                self.primary_stack = self._select_offensive_stack(
                    board,
                    owned_tiles,
                    self.last_seen_enemy,
                )
            if not self.forced_offensive:
                self.force_mission_reselect = True
            self.forced_offensive = True
            self.muster_turns = 0
            self.offensive_hold_turns = 40
            self.gather_mode = None
            return

        if time_step:
            self.turns_since_contact += 1
            self.search_pressure = min(
                600.0,
                self.search_pressure
                + 1.0
                + min(2.0, self.expansion_stall_turns / 90.0),
            )
            if self.campaign_hold_turns > 0:
                known_general = (
                    self.campaign_target is not None
                    and self.known_enemy_generals.get(
                        self.campaign_target_player
                    )
                    == self.campaign_target
                )
                if not known_general:
                    self.campaign_hold_turns -= 1
                    if self.campaign_hold_turns == 0:
                        self._clear_campaign()
            self.aggression_heat = min(120, self.aggression_heat + 2)
            if self.offensive_hold_turns > 0:
                self.offensive_hold_turns -= 1
        if time_step and self.search_waypoint is not None and owned_tiles:
            distance = min(
                Board.manhattan(position, self.search_waypoint)
                for position in owned_tiles
            )
            if distance < self.best_search_distance - 1:
                self.best_search_distance = distance
                self.search_stall_turns = 0
            else:
                self.search_stall_turns += 1
        else:
            self.search_stall_turns = 0

        map_muster = min(55, (board.width + board.height) // 4)
        muster_threshold = max(
            8,
            20 - len(board.active_players) * 2,
            round(map_muster * self.difficulty.muster_ratio),
        )
        if (
            largest_army >= muster_threshold
            or self.turns_since_contact >= self.difficulty.search_contact_turns
            or self.search_stall_turns >= 30
        ):
            if not self.forced_offensive:
                self.force_mission_reselect = True
            self.forced_offensive = True
            if not self._primary_stack_usable(board):
                self.primary_stack = self._select_offensive_stack(
                    board,
                    owned_tiles,
                    self.search_waypoint or general,
                )
            self.muster_turns += 1
            self.offensive_hold_turns = max(self.offensive_hold_turns, 35)
            self.gather_mode = None
        elif not self.forced_offensive and largest_army < 3:
            self.forced_offensive = False
            self.muster_turns = 0
            self.primary_stack = None
            self.offensive_hold_turns = 0
        self._refresh_search_waypoint(board, owned_tiles, general)

    def _home_cells(
        self,
        board: Board,
        general: tuple[int, int],
    ) -> set[tuple[int, int]]:
        return {
            (x, y)
            for y in range(
                max(0, general[1] - HOME_THREAT_RADIUS),
                min(board.height, general[1] + HOME_THREAT_RADIUS + 1),
            )
            for x in range(
                max(0, general[0] - HOME_THREAT_RADIUS),
                min(board.width, general[0] + HOME_THREAT_RADIUS + 1),
            )
        }

    def _home_occupy_cells(
        self,
        board: Board,
        general: tuple[int, int],
    ) -> set[tuple[int, int]]:
        """Return the vision-sized square the AI prioritizes around its capital."""
        radius = int(board.vision_radius)
        return {
            (x, y)
            for y in range(
                max(0, general[1] - radius),
                min(board.height, general[1] + radius + 1),
            )
            for x in range(
                max(0, general[0] - radius),
                min(board.width, general[0] + radius + 1),
            )
        }

    def _general_guard_army(
        self,
        board: Board,
        general: tuple[int, int],
    ) -> int:
        tile = board.tile(*general)
        return tile.army if tile.owner == self.player else 0

    def _capital_depth(
        self,
        board: Board,
        general: tuple[int, int],
    ) -> int:
        gx, gy = general
        for distance in range(1, max(board.width, board.height) + 1):
            for dx in range(-distance, distance + 1):
                remaining = distance - abs(dx)
                offsets = (0,) if remaining == 0 else (-remaining, remaining)
                for dy in offsets:
                    x = gx + dx
                    y = gy + dy
                    if not board.in_bounds(x, y):
                        continue
                    tile = board.tile(x, y)
                    if tile.owner != self.player and tile.terrain != MOUNTAIN:
                        return distance
        return max(board.width, board.height) + 1

    def _capital_depth_requirement(self, turn: int) -> int:
        # Board turns are 1-based, so the first 100 turns keep the lower target.
        if turn <= CAPITAL_DEPTH_STEP_TURNS:
            return CAPITAL_DEPTH_INITIAL_TARGET
        increase = 1 + (
            turn - CAPITAL_DEPTH_STEP_TURNS - 1
        ) // CAPITAL_DEPTH_STEP_TURNS
        return min(
            CAPITAL_DEPTH_FINAL_TARGET,
            CAPITAL_DEPTH_INITIAL_TARGET + increase,
        )

    def _nearest_enemy_territory_distance(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        general: tuple[int, int],
    ) -> int | None:
        nearest: int | None = None
        for position in visible:
            if not self._is_enemy(board.tile(*position).owner):
                continue
            distance = Board.manhattan(position, general)
            if nearest is None or distance < nearest:
                nearest = distance
        for opponent, positions in self.known_enemy_tiles.items():
            if opponent not in board.active_players:
                continue
            for position in positions:
                distance = Board.manhattan(position, general)
                if nearest is None or distance < nearest:
                    nearest = distance
        return nearest

    def _capital_guard_ratios(self) -> tuple[float, float]:
        distance = self.capital_enemy_distance
        if distance is not None and distance > CAPITAL_ENEMY_GUARD_DEEP_DISTANCE:
            return CAPITAL_GUARD_DEEP_RATIO, CAPITAL_GUARD_DEEP_RATIO
        if distance is not None and distance > CAPITAL_ENEMY_GUARD_MEDIUM_DISTANCE:
            return CAPITAL_GUARD_MEDIUM_RATIO, CAPITAL_GUARD_MEDIUM_RATIO
        return HOME_GUARD_HARD_RATIO, HOME_GUARD_RATIO * self.genome.home_guard

    def _all_in_active(self) -> bool:
        return (
            self.all_in_target is not None
            or self.overwhelming_hold_turns > 0
        )

    def _all_in_penalties_suspended(self) -> bool:
        if (
            self.archetype == ARCHETYPE_ATTACK
            and self._all_in_active()
        ):
            return True
        return (
            self.archetype == ARCHETYPE_TURTLE
            and self.turtle_released
            and self.turtle_all_in_turns > 0
        )

    def _archetype_all_in_multiplier(self) -> float:
        if self.archetype == ARCHETYPE_ATTACK:
            return ARCHETYPE_ATTACK_ALL_IN_MULTIPLIER
        if (
            self.archetype == ARCHETYPE_TURTLE
            and self.turtle_released
            and self.turtle_all_in_turns > 0
        ):
            return ARCHETYPE_TURTLE_DECAPITATION_MULTIPLIER
        return 1.0

    def _is_weak_enemy_city(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> bool:
        if not board.in_bounds(*position):
            return False
        tile = board.tile(*position)
        if tile.terrain != CITY or not self._is_enemy(tile.owner):
            return False
        support = self._nearby_owned_army(
            board,
            position[0],
            position[1],
            3,
        )
        return (
            tile.army <= ARCHETYPE_FORT_WEAK_CITY_ARMY
            or tile.army * 4 <= support * 3
        )

    def _archetype_mission_multiplier(
        self,
        board: Board,
        mission_kind: str,
        target: tuple[int, int],
    ) -> float:
        if self.archetype == ARCHETYPE_ATTACK:
            if mission_kind in (ATTACK_ENEMY_CITY, ATTACK_ENEMY_LAND):
                return ARCHETYPE_ATTACK_MISSION_MULTIPLIER
            return 1.0

        if self.archetype == ARCHETYPE_FORT:
            if mission_kind == EXPLORE_EXPAND:
                return ARCHETYPE_FORT_EXPLORE_MULTIPLIER
            if mission_kind == ATTACK_NEUTRAL_CITY:
                return ARCHETYPE_FORT_CITY_MULTIPLIER
            if (
                mission_kind == ATTACK_ENEMY_CITY
                and self._is_weak_enemy_city(board, target)
            ):
                return ARCHETYPE_FORT_CITY_MULTIPLIER
            return 1.0

        if self.archetype != ARCHETYPE_TURTLE:
            return 1.0
        target_tile = (
            board.tile(*target)
            if board.in_bounds(*target)
            else None
        )
        is_decapitation = (
            mission_kind == ATTACK_ENEMY_CITY
            and target_tile is not None
            and target_tile.terrain == GENERAL
        )
        if self.turtle_accumulating:
            if mission_kind == EXPLORE_EXPAND:
                return ARCHETYPE_TURTLE_EARLY_EXPLORE_MULTIPLIER
            if mission_kind in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
                ATTACK_NEUTRAL_CITY,
            ):
                return ARCHETYPE_TURTLE_EARLY_ATTACK_MULTIPLIER
            return 1.0
        multiplier = 1.0
        if mission_kind in (
            ATTACK_ENEMY_CITY,
            ATTACK_ENEMY_LAND,
            EXPLORE_EXPAND,
        ):
            multiplier *= ARCHETYPE_TURTLE_RELEASED_ATTACK_MULTIPLIER
        if is_decapitation:
            multiplier *= ARCHETYPE_TURTLE_DECAPITATION_MULTIPLIER
        return multiplier

    def _update_archetype_state(
        self,
        board: Board,
        general: tuple[int, int] | None,
    ) -> None:
        if self.archetype != ARCHETYPE_TURTLE or general is None:
            return
        area = board.width * board.height
        self.archetype_capital_target_army = max(
            ARCHETYPE_TURTLE_CAPITAL_TARGET_MIN,
            min(
                ARCHETYPE_TURTLE_CAPITAL_TARGET_MAX,
                20 + area // 250,
            ),
        )
        if self.archetype_update_turn == board.turn:
            return
        self.archetype_update_turn = board.turn
        if self.turtle_released:
            self.turtle_all_in_turns = max(
                0,
                self.turtle_all_in_turns - 1,
            )
            return

        capital_army = board.tile(*general).army
        if capital_army < self.archetype_capital_target_army:
            self.turtle_accumulating = True
            return

        self.turtle_accumulating = False
        self.turtle_released = True
        self.turtle_all_in_turns = ARCHETYPE_TURTLE_ALL_IN_TURNS
        self.forced_offensive = True
        self.force_mission_reselect = True
        self.primary_reinforce_target = None

    def _capital_depth_target(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
    ) -> Mission | None:
        general = board.general_position(self.player)
        required_depth = self._capital_depth_requirement(board.turn)
        if general is None or self.capital_depth >= required_depth:
            return None

        owned_lookup = set(owned_tiles)
        gx, gy = general
        frontier_targets: list[tuple[tuple[int, int], int, int, str]] = []
        distance = self.capital_depth
        for dx in range(-distance, distance + 1):
            remaining = distance - abs(dx)
            offsets = (0,) if remaining == 0 else (-remaining, remaining)
            for dy in offsets:
                position = (gx + dx, gy + dy)
                if not board.in_bounds(*position):
                    continue
                tile = board.tile(*position)
                if tile.owner == self.player or tile.terrain == MOUNTAIN:
                    continue
                adjacent_frontier = any(
                    (position[0] + nx, position[1] + ny) in owned_lookup
                    for nx, ny in DIRECTIONS
                )
                if tile.owner == NEUTRAL:
                    mission_kind = (
                        ATTACK_NEUTRAL_CITY
                        if tile.terrain == CITY
                        else EXPLORE_EXPAND
                    )
                elif position in visible:
                    mission_kind = (
                        ATTACK_ENEMY_CITY
                        if tile.terrain in (CITY, GENERAL)
                        else ATTACK_ENEMY_LAND
                    )
                else:
                    mission_kind = EXPLORE_EXPAND
                frontier_targets.append(
                    (
                        position,
                        int(adjacent_frontier),
                        int(position in visible),
                        mission_kind,
                    )
                )

        if not frontier_targets:
            return None
        position, _, _, mission_kind = max(
            frontier_targets,
            key=lambda item: (
                item[1],
                item[2],
                -item[0][0] - item[0][1],
            ),
        )
        return Mission(
            mission_kind,
            position,
            (
                CAPITAL_DEPTH_PRIORITY_SCORE
                + min(
                    CAPITAL_DEPTH_DEFICIT_BONUS_CAP,
                    max(1, required_depth - self.capital_depth)
                    * CAPITAL_DEPTH_DEFICIT_BONUS,
                )
            ),
            "all",
            f"deepen capital buffer depth {self.capital_depth}/{required_depth}",
        )

    def _nearest_enemy_distance(
        self,
        position: tuple[int, int],
    ) -> int:
        if self.last_seen_enemy is None:
            return 1
        return max(1, Board.manhattan(position, self.last_seen_enemy))

    @staticmethod
    def _idle_distance_discount(idle_turns: int) -> float:
        for threshold, discount in GATHER_IDLE_DISTANCE_DISCOUNTS:
            if idle_turns < threshold:
                return discount
        return GATHER_IDLE_DISTANCE_DISCOUNTS[-1][1]

    def _gather_idle_region_key(
        self,
        position: tuple[int, int],
    ) -> tuple[int, int]:
        return (
            position[0] // GATHER_IDLE_REGION_SIZE,
            position[1] // GATHER_IDLE_REGION_SIZE,
        )

    def _record_gather_activity(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> None:
        self.gather_idle_region_last_move[
            self._gather_idle_region_key(position)
        ] = board.turn

    def _record_main_move_activity(self, board: Board, move: Move) -> None:
        self._record_gather_activity(board, move.source)
        self.army_controller.touch_main_army(board, move.source)

    def _gather_idle_turns(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> int:
        key = self._gather_idle_region_key(position)
        last_move = self.gather_idle_region_last_move.setdefault(
            key,
            board.turn,
        )
        return max(0, board.turn - last_move)

    def _effective_reinforcement_distance(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> float:
        distance = self._nearest_enemy_distance(position)
        if distance <= 1:
            return 1.0
        idle_turns = self._gather_idle_turns(board, position)
        discount = self._idle_distance_discount(idle_turns)
        return max(1.0, distance * (1.0 - discount))

    def _regional_average_army(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> float:
        total = 0
        count = 0
        x, y = position
        width = board.width
        height = board.height
        grid = board.grid
        player = self.player
        for dx, dy in REGION_REINFORCEMENT_OFFSETS:
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            tile = grid[ny][nx]
            if tile.owner != player:
                continue
            total += tile.army
            count += 1
        return total / count if count else 0.0

    def _regional_reinforcement_ratio(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> float:
        average = self._regional_average_army(board, position)
        return average / self._effective_reinforcement_distance(
            board,
            position,
        )

    def _update_reinforcement_priority(
        self,
        board: Board,
        owned_tiles: list[tuple[int, int]],
    ) -> None:
        priority: dict[tuple[int, int], float] = {}
        for position in owned_tiles:
            if board.tile(*position).army <= 1:
                continue
            ratio = self._regional_reinforcement_ratio(board, position)
            priority[position] = ratio
        self.reinforcement_priority = priority

    def _reinforcement_priority_bonus(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> float:
        ratio = self.reinforcement_priority.get(position)
        if ratio is None:
            ratio = self._regional_reinforcement_ratio(board, position)
        if ratio <= REGION_REINFORCEMENT_RATIO:
            return 0.0
        return (
            REGION_REINFORCEMENT_BONUS
            + min(20_000.0, (ratio - REGION_REINFORCEMENT_RATIO) * 2_500.0)
        )

    def _home_guard_priority_active(self) -> bool:
        return (
            self.last_contact_turn >= 0
            or self.home_threat_target is not None
            or self.home_guard_target_army > 0
            or self.home_guard_hard_target_army > 0
        )

    def _cap_home_guard_move(self, board: Board, move: Move) -> Move | None:
        general = board.general_position(self.player)
        if general is None:
            return move
        if self.army_controller.has_main_army_tag(move.source):
            return move
        if move.source != general or move.target == general:
            return move
        if not self._home_guard_priority_active():
            return move
        if self._all_in_penalties_suspended():
            return move
        current_guard = self._general_guard_army(board, general)
        reserve = self.home_guard_hard_target_army
        if reserve <= 0:
            total_army = sum(
                tile.army
                for row in board.grid
                for tile in row
                if tile.owner == self.player
            )
            hard_ratio, _ = self._capital_guard_ratios()
            reserve = (
                math.ceil(total_army * hard_ratio)
                if total_army >= 5
                else 0
            )
        max_outgoing = max(0, current_guard - reserve)
        amount = min(move.amount, max_outgoing)
        if amount <= 0:
            return None
        return Move(move.sx, move.sy, move.tx, move.ty, amount)

    def _update_defense_intent(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        if general is None:
            self.capital_depth = 1
            self.capital_enemy_distance = None
            self.home_threat_target = None
            self.major_invasion = False
            self.major_invasion_player = -1
            self.major_invasion_army = 0
            self.major_invasion_turn = -1
            self.home_guard_target_army = 0
            self.home_guard_deficit = 0
            self.home_guard_hard_target_army = 0
            self.home_guard_hard_deficit = 0
            self.home_support_need = 0
            self.home_support_last_source = None
            self.home_support_last_score = 0.0
            self.all_in_target = None
            self.all_in_hold_turns = 0
            return

        self.capital_depth = self._capital_depth(board, general)
        self.capital_enemy_distance = self._nearest_enemy_territory_distance(
            board,
            visible,
            general,
        )
        home_cells = self._home_cells(board, general)
        threats = [
            position
            for position in visible
            if position in home_cells
            and self._is_enemy(board.tile(*position).owner)
        ]
        threat_target = (
            min(
                threats,
                key=lambda position: (
                    Board.manhattan(position, general),
                    -board.tile(*position).army,
                    position,
                ),
            )
            if threats
            else None
        )
        if threat_target != self.home_threat_target:
            self.force_mission_reselect = True
        self.home_threat_target = threat_target

        total_army = sum(board.tile(*position).army for position in owned_tiles)
        if self.major_invasion_turn != board.turn:
            owned_lookup = set(owned_tiles)
            invasion_by_player: dict[int, int] = {}
            for position in visible:
                tile = board.tile(*position)
                if not self._is_enemy(tile.owner):
                    continue
                x, y = position
                near_owned = False
                for ny in range(
                    y - MAJOR_INVASION_RADIUS,
                    y + MAJOR_INVASION_RADIUS + 1,
                ):
                    for nx in range(
                        x - MAJOR_INVASION_RADIUS,
                        x + MAJOR_INVASION_RADIUS + 1,
                    ):
                        if (nx, ny) in owned_lookup:
                            near_owned = True
                            break
                    if near_owned:
                        break
                if near_owned:
                    invasion_by_player[tile.owner] = (
                        invasion_by_player.get(tile.owner, 0) + tile.army
                    )

            major_invasion_player = -1
            major_invasion_army = 0
            invasion_threshold = total_army * MAJOR_INVASION_ARMY_RATIO
            for opponent, army in invasion_by_player.items():
                if army <= invasion_threshold:
                    continue
                if army > major_invasion_army:
                    major_invasion_player = opponent
                    major_invasion_army = army
            major_invasion = major_invasion_player >= 0
            if major_invasion != self.major_invasion:
                self.force_mission_reselect = True
            self.major_invasion = major_invasion
            self.major_invasion_player = major_invasion_player
            self.major_invasion_army = major_invasion_army
            self.major_invasion_turn = board.turn

        current_guard = self._general_guard_army(board, general)
        guard_priority_active = (
            threat_target is not None or self.last_contact_turn >= 0
        )
        desired_guard = 0
        hard_guard = 0
        if guard_priority_active:
            hard_ratio, preferred_ratio = self._capital_guard_ratios()
            desired_guard = (
                math.ceil(
                    total_army
                    * max(hard_ratio, preferred_ratio)
                )
                if total_army >= 5
                else 0
            )
            hard_guard = (
                math.ceil(
                    total_army * hard_ratio
                )
                if total_army >= 5
                else 0
            )
        self.home_guard_target_army = desired_guard
        self.home_guard_deficit = max(0, desired_guard - current_guard)
        self.home_guard_hard_target_army = hard_guard
        self.home_guard_hard_deficit = max(0, hard_guard - current_guard)
        if threat_target is not None:
            threat_army = board.tile(*threat_target).army
            self.home_support_need = max(
                self.home_guard_deficit,
                max(0, threat_army + 1 - current_guard),
            )
        else:
            self.home_support_need = self.home_guard_hard_deficit
        self.home_support_last_source = None
        self.home_support_last_score = 0.0

        visible_generals = [
            position
            for position in visible
            if board.tile(*position).terrain == GENERAL
            and self._is_enemy(board.tile(*position).owner)
        ]
        mobile_army = sum(
            max(0, board.tile(*position).army - 1)
            for position in owned_tiles
        )
        if visible_generals:
            target = min(
                visible_generals,
                key=lambda position: (
                    Board.manhattan(position, general),
                    position,
                ),
            )
            self.exposed_general_player = board.tile(*target).owner
            target_army = board.tile(*target).army
            required = target_army + max(2, math.ceil(target_army * 0.20))
            if mobile_army >= required:
                if self.all_in_target != target:
                    self.force_mission_reselect = True
                self.all_in_target = target
                self.all_in_hold_turns = 45
                self.campaign_target = target
                self.campaign_target_player = board.tile(*target).owner
                self.campaign_target_kind = ATTACK_ENEMY_CITY
                self.campaign_hold_turns = max(
                    self.campaign_hold_turns,
                    GENERAL_CAMPAIGN_HOLD_TURNS,
                )
            elif self.all_in_target == target:
                self.all_in_target = None
                self.all_in_hold_turns = 0
        elif self.exposed_general_player not in self.known_enemy_generals:
            self.exposed_general_player = -1

        if self.all_in_hold_turns > 0:
            self.all_in_hold_turns -= 1
            target = self.all_in_target
            if (
                target is not None
                and target in visible
                and not self._is_enemy(board.tile(*target).owner)
            ):
                self.all_in_target = None
                self.all_in_hold_turns = 0
            elif self.all_in_hold_turns == 0:
                self.all_in_target = None

        if self.home_threat_target is not None:
            self.passive_defense_turns = 0
        elif self.mission is not None and self.mission.kind in (
            DEFEND_HOME_CITY,
            DEFEND_TERRITORY,
            EXPLORE_EXPAND,
        ):
            self.passive_defense_turns = min(
                900,
                self.passive_defense_turns + 1,
            )
        else:
            self.passive_defense_turns = max(
                0,
                self.passive_defense_turns - 3,
            )

    def _update_war_state(
        self,
        board: Board,
        total_army: int,
    ) -> None:
        opponents = [
            opponent
            for opponent in board.active_players
            if opponent != self.player
        ]
        strengths = [
            (
                self.player,
                max(0, total_army),
            )
        ]
        strengths.extend(
            (
                opponent,
                max(1, self.enemy_army_estimate.get(opponent, 1)),
            )
            for opponent in opponents
        )
        strengths.sort(key=lambda item: (-item[1], item[0]))
        total_strength = max(1, len(strengths))
        self.global_army_rank = next(
            (
                index + 1
                for index, (player, _strength) in enumerate(strengths)
                if player == self.player
            ),
            total_strength,
        )
        self.global_army_rank_fraction = (
            self.global_army_rank / total_strength
        )
        self.global_army_rank_high = (
            self.global_army_rank
            <= max(1, math.ceil(total_strength / 3))
        )
        self.global_army_rank_low = (
            self.global_army_rank
            >= max(2, math.floor(total_strength * 2 / 3) + 1)
        )

        opponent = self.war_last_exchange_opponent
        if (
            opponent not in board.active_players
            or board.turn - self.war_last_exchange_turn
            > WAR_STATE_PEACE_AFTER_TURNS
        ):
            if self.war_state != "peace":
                self.war_state_started_turn = board.turn
            self.war_state = "peace"
            self.war_state_opponent = -1
            self.war_force_ratio = 1.0
            return

        if (
            self.war_state != "war"
            or self.war_state_opponent != opponent
        ):
            self.war_state_started_turn = board.turn
        self.war_state = "war"
        self.war_state_opponent = opponent
        opponent_army = max(
            1,
            self.enemy_army_estimate.get(opponent, 1),
        )
        self.war_force_ratio = total_army / opponent_army

    def _peace_rank_state(self) -> str | None:
        if self.war_state == "war":
            return None
        if self.global_army_rank_low:
            return STRATEGY_DEVELOPMENT
        if self.global_army_rank_high:
            return STRATEGY_EXPLORATION
        return None

    def _war_strategy_multiplier(self, mission_kind: str) -> float:
        if self.war_state != "war":
            return 1.0
        if (
            self.war_force_ratio < 1.0
            and mission_kind in (
                DEFEND_HOME_CITY,
                DEFEND_TERRITORY,
            )
        ):
            return WAR_STRATEGY_MULTIPLIER
        if (
            self.war_force_ratio >= 1.0
            and mission_kind in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
            )
        ):
            return WAR_STRATEGY_MULTIPLIER
        return 1.0

    def _update_self_state(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        if self.strategy_update_turn != board.turn:
            self.strategy_update_turn = board.turn
            self.development_frozen_turns = max(
                0,
                self.development_frozen_turns - 1,
            )
            self.border_muster_cooldown_turns = max(
                0,
                self.border_muster_cooldown_turns - 1,
            )
            self.war_tempo_cooldown_turns = max(
                0,
                self.war_tempo_cooldown_turns - 1,
            )
            if self.border_muster_hold_turns > 0:
                self.border_muster_hold_turns -= 1
                if self.border_muster_hold_turns <= 0:
                    self._clear_border_muster()
            if self.overwhelming_hold_turns > 0:
                self.overwhelming_hold_turns -= 1
                target = self.overwhelming_target
                if (
                    target is not None
                    and target in visible
                    and not self._is_enemy(board.tile(*target).owner)
                ):
                    self._clear_overwhelming_attack()
            self._refresh_enemy_estimates(board, visible, owned_tiles)

        total_army = sum(board.tile(*position).army for position in owned_tiles)
        self._update_war_state(board, total_army)
        home_army = (
            board.tile(*general).army
            if general is not None and board.tile(*general).owner == self.player
            else 0
        )
        field_army = max(0, total_army - home_army)
        home_pressure = (
            general is not None
            and self._threat_at(board, general[0], general[1], visible, 3) > 0
        )
        defense_active = self.home_threat_target is not None or home_pressure

        attack_player = self._current_attack_opponent(board, visible)
        if attack_player < 0 and self.overwhelming_target_player in board.active_players:
            attack_player = self.overwhelming_target_player
        target_estimate = self.enemy_army_estimate.get(attack_player, 0)
        overwhelming_ready = (
            attack_player in board.active_players
            and attack_player != self.player
            and target_estimate > 0
            and field_army
            >= target_estimate
            * max(
                0.94,
                OVERWHELMING_ARMY_RATIO
                / max(0.75, self.genome.aggression),
            )
        )

        if defense_active and not self._all_in_penalties_suspended():
            self.overwhelming_streak = 0
            self._clear_overwhelming_attack()
            state = STRATEGY_DEFENSE
            reason = "home or capital under threat"
        elif self.overwhelming_hold_turns > 0:
            state = STRATEGY_ATTACK
            reason = f"overwhelming campaign against player {attack_player + 1}"
        elif self.forced_offensive or (
            attack_player in board.active_players
            and target_estimate > 0
            and field_army >= target_estimate * 0.95
        ):
            state = STRATEGY_ATTACK
            reason = "offensive army advantage"
        elif self._peace_rank_state() == STRATEGY_DEVELOPMENT:
            state = STRATEGY_DEVELOPMENT
            reason = (
                f"peace and low global army rank "
                f"{self.global_army_rank}/{len(board.active_players)}"
            )
        elif self._peace_rank_state() == STRATEGY_EXPLORATION:
            state = STRATEGY_EXPLORATION
            reason = (
                f"peace and high global army rank "
                f"{self.global_army_rank}/{len(board.active_players)}"
            )
        elif (
            self.development_hold_turns > 0
            or self.gather_mode is not None
            or (self.known_neutral_cities and total_army < 120)
        ):
            state = STRATEGY_DEVELOPMENT
            reason = "gather troops or secure neutral income"
        else:
            state = STRATEGY_EXPLORATION
            reason = "search for contact and expansion"

        if overwhelming_ready and attack_player in board.active_players:
            if attack_player != self.overwhelming_target_player:
                self.overwhelming_streak = 1
                self.overwhelming_last_turn = board.turn
            elif self.overwhelming_last_turn != board.turn:
                self.overwhelming_streak += 1
                self.overwhelming_last_turn = board.turn
            if state != STRATEGY_DEFENSE:
                target = self._select_attack_target(
                    board,
                    visible,
                    attack_player,
                    general,
                )
                if target is not None:
                    chance = OVERWHELMING_TRIGGER_CHANCE[self.difficulty.key]
                    chance = min(
                        0.98,
                        chance
                        * self.genome.all_in
                        * self._archetype_all_in_multiplier(),
                    )
                    if attack_player == self.exposed_general_player:
                        chance = min(
                            0.98,
                            chance * EXPOSED_GENERAL_ALL_IN_MULTIPLIER,
                        )
                    if (
                        self.overwhelming_hold_turns > 0
                        and self.overwhelming_target_player == attack_player
                    ):
                        self.overwhelming_hold_turns = max(
                            self.overwhelming_hold_turns,
                            45,
                        )
                    elif (
                        self.overwhelming_streak >= 2
                        or self.rng.random() < chance
                    ):
                        self._start_overwhelming_attack(
                            board,
                            visible,
                            owned_tiles,
                            target,
                            attack_player,
                        )
                        state = STRATEGY_ATTACK
                        reason = (
                            f"overwhelming army ratio "
                            f"{field_army / max(1, target_estimate):.2f}"
                        )
        elif self.overwhelming_hold_turns <= 0:
            self.overwhelming_streak = 0

        if state == self.strategy_state:
            self.strategy_state_turns += 1
        else:
            self.strategy_state = state
            self.strategy_state_turns = 0
        self.strategy_state_reason = reason
        self.strategy_state_ready = True

    def _refresh_enemy_estimates(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
    ) -> None:
        visible_army: dict[int, int] = {}
        visible_tiles: dict[int, int] = {}
        for position in visible:
            tile = board.tile(*position)
            if not self._is_enemy(tile.owner):
                continue
            visible_army[tile.owner] = visible_army.get(tile.owner, 0) + tile.army
            visible_tiles[tile.owner] = visible_tiles.get(tile.owner, 0) + 1

        area_per_player = max(
            1,
            board.width * board.height // max(1, len(board.active_players)),
        )
        reference_tiles = min(
            area_per_player,
            max(
                1,
                int(len(owned_tiles) * 0.9),
                min(area_per_player, 20 + board.turn * 2),
            ),
        )
        for opponent in list(self.enemy_army_estimate):
            if opponent not in board.active_players:
                self.enemy_army_estimate.pop(opponent, None)
                self.enemy_visible_tiles.pop(opponent, None)

        for opponent in sorted(board.active_players):
            if opponent == self.player:
                continue
            count = visible_tiles.get(opponent, 0)
            previous = self.enemy_army_estimate.get(opponent, 0)
            if count > 0:
                inferred_tiles = min(
                    area_per_player,
                    max(count, reference_tiles),
                )
                observation = math.ceil(
                    visible_army.get(opponent, 0) / count * inferred_tiles
                )
                estimate = max(observation, math.ceil(previous * 0.94))
            else:
                estimate = math.ceil(previous * 0.94)
            if estimate > 0:
                self.enemy_army_estimate[opponent] = estimate
            else:
                self.enemy_army_estimate.pop(opponent, None)
            self.enemy_visible_tiles[opponent] = count

    def _current_attack_opponent(
        self,
        board: Board,
        visible: set[tuple[int, int]],
    ) -> int:
        if self.exposed_general_player in board.active_players:
            return self.exposed_general_player
        if (
            self.enemy_hunt_player in board.active_players
            and self.enemy_hunt_target is not None
        ):
            return self.enemy_hunt_player
        if (
            self.mission is not None
            and self.mission.kind in (ATTACK_ENEMY_CITY, ATTACK_ENEMY_LAND)
            and self.mission.target in visible
        ):
            owner = board.tile(*self.mission.target).owner
            if self._is_enemy(owner):
                return owner
        if self.campaign_target_player in board.active_players:
            return self.campaign_target_player
        if self.border_muster_player in board.active_players:
            return self.border_muster_player
        if self.overwhelming_target_player in board.active_players:
            return self.overwhelming_target_player

        visible_strength: dict[int, int] = {}
        for position in visible:
            tile = board.tile(*position)
            if self._is_enemy(tile.owner):
                visible_strength[tile.owner] = (
                    visible_strength.get(tile.owner, 0) + tile.army
                )
        if not visible_strength:
            return -1
        return max(
            visible_strength,
            key=lambda opponent: (
                visible_strength[opponent],
                self.enemy_visible_tiles.get(opponent, 0),
                -opponent,
            ),
        )

    def _select_attack_target(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        opponent: int,
        general: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        candidates = [
            position
            for position in visible
            if board.tile(*position).owner == opponent
        ]
        if candidates:
            return max(
                candidates,
                key=lambda position: (
                    board.tile(*position).terrain == GENERAL,
                    board.tile(*position).terrain == CITY,
                    board.tile(*position).army,
                    -(
                        Board.manhattan(position, general)
                        if general is not None
                        else 0
                    ),
                    -position[0] - position[1],
                ),
            )
        if (
            self.campaign_target is not None
            and self.campaign_target_player == opponent
            and board.in_bounds(*self.campaign_target)
        ):
            return self.campaign_target
        known_general = self.known_enemy_generals.get(opponent)
        if known_general is not None and board.in_bounds(*known_general):
            return known_general
        if self.enemy_hunt_player == opponent and self.enemy_hunt_target is not None:
            return self.enemy_hunt_target
        if (
            self.overwhelming_target is not None
            and self.overwhelming_target_player == opponent
            and board.in_bounds(*self.overwhelming_target)
            and self._is_enemy(board.tile(*self.overwhelming_target).owner)
        ):
            return self.overwhelming_target
        return None

    def _exposed_general_multiplier(self, opponent: int) -> float:
        if (
            opponent == self.exposed_general_player
            or opponent in self.known_enemy_generals
        ):
            return EXPOSED_GENERAL_ALL_IN_MULTIPLIER
        return 1.0

    def _start_overwhelming_attack(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        target: tuple[int, int],
        opponent: int,
    ) -> None:
        self._freeze_development()
        self.overwhelming_target = target
        self.overwhelming_target_player = opponent
        self.overwhelming_hold_turns = max(
            self.overwhelming_hold_turns,
            OVERWHELMING_HOLD_TURNS,
        )
        self.force_mission_reselect = True
        self.forced_offensive = True
        self.primary_reinforce_target = None
        self.primary_stack = self._select_offensive_stack(
            board,
            owned_tiles,
            target,
        )
        self.muster_turns = 0
        self.offensive_hold_turns = max(self.offensive_hold_turns, 80)
        self.gather_mode = "overwhelming"
        self.gather_army_mode = "all"
        if (
            target in visible
            and board.tile(*target).terrain == GENERAL
        ):
            self.all_in_target = target
            self.all_in_hold_turns = max(self.all_in_hold_turns, 80)

    def _freeze_development(self) -> None:
        self.development_hold_turns = 0
        self.development_target = None
        self.development_target_kind = None
        self.development_trigger_opponent = -1
        self.development_stack = None
        self.development_transit_target = None
        self.development_frozen_turns = max(
            self.development_frozen_turns,
            OVERWHELMING_DEVELOPMENT_FREEZE_TURNS,
        )

    def _clear_overwhelming_attack(self) -> None:
        self.overwhelming_target = None
        self.overwhelming_target_player = -1
        self.overwhelming_hold_turns = 0
        self.overwhelming_streak = 0
        if self.gather_mode == "overwhelming":
            self.gather_mode = None
            self.gather_army_mode = "half"

    def _update_border_muster(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        if self.overwhelming_hold_turns > 0:
            self._clear_border_muster()
            return
        if (
            self.mission is None
            or self.mission.kind not in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
            )
            or self.home_threat_target is not None
            or self.all_in_target is not None
        ):
            if self.border_muster_target is not None:
                self._clear_border_muster()
            return

        target = self.mission.target
        if not board.in_bounds(*target):
            self._clear_border_muster()
            return
        target_owner = board.tile(*target).owner
        total_army = sum(
            board.tile(*position).army
            for position in owned_tiles
        )
        if (
            self.border_muster_hold_turns <= 0
            and self.border_muster_cooldown_turns <= 0
            and total_army >= BORDER_MUSTER_MIN_TOTAL_ARMY
            and self._is_enemy(target_owner)
        ):
            staging = self._select_border_muster_target(
                board,
                visible,
                owned_tiles,
                target,
            )
            if staging is not None:
                self.border_muster_target = staging
                self.border_muster_player = target_owner
                self.border_muster_hold_turns = BORDER_MUSTER_HOLD_TURNS
                self.border_muster_ready = False
                self.border_muster_army_target = BORDER_MUSTER_MIN_ARMY
                self.primary_reinforce_target = None
                self.forced_offensive = True

        if self.border_muster_hold_turns <= 0:
            return
        if not self._is_enemy(board.tile(*target).owner):
            self._clear_border_muster()
            return
        if (
            self.border_muster_player not in board.active_players
            or self.border_muster_target is None
            or board.tile(*self.border_muster_target).owner != self.player
        ):
            self.border_muster_target = self._select_border_muster_target(
                board,
                visible,
                owned_tiles,
                target,
            )
            if self.border_muster_target is None:
                self._clear_border_muster()
                return

        self.border_muster_army_target = BORDER_MUSTER_MIN_ARMY
        gathered = self._border_muster_gathered_army(
            board,
            owned_tiles,
        )
        local_enemy_army = self._border_muster_enemy_army(
            board,
            visible,
        )
        was_ready = self.border_muster_ready
        self.border_muster_ready = (
            gathered >= self.border_muster_army_target
            and local_enemy_army
            <= gathered * BORDER_MUSTER_LOCAL_ENEMY_RATIO
        )
        if self.border_muster_ready and not was_ready:
            self.border_muster_hold_turns = BORDER_MUSTER_HOLD_TURNS
            self.primary_reinforce_target = None

    def _select_border_muster_target(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        objective: tuple[int, int],
    ) -> tuple[int, int] | None:
        if not owned_tiles:
            return None
        frontier = [
            position
            for position in owned_tiles
            if self._is_frontier(
                board,
                position[0],
                position[1],
                visible,
            )
        ]
        choices = frontier or owned_tiles
        distances = self._distance_map(board, [objective])
        max_distance = board.width + board.height

        def score(position: tuple[int, int]) -> tuple[int, int, int, int]:
            distance = distances.get(position, max_distance)
            tile = board.tile(*position)
            support = self._friendly_neighbor_count(
                board,
                position[0],
                position[1],
            )
            return (
                distance,
                -tile.army,
                -support,
                -position[0] - position[1],
            )

        return min(choices, key=score)

    def _border_muster_gathered_army(
        self,
        board: Board,
        owned_tiles: list[tuple[int, int]],
    ) -> int:
        target = self.border_muster_target
        if target is None:
            return 0
        return sum(
            board.tile(*position).army
            for position in owned_tiles
            if Board.manhattan(position, target) <= BORDER_MUSTER_RADIUS
        )

    def _border_muster_enemy_army(
        self,
        board: Board,
        visible: set[tuple[int, int]],
    ) -> int:
        target = self.border_muster_target
        if target is None:
            return 0
        return sum(
            board.tile(*position).army
            for position in visible
            if self._is_enemy(board.tile(*position).owner)
            and Board.manhattan(position, target) <= BORDER_MUSTER_RADIUS
        )

    def _clear_border_muster(self) -> None:
        was_active = (
            self.border_muster_target is not None
            or self.border_muster_hold_turns > 0
        )
        self.border_muster_target = None
        self.border_muster_player = -1
        self.border_muster_hold_turns = 0
        self.border_muster_army_target = 0
        self.border_muster_ready = False
        if was_active:
            self.border_muster_cooldown_turns = max(
                self.border_muster_cooldown_turns,
                BORDER_MUSTER_COOLDOWN_TURNS,
            )

    def _update_war_tempo(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        if (
            board.turn < WAR_TEMPO_START_TURN
            or self.war_tempo_cooldown_turns > 0
            or self.overwhelming_hold_turns > 0
        ):
            return
        active_count = len(board.active_players)
        required_idle = (
            WAR_TEMPO_ENDGAME_IDLE_TURNS
            if active_count <= WAR_TEMPO_ENDGAME_PLAYERS
            else WAR_TEMPO_IDLE_TURNS
        )
        if board.turn - self.last_attrition_turn < required_idle:
            if (
                active_count
                > WAR_TEMPO_SIGNIFICANT_ENDGAME_PLAYERS
                or board.turn - self.last_significant_attrition_turn
                < required_idle
            ):
                return

        opponent = self._current_attack_opponent(board, visible)
        target = self._select_attack_target(
            board,
            visible,
            opponent,
            general,
        )
        if target is None:
            enemies = [
                position
                for position in visible
                if self._is_enemy(board.tile(*position).owner)
            ]
            if enemies:
                target = max(
                    enemies,
                    key=lambda position: (
                        board.tile(*position).terrain == GENERAL,
                        board.tile(*position).terrain == CITY,
                        board.tile(*position).army,
                        -Board.manhattan(
                            position,
                            general or position,
                        ),
                        -position[0] - position[1],
                    ),
                )
                opponent = board.tile(*target).owner
        if target is None and self.campaign_target is not None:
            target = self.campaign_target
            opponent = self.campaign_target_player
        if (
            target is None
            or not board.in_bounds(*target)
            or not self._is_enemy(board.tile(*target).owner)
        ):
            self.long_range_search = True
            if self.enemy_hunt_target is not None:
                target = self.enemy_hunt_target
            elif self.known_enemy_tiles:
                if self.enemy_hunt_player not in board.active_players:
                    self.enemy_hunt_player = max(
                        (
                            player
                            for player, positions in self.known_enemy_tiles.items()
                            if player in board.active_players and positions
                        ),
                        default=-1,
                    )
                if self.enemy_hunt_player in board.active_players:
                    target = self._select_enemy_search_target(board, visible)
                    self.enemy_hunt_target = target
                    self.enemy_hunt_hold_turns = 0
            if target is None:
                target = self._select_global_search_target(
                    board,
                    owned_tiles,
                    preferred=(
                        position
                        for positions in self.known_enemy_tiles.values()
                        for position in positions
                    ),
                )
            if target is not None:
                self.search_waypoint = target
                mission_kind = (
                    ATTACK_ENEMY_LAND
                    if (
                        board.in_bounds(*target)
                        and self._is_enemy(board.tile(*target).owner)
                    )
                    else EXPLORE_EXPAND
                )
                self.mission = Mission(
                    mission_kind,
                    target,
                    ENEMY_SEARCH_ATTACK_SCORE,
                    "all",
                    "war tempo active enemy search",
                )
                self.mission_evaluated_turn = board.turn
                self.forced_offensive = True
                self.development_frozen_turns = max(
                    self.development_frozen_turns,
                    WAR_TEMPO_PUSH_TURNS,
                )
                self.war_tempo_cooldown_turns = WAR_TEMPO_PUSH_TURNS
                return
            self.force_mission_reselect = True
            return

        target_tile = board.tile(*target)
        mission_kind = (
            ATTACK_ENEMY_CITY
            if target_tile.terrain in (CITY, GENERAL)
            else ATTACK_ENEMY_LAND
        )
        self.mission = Mission(
            mission_kind,
            target,
            2_500_000,
            "all",
            "war tempo forced contact",
        )
        self.mission_evaluated_turn = board.turn
        self.campaign_target = target
        self.campaign_target_player = opponent
        self.campaign_target_kind = mission_kind
        self.campaign_hold_turns = max(self.campaign_hold_turns, 120)
        self.forced_offensive = True
        self.development_frozen_turns = max(
            self.development_frozen_turns,
            WAR_TEMPO_PUSH_TURNS,
        )
        if self.border_muster_target is None:
            self.border_muster_target = self._select_border_muster_target(
                board,
                visible,
                owned_tiles,
                target,
            )
        self.border_muster_player = opponent
        self.border_muster_hold_turns = max(
            self.border_muster_hold_turns,
            WAR_TEMPO_PUSH_TURNS,
        )
        self.border_muster_ready = True
        self.war_tempo_cooldown_turns = WAR_TEMPO_PUSH_TURNS

    def _state_score_multiplier(self, mission_kind: str) -> float:
        if not self.strategy_state_ready:
            return 1.0
        weights = STATE_MISSION_WEIGHTS.get(
            self.strategy_state,
            STATE_MISSION_WEIGHTS[STRATEGY_EXPLORATION],
        )
        multiplier = weights.get(mission_kind, 1.0)
        if self.overwhelming_hold_turns > 0:
            if mission_kind in (ATTACK_ENEMY_CITY, ATTACK_ENEMY_LAND):
                multiplier *= 1.35
            elif mission_kind in (
                ATTACK_NEUTRAL_CITY,
                EXPLORE_EXPAND,
            ):
                multiplier *= 0.35
        if (
            mission_kind == ATTACK_ENEMY_CITY
            and self.all_in_target is not None
        ):
            multiplier *= 1.30
        if self.border_muster_hold_turns > 0:
            if mission_kind in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
                ATTACK_NEUTRAL_CITY,
            ):
                multiplier *= BORDER_MUSTER_MISSION_MULTIPLIER
            elif mission_kind == EXPLORE_EXPAND:
                multiplier *= 0.35
        if self.major_invasion and mission_kind in (
            DEFEND_HOME_CITY,
            DEFEND_TERRITORY,
        ):
            multiplier *= MAJOR_INVASION_DEFENSE_MULTIPLIER
        multiplier *= self._war_strategy_multiplier(mission_kind)
        return multiplier

    def _genome_mission_multiplier(self, mission_kind: str) -> float:
        genome = self.genome
        if mission_kind == DEFEND_HOME_CITY:
            return genome.defense * genome.home_guard
        if mission_kind == DEFEND_TERRITORY:
            return genome.defense
        if mission_kind == ATTACK_ENEMY_CITY:
            return genome.aggression * genome.all_in
        if mission_kind == ATTACK_ENEMY_LAND:
            return genome.aggression * genome.risk_tolerance
        if mission_kind == ATTACK_NEUTRAL_CITY:
            return genome.development * genome.expansion
        if mission_kind == EXPLORE_EXPAND:
            return math.sqrt(genome.expansion * genome.exploration)
        return 1.0

    def _genome_action_multiplier(
        self,
        target_owner: int,
        target_terrain: int,
        target_visible: bool,
        progress: int,
    ) -> float:
        if target_owner == self.player:
            return self.genome.consolidation
        if target_owner == NEUTRAL:
            return self.genome.expansion
        if target_owner >= 0:
            if target_terrain == GENERAL:
                return self.genome.aggression * self.genome.all_in
            return self.genome.aggression * (
                self.genome.risk_tolerance if progress >= 0 else 1.0
            )
        if not target_visible:
            return self.genome.exploration
        return 1.0

    def _forced_home_defense_move(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        movable: list[tuple[int, int]],
        general: tuple[int, int],
    ) -> Move | None:
        threat = self.home_threat_target
        if self._all_in_penalties_suspended():
            return None
        if threat is None:
            if self.home_guard_hard_deficit <= 0:
                return None
            if self.mission is not None and self.mission.kind != DEFEND_HOME_CITY:
                return None
        current_guard = self._general_guard_army(board, general)
        support_need = max(
            self.home_support_need,
            self.home_guard_hard_deficit,
            self.home_guard_deficit if threat is not None else 0,
        )
        if threat is not None:
            threat_army = board.tile(*threat).army
            support_need = max(
                support_need,
                max(1, threat_army + 1 - current_guard),
            )
        if support_need <= 0 and threat is None:
            return None

        distances = self._distance_map(board, [general])
        max_distance = board.width + board.height
        emergency = (
            threat is not None
            and current_guard <= board.tile(*threat).army
        )
        best_support: tuple[float, Move] | None = None
        best_fight: tuple[float, Move] | None = None
        army_positions = self.army_controller.positions
        threat_army = (
            board.tile(*threat).army
            if threat is not None
            else 0
        )

        for sx, sy in movable:
            if (sx, sy) in army_positions:
                continue
            source = board.tile(sx, sy)
            available = max(0, source.army - 1)
            if available <= 0:
                continue
            capital_distance = Board.manhattan((sx, sy), general)
            reinforcement_ratio = (
                source.army / max(1, capital_distance)
            )
            ratio_bonus = min(
                HOME_SUPPORT_RATIO_BONUS_CAP,
                reinforcement_ratio * HOME_SUPPORT_RATIO_WEIGHT,
            )

            source_distance = distances.get((sx, sy), max_distance)
            if emergency:
                local_threat = 0
                source_frontier = False
            else:
                local_threat = self._threat_at(board, sx, sy, visible, 1)
                source_frontier = self._is_frontier(board, sx, sy, visible)
            legal_targets = board.legal_targets(sx, sy, self.player)

            source_fight_score = 0.0
            source_fight_move: Move | None = None
            for tx, ty in legal_targets:
                if (tx, ty) in army_positions:
                    continue
                target = board.tile(tx, ty)
                if not self._is_enemy(target.owner):
                    continue
                if source.army - 1 <= target.army:
                    continue
                advantage = source.army - 1 - target.army
                fight_score = (
                    HOME_SUPPORT_FIGHT_VALUE
                    + advantage * HOME_SUPPORT_FIGHT_ADVANTAGE
                )
                if (tx, ty) == threat:
                    fight_score += HOME_SUPPORT_COUNTER_SCORE
                if fight_score > source_fight_score:
                    source_fight_score = fight_score
                    source_fight_move = Move(
                        sx,
                        sy,
                        tx,
                        ty,
                        source.army - 1,
                    )

            if source_fight_move is not None:
                if (
                    best_fight is None
                    or source_fight_score > best_fight[0]
                ):
                    best_fight = (source_fight_score, source_fight_move)

            capital_defense_score = (
                HOME_SUPPORT_BASE_SCORE
                + support_need * HOME_SUPPORT_NEED_WEIGHT
                + threat_army * HOME_SUPPORT_THREAT_WEIGHT
                + (
                    HOME_SUPPORT_EMERGENCY_BONUS
                    if emergency
                    else 0.0
                )
                - capital_distance * HOME_SUPPORT_DISTANCE_PENALTY
                + ratio_bonus
            )
            if source_frontier and not emergency:
                capital_defense_score -= HOME_SUPPORT_FRONTIER_PENALTY
            if local_threat > 0 and not emergency:
                capital_defense_score -= (
                    local_threat * HOME_SUPPORT_LOCAL_THREAT_PENALTY
                )
            keep_fight_weight = 0.18 if emergency else 0.82
            capital_defense_score -= (
                source_fight_score * keep_fight_weight
            )

            best_step: tuple[int, Move] | None = None
            for tx, ty in legal_targets:
                if (tx, ty) in army_positions:
                    continue
                target = board.tile(tx, ty)
                target_distance = distances.get((tx, ty), max_distance)
                progress = source_distance - target_distance
                if progress <= 0:
                    continue
                if self._is_enemy(target.owner) and (
                    source.army - 1 <= target.army
                ):
                    continue
                if emergency:
                    amount = available
                else:
                    amount = min(available, max(1, support_need))
                    if source_frontier and local_threat > 0:
                        amount = min(
                            amount,
                            max(1, (available + 1) // 2),
                        )
                if amount <= 0:
                    continue
                move = Move(sx, sy, tx, ty, amount)
                step_score = (
                    progress * 8_000.0
                    + (60_000.0 if (tx, ty) == general else 0.0)
                    + min(amount, 30) * 320.0
                )
                if best_step is None or step_score > best_step[0]:
                    best_step = (step_score, move)

            support_move: Move | None = None
            support_score = capital_defense_score
            if (
                threat is not None
                and source_fight_move is not None
                and source_fight_move.target == threat
            ):
                support_move = source_fight_move
                support_score += HOME_SUPPORT_COUNTER_SCORE
            elif best_step is not None:
                support_move = best_step[1]
                support_score += best_step[0]
            if support_move is None:
                continue
            if (
                not board.move_legal(support_move, self.player)
                or support_move.source in army_positions
                or support_move.target in army_positions
            ):
                continue
            if (
                best_support is None
                or support_score > best_support[0]
            ):
                best_support = (support_score, support_move)

        if best_support is None:
            self.home_support_last_source = None
            self.home_support_last_score = 0.0
            return None
        if (
            best_fight is not None
            and not emergency
            and best_support[1].target != threat
            and best_fight[0]
            > best_support[0] + HOME_SUPPORT_FIGHT_MARGIN
        ):
            self.home_support_last_source = None
            self.home_support_last_score = best_fight[0]
            return None
        self.home_support_last_source = best_support[1].source
        self.home_support_last_score = best_support[0]
        return best_support[1]

    def _clear_campaign(self) -> None:
        self.campaign_target = None
        self.campaign_target_player = -1
        self.campaign_target_kind = None
        self.campaign_hold_turns = 0

    def _update_attrition_state(
        self,
        board: Board,
        owned_tiles: list[tuple[int, int]],
    ) -> None:
        if self.attrition_ingested_turn == board.turn:
            return
        self.attrition_ingested_turn = board.turn
        self.war_last_exchange_amount = 0
        total_exchange = 0

        for opponent, last_update in list(self.attrition_last_update.items()):
            if board.turn - last_update > ATTRITION_STALE_TURNS:
                self.attrition_last_update.pop(opponent, None)
                self.attrition_by_opponent.pop(opponent, None)

        if self.development_hold_turns > 0:
            self.development_hold_turns -= 1

        for pair, raw_amount in (board.last_report.attrition_by_pair or {}).items():
            if self.player not in pair:
                continue
            opponent = pair[0] if pair[1] == self.player else pair[1]
            if (
                opponent < 0
                or opponent == self.player
                or opponent not in board.active_players
            ):
                continue
            amount = max(0, int(raw_amount))
            if amount <= 0:
                continue
            total_exchange += amount
            self.last_attrition_turn = board.turn
            self.attrition_by_opponent[opponent] = (
                self.attrition_by_opponent.get(opponent, 0) + amount
            )
            self.attrition_last_update[opponent] = board.turn
            if amount > self.war_last_exchange_amount:
                self.war_last_exchange_amount = amount
                self.war_last_exchange_opponent = opponent
                self.war_last_exchange_turn = board.turn

        total_army = sum(
            board.tile(*position).army
            for position in owned_tiles
            if board.tile(*position).owner == self.player
        )
        if total_army <= 0:
            return
        significant_threshold = max(
            WAR_TEMPO_SIGNIFICANT_BATTLE_MIN,
            math.ceil(
                total_army * WAR_TEMPO_SIGNIFICANT_BATTLE_RATIO
            ),
        )
        if total_exchange >= significant_threshold:
            self.last_significant_attrition_turn = board.turn
        threshold = total_army * ATTRITION_DEVELOPMENT_MULTIPLIER
        opponent = max(
            self.attrition_by_opponent,
            key=lambda player: (
                self.attrition_by_opponent.get(player, 0),
                -player,
            ),
            default=None,
        )
        if (
            opponent is not None
            and opponent in board.active_players
            and self.attrition_by_opponent.get(opponent, 0) > threshold
        ):
            if (
                self.development_frozen_turns <= 0
                and self.overwhelming_hold_turns <= 0
            ):
                self._start_development(board, opponent)

    def _start_development(self, board: Board, opponent: int) -> None:
        duration = max(
            DEVELOPMENT_MIN_HOLD_TURNS,
            min(DEVELOPMENT_MAX_HOLD_TURNS, (board.width + board.height) // 2),
        )
        self.development_hold_turns = max(self.development_hold_turns, duration)
        self.development_trigger_opponent = opponent
        self.development_phase = board.turn % 2
        self.development_stack = None
        self.development_transit_target = None
        self.development_target_kind = None
        self.attrition_by_opponent[opponent] = 0
        self.attrition_last_update[opponent] = board.turn

    def _update_development_state(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
    ) -> None:
        if (
            self.development_frozen_turns > 0
            or self.overwhelming_hold_turns > 0
        ):
            self.development_target = None
            self.development_target_kind = None
            self.development_stack = None
            self.development_transit_target = None
            return
        if self.development_hold_turns <= 0:
            self.development_target = None
            self.development_target_kind = None
            self.development_trigger_opponent = -1
            self.development_stack = None
            self.development_transit_target = None
            return

        if (
            self.development_target is not None
            and self.development_target_kind == "city"
            and self.development_target in visible
            and board.tile(*self.development_target).owner != NEUTRAL
        ):
            self.known_neutral_cities.discard(self.development_target)
            self.development_target = None
            self.development_target_kind = None
            self.development_stack = None
            self.development_transit_target = None

        if (
            self.development_target is not None
            and self.development_target_kind == "explore"
            and self.development_target in visible
        ):
            self.development_target = None
            self.development_target_kind = None
            self.development_stack = None
            self.development_transit_target = None

        if (
            self.development_target is not None
            and (
                self.development_target_kind == "city"
                or not self.known_neutral_cities
            )
        ):
            return
        if not owned_tiles:
            return

        distances = self._distance_map(board, owned_tiles)
        max_distance = board.width + board.height

        if self.known_neutral_cities:
            self.development_target_kind = "city"

            def score(position: tuple[int, int]) -> tuple[float, int, int]:
                distance = distances.get(position, max_distance)
                threat = self._threat_at(
                    board,
                    position[0],
                    position[1],
                    visible,
                    3,
                )
                enemy_separation = (
                    Board.manhattan(position, self.last_seen_enemy)
                    if self.last_seen_enemy is not None
                    else 0
                )
                value = -distance * 120.0 - threat * 90.0 + enemy_separation * 18.0
                return value, -distance, -position[0] - position[1]

            self.development_target = max(self.known_neutral_cities, key=score)
            return

        self.development_target_kind = "explore"
        patrol = [
            (board.width // 4, board.height // 4),
            (board.width * 3 // 4, board.height // 4),
            (board.width * 3 // 4, board.height * 3 // 4),
            (board.width // 4, board.height * 3 // 4),
            (board.width // 2, board.height // 2),
        ]
        unexplored = [
            position
            for position in patrol
            if position not in self.known_terrain
        ]
        choices = unexplored or patrol

        def explore_score(position: tuple[int, int]) -> tuple[float, int, int]:
            distance = distances.get(position, max_distance)
            value = distance * 120.0
            if position in self.known_terrain:
                value -= 8_000.0
            return value, -position[0] - position[1], distance

        self.development_target = max(choices, key=explore_score)

    def _should_run_development_phase(self, board: Board) -> bool:
        return (
            self.development_frozen_turns <= 0
            and self.overwhelming_hold_turns <= 0
            and self.border_muster_hold_turns <= 0
            and
            self.development_hold_turns > 0
            and self.development_target is not None
            and self.planning_phase == self.development_phase
        )

    def _forced_development_move(self, board: Board) -> Move | None:
        target = self.development_target
        if target is None or not board.in_bounds(*target):
            return None
        target_tile = board.tile(*target)
        if (
            self.development_target_kind == "city"
            and target in board.visibility(self.player)
            and (target_tile.terrain != CITY or target_tile.owner != NEUTRAL)
        ):
            self.development_target = None
            self.development_target_kind = None
            self.development_stack = None
            return None

        distances = self._distance_map(board, [target])
        max_distance = board.width + board.height
        owned = self.planning_owned_tiles
        if not owned:
            owned = [
                (x, y)
                for y, row in enumerate(board.grid)
                for x, tile in enumerate(row)
                if tile.owner == self.player
            ]
        if not owned:
            return None

        stack = self.development_stack
        if (
            stack is None
            or not board.in_bounds(*stack)
            or board.tile(*stack).owner != self.player
            or board.tile(*stack).army < 2
            or distances.get(stack, max_distance) >= max_distance
        ):
            candidates = [
                position
                for position in owned
                if position != self.primary_stack
                and board.tile(*position).army >= DEVELOPMENT_MIN_DETACHMENT_ARMY
            ]
            if not candidates:
                candidates = [
                    position
                    for position in owned
                    if position != self.primary_stack
                    and board.tile(*position).army > 1
                ]
            if not candidates:
                return None

            def stack_score(position: tuple[int, int]) -> tuple[float, int, int]:
                tile = board.tile(*position)
                distance = distances.get(position, max_distance)
                rear_bonus = (
                    Board.manhattan(position, self.primary_stack)
                    if self.primary_stack is not None
                    else 0
                )
                value = (
                    min(tile.army, GATHER_STACK_ARMY_CAP) * 18.0
                    + max(0, 80 - distance) * 5.0
                    - distance * 2.0
                    + rear_bonus * 2.0
                )
                return value, tile.army, -distance

            stack = max(candidates, key=stack_score)
            self.development_stack = stack

        if any(
            pending.owner == self.player and (pending.sx, pending.sy) == stack
            for pending in board.pending_moves
        ):
            return None

        sx, sy = stack
        source = board.tile(sx, sy)
        source_distance = distances.get(stack, max_distance)
        best: tuple[float, Move] | None = None
        fallback: tuple[float, Move] | None = None

        for tx, ty in board.legal_targets(sx, sy, self.player):
            target_distance = distances.get((tx, ty), max_distance)
            progress = source_distance - target_distance
            tile = board.tile(tx, ty)
            amount = source.army - 1
            if self._is_enemy(tile.owner) and amount <= tile.army:
                continue
            if (
                tile.owner == NEUTRAL
                and tile.terrain == CITY
                and amount <= tile.army
            ):
                continue
            move = Move(sx, sy, tx, ty, amount)
            if not board.move_legal(move, self.player):
                continue
            value = progress * 2_400.0
            if progress <= 0:
                if (
                    self.last_move is not None
                    and move.target == self.last_move.source
                ):
                    continue
                value -= abs(progress) * 1_200.0
            if (tx, ty) == target:
                value += 120_000.0
            if tile.owner == NEUTRAL:
                value += 4_000.0
            if tile.terrain == CITY and tile.owner == NEUTRAL:
                value += 24_000.0
            if progress > 0 and (best is None or value > best[0]):
                best = (value, move)
            elif progress <= 0 and (fallback is None or value > fallback[0]):
                fallback = (value, move)

        selected = best or fallback
        if selected is None:
            return None
        move = self._cap_home_guard_move(board, selected[1])
        if move is None:
            return None
        if board.movement_turns(
            move.sx,
            move.sy,
            move.tx,
            move.ty,
            self.player,
        ) > 1:
            self.development_transit_target = move.target
        else:
            self.development_stack = move.target
            self.development_transit_target = None
        return move

    def _sync_development_transit(self, board: Board) -> None:
        if self.development_transit_target is None or self.development_stack is None:
            return
        if any(
            pending.owner == self.player
            and (pending.sx, pending.sy) == self.development_stack
            for pending in board.pending_moves
        ):
            return
        target = self.development_transit_target
        if board.in_bounds(*target) and board.tile(*target).owner == self.player:
            self.development_stack = target
        self.development_transit_target = None

    def _forced_vanguard_move(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        allow_reselect: bool = True,
    ) -> Move | None:
        if self.mission is None:
            return None
        if (
            self.primary_reinforce_target is not None
            and (
                not board.in_bounds(*self.primary_reinforce_target)
                or board.tile(*self.primary_reinforce_target).owner != self.player
            )
        ):
            self.primary_reinforce_target = None
        if (
            self.border_muster_hold_turns > 0
            and self.border_muster_target is not None
            and not self.border_muster_ready
        ):
            goal = self.border_muster_target
            if len(board.active_players) <= WAR_TEMPO_ENDGAME_PLAYERS:
                visible_enemies = [
                    position
                    for position in visible
                    if self._is_enemy(board.tile(*position).owner)
                ]
                if visible_enemies and self.primary_stack is not None:
                    goal = min(
                        visible_enemies,
                        key=lambda position: (
                            Board.manhattan(position, self.primary_stack),
                            board.tile(*position).army,
                            position,
                        ),
                    )
        else:
            goal = self.primary_reinforce_target or self.mission.target
        distances = self._mission_distance_map(board, visible, [goal])
        max_distance = board.width + board.height
        if (
            not self._primary_stack_usable(board)
            or distances.get(self.primary_stack, max_distance) >= max_distance
        ):
            self.primary_stack = self._select_offensive_stack(
                board,
                self.planning_owned_tiles,
                goal,
            )
        if self.primary_stack is None:
            return None

        sx, sy = self.primary_stack
        source = board.tile(sx, sy)
        owned_positions = self.planning_owned_tiles
        if not owned_positions:
            owned_positions = [
                (x, y)
                for y, row in enumerate(board.grid)
                for x, tile in enumerate(row)
                if tile.owner == self.player
            ]
        if (
            allow_reselect
            and source.army <= self.difficulty.exploration_army
        ):
            reserves = [
                (x, y)
                for x, y in owned_positions
                if board.tile(x, y).army > 1
                and (x, y) != self.primary_stack
            ]
            reserve = self._select_strategic_reserve(board, reserves, goal)
            if (
                reserve is not None
                and board.tile(*reserve).army >= max(6, source.army + 2)
            ):
                self.primary_reinforce_target = self.primary_stack
                self.primary_stack = reserve
                return self._forced_vanguard_move(
                    board,
                    visible,
                    allow_reselect=False,
                )
        if source.army <= 1 and allow_reselect:
            self.primary_reinforce_target = self.primary_stack
            replacements = [
                (x, y)
                for x, y in owned_positions
                if board.tile(x, y).army > 1
                and (x, y) != self.primary_stack
            ]
            self.primary_stack = self._select_support_stack(
                board,
                replacements,
                goal,
                stalled=self.frontline_stall_turns >= 3,
            )
            if self.primary_stack is None:
                return None
            return self._forced_vanguard_move(
                board,
                visible,
                allow_reselect=False,
            )
        if any(
            pending.owner == self.player
            and (pending.sx, pending.sy) == self.primary_stack
            for pending in board.pending_moves
        ):
            return None
        source_distance = distances.get(self.primary_stack, max_distance)
        if source_distance >= max_distance:
            return None

        best: tuple[float, Move] | None = None
        for tx, ty in board.legal_targets(sx, sy, self.player):
            target_distance = distances.get((tx, ty), max_distance)
            progress = source_distance - target_distance
            if progress <= 0:
                continue
            target = board.tile(tx, ty)
            if (
                self._is_enemy(target.owner)
                and source.army - 1 <= target.army
            ):
                continue
            if (
                target.owner == NEUTRAL
                and target.terrain == CITY
                and source.army - 1 <= target.army
            ):
                continue
            move = Move(sx, sy, tx, ty, source.army - 1)
            if not board.move_legal(move, self.player):
                continue
            value = progress * 1_800.0
            if target_distance == 0:
                value += 80_000.0
            if self._is_enemy(target.owner):
                value += 35_000.0 + target.army * 120.0
            elif target.owner == NEUTRAL:
                value += 12_000.0
            if best is None or value > best[0]:
                best = (value, move)
        if best is None:
            self.frontline_stall_turns += 1
            if allow_reselect:
                front = self.primary_stack
                reinforcements = [
                    (x, y)
                    for y, row in enumerate(board.grid)
                    for x, tile in enumerate(row)
                    if tile.owner == self.player
                    and tile.army > 1
                    and (x, y) != front
                ]
                support = self._select_reinforcement_stack(
                    board,
                    reinforcements,
                    front,
                )
                if self.frontline_stall_turns >= 3:
                    support = self._select_strategic_reserve(
                        board,
                        reinforcements,
                        front,
                    ) or support
                if support is not None:
                    self.primary_reinforce_target = front
                    self.primary_stack = support
                    return self._forced_vanguard_move(
                        board,
                        visible,
                        allow_reselect=False,
                    )
            detour: tuple[float, Move] | None = None
            for tx, ty in board.legal_targets(sx, sy, self.player):
                target_distance = distances.get((tx, ty), max_distance)
                if target_distance > source_distance + 1:
                    continue
                target = board.tile(tx, ty)
                if (
                    self._is_enemy(target.owner)
                    and source.army - 1 <= target.army
                ):
                    continue
                if (
                    target.owner == NEUTRAL
                    and target.terrain == CITY
                    and source.army - 1 <= target.army
                ):
                    continue
                move = Move(sx, sy, tx, ty, source.army - 1)
                if not board.move_legal(move, self.player):
                    continue
                value = -target_distance * 1_000.0
                if (
                    self.last_move is not None
                    and move.target == self.last_move.source
                ):
                    value -= 8_000.0
                if detour is None or value > detour[0]:
                    detour = (value, move)
            if detour is None:
                return None
            best = detour

        self.frontline_stall_turns = 0
        capped = self._cap_home_guard_move(board, best[1])
        if capped is None:
            return None
        self._advance_primary(board, capped)
        return capped

    def _select_support_stack(
        self,
        board: Board,
        candidates: list[tuple[int, int]],
        front: tuple[int, int],
        stalled: bool,
    ) -> tuple[int, int] | None:
        if stalled:
            reserve = self._select_strategic_reserve(
                board,
                candidates,
                front,
            )
            if reserve is not None:
                return reserve
        return self._select_reinforcement_stack(
            board,
            candidates,
            front,
        )

    def _select_strategic_reserve(
        self,
        board: Board,
        candidates: list[tuple[int, int]],
        front: tuple[int, int],
    ) -> tuple[int, int] | None:
        reserves = [
            position
            for position in candidates
            if board.tile(*position).army >= 4
            and Board.manhattan(position, front) >= 4
        ]
        if not reserves:
            return None

        def score(position: tuple[int, int]) -> tuple[float, int, int, int]:
            tile = board.tile(*position)
            distance = Board.manhattan(position, front)
            value = (
                min(tile.army, GATHER_STACK_ARMY_CAP) * 18.0
                + max(0, 80 - distance) * 5.0
                - distance * 2.0
                + self._reinforcement_priority_bonus(board, position)
            )
            return value, tile.army, distance, -position[0] - position[1]

        return max(reserves, key=score)

    def _select_reinforcement_stack(
        self,
        board: Board,
        candidates: list[tuple[int, int]],
        front: tuple[int, int],
    ) -> tuple[int, int] | None:
        if not candidates:
            return None

        def score(position: tuple[int, int]) -> tuple[float, int, int, int]:
            tile = board.tile(*position)
            distance = Board.manhattan(position, front)
            value = (
                min(tile.army, GATHER_STACK_ARMY_CAP) * 16.0
                + max(0, 64 - distance) * 6.0
                - distance * 4.0
                + self._reinforcement_priority_bonus(board, position)
            )
            return value, tile.army, -distance, -position[0] - position[1]

        return max(candidates, key=score)

    def _sync_primary_transit(self, board: Board) -> None:
        if self.primary_transit_target is None or self.primary_stack is None:
            return
        if any(
            pending.owner == self.player
            and (pending.sx, pending.sy) == self.primary_stack
            for pending in board.pending_moves
        ):
            return
        target = self.primary_transit_target
        if board.in_bounds(*target) and board.tile(*target).owner == self.player:
            self.primary_stack = target
        self.primary_transit_target = None

    def _advance_primary(self, board: Board, move: Move) -> None:
        if self.primary_stack != move.source:
            return
        if board.movement_turns(
            move.sx,
            move.sy,
            move.tx,
            move.ty,
            self.player,
        ) > 1:
            self.primary_transit_target = move.target
            return
        self.primary_stack = move.target
        self.primary_transit_target = None

    def _primary_stack_usable(self, board: Board) -> bool:
        if self.primary_stack is None:
            return False
        if not board.in_bounds(*self.primary_stack):
            return False
        tile = board.tile(*self.primary_stack)
        return (
            tile.owner == self.player
            and tile.terrain != MOUNTAIN
            and tile.army >= 1
        )

    def _select_offensive_stack(
        self,
        board: Board,
        owned_tiles: list[tuple[int, int]],
        goal: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        movable = [
            position
            for position in owned_tiles
            if board.tile(*position).terrain != MOUNTAIN
            and board.tile(*position).army > 1
        ]
        if not movable:
            return None
        distances = self._distance_map(board, [goal]) if goal is not None else {}
        max_distance = board.width + board.height
        visible = self._cached_visibility(board, owned_tiles)
        general = board.general_position(self.player)
        hard_guard = self.home_guard_hard_target_army
        home_cells = (
            self._home_cells(board, general)
            if general is not None
            else set()
        )

        def score(position: tuple[int, int]) -> tuple[float, int, int, int, int]:
            tile = board.tile(*position)
            distance = distances.get(position, max_distance)
            frontier = int(
                self._is_frontier(
                    board,
                    position[0],
                    position[1],
                    visible,
                )
            )
            value = (
                min(tile.army, GATHER_STACK_ARMY_CAP) * 16.0
                - distance * 10.0
                + frontier * 120.0
            )
            if (
                not self._all_in_penalties_suspended()
                and position == general
                and (
                    self.home_guard_deficit > 0
                    or tile.army <= hard_guard
                )
            ):
                value -= 1_000_000.0
            elif (
                not self._all_in_penalties_suspended()
                and position in home_cells
                and self.overwhelming_hold_turns <= 0
            ):
                value -= 5_000.0
            return value, tile.army, frontier, -distance, -position[0] - position[1]

        if (
            self.overwhelming_hold_turns > 0
            and general is not None
            and not self._all_in_penalties_suspended()
        ):
            protected = [
                position
                for position in movable
                if not (
                    position == general
                    and board.tile(*position).army <= hard_guard
                )
            ]
            if protected:
                movable = protected
        return max(movable, key=score)

    def _refresh_search_waypoint(
        self,
        board: Board,
        owned_tiles: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> None:
        if (
            self.enemy_hunt_target is not None
            and self.enemy_hunt_player in board.active_players
            and not self.planning_has_visible_enemy
        ):
            if self.enemy_hunt_hold_turns >= ENEMY_SEARCH_TARGET_HOLD_TURNS:
                refreshed = self._select_enemy_search_target(
                    board,
                    set(),
                )
                if refreshed is not None:
                    self.enemy_hunt_target = refreshed
                    self.enemy_hunt_hold_turns = 0
                    self.force_mission_reselect = True
            self.search_waypoint = self.enemy_hunt_target
            return
        if self.campaign_target is not None and self.campaign_hold_turns > 0:
            self.search_waypoint = self.campaign_target
            return
        if self.last_seen_enemy is not None and board.turn - self.last_contact_turn <= 60:
            self.search_waypoint = self.last_seen_enemy
            return
        if not self.long_range_search and self.turns_since_contact < 80:
            self.search_waypoint = None
            return

        center = (board.width // 2, board.height // 2)
        if board.tile(*center).terrain == MOUNTAIN:
            nearby = [
                (x, y)
                for y in range(
                    max(0, center[1] - 4),
                    min(board.height, center[1] + 5),
                )
                for x in range(
                    max(0, center[0] - 4),
                    min(board.width, center[0] + 5),
                )
                if board.tile(x, y).terrain != MOUNTAIN
            ]
            if nearby:
                center = min(
                    nearby,
                    key=lambda position: (
                        Board.manhattan(position, center),
                        position,
                    ),
                )
        if (
            len(board.active_players) <= 2
            and self.turns_since_contact < 260
        ):
            if self.search_waypoint != center:
                self.search_turn = board.turn
                self.best_search_distance = 10**9
                self.search_stall_turns = 0
            self.search_waypoint = center
            self.global_search_turn = board.turn
            self.search_phase = 1
            return

        patrol = (
            center,
            (board.width // 4, board.height // 4),
            (board.width * 3 // 4, board.height // 4),
            (board.width * 3 // 4, board.height * 3 // 4),
            (board.width // 4, board.height * 3 // 4),
        )
        interval = max(140, (board.width + board.height) * 2)
        waypoint_index = (board.turn // interval) % len(patrol)

        current_is_unknown = (
            self.search_waypoint is not None
            and board.in_bounds(*self.search_waypoint)
            and self.search_waypoint not in self.seen_land
            and self.known_terrain.get(self.search_waypoint) != MOUNTAIN
        )
        if (
            current_is_unknown
            and self.search_stall_turns < 12
            and board.turn - self.global_search_turn
            < GLOBAL_SEARCH_TARGET_HOLD_TURNS
        ):
            return

        search_beacon_index = waypoint_index
        if len(board.active_players) > 2:
            search_beacon_index += max(0, self.search_stall_turns // 12)
            waypoint = self._select_global_search_target(
            board,
            owned_tiles,
            beacon=patrol[search_beacon_index % len(patrol)],
            )
            if waypoint is not None:
                if waypoint != self.search_waypoint:
                    self.search_turn = board.turn
                self.search_waypoint = waypoint
            self.global_search_turn = board.turn
            self.search_phase = (
                (waypoint[0] * 2 // board.width)
                + 2 * (waypoint[1] * 2 // board.height)
                + 1
            )
            return

        available_patrol = [
            position
            for position in patrol
            if position not in self.seen_land
            and self.known_terrain.get(position) != MOUNTAIN
        ]
        if not available_patrol:
            self.search_waypoint = None
            return
        waypoint = available_patrol[waypoint_index % len(available_patrol)]
        if waypoint != self.search_waypoint:
            self.search_turn = board.turn
            self.best_search_distance = 10**9
            self.search_stall_turns = 0
        self.search_waypoint = waypoint
        self.global_search_turn = board.turn
        self.search_phase = waypoint_index + 1

    def _update_mission(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        land_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> tuple[Mission | None, bool]:
        candidates = self._mission_candidates(
            board,
            visible,
            owned_tiles,
            land_tiles,
            own_cities,
            general,
        )
        if not candidates:
            return self.mission, False

        best = candidates[0]
        current = self.mission
        if self.force_mission_reselect:
            self.force_mission_reselect = False
            changed = (
                current is None
                or (current.kind, current.target) != (best.kind, best.target)
            )
            return best, changed
        if current is None or not self._mission_target_valid(
            board, current, visible
        ):
            changed = current is not None
            return best, changed

        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.kind == current.kind
                and candidate.target == current.target
            ),
            None,
        )
        if match is None:
            current.score *= 0.72
        else:
            current.score = match.score
            current.army_mode = match.army_mode
            current.reason = match.reason
        current.age += 1

        if (best.kind, best.target) == (current.kind, current.target):
            return current, False

        if (
            current.kind == EXPLORE_EXPAND
            and best.kind == EXPLORE_EXPAND
            and best.score >= current.score
        ):
            return best, True

        margin = (
            MISSION_SWITCH_MARGIN[self.difficulty.key]
            * self.genome.task_commitment
        )
        commitment = min(1_800.0, current.age * 120.0)
        if best.score > current.score + margin + commitment:
            return best, True
        return current, False

    def _mission_candidates(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        land_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
        general: tuple[int, int] | None,
    ) -> list[Mission]:
        candidates: list[Mission] = []
        self_distance = self._distance_map(
            board,
            owned_tiles,
            max_depth=board.vision_radius + 2,
        )
        home_distance = (
            self._distance_map(board, [general]) if general is not None else {}
        )
        max_distance = board.width + board.height

        def distance_to(position: tuple[int, int]) -> int:
            distance = self_distance.get(position)
            if distance is not None:
                return distance
            return home_distance.get(position, max_distance)

        grid = board.grid
        visible_enemies = [
            (x, y)
            for x, y in visible
            if self._is_enemy(grid[y][x].owner)
        ]

        home_area_is_strategic_priority = (
            general is not None
            and self.home_threat_target is None
            and self.home_guard_hard_deficit <= 0
            and self.home_guard_deficit <= 0
            and not visible_enemies
            and not self.forced_offensive
            and self.overwhelming_hold_turns <= 0
            and self.border_muster_hold_turns <= 0
            and self.all_in_target is None
            and not (
                self.campaign_target is not None
                and self.campaign_hold_turns > 0
            )
            and not self.known_enemy_generals
        )
        if home_area_is_strategic_priority:
            for position in self._home_occupy_cells(board, general):
                if position == general:
                    continue
                tile = board.tile(*position)
                if (
                    tile.terrain == MOUNTAIN
                    or tile.owner == self.player
                    or position not in visible
                ):
                    continue
                if tile.terrain == CITY and tile.owner == NEUTRAL:
                    mission_kind = ATTACK_NEUTRAL_CITY
                elif self._is_enemy(tile.owner):
                    mission_kind = (
                        ATTACK_ENEMY_CITY
                        if tile.terrain in (CITY, GENERAL)
                        else ATTACK_ENEMY_LAND
                    )
                else:
                    mission_kind = EXPLORE_EXPAND
                candidates.append(
                    Mission(
                        mission_kind,
                        position,
                        HOME_AREA_OCCUPY_SCORE - distance_to(position) * 120,
                        "all",
                        f"priority occupy home vision {board.vision_radius}",
                    )
                )

        if (
            self.home_threat_target is None
            and self.capital_depth < self._capital_depth_requirement(board.turn)
            and self.home_guard_hard_deficit <= 0
            and not visible_enemies
            and not self.forced_offensive
            and self.overwhelming_hold_turns <= 0
            and self.border_muster_hold_turns <= 0
            and self.all_in_target is None
            and not self.known_enemy_generals
            and not (
                self.campaign_target is not None
                and self.campaign_hold_turns > 0
            )
        ):
            depth_mission = self._capital_depth_target(
                board,
                visible,
                owned_tiles,
            )
            if depth_mission is not None:
                candidates.append(depth_mission)

        if (
            self.archetype == ARCHETYPE_TURTLE
            and self.turtle_accumulating
            and general is not None
        ):
            capital_army = board.tile(*general).army
            deficit = max(
                0,
                self.archetype_capital_target_army - capital_army,
            )
            candidates.append(
                Mission(
                    DEFEND_HOME_CITY,
                    general,
                    2_000_000 + deficit * 5_000,
                    "all",
                    f"turtle accumulate capital "
                    f"{capital_army}/{self.archetype_capital_target_army}",
                )
            )

        defended_positions = list(own_cities)
        if general is not None:
            defended_positions.append(general)
        for position in defended_positions:
            tile = board.tile(*position)
            if position == general and self.home_threat_target is not None:
                if self.home_guard_deficit > 0:
                    candidates.append(
                        Mission(
                            DEFEND_HOME_CITY,
                            position,
                            HOME_THREAT_SCORE
                            + 60_000
                            + min(20_000, self.home_guard_deficit * 260),
                            "all",
                            f"garrison before counterattack {self.home_guard_deficit}",
                        )
                    )
                    continue
                threat_position = self.home_threat_target
                threat_tile = board.tile(*threat_position)
                distance = distance_to(threat_position)
                candidates.append(
                    Mission(
                        DEFEND_HOME_CITY,
                        threat_position,
                        HOME_THREAT_SCORE
                        + threat_tile.army * 520
                        - distance * 180,
                        "all",
                        f"home 3x3 intruder army {threat_tile.army}",
                    )
                )
                continue
            if (
                position == general
                and self._home_guard_priority_active()
                and self.home_guard_hard_deficit > 0
                and not visible_enemies
            ):
                hard_guard_score = (
                    HOME_GUARD_HARD_SCORE
                    + min(200_000, self.home_guard_hard_deficit * 1_200)
                )
                if self.overwhelming_hold_turns > 0:
                    hard_guard_score *= 0.10
                elif self.forced_offensive and self.search_pressure >= 120:
                    # Sustained peace must not turn the capital garrison into
                    # a permanent sink. Keep a reserve, but let formed armies
                    # leave to find the next opponent.
                    hard_guard_score *= 0.45
                candidates.append(
                    Mission(
                        DEFEND_HOME_CITY,
                        position,
                        hard_guard_score,
                        "half",
                        f"hard general garrison deficit "
                        f"{self.home_guard_hard_deficit}",
                    )
                )
                continue
            if (
                position == general
                and self._home_guard_priority_active()
                and self.home_guard_deficit > 0
                and not self.forced_offensive
                and not visible_enemies
            ):
                preferred_score = (
                    11_500
                    + min(9_000, self.home_guard_deficit * 180)
                )
                if self.overwhelming_hold_turns > 0:
                    preferred_score *= 0.20
                candidates.append(
                    Mission(
                        DEFEND_HOME_CITY,
                        position,
                        preferred_score,
                        "half",
                        f"preferred 20 percent general garrison "
                        f"deficit {self.home_guard_deficit}",
                    )
                )
                continue
            threat = self._threat_at(board, position[0], position[1], visible, 2)
            if threat <= 0:
                continue
            terrain_bonus = 8_000 if tile.terrain == GENERAL else 2_500
            severity = max(0, threat - tile.army) * 260
            army_mode = "all" if threat >= tile.army else "half"
            candidates.append(
                Mission(
                    DEFEND_HOME_CITY,
                    position,
                    10_000 + threat * 420 + severity + terrain_bonus,
                    army_mode,
                    f"city threat {threat}",
                )
            )

        for position in owned_tiles:
            if not self._is_frontier(board, position[0], position[1], visible):
                continue
            threat = self._threat_at(board, position[0], position[1], visible, 1)
            if threat <= 0:
                continue
            openness = self._open_neighbor_count(board, position[0], position[1])
            candidates.append(
                Mission(
                    DEFEND_TERRITORY,
                    position,
                    5_000 + threat * 360 + openness * 35,
                    "all" if threat >= board.tile(*position).army else "half",
                    f"front threat {threat}",
                )
            )

        for position in visible:
            tile = board.tile(*position)
            if not self._is_enemy(tile.owner):
                continue
            distance = distance_to(position)
            support = self._nearby_owned_army(board, position[0], position[1], 3)
            if tile.terrain == GENERAL:
                all_in_bonus = (
                    220_000 * self._archetype_all_in_multiplier()
                    if position == self.all_in_target
                    else min(80_000, self.passive_defense_turns * 110)
                )
                score = (
                    600_000
                    - distance * 120
                    + support * 30
                    + all_in_bonus
                )
                score *= self._exposed_general_multiplier(tile.owner)
                candidates.append(
                    Mission(
                        ATTACK_ENEMY_CITY,
                        position,
                        score,
                        "all",
                        "enemy general",
                    )
                )
            elif tile.terrain == CITY:
                force_bonus = (
                    10_000 + min(8_000, (support - tile.army) * 180)
                    if support > tile.army
                    else 0
                )
                score = (
                    16_000
                    + max(0, 35 - tile.army) * 85
                    + support * 22
                    - distance * 105
                    + force_bonus
                )
                score *= 1.0 + self.aggression_heat / 120.0
                score *= self._exposed_general_multiplier(tile.owner)
                candidates.append(
                    Mission(
                        ATTACK_ENEMY_CITY,
                        position,
                        score,
                        "all",
                        f"enemy city army {tile.army}",
                    )
                )
            else:
                force_bonus = (
                    6_000 + min(5_000, (support - tile.army) * 160)
                    if support > tile.army
                    else 0
                )
                score = (
                    7_800
                    + max(0, 18 - tile.army) * 95
                    + support * 16
                    - distance * 85
                    + force_bonus
                )
                score *= 1.0 + self.aggression_heat / 100.0
                score *= self._exposed_general_multiplier(tile.owner)
                candidates.append(
                    Mission(
                        ATTACK_ENEMY_LAND,
                        position,
                        score,
                        "all",
                        f"enemy land army {tile.army}",
                    )
                )

        if (
            self.enemy_hunt_target is not None
            and self.enemy_hunt_player in board.active_players
            and not visible_enemies
            and not any(
                opponent in board.active_players
                for opponent in self.known_enemy_generals
            )
        ):
            target = self.enemy_hunt_target
            distance = distance_to(target)
            mass_attack = (
                self.forced_offensive
                or self.overwhelming_hold_turns > 0
                or self.border_muster_hold_turns > 0
                or self.campaign_hold_turns > 0
            )
            base_score = (
                ENEMY_SEARCH_ATTACK_SCORE
                if mass_attack
                else ENEMY_SEARCH_SUPPORT_SCORE
            )
            hidden_bonus = (
                18_000
                if target not in visible
                else 0
            )
            score = (
                base_score
                + min(60_000, self.enemy_hunt_hold_turns * 450)
                + self.aggression_heat * 160
                - distance * 70
                + hidden_bonus
            )
            candidates.append(
                Mission(
                    ATTACK_ENEMY_LAND,
                    target,
                    score,
                    "all",
                    "enemy territory hunt",
                )
            )

        for opponent, position in self.known_enemy_generals.items():
            if opponent not in board.active_players or position in visible:
                continue
            distance = distance_to(position)
            all_in_bonus = (
                (
                    180_000
                    * self._archetype_all_in_multiplier()
                )
                if position == self.all_in_target
                else 0
            )
            score = (
                KNOWN_GENERAL_CAMPAIGN_SCORE
                - distance * 55
                + self.enemy_army_estimate.get(opponent, 0) * 8
                + all_in_bonus
            )
            score *= self._exposed_general_multiplier(opponent)
            candidates.append(
                Mission(
                    ATTACK_ENEMY_CITY,
                    position,
                    score,
                    "all",
                    "known enemy general decapitation",
                )
            )

        if (
            self.campaign_target is not None
            and self.campaign_hold_turns > 0
            and self.campaign_target not in visible
        ):
            target = self.campaign_target
            distance = distance_to(target)
            if self.campaign_target_kind == ATTACK_ENEMY_CITY:
                all_in_bonus = (
                    220_000 * self._archetype_all_in_multiplier()
                    if target == self.all_in_target
                    else min(80_000, self.passive_defense_turns * 110)
                )
                score = 190_000 - distance * 90 + all_in_bonus
                reason = "persistent enemy general campaign"
            else:
                score = 26_000 - distance * 75 + self.aggression_heat * 80
                reason = "persistent enemy territory campaign"
            score *= self._exposed_general_multiplier(
                self.campaign_target_player
            )
            candidates.append(
                Mission(
                    self.campaign_target_kind or ATTACK_ENEMY_LAND,
                    target,
                    score,
                    "all",
                    reason,
                )
            )

        if (
            self.overwhelming_target is not None
            and self.overwhelming_hold_turns > 0
            and self.overwhelming_target_player in board.active_players
            and board.in_bounds(*self.overwhelming_target)
            and self._is_enemy(board.tile(*self.overwhelming_target).owner)
        ):
            target = self.overwhelming_target
            target_tile = board.tile(*target)
            distance = distance_to(target)
            mission_kind = (
                ATTACK_ENEMY_CITY
                if target_tile.terrain in (CITY, GENERAL)
                else ATTACK_ENEMY_LAND
            )
            score = (
                1_650_000
                - distance * 55
                + min(
                    250_000,
                    self.enemy_army_estimate.get(
                        self.overwhelming_target_player,
                        0,
                    ),
                )
            )
            score *= self._archetype_all_in_multiplier()
            score *= self._exposed_general_multiplier(
                self.overwhelming_target_player
            )
            candidates.append(
                Mission(
                    mission_kind,
                    target,
                    score,
                    "all",
                    "overwhelming force campaign",
                )
            )

        for position in visible:
            tile = board.tile(*position)
            if tile.terrain != CITY or tile.owner != NEUTRAL:
                continue
            if self.home_threat_target is not None:
                continue
            distance = distance_to(position)
            support = self._nearby_owned_army(board, position[0], position[1], 3)
            development_bonus = (
                6_000 + min(8_000, (support - tile.army) * 180)
                if support > tile.army
                else 0
            )
            score = (
                10_800
                + max(0, 45 - tile.army) * 90
                + support * 20
                - distance * 95
                + development_bonus
                + min(8_000, self.passive_defense_turns * 80)
            )
            candidates.append(
                Mission(
                    ATTACK_NEUTRAL_CITY,
                    position,
                    score,
                    "all",
                    f"neutral city army {tile.army}",
                )
            )

        for position in owned_tiles:
            if not self._is_frontier(board, position[0], position[1], visible):
                continue
            hidden = 0
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (position[0] + dx, position[1] + dy)
                if board.in_bounds(*neighbor) and neighbor not in visible:
                    hidden += 1
            threat = self._threat_at(board, position[0], position[1], visible, 2)
            openness = self._open_neighbor_count(board, position[0], position[1])
            score = (
                4_200
                + home_distance.get(position, 0) * 110
                + hidden * 280
                + openness * 35
                - threat * 180
            )
            direction = self.exploration_direction or (0, 0)
            outward: list[tuple[float, tuple[int, int]]] = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (position[0] + dx, position[1] + dy)
                if not board.in_bounds(*neighbor):
                    continue
                if board.tile(*neighbor).owner == self.player:
                    continue
                outward_score = (
                    home_distance.get(neighbor, home_distance.get(position, 0) + 1)
                    * 110
                    + (direction[0] * dx + direction[1] * dy)
                    * EXPANSION_TARGET_ALIGNMENT_WEIGHT
                    + self._expansion_shape_score(
                        board,
                        neighbor[0],
                        neighbor[1],
                        visible,
                        self.player,
                    )
                    * EXPANSION_TARGET_SHAPE_WEIGHT
                )
                outward.append((outward_score, neighbor))
            if not outward:
                continue
            outward.sort(key=lambda item: (-item[0], item[1]))
            explore_target = outward[0][1]
            candidates.append(
                Mission(
                    EXPLORE_EXPAND,
                    explore_target,
                    score,
                    "all",
                    "distant frontier",
                )
            )

        if (
            self.search_waypoint is not None
            and not visible_enemies
            and not self.known_enemy_generals
            and not (
                self.campaign_target is not None
                and self.campaign_hold_turns > 0
                and self.campaign_target_kind == ATTACK_ENEMY_CITY
            )
        ):
            urgency = min(
                ACTIVE_SEARCH_MAX_SCORE - ACTIVE_SEARCH_BASE_SCORE,
                self.search_pressure * ACTIVE_SEARCH_URGENCY_RATE
                + self.search_stall_turns * ACTIVE_SEARCH_STALL_RATE,
            )
            pressure_bonus = (
                ENEMY_SEARCH_ATTACK_SCORE - ACTIVE_SEARCH_BASE_SCORE
                if self.forced_offensive
                else 0.0
            )
            sparse_bonus = 2_200 if self.long_range_search else 0
            search_score = (
                ACTIVE_SEARCH_BASE_SCORE
                + urgency
                + pressure_bonus
                + self.aggression_heat * 45
                + sparse_bonus
            )
            candidates.append(
                Mission(
                    EXPLORE_EXPAND,
                    self.search_waypoint,
                    search_score,
                    "all",
                    f"active search {self.turns_since_contact} turns",
                )
            )

        for candidate in candidates:
            candidate.score *= self._state_score_multiplier(candidate.kind)
            candidate.score *= self._genome_mission_multiplier(candidate.kind)
            candidate.score *= self._archetype_mission_multiplier(
                board,
                candidate.kind,
                candidate.target,
            )
            if (
                self.war_tempo_cooldown_turns > 0
                and len(board.active_players) <= WAR_TEMPO_ENDGAME_PLAYERS
                and self.home_threat_target is None
            ):
                if candidate.kind in (
                    DEFEND_HOME_CITY,
                    DEFEND_TERRITORY,
                ):
                    candidate.score *= 0.20
                elif candidate.kind in (
                    ATTACK_ENEMY_CITY,
                    ATTACK_ENEMY_LAND,
                ):
                    candidate.score *= 1.50
        candidates.sort(key=lambda item: (-item.score, item.kind, item.target))
        return candidates[:80]

    def _mission_target_valid(
        self,
        board: Board,
        mission: Mission,
        visible: set[tuple[int, int]],
    ) -> bool:
        if not board.in_bounds(*mission.target):
            return False
        tile = board.tile(*mission.target)
        if mission.kind == DEFEND_HOME_CITY:
            if tile.owner == self.player:
                return True
            if mission.target not in visible or not self._is_enemy(tile.owner):
                return False
            general = board.general_position(self.player)
            return general is not None and (
                max(
                    abs(mission.target[0] - general[0]),
                    abs(mission.target[1] - general[1]),
                )
                <= HOME_THREAT_RADIUS
            )
        if mission.kind == DEFEND_TERRITORY:
            return tile.owner == self.player
        if mission.kind in (ATTACK_ENEMY_CITY, ATTACK_ENEMY_LAND):
            if mission.target in visible:
                return self._is_enemy(tile.owner)
            if (
                mission.kind == ATTACK_ENEMY_CITY
                and any(
                    position == mission.target
                    for opponent, position in self.known_enemy_generals.items()
                    if opponent in board.active_players
                )
            ):
                return True
            if (
                mission.kind == ATTACK_ENEMY_LAND
                and mission.target == self.enemy_hunt_target
                and self.enemy_hunt_player in board.active_players
            ):
                return True
            if (
                mission.target == self.overwhelming_target
                and self.overwhelming_hold_turns > 0
                and self.overwhelming_target_player in board.active_players
            ):
                return True
            return (
                mission.target == self.campaign_target
                and self.campaign_hold_turns > 0
                and self.campaign_target_player in board.active_players
            )
        if mission.kind == ATTACK_NEUTRAL_CITY:
            return mission.target in visible and tile.terrain == CITY and tile.owner == NEUTRAL
        if mission.kind == EXPLORE_EXPAND:
            if mission.target in visible and tile.terrain == MOUNTAIN:
                return False
            if tile.owner == self.player:
                return self._is_frontier(
                    board,
                    mission.target[0],
                    mission.target[1],
                    visible,
                )
            return True
        return False

    @staticmethod
    def _gather_owned_positions(
        board: Board,
        player: int,
        land_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        positions = set(land_tiles)
        positions.update(own_cities)
        general = board.general_position(player)
        if general is not None:
            positions.add(general)
        return list(positions)

    @staticmethod
    def _gather_path_passable(
        board: Board,
        position: tuple[int, int],
        player: int,
        visible: set[tuple[int, int]],
    ) -> bool:
        tile = board.tile(*position)
        if tile.terrain == MOUNTAIN:
            return False
        if tile.owner == player:
            return True
        return (
            position in visible
            and tile.owner == NEUTRAL
            and tile.terrain != CITY
        )

    def _gather_rally_point(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not owned_tiles:
            return None
        border_target = self.border_muster_target
        if (
            border_target is not None
            and board.in_bounds(*border_target)
            and board.tile(*border_target).owner == self.player
        ):
            return border_target
        mission_target = self.mission.target if self.mission is not None else None
        if (
            mission_target is not None
            and board.in_bounds(*mission_target)
            and board.tile(*mission_target).owner == self.player
        ):
            return mission_target
        if mission_target is not None and board.in_bounds(*mission_target):
            staging = self._select_border_muster_target(
                board,
                visible,
                owned_tiles,
                mission_target,
            )
            if staging is not None:
                return staging
        return max(
            owned_tiles,
            key=lambda position: (
                board.tile(*position).army,
                -position[0] - position[1],
            ),
        )

    def _trace_gather_path(
        self,
        board: Board,
        origin: tuple[int, int],
        rally: tuple[int, int],
        visible: set[tuple[int, int]],
        distances: DistanceField | None = None,
    ) -> tuple[tuple[int, int], ...] | None:
        if origin == rally:
            return (origin,)
        if not self._gather_path_passable(
            board,
            origin,
            self.player,
            visible,
        ):
            return None
        distances = distances or self._distance_map(board, [rally])
        current_distance = distances.get(origin)
        if current_distance is None:
            return None

        path = [origin]
        seen = {origin}
        current = origin
        for _step in range(GATHER_ROUTE_MAX_STEPS):
            if current == rally:
                return tuple(path)
            candidates: list[
                tuple[int, int, int, tuple[int, int]]
            ] = []
            for dx, dy in DIRECTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in seen or not board.in_bounds(*neighbor):
                    continue
                if not self._gather_path_passable(
                    board,
                    neighbor,
                    self.player,
                    visible,
                ):
                    continue
                neighbor_distance = distances.get(neighbor)
                if (
                    neighbor_distance is None
                    or neighbor_distance >= current_distance
                ):
                    continue
                tile = board.tile(*neighbor)
                candidates.append(
                    (
                        int(neighbor_distance),
                        0 if tile.terrain == CITY else 1,
                        -tile.army,
                        neighbor,
                    )
                )
            if not candidates:
                return None
            next_distance, _castle_rank, _army_rank, current = min(candidates)
            current_distance = next_distance
            path.append(current)
            seen.add(current)
        return None

    def _gather_path_score(
        self,
        board: Board,
        path: tuple[tuple[int, int], ...],
        rally: tuple[int, int],
    ) -> int:
        collected = 0
        castle_count = 0
        for position in path:
            tile = board.tile(*position)
            if tile.owner != self.player:
                continue
            if position == rally:
                collected += tile.army
            else:
                collected += max(0, tile.army - 1)
            if tile.terrain == CITY:
                castle_count += 1
        return int(
            collected
            + castle_count * GATHER_CASTLE_ROUTE_BONUS
        )

    def _best_gather_path_from_origin(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        origin: tuple[int, int],
        rally: tuple[int, int],
        own_cities: list[tuple[int, int]],
        distance_provider: object,
    ) -> tuple[tuple[tuple[int, int], ...], int] | None:
        direct = self._trace_gather_path(
            board,
            origin,
            rally,
            visible,
            distance_provider(rally),
        )
        best_path = direct
        best_score = (
            self._gather_path_score(board, direct, rally)
            if direct is not None
            else -1
        )

        castle_candidates = sorted(
            (
                castle
                for castle in own_cities
                if castle not in (origin, rally)
                and board.in_bounds(*castle)
                and board.tile(*castle).owner == self.player
            ),
            key=lambda castle: (
                -board.tile(*castle).army,
                Board.manhattan(origin, castle),
                castle,
            ),
        )[:GATHER_ROUTE_CASTLE_LIMIT]
        for castle in castle_candidates:
            first = self._trace_gather_path(
                board,
                origin,
                castle,
                visible,
                distance_provider(castle),
            )
            second = self._trace_gather_path(
                board,
                castle,
                rally,
                visible,
                distance_provider(rally),
            )
            if first is None or second is None:
                continue
            combined = first + second[1:]
            if len(combined) > GATHER_ROUTE_MAX_STEPS:
                continue
            if len(set(combined)) != len(combined):
                continue
            score = self._gather_path_score(board, combined, rally)
            if score > best_score:
                best_path = combined
                best_score = score

        if best_path is None:
            return None
        return best_path, best_score

    def _build_land_gather_route(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
        rally: tuple[int, int] | None = None,
    ) -> GatherRoute | None:
        if rally is None:
            rally = self._gather_rally_point(board, visible, owned_tiles)
        if rally is None:
            return None

        def distance_provider(
            position: tuple[int, int],
        ) -> DistanceField:
            return self._distance_map(board, [position])

        def origin_priority(
            position: tuple[int, int],
        ) -> tuple[float, float, float, int, tuple[int, int]]:
            tile = board.tile(*position)
            ratio = self._regional_reinforcement_ratio(board, position)
            idle_bonus = (
                min(12.0, max(0.0, ratio - REGION_REINFORCEMENT_RATIO))
                * GATHER_IDLE_ORIGIN_WEIGHT
            )
            return (
                -(tile.army + idle_bonus),
                -ratio,
                -self._regional_average_army(board, position),
                Board.manhattan(position, rally),
                position,
            )

        general = board.general_position(self.player)
        candidates = sorted(
            (
                position
                for position in owned_tiles
                if position != rally
                and position != general
                and board.tile(*position).army > 1
            ),
            key=origin_priority,
        )[:GATHER_ROUTE_ORIGIN_LIMIT]

        best_path: tuple[tuple[int, int], ...] | None = None
        best_score = -1
        for origin in candidates:
            result = self._best_gather_path_from_origin(
                board,
                visible,
                origin,
                rally,
                own_cities,
                distance_provider,
            )
            if result is None:
                continue
            path, score = result
            if score > best_score:
                best_path = path
                best_score = score
        if best_path is None:
            return None
        return GatherRoute(
            "land",
            best_path[0],
            rally,
            best_path,
            best_score,
            board.turn,
        )

    def _mountain_effective_army(
        self,
        board: Board,
        position: tuple[int, int],
    ) -> float:
        values: list[int] = []
        for dx, dy in DIRECTIONS:
            neighbor = (position[0] + dx, position[1] + dy)
            if not board.in_bounds(*neighbor):
                continue
            tile = board.tile(*neighbor)
            if tile.terrain == MOUNTAIN or tile.terrain == CITY:
                continue
            values.append(tile.army)
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _castle_adjacent_army_total(
        self,
        board: Board,
        castle: tuple[int, int],
    ) -> float:
        total = 0.0
        for dx, dy in DIRECTIONS:
            neighbor = (castle[0] + dx, castle[1] + dy)
            if not board.in_bounds(*neighbor):
                continue
            tile = board.tile(*neighbor)
            if tile.terrain == CITY:
                continue
            if tile.terrain == MOUNTAIN:
                total += self._mountain_effective_army(board, neighbor)
            else:
                total += tile.army
        return total

    def _select_castle_gather_route(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        owned_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
        rally: tuple[int, int] | None = None,
    ) -> GatherRoute | None:
        if rally is None:
            rally = self._gather_rally_point(board, visible, owned_tiles)
        if rally is None:
            return None
        distances = self._distance_map(board, [rally])
        eligible: list[tuple[float, tuple[int, int]]] = []
        for castle in own_cities:
            if castle == rally or not board.in_bounds(*castle):
                continue
            tile = board.tile(*castle)
            if tile.owner != self.player or tile.terrain != CITY:
                continue
            adjacent_army = self._castle_adjacent_army_total(board, castle)
            if tile.army <= adjacent_army:
                continue
            distance = distances.get(castle)
            if distance is None or distance <= 0:
                continue
            ratio = tile.army / max(1, distance)
            if ratio <= GATHER_CASTLE_DISTANCE_RATIO:
                continue
            eligible.append((ratio, castle))

        eligible.sort(
            key=lambda item: (
                -item[0],
                -board.tile(*item[1]).army,
                item[1],
            )
        )
        best: tuple[float, int, GatherRoute] | None = None

        def distance_provider(
            position: tuple[int, int],
        ) -> DistanceField:
            return self._distance_map(board, [position])

        for ratio, castle in eligible[:GATHER_ROUTE_CASTLE_LIMIT]:
            path = self._best_gather_path_from_origin(
                board,
                visible,
                castle,
                rally,
                own_cities,
                distance_provider,
            )
            if path is None:
                continue
            route_path, collected = path
            route = GatherRoute(
                "castle",
                castle,
                rally,
                route_path,
                collected,
                board.turn,
            )
            candidate = (ratio, collected, route)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return best[2] if best is not None else None

    def _gather_route_move_bonus(
        self,
        board: Board,
        move: Move,
    ) -> float:
        route = self.gather_route
        if route is None:
            return 0.0
        try:
            index = route.path.index(move.source)
        except ValueError:
            return 0.0
        if index + 1 >= len(route.path):
            return 0.0
        if move.target != route.path[index + 1]:
            return 0.0
        source = board.tile(move.sx, move.sy)
        target = board.tile(move.tx, move.ty)
        bonus = (
            GATHER_ROUTE_MOVE_BONUS
            + min(source.army, GATHER_STACK_ARMY_CAP) * 190
        )
        if target.terrain == CITY and target.owner == self.player:
            bonus += 5_000
        return bonus

    def _update_gather_mode(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        land_tiles: list[tuple[int, int]],
        own_cities: list[tuple[int, int]],
    ) -> None:
        self.gather_route = None
        if self.overwhelming_hold_turns > 0:
            self.gather_mode = "overwhelming"
            self.gather_army_mode = "all"
            return
        if self.border_muster_hold_turns > 0:
            self.gather_mode = (
                "border_launch"
                if self.border_muster_ready
                else "border"
            )
            self.gather_army_mode = "all"
            return
        if self.forced_offensive:
            self.gather_mode = None
            self.gather_army_mode = "half"
            return
        ready_land = [
            position
            for position in land_tiles
            if board.tile(*position).army >= LAND_GATHER_MIN_ARMY
        ]
        land_ready = bool(ready_land) and len(ready_land) * 2 > len(land_tiles)
        land_capacity = sum(
            max(0, board.tile(*position).army - 1) for position in ready_land
        )
        if not land_ready and not own_cities:
            self.gather_mode = None
            self.gather_army_mode = "half"
            return

        owned_tiles = self._gather_owned_positions(
            board,
            self.player,
            land_tiles,
            own_cities,
        )
        rally = self._gather_rally_point(board, visible, owned_tiles)
        if own_cities:
            castle_route = self._select_castle_gather_route(
                board,
                visible,
                owned_tiles,
                own_cities,
                rally,
            )
            if castle_route is not None:
                self.gather_mode = "city"
                self.gather_army_mode = "all"
                self.gather_route = castle_route
                return

        ready_cities = [
            position
            for position in own_cities
            if board.tile(*position).army >= CITY_GATHER_MIN_ARMY
        ]
        city_capacity = sum(
            max(0, board.tile(*position).army - 1) for position in ready_cities
        )

        self.gather_mode = None
        self.gather_army_mode = "half"
        if not land_ready and not ready_cities:
            return
        if land_ready and land_capacity >= city_capacity:
            self.gather_mode = "land"
            self.gather_army_mode = "all"
            self.gather_route = self._build_land_gather_route(
                board,
                visible,
                owned_tiles,
                own_cities,
                rally,
            )
            return

        self.gather_mode = "city"
        self.gather_route = None
        strongest = max(ready_cities, key=lambda position: board.tile(*position).army)
        city_army = board.tile(*strongest).army
        pressure = self._threat_at(
            board, strongest[0], strongest[1], visible, 2
        )
        self.gather_army_mode = "all" if pressure * 2 >= city_army else "half"

    def _mission_move_bonus(
        self,
        board: Board,
        move: Move,
        source: object,
        target: object,
        visible: set[tuple[int, int]],
        progress: int,
        source_frontier: bool,
    ) -> float:
        if self.mission is None:
            return 0.0
        bonus = 0.0
        mission = self.mission
        local_expansion = (
            mission.kind == EXPLORE_EXPAND
            and not self.forced_offensive
            and self.search_waypoint != mission.target
        )
        active_search = (
            mission.kind == EXPLORE_EXPAND
            and self.search_waypoint == mission.target
            and not self.planning_has_visible_enemy
        )
        enemy_hunt = (
            mission.target == self.enemy_hunt_target
            and self.enemy_hunt_player in board.active_players
            and not any(
                opponent in board.active_players
                for opponent in self.known_enemy_generals
            )
        )
        if progress > 0:
            if self.strategy_state == STRATEGY_ATTACK and mission.kind in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
            ):
                bonus += progress * 180
            elif self.strategy_state == STRATEGY_DEFENSE and mission.kind in (
                DEFEND_HOME_CITY,
                DEFEND_TERRITORY,
            ):
                bonus += progress * 220
            elif (
                self.strategy_state == STRATEGY_DEVELOPMENT
                and mission.kind == ATTACK_NEUTRAL_CITY
            ):
                bonus += progress * 190
            elif (
                self.strategy_state == STRATEGY_EXPLORATION
                and mission.kind == EXPLORE_EXPAND
            ):
                bonus += progress * (
                    90 if local_expansion else 210
                )
        if progress > 0:
            if mission.kind in (
                ATTACK_ENEMY_CITY,
                ATTACK_ENEMY_LAND,
                ATTACK_NEUTRAL_CITY,
            ):
                bonus += progress * 520
            elif mission.kind in (DEFEND_HOME_CITY, DEFEND_TERRITORY):
                bonus += progress * 600
            else:
                bonus += progress * (
                    110 if local_expansion else 360
                )
        if active_search:
            search_weight = self.genome.search * self.genome.exploration
            if progress > 0:
                bonus += (
                    progress * (1_100 + self.aggression_heat * 10)
                    * search_weight
                    + min(source.army, GATHER_STACK_ARMY_CAP) * 150
                )
            else:
                bonus -= 1_800
        if enemy_hunt:
            hunt_weight = (
                self.genome.search
                * self.genome.aggression
                * self.genome.local_advantage
            )
            if progress > 0:
                bonus += (
                    progress * 1_250 * hunt_weight
                    + min(source.army, GATHER_STACK_ARMY_CAP) * 165
                )
            else:
                bonus -= 900
        if self.forced_offensive:
            if self.primary_stack == move.source:
                bonus += 7_000 + min(source.army, GATHER_STACK_ARMY_CAP) * 220
                if progress > 0:
                    bonus += progress * 1_500
                elif move.target != mission.target:
                    bonus -= 2_500
            else:
                bonus -= 1_800
        if self.overwhelming_hold_turns > 0:
            if progress > 0:
                bonus += 1_200 + min(source.army, GATHER_STACK_ARMY_CAP) * 90
            if move.source == self.primary_stack:
                bonus += 6_000
            if move.target == self.overwhelming_target:
                bonus += 20_000
        if move.target == mission.target:
            bonus += 1_200 if local_expansion else 4_000
        if progress > 0 and mission.kind in (
            ATTACK_ENEMY_CITY,
            ATTACK_ENEMY_LAND,
            DEFEND_HOME_CITY,
            DEFEND_TERRITORY,
        ):
            bonus += self._reinforcement_priority_bonus(board, move.source)

        bonus += self._gather_route_move_bonus(board, move)
        if (
            self.gather_mode == "land"
            and source.terrain in (PLAIN, HILL)
            and source.army >= LAND_GATHER_MIN_ARMY
            and progress > 0
        ):
            bonus += (
                3_200
                + min(source.army, GATHER_STACK_ARMY_CAP) * 120
                + self._gather_proximity_bonus(move.source, mission.target)
            )
        elif (
            self.gather_mode == "city"
            and source.terrain == CITY
            and source.army >= CITY_GATHER_MIN_ARMY
            and progress > 0
        ):
            bonus += (
                3_600
                + min(source.army, GATHER_STACK_ARMY_CAP) * 135
                + self._gather_proximity_bonus(move.source, mission.target)
            )
        elif (
            self.gather_mode in ("border", "border_launch")
            and progress > 0
        ):
            anchor = self.border_muster_target or mission.target
            bonus += (
                min(source.army, GATHER_STACK_ARMY_CAP) * 150
                + self._gather_proximity_bonus(move.source, anchor)
                + (3_000 if self.gather_mode == "border" else 5_000)
            )
        elif self.gather_mode is not None and progress > 0:
            bonus -= 450

        if source_frontier and mission.kind == DEFEND_TERRITORY:
            bonus += 700
        if mission.kind == ATTACK_ENEMY_CITY and target.terrain == CITY:
            bonus += 900
        return bonus

    def _gather_proximity_bonus(
        self,
        source: tuple[int, int],
        anchor: tuple[int, int],
    ) -> float:
        distance = Board.manhattan(source, anchor)
        return max(0, 80 - distance) * 28.0

    def _threat_at(
        self,
        board: Board,
        x: int,
        y: int,
        visible: set[tuple[int, int]],
        radius: int,
    ) -> int:
        strength = 0
        player = self.player
        grid = board.grid
        for ny in range(max(0, y - radius), min(board.height, y + radius + 1)):
            row = grid[ny]
            for nx in range(max(0, x - radius), min(board.width, x + radius + 1)):
                if abs(nx - x) + abs(ny - y) > radius:
                    continue
                if (nx, ny) not in visible:
                    continue
                tile = row[nx]
                if tile.owner >= 0 and tile.owner != player:
                    strength += tile.army
        return strength

    def _nearby_owned_army(
        self,
        board: Board,
        x: int,
        y: int,
        radius: int,
    ) -> int:
        strength = 0
        player = self.player
        grid = board.grid
        for ny in range(max(0, y - radius), min(board.height, y + radius + 1)):
            row = grid[ny]
            for nx in range(max(0, x - radius), min(board.width, x + radius + 1)):
                if abs(nx - x) + abs(ny - y) > radius:
                    continue
                tile = row[nx]
                if tile.owner == player:
                    strength += max(0, tile.army - 1)
        return strength

    def _is_enemy(self, owner: int) -> bool:
        return owner >= 0 and owner != self.player

    @staticmethod
    def _positions_by_owner(
        board: Board,
    ) -> tuple[tuple[tuple[int, int], ...], ...]:
        board_key = id(board)
        cached = BOARD_OWNER_POSITION_CACHE.get(board_key)
        if (
            cached is not None
            and cached[0] is board
            and cached[1] == board.state_revision
        ):
            BOARD_OWNER_POSITION_CACHE.move_to_end(board_key)
            return cached[2]

        positions: list[list[tuple[int, int]]] = [
            [] for _ in range(MAX_PLAYERS)
        ]
        for y, row in enumerate(board.grid):
            for x, tile in enumerate(row):
                if 0 <= tile.owner < MAX_PLAYERS:
                    positions[tile.owner].append((x, y))
        frozen = tuple(tuple(items) for items in positions)
        BOARD_OWNER_POSITION_CACHE[board_key] = (
            board,
            board.state_revision,
            frozen,
        )
        BOARD_OWNER_POSITION_CACHE.move_to_end(board_key)
        while len(BOARD_OWNER_POSITION_CACHE) > BOARD_OWNER_POSITION_CACHE_LIMIT:
            BOARD_OWNER_POSITION_CACHE.popitem(last=False)
        return frozen

    def _cached_visibility(
        self,
        board: Board,
        positions: Iterable[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        stamp = (id(board), board.state_revision, board.vision_radius)
        if stamp != self.visibility_cache_stamp:
            self.visibility_cache = board.visibility_from_positions(positions)
            self.visibility_cache_stamp = stamp
        return self.visibility_cache

    def _friendly_neighbor_count(self, board: Board, x: int, y: int) -> int:
        count = 0
        player = self.player
        width = board.width
        height = board.height
        grid = board.grid
        if x > 0 and grid[y][x - 1].owner == player:
            count += 1
        if x + 1 < width and grid[y][x + 1].owner == player:
            count += 1
        if y > 0 and grid[y - 1][x].owner == player:
            count += 1
        if y + 1 < height and grid[y + 1][x].owner == player:
            count += 1
        return count

    def _is_frontier(
        self,
        board: Board,
        x: int,
        y: int,
        visible: set[tuple[int, int]],
    ) -> bool:
        player = self.player
        width = board.width
        height = board.height
        grid = board.grid
        if x > 0:
            position = (x - 1, y)
            if position not in visible:
                return True
            neighbor = grid[y][x - 1]
            if neighbor.owner != player and neighbor.terrain != MOUNTAIN:
                return True
        if x + 1 < width:
            position = (x + 1, y)
            if position not in visible:
                return True
            neighbor = grid[y][x + 1]
            if neighbor.owner != player and neighbor.terrain != MOUNTAIN:
                return True
        if y > 0:
            position = (x, y - 1)
            if position not in visible:
                return True
            neighbor = grid[y - 1][x]
            if neighbor.owner != player and neighbor.terrain != MOUNTAIN:
                return True
        if y + 1 < height:
            position = (x, y + 1)
            if position not in visible:
                return True
            neighbor = grid[y + 1][x]
            if neighbor.owner != player and neighbor.terrain != MOUNTAIN:
                return True
        return False

    def _nearby_enemy_strength(
        self,
        board: Board,
        x: int,
        y: int,
        visible: set[tuple[int, int]],
    ) -> int:
        strength = 0
        player = self.player
        grid = board.grid
        if (
            (x - 1, y) in visible
            and grid[y][x - 1].owner >= 0
            and grid[y][x - 1].owner != player
        ):
            strength += grid[y][x - 1].army
        if (
            (x + 1, y) in visible
            and grid[y][x + 1].owner >= 0
            and grid[y][x + 1].owner != player
        ):
            strength += grid[y][x + 1].army
        if (
            (x, y - 1) in visible
            and grid[y - 1][x].owner >= 0
            and grid[y - 1][x].owner != player
        ):
            strength += grid[y - 1][x].army
        if (
            (x, y + 1) in visible
            and grid[y + 1][x].owner >= 0
            and grid[y + 1][x].owner != player
        ):
            strength += grid[y + 1][x].army
        return strength

    @staticmethod
    def _open_neighbor_count(board: Board, x: int, y: int) -> int:
        count = 0
        width = board.width
        height = board.height
        grid = board.grid
        if x > 0 and grid[y][x - 1].terrain != MOUNTAIN:
            count += 1
        if x + 1 < width and grid[y][x + 1].terrain != MOUNTAIN:
            count += 1
        if y > 0 and grid[y - 1][x].terrain != MOUNTAIN:
            count += 1
        if y + 1 < height and grid[y + 1][x].terrain != MOUNTAIN:
            count += 1
        return count

    @staticmethod
    def _expansion_shape_score(
        board: Board,
        x: int,
        y: int,
        visible: set[tuple[int, int]],
        player: int,
    ) -> float:
        """Reward frontier cells that fan out instead of extending a corridor."""
        direction_count = 0
        open_cells = 0
        width = board.width
        height = board.height
        grid = board.grid
        for dx, dy in (
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ):
            direction_has_open = False
            for step in (1, 2):
                nx = x + dx * step
                ny = y + dy * step
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    break
                tile = grid[ny][nx]
                if tile.terrain == MOUNTAIN:
                    break
                owner = (
                    tile.owner
                    if (nx, ny) in visible
                    else NEUTRAL
                )
                if owner != NEUTRAL:
                    break
                direction_has_open = True
                open_cells += 1
            if direction_has_open:
                direction_count += 1
        return (
            direction_count * EXPANSION_SHAPE_DIRECTION_BONUS
            + min(open_cells, 12) * EXPANSION_SHAPE_OPEN_BONUS
        )

    @staticmethod
    def _distance_neighbors(
        board: Board,
    ) -> tuple[list[Tile], tuple[tuple[tuple[int, int], ...], ...]]:
        board_key = id(board)
        cached = BOARD_NEIGHBOR_CACHE.get(board_key)
        if cached is not None and cached[0] is board:
            BOARD_NEIGHBOR_CACHE.move_to_end(board_key)
            return cached[1], cached[2]

        width = board.width
        height = board.height
        tiles = [tile for row in board.grid for tile in row]
        neighbors: list[list[tuple[int, int]]] = [
            [] for _ in range(width * height)
        ]
        for y in range(height):
            for x in range(width):
                index = y * width + x
                source = tiles[index]
                if source.terrain == MOUNTAIN:
                    continue
                source_hill = source.terrain == HILL
                cell_neighbors: list[tuple[int, int]] = []
                for dx, dy in DIRECTIONS:
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    neighbor_index = ny * width + nx
                    neighbor = tiles[neighbor_index]
                    if neighbor.terrain != MOUNTAIN:
                        edge_type = (
                            int(source_hill)
                            | (int(neighbor.terrain == HILL) << 1)
                        )
                        cell_neighbors.append((neighbor_index, edge_type))
                neighbors[index] = cell_neighbors

        frozen = [tuple(items) for items in neighbors]
        BOARD_NEIGHBOR_CACHE[board_key] = (board, tiles, frozen)
        BOARD_NEIGHBOR_CACHE.move_to_end(board_key)
        while len(BOARD_NEIGHBOR_CACHE) > BOARD_NEIGHBOR_CACHE_LIMIT:
            BOARD_NEIGHBOR_CACHE.popitem(last=False)
        return tiles, frozen

    def _distance_map(
        self,
        board: Board,
        goals: list[tuple[int, int]],
        max_depth: int | None = None,
    ) -> DistanceField:
        goals_key = tuple(goals)
        terrain_revision = board.terrain_revision
        hill_revision = board.hill_owner_revision.get(self.player, 0)
        cache_key = (terrain_revision, hill_revision, goals_key, max_depth)
        cached = self.distance_cache.get(cache_key)
        if cached is not None:
            return cached
        persistent_key = (
            id(board),
            terrain_revision,
            hill_revision,
            goals_key,
        )
        persistent = (
            self.persistent_distance_cache.get(persistent_key)
            if len(goals_key) <= 2 and max_depth is None
            else None
        )
        if persistent is not None:
            self.persistent_distance_cache.move_to_end(persistent_key)
            self.distance_cache[cache_key] = persistent
            return persistent

        width = board.width
        height = board.height
        infinity = 10**9
        tiles, neighbors = self._distance_neighbors(board)
        sparse = max_depth is not None
        values: list[int] | dict[int, int]
        if sparse:
            values = {}
        else:
            values = [infinity] * (width * height)
        bucket_count = 3
        buckets: list[deque[int]] = [deque() for _ in range(bucket_count)]
        queued = 0
        for goal in goals:
            x, y = goal
            if not (0 <= x < width and 0 <= y < height):
                continue
            index = y * width + x
            if (
                values.get(index, infinity) == 0
                if sparse
                else values[index] == 0
            ):
                continue
            values[index] = 0
            buckets[0].append(index)
            queued += 1

        current_distance = 0
        while queued:
            if max_depth is not None and current_distance > max_depth:
                break
            bucket = buckets[current_distance % bucket_count]
            if not bucket:
                current_distance += 1
                continue
            index = bucket.popleft()
            queued -= 1
            if values[index] != current_distance:
                continue
            for neighbor_index, edge_type in neighbors[index]:
                if edge_type == 1:
                    step_cost = 2
                elif edge_type == 2:
                    step_cost = (
                        1
                        if tiles[neighbor_index].owner == self.player
                        else 2
                    )
                else:
                    step_cost = 1
                next_distance = current_distance + step_cost
                if sparse:
                    previous = values.get(neighbor_index, infinity)
                else:
                    previous = values[neighbor_index]
                if next_distance < previous:
                    values[neighbor_index] = next_distance
                    buckets[next_distance % bucket_count].append(neighbor_index)
                    queued += 1

        result = DistanceField(width, height, values, infinity)
        self.distance_cache[cache_key] = result
        if len(goals_key) <= 2 and max_depth is None:
            self.persistent_distance_cache[persistent_key] = result
            self.persistent_distance_cache.move_to_end(persistent_key)
            while len(self.persistent_distance_cache) > 12:
                self.persistent_distance_cache.popitem(last=False)
        return result

    def _mission_distance_map(
        self,
        board: Board,
        visible: set[tuple[int, int]],
        goals: list[tuple[int, int]],
    ) -> DistanceField:
        mission = self.mission
        if mission is None or mission.kind not in (
            ATTACK_ENEMY_CITY,
            ATTACK_ENEMY_LAND,
            ATTACK_NEUTRAL_CITY,
        ):
            return self._distance_map(board, goals)
        if goals and all(
            board.in_bounds(*goal)
                and board.tile(*goal).owner == self.player
            for goal in goals
        ):
            return self._distance_map(board, goals)
        base_distances = self._distance_map(board, goals)
        visible_enemies = [
            position
            for position in visible
            if self._is_enemy(board.tile(*position).owner)
        ]
        if not visible_enemies:
            return base_distances

        owned_positions = self.planning_owned_tiles
        if not owned_positions:
            owned_positions = [
                (x, y)
                for y, row in enumerate(board.grid)
                for x, tile in enumerate(row)
                if tile.owner == self.player
            ]
        max_mobile_army = self.planning_max_mobile_army
        if max_mobile_army <= 1:
            max_mobile_army = max(
                (
                    board.tile(*position).army - 1
                    for position in owned_positions
                    if board.tile(*position).army > 1
                ),
                default=1,
            )
        max_owned_distance = max(
            (
                base_distances.get(
                    position,
                    board.width + board.height,
                )
                for position in owned_positions
                if board.tile(*position).army > 1
            ),
            default=0,
        )
        route_enemy_armies = [
            board.tile(*position).army
            for position in visible_enemies
            if base_distances.get(
                position,
                board.width + board.height,
            )
            < max_owned_distance
        ]
        if not route_enemy_armies:
            return base_distances
        if (
            max_mobile_army
            > sum(route_enemy_armies)
            * 1.25
            * self.genome.local_advantage
        ):
            return base_distances
        enemy_signature = tuple(
            sorted(
                (
                    x,
                    y,
                    board.tile(x, y).owner,
                    board.tile(x, y).army,
                    board.tile(x, y).terrain,
                )
                for x, y in visible_enemies
            )
        )
        cache_key = (
            board.width,
            board.height,
            board.terrain_revision,
            board.hill_owner_revision.get(self.player, 0),
            max_mobile_army,
            mission.kind,
            tuple(goals),
            enemy_signature,
        )
        cached = self.mission_distance_cache.get(cache_key)
        if cached is not None:
            return cached

        width = board.width
        height = board.height
        infinity = 10**9
        tiles, neighbors = self._distance_neighbors(board)
        goal_indices = {
            y * width + x
            for x, y in goals
            if 0 <= x < width and 0 <= y < height
        }
        visible_indices = {
            y * width + x
            for x, y in visible
            if 0 <= x < width and 0 <= y < height
        }
        risk_costs = [0] * (width * height)
        for index in visible_indices:
            if index in goal_indices:
                continue
            target = tiles[index]
            if not self._is_enemy(target.owner):
                continue
            shortfall = max(0, target.army - max_mobile_army + 1)
            risk = min(
                MISSION_RISK_CAP,
                18 + target.army * 2 + shortfall * 10,
            )
            if target.terrain == CITY:
                risk += 12
            risk_costs[index] = int(
                min(
                    MISSION_RISK_CAP,
                    risk / max(0.70, self.genome.risk_tolerance),
                )
            )
        values = [infinity] * (width * height)
        bucket_count = MISSION_RISK_BUCKETS
        buckets: list[deque[int]] = [deque() for _ in range(bucket_count)]
        queued = 0
        for goal in goals:
            x, y = goal
            if not (0 <= x < width and 0 <= y < height):
                continue
            index = y * width + x
            if values[index] == 0:
                continue
            values[index] = 0
            buckets[0].append(index)
            queued += 1

        current_distance = 0
        while queued:
            bucket = buckets[current_distance % bucket_count]
            if not bucket:
                current_distance += 1
                continue
            index = bucket.popleft()
            queued -= 1
            if values[index] != current_distance:
                continue
            for neighbor_index, edge_type in neighbors[index]:
                if edge_type == 1:
                    step_cost = 2
                elif edge_type == 2:
                    step_cost = (
                        1
                        if tiles[neighbor_index].owner == self.player
                        else 2
                    )
                else:
                    step_cost = 1

                next_distance = (
                    current_distance + step_cost + risk_costs[neighbor_index]
                )
                if next_distance < values[neighbor_index]:
                    values[neighbor_index] = next_distance
                    buckets[next_distance % bucket_count].append(neighbor_index)
                    queued += 1

        result = DistanceField(width, height, values, infinity)
        self.mission_distance_cache[cache_key] = result
        return result
