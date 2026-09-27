from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Iterable


HUMAN = 0
AI = 1
NEUTRAL = -1
MAX_PLAYERS = 37

PLAIN = 0
MOUNTAIN = 1
CITY = 2
GENERAL = 3
HILL = 4

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
VISION_LINE_CACHE: dict[
    int,
    tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...],
] = {}


def _line_offsets(dx: int, dy: int) -> tuple[tuple[int, int], ...]:
    if dx == 0 and dy == 0:
        return ((0, 0),)
    points = [(0, 0)]
    x = 0
    y = 0
    x_distance = abs(dx)
    y_distance = abs(dy)
    step_x = 1 if dx > 0 else -1 if dx < 0 else 0
    step_y = 1 if dy > 0 else -1 if dy < 0 else 0
    error = x_distance - y_distance
    while x != dx or y != dy:
        doubled = error * 2
        if doubled > -y_distance:
            error -= y_distance
            x += step_x
        if doubled < x_distance:
            error += x_distance
            y += step_y
        points.append((x, y))
    return tuple(points)


def _vision_lines(
    radius: int,
) -> tuple[tuple[int, int, tuple[tuple[int, int], ...]], ...]:
    cached = VISION_LINE_CACHE.get(radius)
    if cached is not None:
        return cached
    lines: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            offsets = _line_offsets(dx, dy)
            lines.append((dx, dy, offsets[1:-1]))
    frozen = tuple(lines)
    VISION_LINE_CACHE[radius] = frozen
    return frozen


@dataclass(slots=True)
class Tile:
    terrain: int = PLAIN
    owner: int = NEUTRAL
    army: int = 0

    def clone(self) -> "Tile":
        return Tile(self.terrain, self.owner, self.army)


@dataclass(frozen=True, slots=True)
class Move:
    sx: int
    sy: int
    tx: int
    ty: int
    amount: int

    @property
    def source(self) -> tuple[int, int]:
        return self.sx, self.sy

    @property
    def target(self) -> tuple[int, int]:
        return self.tx, self.ty


@dataclass(slots=True)
class PendingMove:
    sx: int
    sy: int
    tx: int
    ty: int
    amount: int
    owner: int
    remaining: int = 1
    ready_turn: int = 0

    def clone(self) -> "PendingMove":
        return PendingMove(
            self.sx,
            self.sy,
            self.tx,
            self.ty,
            self.amount,
            self.owner,
            self.remaining,
            self.ready_turn,
        )


@dataclass(slots=True)
class TurnReport:
    accepted_moves: int = 0
    captures: list[tuple[int, int, int]] | None = None
    eliminated: list[int] | None = None
    income: dict[int, int] | None = None
    land_growth: dict[int, int] | None = None
    delayed_moves: int = 0
    arrived_moves: int = 0
    actions_by_player: dict[int, int] | None = None
    attrition_by_pair: dict[tuple[int, int], int] | None = None
    winner: int | None = None

    def __post_init__(self) -> None:
        if self.captures is None:
            self.captures = []
        if self.eliminated is None:
            self.eliminated = []
        if self.income is None:
            self.income = {}
        if self.land_growth is None:
            self.land_growth = {}
        if self.actions_by_player is None:
            self.actions_by_player = {}
        if self.attrition_by_pair is None:
            self.attrition_by_pair = {}

    def merge(self, other: "TurnReport") -> None:
        self.accepted_moves += other.accepted_moves
        self.captures.extend(other.captures or [])
        self.eliminated.extend(other.eliminated or [])
        for player, amount in (other.income or {}).items():
            self.income[player] = self.income.get(player, 0) + amount
        for player, amount in (other.land_growth or {}).items():
            self.land_growth[player] = self.land_growth.get(player, 0) + amount
        self.delayed_moves += other.delayed_moves
        self.arrived_moves += other.arrived_moves
        for player, amount in (other.actions_by_player or {}).items():
            self.actions_by_player[player] = (
                self.actions_by_player.get(player, 0) + amount
            )
        for pair, amount in (other.attrition_by_pair or {}).items():
            self.attrition_by_pair[pair] = (
                self.attrition_by_pair.get(pair, 0) + amount
            )
        if other.winner is not None:
            self.winner = other.winner

    def record_attrition(self, first: int, second: int, amount: int) -> None:
        if first < 0 or second < 0 or first == second or amount <= 0:
            return
        pair = (first, second) if first < second else (second, first)
        self.attrition_by_pair[pair] = self.attrition_by_pair.get(pair, 0) + amount


class Board:
    """Rules and mutable state for a multiplayer Generals game."""

    def __init__(
        self,
        width: int = 45,
        height: int = 45,
        player_count: int = 2,
        seed: int | None = None,
        growth_interval: int = 20,
        vision_radius: int = 2,
    ) -> None:
        if width < 20 or height < 20:
            raise ValueError("Board dimensions must both be at least 20.")
        if not 2 <= player_count <= MAX_PLAYERS:
            raise ValueError(f"player_count must be between 2 and {MAX_PLAYERS}.")
        if not 10 <= growth_interval <= 50:
            raise ValueError("growth_interval must be between 10 and 50.")
        if not 0 <= vision_radius <= 5:
            raise ValueError("vision_radius must be between 0 and 5.")
        self.width = width
        self.height = height
        self.player_count = player_count
        self.growth_interval = growth_interval
        self.vision_radius = vision_radius
        self.seed = seed
        self.grid: list[list[Tile]] = []
        self.player_positions: dict[int, tuple[int, int]] = {}
        self.active_players: set[int] = set(range(player_count))
        self.eliminated_players: set[int] = set()
        self.turn = 1
        self.winner: int | None = None
        self.state_revision = 0
        self.terrain_revision = 0
        self.hill_owner_revision: dict[int, int] = {}
        self._vision_offsets_revision = -1
        self._vision_offsets_radius = -1
        self._vision_offsets_cache: dict[
            tuple[int, int],
            tuple[tuple[int, int], ...],
        ] = {}
        self.hill_multiplier = 1.0
        self.pending_moves: list[PendingMove] = []
        self.last_report = TurnReport()
        self.generate(seed)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def tile(self, x: int, y: int) -> Tile:
        return self.grid[y][x]

    def mirrored(self, x: int, y: int) -> tuple[int, int]:
        return self.width - 1 - x, self.height - 1 - y

    @staticmethod
    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_active(self, player: int) -> bool:
        return player in self.active_players

    def general_position(self, owner: int) -> tuple[int, int] | None:
        position = self.player_positions.get(owner)
        if position is None:
            return None
        tile = self.tile(*position)
        if tile.terrain == GENERAL and tile.owner == owner:
            return position
        return None

    def generate(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        rng = random.Random(self.seed)
        spawns = self._spawn_positions(rng)
        hill_multiplier = rng.uniform(1.0, 2.0)

        for _ in range(400):
            grid = [[Tile() for _ in range(self.width)] for _ in range(self.height)]
            safe_cells = {
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if all(self.manhattan((x, y), spawn) > 2 for spawn in spawns)
            }
            area = self.width * self.height
            mountain_count = max(20, int(area * 0.1875))
            city_count = max(7, round(area / 66))

            if self.player_count == 2:
                safe_half = {
                    coord
                    for coord in safe_cells
                    if coord < self.mirrored(*coord)
                    and Board.manhattan(coord, self.mirrored(*coord)) > 1
                }
                mountain_cells = self._mountain_vein_cells(
                    rng,
                    safe_half,
                    mountain_count // 2,
                )
                mountain_cells.update(
                    self.mirrored(x, y)
                    for x, y in tuple(mountain_cells)
                    if self.mirrored(x, y) in safe_cells
                )
                mountain_cells = self._remove_solid_mountain_squares(
                    mountain_cells
                )
                pair_keys: set[tuple[tuple[int, int], tuple[int, int]]] = set()
                for coord in safe_cells:
                    mirror = self.mirrored(*coord)
                    if coord == mirror or mirror not in safe_cells:
                        continue
                    if coord in mountain_cells or mirror in mountain_cells:
                        continue
                    first, second = sorted((coord, mirror))
                    pair_keys.add((first, second))
                pairs = list(pair_keys)
                rng.shuffle(pairs)

                for x, y in mountain_cells:
                    grid[y][x] = Tile(MOUNTAIN, NEUTRAL, 0)
                city_pairs = pairs[:city_count // 2]
                for pair in city_pairs:
                    city_army = rng.randint(20, 36)
                    for x, y in pair:
                        grid[y][x] = Tile(CITY, NEUTRAL, city_army)
            else:
                mountain_cells = self._mountain_vein_cells(
                    rng,
                    safe_cells,
                    mountain_count,
                )
                candidates = list(safe_cells - mountain_cells)
                rng.shuffle(candidates)
                for x, y in mountain_cells:
                    grid[y][x] = Tile(MOUNTAIN, NEUTRAL, 0)
                for x, y in candidates[:city_count]:
                    grid[y][x] = Tile(CITY, NEUTRAL, rng.randint(20, 36))

            self._seal_isolated_passable_regions(grid, spawns)
            self._connect_passable_components(grid, spawns[0])
            self._remove_solid_squares_from_grid(grid)
            self._connect_passable_components(grid, spawns[0])
            self._place_hills(grid, safe_cells, spawns, rng, hill_multiplier)

            for player, (x, y) in enumerate(spawns):
                grid[y][x] = Tile(GENERAL, player, 2)

            if self._all_passable_connected(grid, spawns):
                self.grid = grid
                self.player_positions = dict(enumerate(spawns))
                self.active_players = set(range(self.player_count))
                self.eliminated_players.clear()
                self.turn = 1
                self.winner = None
                self.hill_multiplier = hill_multiplier
                self.pending_moves = []
                self.last_report = TurnReport()
                return

        raise RuntimeError("Could not generate a connected Generals map.")

    def _mountain_vein_cells(
        self,
        rng: random.Random,
        safe_cells: set[tuple[int, int]],
        target_count: int,
    ) -> set[tuple[int, int]]:
        """Place globally distributed organic ridges with open choke gaps."""
        available = set(safe_cells)
        cells: set[tuple[int, int]] = set()
        target_count = max(0, min(target_count, len(available)))
        if target_count <= 0:
            return cells

        area = max(1, self.width * self.height)
        directions = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )
        base_length = max(
            5,
            min(18, round(math.sqrt(area) / 2)),
        )
        ridge_target = max(2, math.ceil(target_count / base_length))
        seed_spacing = max(
            3,
            min(
                12,
                round(
                    math.sqrt(area / ridge_target) * 0.65
                ),
            ),
        )
        max_length = max(6, min(32, round(base_length * 2.5)))
        margins: set[tuple[int, int]] = set()
        starts: list[tuple[int, int]] = []

        def fills_square(
            candidate: tuple[int, int],
            path: set[tuple[int, int]],
        ) -> bool:
            return any(
                all(
                    (
                        square_x + dx,
                        square_y + dy,
                    )
                    in path
                    or (square_x + dx, square_y + dy) == candidate
                    for dy in range(2)
                    for dx in range(2)
                )
                for square_y in (candidate[1] - 1, candidate[1])
                for square_x in (candidate[0] - 1, candidate[0])
            )

        def grow_ridge(
            start: tuple[int, int],
            desired_length: int,
        ) -> set[tuple[int, int]]:
            path = {start}
            x, y = start
            direction = rng.choice(directions)
            for _step in range(1, desired_length):
                options = [direction] * 3
                options.extend(
                    direction_option
                    for direction_option in directions
                    if direction_option != direction
                    and direction_option
                    != (-direction[0], -direction[1])
                )
                rng.shuffle(options)
                for nx, ny in options:
                    candidate = (x + nx, y + ny)
                    if (
                        candidate not in available
                        or candidate in cells
                        or candidate in margins
                        or candidate in path
                        or fills_square(candidate, path)
                    ):
                        continue
                    path.add(candidate)
                    x, y = candidate
                    direction = (nx, ny)
                    break
                else:
                    break
            return path

        rejected: set[tuple[int, int]] = set()

        def next_start() -> tuple[int, int] | None:
            candidates = [
                position
                for position in available
                if position not in cells
                and position not in margins
                and position not in rejected
            ]
            if not candidates:
                return None
            if len(candidates) > 384:
                candidates = rng.sample(candidates, 384)
            choices: list[tuple[float, tuple[int, int]]] = []
            for candidate in candidates:
                distance = (
                    min(
                        Board.manhattan(candidate, accepted)
                        for accepted in starts
                    )
                    if starts
                    else seed_spacing
                )
                choices.append(
                    (
                        distance + rng.random() * seed_spacing * 0.45,
                        candidate,
                    )
                )
            if not choices:
                return None
            choices.sort(key=lambda item: item[0], reverse=True)
            favored = choices[: max(1, min(12, len(choices)))]
            if rng.random() < 0.72:
                return rng.choice(favored)[1]
            return rng.choice(choices[: max(1, len(choices) // 2)])[1]

        generated_ridges = 0
        for _attempt in range(max(8, ridge_target * 3)):
            if len(cells) >= target_count:
                break
            start = next_start()
            if start is None:
                break
            if start in cells or start in margins or start in rejected:
                continue
            remaining = target_count - len(cells)
            remaining_ridges = max(
                1,
                ridge_target - generated_ridges,
            )
            desired_length = (
                math.ceil(remaining / remaining_ridges)
                + rng.randint(-2, 4)
            )
            if rng.random() < 0.18:
                desired_length += rng.randint(6, 16)
            desired_length = min(
                remaining,
                max(1, min(max_length, desired_length)),
            )
            if desired_length <= 0:
                break
            if desired_length == 1:
                path = {start}
            else:
                path = grow_ridge(start, desired_length)
            if not path:
                rejected.add(start)
                continue
            if start not in starts:
                starts.append(start)
            cells.update(path)
            generated_ridges += 1
            for x, y in path:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        neighbor = (x + dx, y + dy)
                        if neighbor not in cells:
                            margins.add(neighbor)
        return self._remove_solid_mountain_squares(cells)

    @staticmethod
    def _remove_solid_squares_from_grid(
        grid: list[list[Tile]],
    ) -> None:
        mountain_cells = {
            (x, y)
            for y, row in enumerate(grid)
            for x, tile in enumerate(row)
            if tile.terrain == MOUNTAIN
        }
        cleaned = Board._remove_solid_mountain_squares(mountain_cells)
        for x, y in mountain_cells - cleaned:
            grid[y][x] = Tile()

    @staticmethod
    def _remove_solid_mountain_squares(
        cells: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        """Remove cells until no 2x2 mountain block remains."""
        cells = set(cells)
        while True:
            squares: list[tuple[tuple[int, int], ...]] = []
            for x, y in cells:
                square = ((x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1))
                if all(cell in cells for cell in square):
                    squares.append(square)
            if not squares:
                return cells

            block_counts: dict[tuple[int, int], int] = {}
            for x, y in cells:
                block = (x // 10, y // 10)
                block_counts[block] = block_counts.get(block, 0) + 1

            square = squares[0]
            candidates: list[
                tuple[tuple[int, int, int, int, int], tuple[int, int]]
            ] = []
            for cell in square:
                x, y = cell
                block = (x // 10, y // 10)
                singleton_penalty = 1 if block_counts.get(block, 0) <= 1 else 0
                broken_squares = sum(
                    all(
                        (square_x + dx, square_y + dy) in cells
                        for dy in range(2)
                        for dx in range(2)
                    )
                    for square_y in (y - 1, y)
                    for square_x in (x - 1, x)
                )
                neighbors = sum(
                    (x + dx, y + dy) in cells
                    for dx, dy in DIRECTIONS
                )
                candidates.append(
                    (
                        (
                            singleton_penalty,
                            -broken_squares,
                            -neighbors,
                            x,
                            y,
                        ),
                        cell,
                    )
                )
            _score, remove_cell = min(candidates, key=lambda item: item[0])
            cells.remove(remove_cell)

    def _place_hills(
        self,
        grid: list[list[Tile]],
        safe_cells: set[tuple[int, int]],
        spawns: list[tuple[int, int]],
        rng: random.Random,
        multiplier: float,
    ) -> None:
        mountains = [
            (x, y)
            for y, row in enumerate(grid)
            for x, tile in enumerate(row)
            if tile.terrain == MOUNTAIN
        ]
        if not mountains:
            return
        target_count = max(1, round(len(mountains) * multiplier))
        empty_cells = {
            (x, y)
            for x, y in safe_cells
            if grid[y][x].terrain == PLAIN and grid[y][x].owner == NEUTRAL
        }
        # Keep hills from becoming the dominant terrain.  The cap is based
        # on the level cells available before hills are placed, so the final
        # number of hills cannot exceed the remaining plains.
        target_count = min(target_count, len(empty_cells) // 2)
        if target_count <= 0:
            return
        near_mountains: set[tuple[int, int]] = set()
        for mx, my in mountains:
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    if dx == 0 and dy == 0:
                        continue
                    coord = (mx + dx, my + dy)
                    if coord in empty_cells:
                        near_mountains.add(coord)
        if self.player_count == 2:
            pair_candidates: dict[tuple[tuple[int, int], tuple[int, int]], bool] = {}
            for coord in empty_cells:
                mirror = self.mirrored(*coord)
                if coord == mirror or mirror not in empty_cells:
                    continue
                first, second = sorted((coord, mirror))
                pair = (first, second)
                pair_candidates[pair] = (
                    first in near_mountains or second in near_mountains
                )

            preferred_pairs = [pair for pair, preferred in pair_candidates.items() if preferred]
            sparse_pairs = [pair for pair, preferred in pair_candidates.items() if not preferred]
            rng.shuffle(preferred_pairs)
            rng.shuffle(sparse_pairs)
            pair_limit = min(
                target_count // 2,
                len(pair_candidates),
                len(empty_cells) // 4,
            )
            for pair in (preferred_pairs + sparse_pairs)[:pair_limit]:
                for x, y in pair:
                    grid[y][x] = Tile(HILL, NEUTRAL, 0)
            return

        preferred = list(near_mountains)
        sparse = list(empty_cells - near_mountains)
        rng.shuffle(preferred)
        rng.shuffle(sparse)
        for x, y in (preferred + sparse)[:target_count]:
            grid[y][x] = Tile(HILL, NEUTRAL, 0)

    def _spawn_positions(self, rng: random.Random) -> list[tuple[int, int]]:
        margin = max(3, min(self.width, self.height) // 18)
        candidates = [
            (x, y)
            for y in range(margin, self.height - margin)
            for x in range(margin, self.width - margin)
        ]
        base_spacing = int(
            min(self.width, self.height)
            / max(3.5, math.sqrt(self.player_count) * 1.8)
        )
        base_spacing = max(3, min(12, base_spacing))

        for spacing in range(base_spacing, 2, -1):
            rng.shuffle(candidates)
            chosen: list[tuple[int, int]] = []
            spacing_squared = spacing * spacing
            for candidate in candidates:
                if all(
                    (candidate[0] - other[0]) ** 2 + (candidate[1] - other[1]) ** 2
                    >= spacing_squared
                    for other in chosen
                ):
                    chosen.append(candidate)
                    if len(chosen) == self.player_count:
                        rng.shuffle(chosen)
                        return chosen

        rng.shuffle(candidates)
        return candidates[:self.player_count]

    def _seal_isolated_passable_regions(
        self,
        grid: list[list[Tile]],
        spawns: list[tuple[int, int]],
    ) -> None:
        spawn_set = set(spawns)
        visited: set[tuple[int, int]] = set()
        for y, row in enumerate(grid):
            for x, tile in enumerate(row):
                start = (x, y)
                if start in visited or tile.terrain == MOUNTAIN:
                    continue
                component: list[tuple[int, int]] = []
                queue = deque([start])
                visited.add(start)
                while queue:
                    cx, cy = queue.popleft()
                    component.append((cx, cy))
                    for dx, dy in DIRECTIONS:
                        neighbor = (cx + dx, cy + dy)
                        if neighbor in visited or not self.in_bounds(*neighbor):
                            continue
                        if grid[neighbor[1]][neighbor[0]].terrain == MOUNTAIN:
                            continue
                        visited.add(neighbor)
                        queue.append(neighbor)
                if any(cell in spawn_set for cell in component):
                    continue
                for cx, cy in component:
                    grid[cy][cx] = Tile(MOUNTAIN, NEUTRAL, 0)

    def _connect_passable_components(
        self,
        grid: list[list[Tile]],
        anchor: tuple[int, int],
    ) -> None:
        """Carve minimal one-cell passages until all passable cells connect."""
        width = self.width
        height = self.height
        directions = DIRECTIONS
        grid_local = grid
        if grid_local[anchor[1]][anchor[0]].terrain == MOUNTAIN:
            return

        def passable_component(
            start: tuple[int, int],
        ) -> set[tuple[int, int]]:
            component = {start}
            queue = deque([start])
            while queue:
                x, y = queue.popleft()
                for dx, dy in directions:
                    neighbor = (x + dx, y + dy)
                    nx, ny = neighbor
                    if (
                        neighbor in component
                        or nx < 0
                        or nx >= width
                        or ny < 0
                        or ny >= height
                    ):
                        continue
                    if grid_local[ny][nx].terrain == MOUNTAIN:
                        continue
                    component.add(neighbor)
                    queue.append(neighbor)
            return component

        connected = passable_component(anchor)
        passable = {
            (x, y)
            for y, row in enumerate(grid_local)
            for x, tile in enumerate(row)
            if tile.terrain != MOUNTAIN
        }
        remaining = passable - connected

        def absorb_reachable(seeds: list[tuple[int, int]]) -> None:
            """Extend the connected component through already passable cells."""
            queue = deque(seeds)
            while queue:
                x, y = queue.popleft()
                for dx, dy in directions:
                    neighbor = (x + dx, y + dy)
                    nx, ny = neighbor
                    if (
                        neighbor not in remaining
                        or nx < 0
                        or nx >= width
                        or ny < 0
                        or ny >= height
                        or grid_local[ny][nx].terrain == MOUNTAIN
                    ):
                        continue
                    remaining.remove(neighbor)
                    connected.add(neighbor)
                    queue.append(neighbor)

        while remaining:
            distances = {position: 0 for position in connected}
            previous: dict[tuple[int, int], tuple[int, int]] = {}
            queue = deque(connected)
            target: tuple[int, int] | None = None

            while queue:
                position = queue.popleft()
                if position in remaining:
                    target = position
                    break
                x, y = position
                for dx, dy in directions:
                    neighbor = (x + dx, y + dy)
                    nx, ny = neighbor
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    mountain_cost = (
                        1
                        if grid_local[ny][nx].terrain == MOUNTAIN
                        else 0
                    )
                    candidate = distances[position] + mountain_cost
                    if candidate >= distances.get(neighbor, 1 << 30):
                        continue
                    distances[neighbor] = candidate
                    previous[neighbor] = position
                    if mountain_cost:
                        queue.append(neighbor)
                    else:
                        queue.appendleft(neighbor)

            if target is None:
                raise RuntimeError("Could not connect map spawn regions.")

            position = target
            carved: list[tuple[int, int]] = []
            while position not in connected:
                x, y = position
                if grid_local[y][x].terrain == MOUNTAIN:
                    grid_local[y][x] = Tile()
                carved.append(position)
                position = previous[position]
            for cell in carved:
                remaining.discard(cell)
                connected.add(cell)
            absorb_reachable(carved)

    def _all_passable_connected(
        self,
        grid: list[list[Tile]],
        spawns: list[tuple[int, int]],
    ) -> bool:
        if not spawns:
            return False
        passable_count = sum(
            tile.terrain != MOUNTAIN
            for row in grid
            for tile in row
        )
        queue = deque([spawns[0]])
        seen = {spawns[0]}
        while queue:
            x, y = queue.popleft()
            for dx, dy in DIRECTIONS:
                nx, ny = x + dx, y + dy
                if not self.in_bounds(nx, ny) or (nx, ny) in seen:
                    continue
                if grid[ny][nx].terrain == MOUNTAIN:
                    continue
                seen.add((nx, ny))
                queue.append((nx, ny))
        return len(seen) == passable_count and all(spawn in seen for spawn in spawns)

    def clone(self) -> "Board":
        clone = Board.__new__(Board)
        clone.width = self.width
        clone.height = self.height
        clone.player_count = self.player_count
        clone.growth_interval = self.growth_interval
        clone.vision_radius = self.vision_radius
        clone.seed = self.seed
        clone.grid = [[tile.clone() for tile in row] for row in self.grid]
        clone.player_positions = dict(self.player_positions)
        clone.active_players = set(self.active_players)
        clone.eliminated_players = set(self.eliminated_players)
        clone.turn = self.turn
        clone.winner = self.winner
        clone.state_revision = self.state_revision
        clone.terrain_revision = self.terrain_revision
        clone.hill_owner_revision = dict(self.hill_owner_revision)
        clone._vision_offsets_revision = self._vision_offsets_revision
        clone._vision_offsets_radius = self._vision_offsets_radius
        clone._vision_offsets_cache = dict(self._vision_offsets_cache)
        clone.hill_multiplier = self.hill_multiplier
        clone.pending_moves = [pending.clone() for pending in self.pending_moves]
        clone.last_report = TurnReport(
            accepted_moves=self.last_report.accepted_moves,
            captures=list(self.last_report.captures or []),
            eliminated=list(self.last_report.eliminated or []),
            income=dict(self.last_report.income or {}),
            land_growth=dict(self.last_report.land_growth or {}),
            delayed_moves=self.last_report.delayed_moves,
            arrived_moves=self.last_report.arrived_moves,
            actions_by_player=dict(self.last_report.actions_by_player or {}),
            attrition_by_pair=dict(self.last_report.attrition_by_pair or {}),
            winner=self.last_report.winner,
        )
        return clone

    def visibility(self, player: int) -> set[tuple[int, int]]:
        return self.visibility_from_positions(
            (x, y)
            for y, row in enumerate(self.grid)
            for x, tile in enumerate(row)
            if tile.owner == player
        )

    def visibility_from_positions(
        self,
        positions: Iterable[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        visible: set[tuple[int, int]] = set()
        for x, y in positions:
            visible.update(self._vision_offsets(x, y))
        return visible

    def _vision_offsets(
        self,
        x: int,
        y: int,
    ) -> tuple[tuple[int, int], ...]:
        if (
            self._vision_offsets_revision != self.terrain_revision
            or self._vision_offsets_radius != self.vision_radius
        ):
            self._vision_offsets_revision = self.terrain_revision
            self._vision_offsets_radius = self.vision_radius
            self._vision_offsets_cache.clear()
        key = (x, y)
        cached = self._vision_offsets_cache.get(key)
        if cached is not None:
            return cached

        radius = self.vision_radius
        grid = self.grid
        offsets = [(x, y)]
        for dx, dy, blockers in _vision_lines(radius):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            if any(
                grid[y + block_y][x + block_x].terrain == HILL
                for block_x, block_y in blockers
            ):
                continue
            offsets.append((nx, ny))
        frozen = tuple(offsets)
        self._vision_offsets_cache[key] = frozen
        return frozen

    def is_visible(self, x: int, y: int, player: int) -> bool:
        tile = self.tile(x, y)
        if tile.owner == player:
            return True
        radius = self.vision_radius
        for ny in range(y - radius, y + radius + 1):
            for nx in range(x - radius, x + radius + 1):
                if (
                    self.in_bounds(nx, ny)
                    and self.tile(nx, ny).owner == player
                    and (x, y) in self.visibility_from_positions([(nx, ny)])
                ):
                    return True
        return False

    def move_legal(self, move: Move, player: int | None = None) -> bool:
        if not self.in_bounds(move.sx, move.sy) or not self.in_bounds(move.tx, move.ty):
            return False
        source = self.tile(move.sx, move.sy)
        target = self.tile(move.tx, move.ty)
        if player is not None and source.owner != player:
            return False
        if source.owner < 0 or target.terrain == MOUNTAIN:
            return False
        if any(
            pending.owner == source.owner and (pending.sx, pending.sy) == move.source
            for pending in self.pending_moves
        ):
            return False
        if abs(move.sx - move.tx) + abs(move.sy - move.ty) != 1:
            return False
        return 1 <= move.amount <= source.army - 1

    def cancel_pending_moves(
        self,
        owner: int,
        source: tuple[int, int] | None = None,
    ) -> int:
        before = len(self.pending_moves)
        self.pending_moves = [
            pending
            for pending in self.pending_moves
            if pending.owner != owner
            or (source is not None and (pending.sx, pending.sy) != source)
        ]
        return before - len(self.pending_moves)

    def movement_turns(self, sx: int, sy: int, tx: int, ty: int, owner: int) -> int:
        source = self.tile(sx, sy)
        target = self.tile(tx, ty)
        if target.terrain == HILL:
            if target.owner == owner:
                return 1
            if source.terrain == HILL and target.owner == NEUTRAL:
                return 1
            return 2
        if source.terrain == HILL:
            return 2
        return 1

    def legal_targets(self, sx: int, sy: int, player: int) -> list[tuple[int, int]]:
        if not self.in_bounds(sx, sy):
            return []
        source = self.tile(sx, sy)
        if source.owner != player or source.army < 2:
            return []
        targets: list[tuple[int, int]] = []
        for dx, dy in DIRECTIONS:
            x, y = sx + dx, sy + dy
            if self.in_bounds(x, y) and self.tile(x, y).terrain != MOUNTAIN:
                targets.append((x, y))
        return targets

    def resolve_turn(
        self,
        moves: Iterable[Move],
        follow_up_moves: Iterable[Move] | None = None,
    ) -> TurnReport:
        report = TurnReport()
        if self.winner is not None:
            return report

        report = self.resolve_movement(moves)
        if follow_up_moves is not None and self.winner is None:
            report.merge(self.resolve_movement(follow_up_moves))
        self.finish_turn(report)
        return report

    def resolve_movement(self, moves: Iterable[Move]) -> TurnReport:
        report = TurnReport()
        if self.winner is not None:
            return report

        arriving: list[tuple[Move, int]] = []
        waiting: list[PendingMove] = []
        for pending in self.pending_moves:
            if pending.ready_turn > self.turn:
                waiting.append(pending)
            else:
                source = self.tile(pending.sx, pending.sy)
                if source.owner != pending.owner or source.army < 2:
                    continue
                amount = min(pending.amount, source.army - 1)
                if amount <= 0:
                    continue
                arriving.append(
                    (
                        Move(pending.sx, pending.sy, pending.tx, pending.ty, amount),
                        pending.owner,
                    )
                )
        self.pending_moves = waiting
        report.arrived_moves = len(arriving)
        arriving_sources = {move.source for move, _ in arriving}

        deduplicated: dict[tuple[int, int], Move] = {}
        for move in moves:
            if move.source not in arriving_sources and self.move_legal(move):
                deduplicated[move.source] = move
        valid_moves = list(deduplicated.values())
        report.accepted_moves = len(valid_moves)

        committed = list(arriving)
        outgoing: dict[tuple[int, int], int] = {
            move.source: move.amount
            for move, _ in arriving
        }
        for move in valid_moves:
            owner = self.tile(move.sx, move.sy).owner
            report.actions_by_player[owner] = report.actions_by_player.get(owner, 0) + 1
            delay = self.movement_turns(move.sx, move.sy, move.tx, move.ty, owner)
            if delay > 1:
                self.pending_moves.append(
                    PendingMove(
                        move.sx,
                        move.sy,
                        move.tx,
                        move.ty,
                        move.amount,
                        owner,
                        remaining=1,
                        ready_turn=self.turn + 1,
                    )
                )
                report.delayed_moves += 1
            else:
                outgoing[move.source] = outgoing.get(move.source, 0) + move.amount
                committed.append((move, owner))

        for (x, y), amount in outgoing.items():
            self.tile(x, y).army -= amount

        incoming: dict[tuple[int, int], dict[int, int]] = {}
        for move, owner in committed:
            target_forces = incoming.setdefault(move.target, {})
            target_forces[owner] = target_forces.get(owner, 0) + move.amount

        generals = {
            position: owner
            for owner, position in self.player_positions.items()
            if owner in self.active_players
        }

        for (x, y), attacking_forces in incoming.items():
            target = self.tile(x, y)
            if target.terrain == MOUNTAIN:
                continue

            if target.owner in attacking_forces:
                friendly = attacking_forces.pop(target.owner)
                target.army += friendly
            if not attacking_forces:
                continue

            forces = sorted(
                attacking_forces.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if len(forces) == 1:
                winner_owner, winner_army = forces[0]
            else:
                winner_owner, winner_amount = forces[0]
                second_owner, second_amount = forces[1]
                winner_army = winner_amount - second_amount
                report.record_attrition(
                    winner_owner,
                    second_owner,
                    winner_amount + second_amount,
                )
                if winner_army <= 0:
                    continue

            defense = target.army
            previous_owner = target.owner
            if winner_army > defense:
                report.record_attrition(
                    winner_owner,
                    previous_owner,
                    defense * 2,
                )
                target.owner = winner_owner
                target.army = winner_army - defense
                if previous_owner != winner_owner:
                    self._record_ownership_change(target, previous_owner)
                    report.captures.append((x, y, winner_owner))
            elif winner_army < defense:
                report.record_attrition(
                    winner_owner,
                    previous_owner,
                    winner_army * 2,
                )
                target.army = defense - winner_army
            else:
                report.record_attrition(
                    winner_owner,
                    previous_owner,
                    winner_army * 2,
                )
                target.army = 0

        for position, original_owner in generals.items():
            if original_owner not in self.active_players:
                continue
            tile = self.tile(*position)
            if tile.owner != original_owner:
                attacker = tile.owner if tile.owner in self.active_players else NEUTRAL
                self._eliminate_player(original_owner, position, attacker)
                report.eliminated.append(original_owner)

        if len(self.active_players) == 1:
            self.winner = next(iter(self.active_players))
            report.winner = self.winner
        if report.captures or report.eliminated:
            self.state_revision += 1

        return report

    def finish_turn(self, report: TurnReport) -> None:
        for row in self.grid:
            for tile in row:
                if (
                    tile.terrain in (CITY, GENERAL)
                    and tile.owner in self.active_players
                ):
                    tile.army += 1
                    report.income[tile.owner] = report.income.get(tile.owner, 0) + 1

        if self.turn % self.growth_interval == 0:
            for row in self.grid:
                for tile in row:
                    if (
                        tile.terrain == PLAIN
                        and tile.owner in self.active_players
                    ):
                        tile.army += 1
                        report.land_growth[tile.owner] = report.land_growth.get(tile.owner, 0) + 1

        hill_interval = self.growth_interval * 2
        if self.turn % hill_interval == 0:
            for row in self.grid:
                for tile in row:
                    if (
                        tile.terrain == HILL
                        and tile.owner in self.active_players
                    ):
                        tile.army += 1
                        report.land_growth[tile.owner] = report.land_growth.get(tile.owner, 0) + 1

        self.last_report = report
        self.turn += 1

    def _eliminate_player(
        self,
        player: int,
        captured_general: tuple[int, int],
        attacker: int,
    ) -> None:
        self.active_players.discard(player)
        self.eliminated_players.add(player)
        self.pending_moves = [pending for pending in self.pending_moves if pending.owner != player]
        captured_army = self.tile(*captured_general).army

        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                if tile.owner != player or (x, y) == captured_general:
                    continue
                previous_owner = tile.owner
                tile.owner = attacker if attacker in self.active_players else NEUTRAL
                tile.army = (tile.army + 1) // 2
                self._record_ownership_change(tile, previous_owner)

        captured_tile = self.tile(*captured_general)
        captured_tile.terrain = CITY
        captured_tile.owner = attacker if attacker in self.active_players else NEUTRAL
        captured_tile.army = captured_army
        self.terrain_revision += 1

    def _record_ownership_change(self, tile: Tile, previous_owner: int) -> None:
        if tile.terrain != HILL:
            return
        for owner in (previous_owner, tile.owner):
            if owner >= 0:
                self.hill_owner_revision[owner] = (
                    self.hill_owner_revision.get(owner, 0) + 1
                )

    def stats(self, owner: int) -> dict[str, int]:
        tiles = 0
        armies = 0
        cities = 0
        generals = 0
        for row in self.grid:
            for tile in row:
                if tile.owner != owner:
                    continue
                tiles += 1
                armies += tile.army
                cities += int(tile.terrain == CITY)
                generals += int(tile.terrain == GENERAL)
        return {"tiles": tiles, "armies": armies, "cities": cities, "generals": generals}


def opponents_of(player: int, player_count: int) -> list[int]:
    return [owner for owner in range(player_count) if owner != player]


def enemy_of(player: int) -> int:
    return AI if player == HUMAN else HUMAN
