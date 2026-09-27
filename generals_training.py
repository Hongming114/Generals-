from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import json
from multiprocessing import shared_memory
import os
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from typing import Any

from generals_ai import AIGenome, GeneralsAI
from generals_core import CITY, Board, TurnReport


TRAINING_PLAYERS = 12
TRAINING_MAX_PLAYERS = 24
TRAINING_MAP_SIZE = 39
TRAINING_MAX_PARALLEL_WORKERS = 24
# 70% FFA is split across small, medium and large maps; 30% runs as 1v1.
TRAINING_FFA_MATCH_TYPES = (
    (4, 25),
    (12, 39),
    (20, 55),
)
TRAINING_DUEL_MATCH_TYPE = (2, 25)
TRAINING_LIVE_MAP_SIZE = max(
    [
        TRAINING_MAP_SIZE,
        TRAINING_DUEL_MATCH_TYPE[1],
        *[map_size for _players, map_size in TRAINING_FFA_MATCH_TYPES],
    ]
)
TRAINING_MAX_TURNS = 900
TRAINING_POPULATION_SIZE = 12
TRAINING_ELITE_COUNT = 3
TRAINING_UPDATE_INTERVAL_MATCHES = 5
TRAINING_CHAMPION_WINDOW_MATCHES = 100
TRAINING_HISTORY_IN_MEMORY = 120
TRAINING_SNAPSHOT_TURNS = 60
LIVE_SLOT_HEADER = 7
TRAINING_FFA_WEIGHT = 0.70
TRAINING_DUEL_WEIGHT = 0.30
TRAINING_ONLINE_WEIGHT = 0.10
TRAINING_DUEL_MAX_TURNS = 300
TRAINING_ONLINE_HISTORY = 40
_LIVE_MEMORY_HANDLES: dict[str, shared_memory.SharedMemory] = {}


def _default_training_data_dir() -> Path:
    configured = os.environ.get("GENERALS_TRAINING_DIR")
    if configured:
        return Path(configured)
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().with_name("training_data")

    executable_dir = Path(sys.executable).resolve().parent
    data_dir = executable_dir / "training_data"
    bundled_dir = Path(getattr(sys, "_MEIPASS", executable_dir)) / "training_data"
    if bundled_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
        for source in bundled_dir.rglob("*"):
            if not source.is_file():
                continue
            destination = data_dir / source.relative_to(bundled_dir)
            if destination.exists():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return data_dir


def _training_episode_task(
    payload: tuple[
        list[dict[str, float]],
        int,
        int,
        int,
        str | None,
        int,
        str,
        list[int],
        int,
        int,
    ],
) -> dict[str, Any]:
    (
        genomes,
        seed,
        map_size,
        max_turns,
        shared_name,
        shared_slot,
        mode,
        genome_indices,
        player_count,
        live_map_size,
    ) = payload
    return TrainingEngine.simulate_episode(
        genomes,
        seed,
        map_size=map_size,
        max_turns=max_turns,
        shared_name=shared_name,
        shared_slot=shared_slot,
        mode=mode,
        genome_indices=genome_indices,
        player_count=player_count,
        live_map_size=live_map_size,
    )


@dataclass(slots=True)
class TrainingMatch:
    board: Board
    ais: dict[int, GeneralsAI]
    genomes: list[AIGenome]
    seed: int
    genome_indices: list[int] = field(default_factory=list)
    mode: str = "ffa"
    source: str = "self_play"
    collect_diagnostics: bool = False
    stats: dict[int, dict[str, float]] = field(default_factory=dict)
    elimination_order: list[int] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    turns_in_window: int = 0
    window_started_at: float = field(default_factory=time.perf_counter)

    @property
    def turn(self) -> int:
        return self.board.turn

    @property
    def winner(self) -> int | None:
        return self.board.winner


class TrainingEngine:
    """Persistent 12-player self-play with evolutionary policy selection."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        *,
        seed: int | None = None,
        population_size: int = TRAINING_POPULATION_SIZE,
        map_size: int = TRAINING_MAP_SIZE,
        max_turns: int = TRAINING_MAX_TURNS,
        parallel_workers: int | None = None,
        live_render: bool = True,
        live_map_size: int | None = None,
    ) -> None:
        if not 2 <= population_size <= TRAINING_PLAYERS:
            raise ValueError(
                f"population_size must be between 2 and {TRAINING_PLAYERS}."
            )
        if map_size < 20:
            raise ValueError("Training map size must be at least 20.")
        self.population_size = population_size
        self.map_size = map_size
        self.live_map_size = max(
            int(map_size),
            int(live_map_size or map_size),
        )
        self.max_turns = max_turns
        detected_workers = max(1, os.cpu_count() or 2)
        self.parallel_workers = max(
            1,
            int(
                detected_workers
                if parallel_workers is None
                else parallel_workers
            ),
        )
        self.live_render = live_render
        self.rng = random.Random(seed)
        self.data_dir = Path(data_dir) if data_dir else _default_training_data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.generation = 0
        self.completed_matches = 0
        self.completed_turns = 0
        self.best_fitness = float("-inf")
        self.best_match: dict[str, Any] | None = None
        self.champion = AIGenome.baseline()
        self.has_recent_champion = False
        self.population: list[AIGenome] = []
        self.history: list[dict[str, Any]] = []
        self.online_history: list[dict[str, Any]] = []
        self.pending_results: list[dict[str, Any]] = []
        self.latest_results: list[dict[str, Any]] = []
        self.current: TrainingMatch | None = None
        self._duel_pair_pool: list[tuple[int, int]] = []
        self.running = False
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._shared_memory: shared_memory.SharedMemory | None = None
        self._live_worker_count = 0
        self.live_slot = 0
        self._live_caches: dict[int, dict[str, Any]] = {}
        self._live_cache_stamp: tuple[int, int, int] | None = None
        self._live_cache: dict[str, Any] | None = None
        self.last_error = ""
        self.turns_per_second = 0.0
        self.matches_per_hour = 0.0
        self._match_duration_total = 0.0
        self._load_state()
        self._ensure_population()

    def _next_duel_pairs(self, count: int) -> list[tuple[int, int]]:
        pairs: list[tuple[int, int]] = []
        if self.population_size < 2:
            return pairs
        for _ in range(count):
            opponent = self.rng.randrange(1, self.population_size)
            pairs.append((0, opponent))
        self._duel_pair_pool.clear()
        return pairs

    @property
    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        self.running = True
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._worker_main,
                name="generals-training",
                daemon=True,
            )
            self._worker.start()

    def _worker_main(self) -> None:
        try:
            self._worker_loop()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.running = False
            self.save_state()

    def pause(self) -> None:
        self.running = False

    def wait_for_stop(self, timeout: float | None = None) -> bool:
        """Wait until the training worker has left its current batch."""
        worker = self._worker
        if (
            worker is not None
            and worker.is_alive()
            and worker is not threading.current_thread()
        ):
            worker.join(timeout)
        return worker is None or not worker.is_alive()

    def toggle(self) -> None:
        if self.running:
            self.pause()
        else:
            self.start()

    def _live_slot_ints(self) -> int:
        return (
            LIVE_SLOT_HEADER
            + self.live_map_size * self.live_map_size * 2
        )

    @staticmethod
    def _write_live_snapshot(
        match: TrainingMatch,
        shared_name: str | None,
        shared_slot: int,
        slot_ints: int,
        *,
        write_terrain: bool,
    ) -> None:
        if shared_name is None:
            return
        memory = _LIVE_MEMORY_HANDLES.get(shared_name)
        if memory is None:
            try:
                memory = shared_memory.SharedMemory(name=shared_name)
            except FileNotFoundError:
                return
            _LIVE_MEMORY_HANDLES[shared_name] = memory
        try:
            values = memory.buf.cast("i")
        except (BufferError, ValueError):
            return
        offset = shared_slot * slot_ints
        sequence = int(values[offset])
        values[offset] = sequence + 1
        values[offset + 1] = match.board.turn
        values[offset + 2] = len(match.board.active_players)
        values[offset + 3] = shared_slot
        values[offset + 4] = (
            time.monotonic_ns() // 1_000_000
        ) & 0x7FFF_FFFF
        values[offset + 5] = match.board.width
        values[offset + 6] = match.board.height
        tile_offset = offset + LIVE_SLOT_HEADER
        if write_terrain:
            for row in match.board.grid:
                for tile in row:
                    values[tile_offset] = tile.terrain
                    values[tile_offset + 1] = tile.owner
                    tile_offset += 2
        else:
            for row in match.board.grid:
                for tile in row:
                    values[tile_offset + 1] = tile.owner
                    tile_offset += 2
        values[offset] = sequence + 2

    def _worker_loop(self) -> None:
        if self.parallel_workers <= 1:
            while self.running:
                with self._lock:
                    if self.current is None:
                        self.current = self._new_match()
                    match = self.current
                    self._step_turn(match)
                    match.turns_in_window += 1
                    now = time.perf_counter()
                    window = now - match.window_started_at
                    if window >= 0.40:
                        self.turns_per_second = (
                            match.turns_in_window / window
                        )
                        match.turns_in_window = 0
                        match.window_started_at = now
                    if self._match_finished(match):
                        self._finish_match(match)
                        self.current = None
                time.sleep(0)
            return

        workers = min(
            self.parallel_workers,
            TRAINING_MAX_PARALLEL_WORKERS,
        )
        self._live_worker_count = workers
        self.live_slot = min(max(0, self.live_slot), workers - 1)
        self._live_caches.clear()
        self.live_map_size = max(
            TRAINING_LIVE_MAP_SIZE,
            self.map_size,
        )
        slot_ints = self._live_slot_ints()
        if self.live_render:
            self._shared_memory = shared_memory.SharedMemory(
                create=True,
                size=slot_ints * workers * 4,
            )
        shared_name = (
            self._shared_memory.name
            if self._shared_memory is not None
            else None
        )
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                while self.running:
                    with self._lock:
                        genomes = [
                            genome.to_dict() for genome in self.population
                        ]
                    payloads = []
                    for _slot in range(workers):
                        seed = self.rng.randrange(1, 1_000_000_000)
                        if self.rng.random() < TRAINING_DUEL_WEIGHT:
                            pair = self._next_duel_pairs(1)[0]
                            payloads.append(
                                (
                                    genomes,
                                    seed,
                                    TRAINING_DUEL_MATCH_TYPE[1],
                                    self.max_turns,
                                    shared_name,
                                    len(payloads),
                                    "duel",
                                    [pair[0], pair[1]],
                                    TRAINING_DUEL_MATCH_TYPE[0],
                                    self.live_map_size,
                                )
                            )
                            continue
                        indices = list(range(len(genomes)))
                        self.rng.shuffle(indices)
                        ffa_players, ffa_map_size = self.rng.choice(
                            TRAINING_FFA_MATCH_TYPES
                        )
                        payloads.append(
                            (
                                genomes,
                                seed,
                                ffa_map_size,
                                self.max_turns,
                                shared_name,
                                len(payloads),
                                "ffa",
                                indices,
                                ffa_players,
                                self.live_map_size,
                            )
                        )
                    batch_started = time.perf_counter()
                    items = list(executor.map(_training_episode_task, payloads))
                    if not self.running:
                        break
                    elapsed = max(
                        0.001,
                        time.perf_counter() - batch_started,
                    )
                    with self._lock:
                        for item in items:
                            item["duration_seconds"] = round(
                                float(item.get("duration_seconds", 0.0))
                                + elapsed / max(1, len(items)),
                                4,
                            )
                        self._commit_batch(
                            items,
                            batch_elapsed=elapsed,
                        )
        finally:
            if self._shared_memory is not None:
                self._shared_memory.close()
                try:
                    self._shared_memory.unlink()
                except FileNotFoundError:
                    pass
                self._shared_memory = None
            self._live_worker_count = 0

    def _load_json(self, name: str) -> dict[str, Any] | None:
        path = self.data_dir / name
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _load_state(self) -> None:
        manifest = self._load_json("manifest.json")
        raw_best_match = (
            manifest.get("best_match")
            if isinstance(manifest, dict)
            else None
        )
        hybrid_metric = bool(
            manifest
            and (
                manifest.get("training_mode") == "hybrid_fa_duel"
                or (
                    "ffa_weight" in manifest
                    and "duel_weight" in manifest
                )
                or (
                    isinstance(raw_best_match, dict)
                    and "results" in raw_best_match
                )
            )
        )
        if manifest:
            self.generation = max(0, int(manifest.get("generation", 0)))
            self.completed_matches = max(
                0,
                int(manifest.get("completed_matches", 0)),
            )
            raw_pending = manifest.get("pending_results")
            if isinstance(raw_pending, list):
                self.pending_results = [
                    item for item in raw_pending if isinstance(item, dict)
                ]
            raw_fitness = manifest.get("best_fitness")
            if isinstance(raw_fitness, (int, float)):
                self.best_fitness = float(raw_fitness)
            raw_match = manifest.get("best_match")
            if isinstance(raw_match, dict):
                self.best_match = raw_match
            raw_speed = manifest.get("matches_per_hour")
            if isinstance(raw_speed, (int, float)):
                self.matches_per_hour = float(raw_speed)
            if not hybrid_metric:
                self.best_fitness = float("-inf")
                self.best_match = None

        champion_data = self._load_json("champion.json")
        if champion_data:
            raw_genome = champion_data.get("genome")
            if isinstance(raw_genome, dict):
                self.champion = AIGenome.from_mapping(raw_genome)
            raw_fitness = champion_data.get("fitness")
            if hybrid_metric and isinstance(raw_fitness, (int, float)):
                self.best_fitness = max(
                    self.best_fitness,
                    float(raw_fitness),
                )

        population_data = self._load_json("population.json")
        if population_data:
            raw_population = population_data.get("genomes")
            if isinstance(raw_population, list):
                self.population = [
                    AIGenome.from_mapping(genome)
                    for genome in raw_population[: self.population_size]
                    if isinstance(genome, dict)
                ]

        history_path = self.data_dir / "history.jsonl"
        if history_path.exists():
            try:
                lines = history_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            except OSError:
                lines = []
            for line in lines[-TRAINING_HISTORY_IN_MEMORY:]:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    self.history.append(item)

        online_path = self.data_dir / "online_history.jsonl"
        if online_path.exists():
            try:
                online_lines = online_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                ).splitlines()
            except OSError:
                online_lines = []
            for line in online_lines[-TRAINING_ONLINE_HISTORY:]:
                try:
                    item = json.loads(line)
                except ValueError:
                    continue
                if isinstance(item, dict):
                    self.online_history.append(item)
        self._refresh_recent_champion()

    @staticmethod
    def _result_fitness(result: dict[str, Any]) -> float:
        try:
            return float(result.get("fitness", float("-inf")))
        except (TypeError, ValueError):
            return float("-inf")

    def _refresh_recent_champion(self) -> bool:
        window = self.history[-TRAINING_CHAMPION_WINDOW_MATCHES:]
        best_fitness = float("-inf")
        best_item: dict[str, Any] | None = None
        best_result: dict[str, Any] | None = None
        for item in reversed(window):
            results = item.get("results")
            if not isinstance(results, list):
                continue
            for result in results:
                if not isinstance(result, dict):
                    continue
                fitness = self._result_fitness(result)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_item = item
                    best_result = result
        if best_result is None or best_item is None:
            return False
        genome = best_result.get("genome")
        if not isinstance(genome, dict):
            return False
        self.champion = AIGenome.from_mapping(genome)
        self.best_fitness = best_fitness
        self.best_match = {
            "match": best_item.get("match"),
            "generation": best_item.get("generation"),
            "mode": best_item.get("mode"),
            "fitness": best_fitness,
            "genome": dict(genome),
            "results": [dict(best_result)],
            "window_matches": min(
                TRAINING_CHAMPION_WINDOW_MATCHES,
                len(window),
            ),
        }
        self.has_recent_champion = True
        return True

    def _ensure_population(self) -> None:
        if self.population:
            while len(self.population) < self.population_size:
                self.population.append(
                    AIGenome.random(
                        self.rng,
                        center=self.champion.to_dict(),
                        scale=0.13,
                    )
                )
            self.population = self.population[: self.population_size]
            if self.has_recent_champion:
                self.population[0] = self.champion
            return
        self.population.append(self.champion)
        while len(self.population) < self.population_size:
            scale = 0.08 if len(self.population) == 1 else 0.16
            self.population.append(
                AIGenome.random(
                    self.rng,
                    center=self.champion.to_dict(),
                    scale=scale,
                )
            )
        if self.has_recent_champion:
            self.population[0] = self.champion

    def _atomic_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.data_dir / name
        temp = path.with_suffix(path.suffix + ".tmp")
        text = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        temp.write_text(text, encoding="utf-8")
        temp.replace(path)

    def save_state(self) -> None:
        try:
            self._atomic_json(
                "manifest.json",
                {
                    "version": 1,
                    "generation": self.generation,
                    "completed_matches": self.completed_matches,
                    "best_fitness": (
                        self.best_fitness
                        if self.best_fitness != float("-inf")
                        else None
                    ),
                    "best_match": self.best_match,
                    "pending_results": list(self.pending_results),
                    "matches_per_hour": round(self.matches_per_hour, 3),
                    "updated_at": time.time(),
                    "players_per_match": TRAINING_PLAYERS,
                    "training_mode": "hybrid_fa_duel",
                    "ffa_weight": TRAINING_FFA_WEIGHT,
                    "duel_weight": TRAINING_DUEL_WEIGHT,
                    "online_weight": TRAINING_ONLINE_WEIGHT,
                    "online_matches": len(self.online_history),
                    "champion_window_matches": (
                        TRAINING_CHAMPION_WINDOW_MATCHES
                    ),
                    "update_interval_matches": (
                        TRAINING_UPDATE_INTERVAL_MATCHES
                    ),
                    "map_size": self.map_size,
                    "max_turns": self.max_turns,
                },
            )
            self._atomic_json(
                "champion.json",
                {
                    "version": 1,
                    "generation": self.generation,
                    "completed_matches": self.completed_matches,
                    "fitness": (
                        self.best_fitness
                        if self.best_fitness != float("-inf")
                        else None
                    ),
                    "genome": self.champion.to_dict(),
                    "best_match": self.best_match,
                    "window_matches": (
                        TRAINING_CHAMPION_WINDOW_MATCHES
                    ),
                    "saved_at": time.time(),
                },
            )
            self._atomic_json(
                "population.json",
                {
                    "version": 1,
                    "generation": self.generation,
                    "genomes": [
                        genome.to_dict() for genome in self.population
                    ],
                },
            )
            self.last_error = ""
        except OSError as exc:
            self.last_error = str(exc)

    def _append_history(self, item: dict[str, Any]) -> None:
        self.history.append(item)
        self.history = self.history[-TRAINING_HISTORY_IN_MEMORY:]
        try:
            with (self.data_dir / "history.jsonl").open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
        except OSError as exc:
            self.last_error = str(exc)

    def record_live_match(
        self,
        results: list[dict[str, Any]],
        *,
        turn: int = 0,
        player_count: int | None = None,
        seed: int | None = None,
    ) -> dict[str, Any] | None:
        if not results:
            return None
        count = max(
            int(player_count or len(results)),
            2,
        )
        normalized_results: list[dict[str, Any]] = []
        for raw in results:
            genome = AIGenome.from_mapping(raw.get("genome"))
            rank = max(1, int(raw.get("rank", count)))
            tiles = max(0, int(raw.get("tiles", 0)))
            armies = max(0, int(raw.get("armies", 0)))
            winner = bool(raw.get("winner", rank == 1))
            fitness = (
                (120_000.0 if winner else 0.0)
                + (count - rank) ** 2 * 520.0
                + tiles * 38.0
                + armies * 4.0
            )
            normalized_results.append(
                {
                    "player": int(raw.get("player", -1)),
                    "rank": rank,
                    "fitness": round(fitness, 3),
                    "genome": genome.to_dict(),
                    "winner": winner,
                    "tiles": tiles,
                    "armies": armies,
                }
            )
        normalized_results.sort(
            key=lambda item: (
                -float(item["fitness"]),
                int(item["rank"]),
            )
        )
        item = {
            "match": "LIVE",
            "generation": self.generation,
            "mode": "online",
            "source": "live_game",
            "match_count": 1,
            "seed": seed,
            "turn": max(0, int(turn)),
            "simulated_turns": max(0, int(turn)),
            "winner": next(
                (
                    int(result["player"])
                    for result in normalized_results
                    if result.get("winner")
                ),
                None,
            ),
            "best_fitness": normalized_results[0]["fitness"],
            "best_player": normalized_results[0]["player"],
            "results": normalized_results,
            "saved_at": time.time(),
        }
        with self._lock:
            self.online_history.append(item)
            self.online_history = self.online_history[
                -TRAINING_ONLINE_HISTORY:
            ]
            self._append_history(item)
            try:
                with (self.data_dir / "online_history.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            except OSError as exc:
                self.last_error = str(exc)
        return item

    def _online_evaluation_items(self) -> list[dict[str, Any]]:
        genome_lookup: dict[AIGenome, int] = {}
        for index, genome in enumerate(self.population):
            genome_lookup.setdefault(genome, index)
        mapped_items: list[dict[str, Any]] = []
        with self._lock:
            history = list(self.online_history)
        for item in history:
            mapped_results: list[dict[str, Any]] = []
            for result in item.get("results", []):
                genome = AIGenome.from_mapping(result.get("genome"))
                genome_index = genome_lookup.get(genome)
                if genome_index is None:
                    continue
                mapped = dict(result)
                mapped["player"] = genome_index
                mapped["genome_index"] = genome_index
                mapped["mode"] = "online"
                mapped_results.append(mapped)
            if mapped_results:
                mapped_item = dict(item)
                mapped_item["results"] = mapped_results
                mapped_items.append(mapped_item)
        return mapped_items

    def _new_match(
        self,
        seed: int | None = None,
        *,
        shuffle_genomes: bool = True,
        player_count: int = TRAINING_PLAYERS,
        population_indices: list[int] | None = None,
        mode: str = "ffa",
        source: str = "self_play",
        collect_diagnostics: bool = False,
    ) -> TrainingMatch:
        if not 2 <= player_count <= TRAINING_MAX_PLAYERS:
            raise ValueError(
                f"player_count must be between 2 and {TRAINING_MAX_PLAYERS}."
            )
        match_seed = (
            int(seed)
            if seed is not None
            else self.rng.randrange(1, 1_000_000_000)
        )
        if population_indices is None:
            population_indices = list(range(self.population_size))
        else:
            population_indices = [
                int(index) % self.population_size
                for index in population_indices
            ]
        if not population_indices:
            population_indices = [0]
        while len(population_indices) < player_count:
            population_indices.append(
                population_indices[
                    len(population_indices) % len(population_indices)
                ]
            )
        if shuffle_genomes:
            self.rng.shuffle(population_indices)
        population_indices = population_indices[:player_count]
        genomes = [
            self.population[index]
            for index in population_indices
        ]
        board = Board(
            self.map_size,
            self.map_size,
            player_count,
            match_seed,
            growth_interval=20,
            vision_radius=2,
        )
        ais = {
            player: GeneralsAI(
                player,
                "normal",
                match_seed * 10_000 + player * 7919,
                genome=genomes[player],
            )
            for player in range(player_count)
        }
        stats = {
            player: {
                "peak_tiles": 0.0,
                "peak_army": 0.0,
                "attrition_for": 0.0,
                "attrition_against": 0.0,
                "captures": 0.0,
                "castles_captured": 0.0,
                "actions": 0.0,
                "survival_turns": 0.0,
                "idle_turns": 0.0,
                "battle_turns": 0.0,
                "major_battle_turns": 0.0,
                "home_guard_active_turns": 0.0,
                "home_guard_deficit_turns": 0.0,
                "development_turns": 0.0,
                "border_muster_turns": 0.0,
                "all_in_turns": 0.0,
                "capital_depth_short_turns": 0.0,
                "max_capital_depth_shortfall": 0.0,
                "first_contact_turn": -1.0,
                "first_attrition_turn": -1.0,
                "armies_created": 0.0,
                "armies_disbanded": 0.0,
                "peak_army_units": 0.0,
                "army_attack_turns": 0.0,
                "army_defend_turns": 0.0,
                "army_resupply_turns": 0.0,
                "army_displacement_rewards": 0.0,
                "strategy_state_turns": {},
                "mission_turns": {},
                "gather_mode_turns": {},
            }
            for player in range(player_count)
        }
        return TrainingMatch(
            board=board,
            ais=ais,
            genomes=genomes,
            seed=match_seed,
            genome_indices=population_indices,
            mode=mode,
            source=source,
            collect_diagnostics=collect_diagnostics,
            stats=stats,
        )

    def _step_turn(self, match: TrainingMatch) -> TurnReport:
        turn_number = match.board.turn
        active_before = set(match.board.active_players)
        movable_before: dict[int, bool] = {}
        if match.collect_diagnostics:
            movable_before = {
                player: any(
                    tile.owner == player and tile.army > 1
                    for row in match.board.grid
                    for tile in row
                )
                for player in active_before
            }

        total = TurnReport()
        for player, ai in match.ais.items():
            if player not in match.board.active_players:
                continue
            controller = ai.army_controller
            displacement_before = {
                army.unit_id: army.displacement_reward
                for army in controller.armies
            }
            controller.begin_turn(match.board, ai)
            before_units = {army.unit_id for army in controller.armies}
            values = match.stats.get(player)
            if values is not None:
                values["armies_disbanded"] += sum(
                    1
                    for army in controller.armies
                    if army.force_disband
                )
            for army in controller.armies:
                army.force_disband = False
            controller.review_ai_armies(match.board, ai)
            controller.maybe_create_auto(match.board, ai)
            if values is not None:
                created_ids = {
                    army.unit_id for army in controller.armies
                } - before_units
                values["armies_created"] += float(len(created_ids))
                values["peak_army_units"] = max(
                    values["peak_army_units"],
                    float(controller.count),
                )
                values["army_displacement_rewards"] += float(
                    sum(
                        army.displacement_reward
                        - displacement_before.get(army.unit_id, 0)
                        for army in controller.armies
                    )
                )
                for army in controller.armies:
                    if army.state == "attack":
                        values["army_attack_turns"] += 1.0
                    elif army.state == "defend":
                        values["army_defend_turns"] += 1.0
                    else:
                        values["army_resupply_turns"] += 1.0
        for phase in range(2):
            moves = []
            for player, ai in match.ais.items():
                if player in match.board.active_players:
                    moves.extend(ai.plan_moves(match.board, phase=phase))
            total.merge(match.board.resolve_movement(moves))
            if match.board.winner is not None:
                break
        match.board.finish_turn(total)

        for pair, amount in (total.attrition_by_pair or {}).items():
            if len(pair) != 2 or amount <= 0:
                continue
            first, second = pair
            if first not in match.stats or second not in match.stats:
                continue
            split = amount * 0.5
            match.stats[first]["attrition_for"] += split
            match.stats[first]["attrition_against"] += split
            match.stats[second]["attrition_for"] += split
            match.stats[second]["attrition_against"] += split
        for _x, _y, owner in total.captures or []:
            if owner in match.stats:
                match.stats[owner]["captures"] += 1.0
                if match.board.tile(_x, _y).terrain == CITY:
                    match.stats[owner]["castles_captured"] += 1.0
        for player, actions in (total.actions_by_player or {}).items():
            if player in match.stats:
                match.stats[player]["actions"] += actions
        match.elimination_order.extend(
            player
            for player in (total.eliminated or [])
            if player not in match.elimination_order
        )
        for player in match.board.active_players:
            own = match.board.stats(player)
            values = match.stats[player]
            values["peak_tiles"] = max(values["peak_tiles"], own["tiles"])
            values["peak_army"] = max(values["peak_army"], own["armies"])
            values["survival_turns"] += 1.0
        self._record_turn_diagnostics(
            match,
            total,
            turn_number,
            active_before,
            movable_before,
        )
        return total

    def _record_turn_diagnostics(
        self,
        match: TrainingMatch,
        report: TurnReport,
        turn_number: int,
        active_before: set[int],
        movable_before: dict[int, bool],
    ) -> None:
        board = match.board
        full_diagnostics = match.collect_diagnostics
        for player in active_before & board.active_players:
            ai = match.ais[player]
            values = match.stats[player]
            if values["first_contact_turn"] < 0:
                visible = board.visibility(player)
                if any(
                    board.tile(*position).owner >= 0
                    and board.tile(*position).owner != player
                    for position in visible
                ):
                    values["first_contact_turn"] = float(turn_number)
            if ai.home_guard_target_army > 0:
                values["home_guard_active_turns"] += 1.0
            if ai.home_guard_hard_deficit > 0:
                values["home_guard_deficit_turns"] += 1.0
            if full_diagnostics:
                state_turns = values["strategy_state_turns"]
                state_turns[ai.strategy_state] = (
                    int(state_turns.get(ai.strategy_state, 0)) + 1
                )
                if ai.mission is not None:
                    mission_turns = values["mission_turns"]
                    mission_turns[ai.mission.kind] = (
                        int(mission_turns.get(ai.mission.kind, 0)) + 1
                    )
                if ai.gather_mode:
                    gather_turns = values["gather_mode_turns"]
                    gather_turns[ai.gather_mode] = (
                        int(gather_turns.get(ai.gather_mode, 0)) + 1
                    )
                if ai.development_hold_turns > 0:
                    values["development_turns"] += 1.0
                if ai.border_muster_hold_turns > 0:
                    values["border_muster_turns"] += 1.0
                if ai.all_in_hold_turns > 0:
                    values["all_in_turns"] += 1.0
                depth_shortfall = max(
                    0,
                    ai._capital_depth_requirement(turn_number)
                    - ai.capital_depth,
                )
                if depth_shortfall > 0:
                    values["capital_depth_short_turns"] += 1.0
                    values["max_capital_depth_shortfall"] = max(
                        values["max_capital_depth_shortfall"],
                        float(depth_shortfall),
                    )
                if (
                    movable_before.get(player, False)
                    and int((report.actions_by_player or {}).get(player, 0)) <= 0
                ):
                    values["idle_turns"] += 1.0

        battle_players: set[int] = set()
        major_players: set[int] = set()
        for pair, raw_amount in (report.attrition_by_pair or {}).items():
            if len(pair) != 2 or raw_amount <= 0:
                continue
            amount = float(raw_amount)
            for player in pair:
                if player not in active_before:
                    continue
                battle_players.add(player)
                current_army = board.stats(player)["armies"]
                major_threshold = max(10.0, current_army * 0.10)
                if amount >= major_threshold:
                    major_players.add(player)
        for player in battle_players:
            values = match.stats[player]
            values["battle_turns"] += 1.0
            if values["first_attrition_turn"] < 0:
                values["first_attrition_turn"] = float(turn_number)
        for player in major_players:
            match.stats[player]["major_battle_turns"] += 1.0

    def _match_finished(self, match: TrainingMatch) -> bool:
        return (
            match.board.winner is not None
            or len(match.board.active_players) <= 1
            or match.turn > self.max_turns
        )

    def _player_fitness(
        self,
        match: TrainingMatch,
        player: int,
        rank: int,
    ) -> float:
        values = match.stats[player]
        final = match.board.stats(player)
        player_count = match.board.player_count
        rank_points = (player_count - rank) ** 2 * 520.0
        win_bonus = 120_000.0 if match.winner == player else 0.0
        combat = (
            values["attrition_for"] * 1.55
            - values["attrition_against"] * 0.58
        )
        efficiency = (
            values["attrition_for"]
            / max(1.0, values["attrition_against"])
        )
        army_action_turns = (
            values["army_attack_turns"]
            + values["army_defend_turns"]
        )
        army_effectiveness = min(
            1.0,
            army_action_turns / max(1.0, values["survival_turns"]),
        )
        army_value = min(
            45_000.0,
            values["armies_created"] * 260.0
            + values["peak_army_units"] * 180.0
            + army_action_turns * 25.0
            + values["army_displacement_rewards"] * 400.0,
        )
        defense_value = (
            min(12_000.0, values["home_guard_active_turns"] * 6.0)
            - values["home_guard_deficit_turns"] * 35.0
        )
        battle_value = (
            values["major_battle_turns"] * 180.0
            + values["battle_turns"] * 12.0
        )
        return (
            win_bonus
            + rank_points
            + values["peak_tiles"] * 72.0
            + final["tiles"] * 38.0
            + values["peak_army"] * 8.0
            + final["armies"] * 4.0
            + values["captures"] * 180.0
            + values["castles_captured"] * 2_400.0
            + values["actions"] * 1.8
            + values["survival_turns"] * 12.0
            + combat
            + min(30_000.0, efficiency * 1_500.0)
            + army_effectiveness * 6_000.0
            + army_value
            + defense_value
            + battle_value
        )

    def _ranking(self, match: TrainingMatch) -> list[int]:
        if match.winner is not None:
            survivors = [
                match.winner,
                *sorted(
                    match.board.active_players - {match.winner},
                    key=lambda player: (
                        match.board.stats(player)["armies"],
                        match.board.stats(player)["tiles"],
                    ),
                    reverse=True,
                ),
            ]
        else:
            survivors = sorted(
                match.board.active_players,
                key=lambda player: (
                    match.board.stats(player)["armies"],
                    match.board.stats(player)["tiles"],
                ),
                reverse=True,
            )
        eliminated_worst_first = [
            player
            for player in reversed(match.elimination_order)
            if player not in survivors
        ]
        remaining = [
            player
            for player in range(match.board.player_count)
            if player not in survivors and player not in eliminated_worst_first
        ]
        return survivors + eliminated_worst_first + remaining

    def _build_match_result(
        self,
        match: TrainingMatch,
        *,
        match_number: int,
        generation: int,
    ) -> dict[str, Any]:
        ranking = self._ranking(match)
        results: list[dict[str, Any]] = []
        for rank, player in enumerate(ranking):
            fitness = self._player_fitness(match, player, rank)
            values = match.stats[player]
            result = {
                "player": player,
                "genome_index": match.genome_indices[player],
                "rank": rank + 1,
                "fitness": round(fitness, 3),
                "genome": match.genomes[player].to_dict(),
                "mode": match.mode,
                "winner": match.winner == player,
                "tiles": match.board.stats(player)["tiles"],
                "armies": match.board.stats(player)["armies"],
                "peak_tiles": int(values["peak_tiles"]),
                "peak_army": int(values["peak_army"]),
                "actions": int(values["actions"]),
                "castles_captured": int(values["castles_captured"]),
                "survival_turns": int(values["survival_turns"]),
                "attrition_for": round(values["attrition_for"], 2),
                "attrition_against": round(
                    values["attrition_against"],
                    2,
                ),
                "idle_turns": int(values["idle_turns"]),
                "battle_turns": int(values["battle_turns"]),
                "major_battle_turns": int(
                    values["major_battle_turns"]
                ),
                "home_guard_active_turns": int(
                    values["home_guard_active_turns"]
                ),
                "home_guard_deficit_turns": int(
                    values["home_guard_deficit_turns"]
                ),
                "development_turns": int(values["development_turns"]),
                "border_muster_turns": int(values["border_muster_turns"]),
                "all_in_turns": int(values["all_in_turns"]),
                "capital_depth_short_turns": int(
                    values["capital_depth_short_turns"]
                ),
                "max_capital_depth_shortfall": int(
                    values["max_capital_depth_shortfall"]
                ),
                "first_contact_turn": (
                    int(values["first_contact_turn"])
                    if values["first_contact_turn"] >= 0
                    else None
                ),
                "first_attrition_turn": (
                    int(values["first_attrition_turn"])
                    if values["first_attrition_turn"] >= 0
                    else None
                ),
                "armies_created": int(values["armies_created"]),
                "armies_disbanded": int(values["armies_disbanded"]),
                "peak_army_units": int(values["peak_army_units"]),
                "army_attack_turns": int(values["army_attack_turns"]),
                "army_defend_turns": int(values["army_defend_turns"]),
                "army_resupply_turns": int(values["army_resupply_turns"]),
                "army_displacement_rewards": int(
                    values["army_displacement_rewards"]
                ),
                "strategy_state_turns": dict(
                    values["strategy_state_turns"]
                ),
                "mission_turns": dict(values["mission_turns"]),
                "gather_mode_turns": dict(values["gather_mode_turns"]),
            }
            results.append(result)
        results.sort(key=lambda item: (-float(item["fitness"]), item["player"]))
        best = results[0]
        duration = max(0.001, time.perf_counter() - match.started_at)
        return {
            "match": match_number,
            "generation": generation,
            "mode": match.mode,
            "source": match.source,
            "match_count": 1,
            "seed": match.seed,
            "player_count": match.board.player_count,
            "turn": match.turn - 1,
            "simulated_turns": max(0, match.turn - 1),
            "winner": match.winner,
            "winner_genome_index": (
                match.genome_indices[match.winner]
                if match.winner is not None
                else None
            ),
            "duration_seconds": round(duration, 4),
            "best_fitness": best["fitness"],
            "best_player": best["player"],
            "results": results,
            "saved_at": time.time(),
        }

    def _aggregate_results(
        self,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        totals: dict[int, dict[str, Any]] = {}
        for item in items:
            mode = str(item.get("mode", "ffa"))
            for result in item["results"]:
                genome_index = int(
                    result.get("genome_index", result["player"])
                )
                entry = totals.setdefault(
                    genome_index,
                    {
                        "genome": result["genome"],
                        "modes": {},
                        "wins": 0.0,
                    },
                )
                mode_values = entry["modes"].setdefault(
                    mode,
                    {
                        "count": 0.0,
                        "fitness": 0.0,
                        "rank": 0.0,
                        "tiles": 0.0,
                        "armies": 0.0,
                    },
                )
                mode_values["count"] += 1.0
                mode_values["fitness"] += float(result["fitness"])
                mode_values["rank"] += float(result["rank"])
                mode_values["tiles"] += float(result["tiles"])
                mode_values["armies"] += float(result["armies"])
                entry["wins"] += 1.0 if result.get("winner") else 0.0

        aggregated: list[dict[str, Any]] = []
        for genome_index, entry in totals.items():
            mode_values = entry["modes"]
            weighted_fitness = 0.0
            weight_total = 0.0
            for mode, weight in (
                ("ffa", TRAINING_FFA_WEIGHT),
                ("duel", TRAINING_DUEL_WEIGHT),
                ("online", TRAINING_ONLINE_WEIGHT),
            ):
                values = mode_values.get(mode)
                if not values or float(values["count"]) <= 0.0:
                    continue
                weighted_fitness += (
                    weight * float(values["fitness"]) / float(values["count"])
                )
                weight_total += weight
            if weight_total <= 0.0:
                continue
            fitness = weighted_fitness / weight_total
            total_count = sum(
                float(values["count"])
                for values in mode_values.values()
            )
            rank = sum(
                float(values["rank"])
                for values in mode_values.values()
            ) / max(1.0, total_count)
            tiles = sum(
                float(values["tiles"])
                for values in mode_values.values()
            ) / max(1.0, total_count)
            armies = sum(
                float(values["armies"])
                for values in mode_values.values()
            ) / max(1.0, total_count)
            aggregated.append(
                {
                    "player": genome_index,
                    "genome_index": genome_index,
                    "rank": round(rank, 3),
                    "fitness": round(fitness, 3),
                    "genome": entry["genome"],
                    "winner": entry["wins"] > total_count / 2.0,
                    "wins": int(entry["wins"]),
                    "tiles": int(round(tiles)),
                    "armies": int(round(armies)),
                    "mode_fitness": {
                        mode: round(
                            float(values["fitness"])
                            / max(1.0, float(values["count"])),
                            3,
                        )
                        for mode, values in mode_values.items()
                    },
                    "samples": {
                        mode: int(values["count"])
                        for mode, values in mode_values.items()
                    },
                }
            )
        return aggregated

    @staticmethod
    def _match_count(item: dict[str, Any]) -> int:
        return max(1, int(item.get("match_count", 1)))

    def _pending_match_count(self) -> int:
        return sum(self._match_count(item) for item in self.pending_results)

    def _take_pending_chunk(self) -> list[dict[str, Any]]:
        chunk: list[dict[str, Any]] = []
        matches = 0
        while (
            self.pending_results
            and matches < TRAINING_UPDATE_INTERVAL_MATCHES
        ):
            item = self.pending_results.pop(0)
            chunk.append(item)
            matches += self._match_count(item)
        return chunk

    def _apply_training_update(
        self,
        items: list[dict[str, Any]],
    ) -> None:
        aggregated = self._aggregate_results(
            items + self._online_evaluation_items()
        )
        aggregated.sort(
            key=lambda item: (-float(item["fitness"]), item["player"])
        )
        self.latest_results = aggregated
        self._refresh_recent_champion()
        self._evolve(aggregated)
        self.generation += 1
        self.save_state()

    def _commit_batch(
        self,
        items: list[dict[str, Any]],
        *,
        batch_elapsed: float | None = None,
    ) -> dict[str, Any]:
        if not items:
            raise ValueError("Training batch cannot be empty.")
        simulated_turns = sum(
            max(0, int(item.get("simulated_turns", 0)))
            for item in items
        )
        self.completed_turns += simulated_turns
        for item in items:
            match_count = max(1, int(item.get("match_count", 1)))
            self.completed_matches += match_count
            item["match"] = self.completed_matches
            item["generation"] = self.generation
            self._append_history(item)
            duration = float(item.get("duration_seconds", 0.0))
            self._match_duration_total += duration
        if batch_elapsed is not None and batch_elapsed > 0.0:
            self.turns_per_second = simulated_turns / batch_elapsed
            matches = sum(
                max(1, int(item.get("match_count", 1)))
                for item in items
            )
            recent_rate = matches * 3_600.0 / batch_elapsed
            if self.matches_per_hour <= 0:
                self.matches_per_hour = recent_rate
            else:
                self.matches_per_hour = (
                    self.matches_per_hour * 0.85 + recent_rate * 0.15
                )
        self.pending_results.extend(items)
        updated = False
        while (
            self._pending_match_count()
            >= TRAINING_UPDATE_INTERVAL_MATCHES
        ):
            self._apply_training_update(self._take_pending_chunk())
            updated = True
        if not updated:
            self.save_state()
        return items[-1]

    def _finish_match(self, match: TrainingMatch) -> dict[str, Any]:
        item = self._build_match_result(
            match,
            match_number=self.completed_matches + 1,
            generation=self.generation,
        )
        return self._commit_batch([item])

    @classmethod
    def _simulate_match(
        cls,
        simulation: "TrainingEngine",
        match: TrainingMatch,
        *,
        shared_name: str | None,
        shared_slot: int,
    ) -> dict[str, Any]:
        slot_ints = simulation._live_slot_ints()
        if shared_name is not None:
            simulation._write_live_snapshot(
                match,
                shared_name,
                shared_slot,
                slot_ints,
                write_terrain=True,
            )
        while not simulation._match_finished(match):
            simulation._step_turn(match)
            if (
                shared_name is not None
                and match.board.turn % TRAINING_SNAPSHOT_TURNS == 0
            ):
                simulation._write_live_snapshot(
                    match,
                    shared_name,
                    shared_slot,
                    slot_ints,
                    write_terrain=False,
                )
        return simulation._build_match_result(
            match,
            match_number=1,
            generation=0,
        )

    @staticmethod
    def _combine_duel_results(
        items: list[dict[str, Any]],
        *,
        seed: int,
        genome_indices: list[int],
    ) -> dict[str, Any]:
        totals: dict[int, dict[str, Any]] = {}
        average_fields = (
            "fitness",
            "rank",
            "tiles",
            "armies",
            "peak_tiles",
            "peak_army",
            "attrition_for",
            "attrition_against",
            "castles_captured",
            "survival_turns",
            "first_contact_turn",
            "first_attrition_turn",
            "battle_turns",
            "major_battle_turns",
            "home_guard_active_turns",
            "home_guard_deficit_turns",
            "armies_created",
            "armies_disbanded",
            "peak_army_units",
            "army_attack_turns",
            "army_defend_turns",
            "army_resupply_turns",
            "army_displacement_rewards",
        )
        for item in items:
            for result in item["results"]:
                genome_index = int(result["genome_index"])
                entry = totals.setdefault(
                    genome_index,
                    {
                        "genome": result["genome"],
                        "count": 0.0,
                        "wins": 0.0,
                        **{key: 0.0 for key in average_fields},
                    },
                )
                entry["count"] += 1.0
                for key in average_fields:
                    value = result.get(key)
                    if value is None:
                        value = -1.0 if key.startswith("first_") else 0.0
                    entry[key] += float(value)
                entry["wins"] += 1.0 if result.get("winner") else 0.0

        results: list[dict[str, Any]] = []
        for genome_index, entry in totals.items():
            count = max(1.0, float(entry["count"]))
            result = {
                "player": genome_index,
                "genome_index": genome_index,
                "rank": round(float(entry["rank"]) / count, 3),
                "fitness": round(float(entry["fitness"]) / count, 3),
                "genome": entry["genome"],
                "mode": "duel",
                "winner": float(entry["wins"]) > count / 2.0,
                "tiles": int(round(float(entry["tiles"]) / count)),
                "armies": int(round(float(entry["armies"]) / count)),
                "peak_tiles": int(
                    round(float(entry["peak_tiles"]) / count)
                ),
                "peak_army": int(
                    round(float(entry["peak_army"]) / count)
                ),
                "attrition_for": round(
                    float(entry["attrition_for"]) / count,
                    2,
                ),
                "attrition_against": round(
                    float(entry["attrition_against"]) / count,
                    2,
                ),
                "castles_captured": int(
                    round(float(entry["castles_captured"]) / count)
                ),
                "survival_turns": round(
                    float(entry["survival_turns"]) / count,
                    2,
                ),
                "first_contact_turn": (
                    int(round(float(entry["first_contact_turn"]) / count))
                    if float(entry["first_contact_turn"]) >= 0.0
                    else None
                ),
                "first_attrition_turn": (
                    int(round(float(entry["first_attrition_turn"]) / count))
                    if float(entry["first_attrition_turn"]) >= 0.0
                    else None
                ),
                "battle_turns": int(
                    round(float(entry["battle_turns"]) / count)
                ),
                "major_battle_turns": int(
                    round(float(entry["major_battle_turns"]) / count)
                ),
                "home_guard_active_turns": int(
                    round(float(entry["home_guard_active_turns"]) / count)
                ),
                "home_guard_deficit_turns": int(
                    round(float(entry["home_guard_deficit_turns"]) / count)
                ),
                "armies_created": int(
                    round(float(entry["armies_created"]) / count)
                ),
                "armies_disbanded": int(
                    round(float(entry["armies_disbanded"]) / count)
                ),
                "peak_army_units": int(
                    round(float(entry["peak_army_units"]) / count)
                ),
                "army_attack_turns": int(
                    round(float(entry["army_attack_turns"]) / count)
                ),
                "army_defend_turns": int(
                    round(float(entry["army_defend_turns"]) / count)
                ),
                "army_resupply_turns": int(
                    round(float(entry["army_resupply_turns"]) / count)
                ),
                "army_displacement_rewards": int(
                    round(
                        float(entry["army_displacement_rewards"]) / count
                    )
                ),
            }
            results.append(result)
        results.sort(
            key=lambda result: (
                -float(result["fitness"]),
                int(result["genome_index"]),
            )
        )
        best = results[0]
        winning_index = (
            int(best["genome_index"]) if best.get("winner") else None
        )
        return {
            "match": 1,
            "generation": 0,
            "mode": "duel",
            "source": str(items[0].get("source", "self_play"))
            if items
            else "self_play",
            "match_count": len(items),
            "seed": seed,
            "turn": max(
                (int(item.get("turn", 0)) for item in items),
                default=0,
            ),
            "simulated_turns": sum(
                int(item.get("simulated_turns", 0))
                for item in items
            ),
            "winner": winning_index,
            "winner_genome_index": winning_index,
            "duration_seconds": round(
                sum(
                    float(item.get("duration_seconds", 0.0))
                    for item in items
                ),
                4,
            ),
            "best_fitness": best["fitness"],
            "best_player": best["genome_index"],
            "results": results,
            "duel_participants": list(genome_indices),
            "saved_at": time.time(),
        }

    @classmethod
    def simulate_episode(
        cls,
        genomes: list[dict[str, float]],
        seed: int,
        *,
        map_size: int = TRAINING_MAP_SIZE,
        max_turns: int = TRAINING_MAX_TURNS,
        shared_name: str | None = None,
        shared_slot: int = 0,
        mode: str = "ffa",
        genome_indices: list[int] | None = None,
        player_count: int = TRAINING_PLAYERS,
        source: str = "self_play",
        collect_diagnostics: bool = False,
        live_map_size: int | None = None,
    ) -> dict[str, Any]:
        if mode not in ("ffa", "duel"):
            raise ValueError("mode must be 'ffa' or 'duel'.")
        if not genomes:
            raise ValueError("At least one genome is required.")
        if not 2 <= player_count <= TRAINING_MAX_PLAYERS:
            raise ValueError(
                f"player_count must be between 2 and {TRAINING_MAX_PLAYERS}."
            )
        simulation = cls.__new__(cls)
        simulation.rng = random.Random(seed)
        simulation.population_size = len(genomes)
        simulation.map_size = map_size
        simulation.live_map_size = max(
            int(map_size),
            int(live_map_size or map_size),
        )
        simulation.max_turns = max_turns
        simulation.population = [
            AIGenome.from_mapping(genome) for genome in genomes
        ]
        if genome_indices is None:
            genome_indices = list(range(len(genomes)))
        else:
            genome_indices = [
                int(index) % len(genomes) for index in genome_indices
            ]
        if mode == "ffa":
            match = simulation._new_match(
                seed,
                shuffle_genomes=False,
                player_count=player_count,
                population_indices=genome_indices,
                mode="ffa",
                source=source,
                collect_diagnostics=collect_diagnostics,
            )
            return cls._simulate_match(
                simulation,
                match,
                shared_name=shared_name,
                shared_slot=shared_slot,
            )

        if len(genome_indices) < 2:
            genome_indices = [
                genome_indices[0],
                genome_indices[0],
            ]
        simulation.max_turns = min(max_turns, TRAINING_DUEL_MAX_TURNS)
        duel_items: list[dict[str, Any]] = []
        for offset in range(0, len(genome_indices) - 1, 2):
            pair = genome_indices[offset : offset + 2]
            duel_seed = seed + offset * 7919
            match = simulation._new_match(
                duel_seed,
                shuffle_genomes=False,
                player_count=2,
                population_indices=pair,
                mode="duel",
                source=source,
                collect_diagnostics=collect_diagnostics,
            )
            duel_items.append(
                cls._simulate_match(
                    simulation,
                    match,
                    shared_name=shared_name,
                    shared_slot=shared_slot,
                )
            )
        return cls._combine_duel_results(
            duel_items,
            seed=seed,
            genome_indices=genome_indices,
        )

    def _select_parent(
        self,
        ranked_results: list[dict[str, Any]],
    ) -> AIGenome:
        sample_size = min(4, len(ranked_results))
        sample = self.rng.sample(ranked_results, sample_size)
        selected = max(sample, key=lambda item: float(item["fitness"]))
        return AIGenome.from_mapping(selected["genome"])

    def _evolve(self, ranked_results: list[dict[str, Any]]) -> None:
        ranked_results = sorted(
            ranked_results,
            key=lambda item: (-float(item["fitness"]), item["player"]),
        )
        next_population = [
            AIGenome.from_mapping(item["genome"])
            for item in ranked_results[:TRAINING_ELITE_COUNT]
        ]
        if not next_population:
            next_population.append(self.champion)
        next_population[0] = self.champion
        while len(next_population) < self.population_size:
            first = self._select_parent(ranked_results)
            second = self._select_parent(ranked_results)
            child = first.crossed(second, self.rng)
            child = child.mutate(
                self.rng,
                rate=0.20,
                scale=0.10,
                reset_chance=0.012,
            )
            next_population.append(child)
        self.population = next_population[: self.population_size]

    def advance(self, budget_seconds: float = 0.03) -> int:
        if not self.running:
            return 0
        deadline = time.perf_counter() + max(0.0, budget_seconds)
        turns = 0
        while self.running and time.perf_counter() < deadline:
            with self._lock:
                if self.current is None:
                    self.current = self._new_match()
                match = self.current
                self._step_turn(match)
                turns += 1
                match.turns_in_window += 1
                now = time.perf_counter()
                window = now - match.window_started_at
                if window >= 0.40:
                    self.turns_per_second = match.turns_in_window / window
                    match.turns_in_window = 0
                    match.window_started_at = now
                if self._match_finished(match):
                    self._finish_match(match)
                    self.current = None
                    if time.perf_counter() < deadline:
                        self.current = self._new_match()
        return turns

    def run_match(
        self,
        *,
        seed: int | None = None,
        max_turns: int | None = None,
    ) -> dict[str, Any]:
        previous_max = self.max_turns
        if max_turns is not None:
            self.max_turns = max(1, int(max_turns))
        with self._lock:
            match = self._new_match(seed)
            try:
                while not self._match_finished(match):
                    self._step_turn(match)
                return self._finish_match(match)
            finally:
                self.max_turns = previous_max

    def record_training_batch(
        self,
        items: list[dict[str, Any]],
        *,
        batch_elapsed: float | None = None,
    ) -> dict[str, Any]:
        if not items:
            raise ValueError("Training batch cannot be empty.")
        if self.running:
            raise RuntimeError(
                "Pause training before recording an external batch."
            )
        with self._lock:
            return self._commit_batch(
                items,
                batch_elapsed=batch_elapsed,
            )

    def snapshot(
        self,
        *,
        live_view: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        live = self.live_view() if live_view is None else live_view
        with self._lock:
            current_turn = (
                self.current.turn
                if self.current is not None
                else int((live or {}).get("turn", 0))
            )
            current_active = (
                len(self.current.board.active_players)
                if self.current is not None
                else int((live or {}).get("active", 0))
            )
            return {
                "running": self.running,
                "training_mode": "hybrid_fa_duel",
                "ffa_weight": TRAINING_FFA_WEIGHT,
                "duel_weight": TRAINING_DUEL_WEIGHT,
                "update_interval_matches": (
                    TRAINING_UPDATE_INTERVAL_MATCHES
                ),
                "champion_window_matches": (
                    TRAINING_CHAMPION_WINDOW_MATCHES
                ),
                "generation": self.generation,
                "completed_matches": self.completed_matches,
                "completed_turns": self.completed_turns,
                "pending_matches": self._pending_match_count(),
                "online_matches": len(self.online_history),
                "best_fitness": self.best_fitness,
                "champion": self.champion.to_dict(),
                "current_turn": current_turn,
                "current_active": current_active,
                "live_slot": self.live_slot,
                "live_slot_count": max(1, self._live_worker_count),
                "turns_per_second": self.turns_per_second,
                "matches_per_hour": self.matches_per_hour,
                "history": list(self.history),
                "latest_results": list(self.latest_results),
                "last_error": self.last_error,
            }

    def cycle_live_slot(self) -> int:
        """Select the next parallel training match without auto-switching."""
        with self._lock:
            count = max(1, self._live_worker_count)
            self.live_slot = (self.live_slot + 1) % count
            self._live_cache = self._live_caches.get(self.live_slot)
            self._live_cache_stamp = (
                self._live_cache.get("stamp")
                if self._live_cache is not None
                else None
            )
            return self.live_slot

    def live_view(self) -> dict[str, Any] | None:
        with self._lock:
            if self.current is not None:
                board = self.current.board
                return {
                    "width": board.width,
                    "height": board.height,
                    "turn": board.turn,
                    "active": len(board.active_players),
                    "stamp": (board.turn, -1),
                    "tiles": [
                        (tile.terrain, tile.owner)
                        for row in board.grid
                        for tile in row
                    ],
                }
        memory = self._shared_memory
        if memory is None:
            return self._live_caches.get(self.live_slot)
        try:
            values = memory.buf.cast("i")
            total_ints = len(values)
            slot_ints = self._live_slot_ints()
            if slot_ints <= LIVE_SLOT_HEADER or total_ints < slot_ints:
                return self._live_caches.get(self.live_slot)
            available_slots = total_ints // slot_ints
            if available_slots <= 0:
                return self._live_caches.get(self.live_slot)
            slot = min(
                max(0, self.live_slot),
                max(0, self._live_worker_count - 1),
                available_slots - 1,
            )
            offset = slot * slot_ints
            if offset < 0 or offset + slot_ints > total_ints:
                return self._live_caches.get(slot)
            sequence = int(values[offset])
            if sequence <= 0 or sequence % 2:
                return self._live_caches.get(slot)
            width = int(values[offset + 5])
            height = int(values[offset + 6])
            if (
                width <= 0
                or height <= 0
                or width > self.live_map_size
                or height > self.live_map_size
                or width * height * 2 > slot_ints - LIVE_SLOT_HEADER
            ):
                return self._live_caches.get(slot)
            freshness = int(values[offset + 4])
            best_stamp = (freshness, sequence, slot)
            if self._live_cache_stamp == best_stamp:
                return self._live_cache
            first_sequence = int(values[offset])
            turn = int(values[offset + 1])
            active = int(values[offset + 2])
            tile_offset = offset + LIVE_SLOT_HEADER
            tiles = [
                (
                    int(values[tile_offset + index * 2]),
                    int(values[tile_offset + index * 2 + 1]),
                )
                for index in range(width * height)
            ]
            if int(values[offset]) != first_sequence:
                return self._live_caches.get(slot)
            self._live_cache_stamp = best_stamp
            self._live_cache = {
                "width": width,
                "height": height,
                "turn": turn,
                "active": active,
                "tiles": tiles,
                "stamp": best_stamp,
            }
            self._live_caches[slot] = self._live_cache
            return self._live_cache
        except (BufferError, IndexError, RuntimeError, ValueError):
            return self._live_caches.get(self.live_slot)
