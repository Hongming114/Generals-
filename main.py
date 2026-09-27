from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import random
import time

import pygame

from generals_ai import (
    AIGenome,
    DIFFICULTIES,
    GeneralsAI,
    assign_ai_archetypes,
)
from generals_core import (
    CITY,
    GENERAL,
    HILL,
    HUMAN,
    MAX_PLAYERS,
    MOUNTAIN,
    NEUTRAL,
    Board,
    Move,
    PendingMove,
    TurnReport,
)
from generals_training import TrainingEngine


WINDOW_W = 1280
WINDOW_H = 840
PADDING = 18
PANEL_W = 340
HEADER_H = 66
DEFAULT_TURN_SECONDS = 1.2
MIN_TURN_SECONDS = 0.0
MAX_TURN_SECONDS = 3.0
MAX_ROUTE_STEPS = 64
MAX_STALLED_TURNS = 8
MIN_VIEW_CELLS = 36
AI_COUNT_MIN = 1
AI_COUNT_MAX = MAX_PLAYERS - 1
MAX_ZOOM = 9.0
MAP_SIZES = (20, 39, 45, 51, 75, 100)
VIEW_PAN_PIXELS_PER_SECOND = 1480.0
VIEW_PAN_SPRINT_MULTIPLIER = 1.8
VIEW_PAN_RESPONSE = 13.0
VIEW_PAN_DAMPING = 16.0
BOARD_CHUNK_TILES = 16
BOARD_CACHE_MAX_PIXELS = 8_000_000
BOARD_CHUNK_CACHE_LIMIT = 24
INDEPENDENT_ARMY_LABEL_COLOR = (221, 74, 55)
TEXT_RENDER_CACHE_LIMIT = 768
TEXT_RENDER_CACHE: OrderedDict[
    tuple[int, str, tuple[int, int, int]],
    pygame.Surface,
] = OrderedDict()

THEMES = {
    "night": {
        "BG": (17, 20, 26),
        "PANEL": (27, 31, 39),
        "PANEL_LIGHT": (36, 41, 51),
        "LINE": (58, 65, 78),
        "TEXT": (235, 238, 244),
        "MUTED": (151, 160, 176),
        "LAND": (47, 52, 61),
        "LAND_ALT": (52, 58, 68),
        "MOUNTAIN_COLOR": (78, 82, 88),
        "MOUNTAIN_LIGHT": (136, 141, 147),
        "MOUNTAIN_EDGE": (39, 43, 48),
        "HILL_COLOR": (47, 52, 61),
        "HILL_LIGHT": (139, 119, 83),
        "HILL_EDGE": (77, 66, 49),
        "CITY_COLOR": (196, 151, 55),
        "CITY_LIGHT": (245, 205, 100),
        "CITY_DARK": (65, 57, 39),
        "SELECT_COLOR": (255, 218, 92),
        "FOG": (9, 11, 15, 175),
        "SUCCESS": (77, 202, 142),
        "BUTTON": (49, 56, 69),
        "BUTTON_HOVER": (66, 75, 91),
        "BUTTON_ACTIVE": (218, 166, 52),
        "GRID_LINE": (22, 26, 33),
        "BOARD_BG": (12, 15, 20),
        "BOARD_SURFACE": (18, 21, 27),
        "TEXT_SHADOW": (7, 9, 13),
        "ICON_DARK": (18, 22, 27),
        "MINIMAP_FOG": (28, 32, 39),
        "HIGHLIGHT_ROW": (38, 45, 59),
        "TRACK": (20, 24, 30),
    },
    "day": {
        "BG": (235, 238, 243),
        "PANEL": (250, 251, 253),
        "PANEL_LIGHT": (233, 237, 243),
        "LINE": (191, 200, 213),
        "TEXT": (28, 36, 47),
        "MUTED": (93, 105, 122),
        "LAND": (222, 226, 232),
        "LAND_ALT": (214, 220, 228),
        "MOUNTAIN_COLOR": (128, 132, 136),
        "MOUNTAIN_LIGHT": (180, 184, 188),
        "MOUNTAIN_EDGE": (75, 80, 86),
        "HILL_COLOR": (222, 226, 232),
        "HILL_LIGHT": (196, 154, 88),
        "HILL_EDGE": (128, 96, 56),
        "CITY_COLOR": (190, 138, 35),
        "CITY_LIGHT": (236, 175, 43),
        "CITY_DARK": (244, 222, 169),
        "SELECT_COLOR": (235, 165, 20),
        "FOG": (244, 247, 250, 185),
        "SUCCESS": (38, 157, 103),
        "BUTTON": (221, 227, 234),
        "BUTTON_HOVER": (204, 214, 225),
        "BUTTON_ACTIVE": (239, 178, 55),
        "GRID_LINE": (225, 230, 236),
        "BOARD_BG": (205, 211, 220),
        "BOARD_SURFACE": (243, 246, 249),
        "TEXT_SHADOW": (255, 255, 255),
        "ICON_DARK": (246, 248, 250),
        "MINIMAP_FOG": (212, 218, 226),
        "HIGHLIGHT_ROW": (218, 230, 244),
        "TRACK": (208, 214, 223),
    },
}

BG = THEMES["day"]["BG"]
PANEL = THEMES["day"]["PANEL"]
PANEL_LIGHT = THEMES["day"]["PANEL_LIGHT"]
LINE = THEMES["day"]["LINE"]
TEXT = THEMES["day"]["TEXT"]
MUTED = THEMES["day"]["MUTED"]
LAND = THEMES["day"]["LAND"]
LAND_ALT = THEMES["day"]["LAND_ALT"]
MOUNTAIN_COLOR = THEMES["day"]["MOUNTAIN_COLOR"]
MOUNTAIN_LIGHT = THEMES["day"]["MOUNTAIN_LIGHT"]
MOUNTAIN_EDGE = THEMES["day"]["MOUNTAIN_EDGE"]
HILL_COLOR = THEMES["day"]["HILL_COLOR"]
HILL_LIGHT = THEMES["day"]["HILL_LIGHT"]
HILL_EDGE = THEMES["day"]["HILL_EDGE"]
CITY_COLOR = THEMES["day"]["CITY_COLOR"]
CITY_LIGHT = THEMES["day"]["CITY_LIGHT"]
CITY_DARK = THEMES["day"]["CITY_DARK"]
SELECT_COLOR = THEMES["day"]["SELECT_COLOR"]
FOG = THEMES["day"]["FOG"]
SUCCESS = THEMES["day"]["SUCCESS"]
BUTTON = THEMES["day"]["BUTTON"]
BUTTON_HOVER = THEMES["day"]["BUTTON_HOVER"]
BUTTON_ACTIVE = THEMES["day"]["BUTTON_ACTIVE"]
GRID_LINE = THEMES["day"]["GRID_LINE"]
BOARD_BG = THEMES["day"]["BOARD_BG"]
BOARD_SURFACE = THEMES["day"]["BOARD_SURFACE"]
TEXT_SHADOW = THEMES["day"]["TEXT_SHADOW"]
ICON_DARK = THEMES["day"]["ICON_DARK"]
MINIMAP_FOG = THEMES["day"]["MINIMAP_FOG"]
HIGHLIGHT_ROW = THEMES["day"]["HIGHLIGHT_ROW"]
TRACK = THEMES["day"]["TRACK"]


def apply_theme(name: str) -> None:
    global BG, PANEL, PANEL_LIGHT, LINE, TEXT, MUTED, LAND, LAND_ALT
    global MOUNTAIN_COLOR, MOUNTAIN_LIGHT, MOUNTAIN_EDGE, CITY_COLOR, CITY_LIGHT
    global HILL_COLOR, HILL_LIGHT, HILL_EDGE
    global CITY_DARK, SELECT_COLOR, FOG, SUCCESS, BUTTON, BUTTON_HOVER
    global BUTTON_ACTIVE, GRID_LINE, BOARD_BG, BOARD_SURFACE, TEXT_SHADOW
    global ICON_DARK, MINIMAP_FOG, HIGHLIGHT_ROW, TRACK

    colors = THEMES[name]
    BG = colors["BG"]
    PANEL = colors["PANEL"]
    PANEL_LIGHT = colors["PANEL_LIGHT"]
    LINE = colors["LINE"]
    TEXT = colors["TEXT"]
    MUTED = colors["MUTED"]
    LAND = colors["LAND"]
    LAND_ALT = colors["LAND_ALT"]
    MOUNTAIN_COLOR = colors["MOUNTAIN_COLOR"]
    MOUNTAIN_LIGHT = colors["MOUNTAIN_LIGHT"]
    MOUNTAIN_EDGE = colors["MOUNTAIN_EDGE"]
    HILL_COLOR = colors["HILL_COLOR"]
    HILL_LIGHT = colors["HILL_LIGHT"]
    HILL_EDGE = colors["HILL_EDGE"]
    CITY_COLOR = colors["CITY_COLOR"]
    CITY_LIGHT = colors["CITY_LIGHT"]
    CITY_DARK = colors["CITY_DARK"]
    SELECT_COLOR = colors["SELECT_COLOR"]
    FOG = colors["FOG"]
    SUCCESS = colors["SUCCESS"]
    BUTTON = colors["BUTTON"]
    BUTTON_HOVER = colors["BUTTON_HOVER"]
    BUTTON_ACTIVE = colors["BUTTON_ACTIVE"]
    GRID_LINE = colors["GRID_LINE"]
    BOARD_BG = colors["BOARD_BG"]
    BOARD_SURFACE = colors["BOARD_SURFACE"]
    TEXT_SHADOW = colors["TEXT_SHADOW"]
    ICON_DARK = colors["ICON_DARK"]
    MINIMAP_FOG = colors["MINIMAP_FOG"]
    HIGHLIGHT_ROW = colors["HIGHLIGHT_ROW"]
    TRACK = colors["TRACK"]

PLAYER_COLORS = (
    (45, 145, 231),
    (224, 77, 88),
    (73, 194, 132),
    (232, 155, 58),
    (168, 116, 232),
    (61, 184, 201),
    (226, 105, 176),
    (177, 199, 70),
    (121, 139, 239),
    (222, 126, 88),
    (91, 179, 112),
    (203, 175, 90),
    (111, 118, 137),
    (205, 93, 122),
    (78, 157, 169),
    (156, 137, 88),
    (127, 103, 181),
    (205, 132, 53),
    (87, 146, 87),
    (177, 101, 157),
    (84, 125, 190),
    (194, 87, 69),
    (82, 170, 143),
    (146, 126, 178),
    (53, 103, 126),
    (132, 83, 61),
    (119, 172, 207),
    (198, 144, 186),
    (107, 166, 84),
    (215, 188, 126),
    (88, 107, 160),
    (175, 90, 92),
    (75, 145, 131),
    (145, 114, 65),
    (164, 164, 164),
    (118, 83, 142),
    (198, 111, 82),
)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def mix(
    color_a: tuple[int, int, int],
    color_b: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * amount) for a, b in zip(color_a, color_b))


def player_color(player: int) -> tuple[int, int, int]:
    return PLAYER_COLORS[player % len(PLAYER_COLORS)]


def player_light(player: int) -> tuple[int, int, int]:
    return mix(player_color(player), TEXT, 0.42)


def player_label(player: int) -> str:
    return "你" if player == HUMAN else f"AI {player:02d}"


def load_font(size: int, bold: bool = False) -> pygame.font.Font:
    windows = os.environ.get("WINDIR", "C:\\Windows")
    candidates = [
        os.path.join(windows, "Fonts", "msyhbd.ttc" if bold else "msyh.ttc"),
        os.path.join(windows, "Fonts", "simhei.ttf"),
        os.path.join(windows, "Fonts", "simsun.ttc"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont("arial", size, bold=bold)


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    color: tuple[int, int, int],
    pos: tuple[int, int],
    *,
    center: bool = False,
    right: bool = False,
) -> pygame.Rect:
    cache_key = (id(font), text, color)
    image = TEXT_RENDER_CACHE.get(cache_key)
    if image is None:
        image = font.render(text, True, color)
        TEXT_RENDER_CACHE[cache_key] = image
        if len(TEXT_RENDER_CACHE) > TEXT_RENDER_CACHE_LIMIT:
            TEXT_RENDER_CACHE.popitem(last=False)
    else:
        TEXT_RENDER_CACHE.move_to_end(cache_key)
    rect = image.get_rect()
    if center:
        rect.center = pos
    elif right:
        rect.topright = pos
    else:
        rect.topleft = pos
    surface.blit(image, rect)
    return rect


def draw_button(
    surface: pygame.Surface,
    rect: pygame.Rect,
    label: str,
    font: pygame.font.Font,
    *,
    active: bool = False,
    hovered: bool = False,
    enabled: bool = True,
) -> None:
    color = BUTTON_ACTIVE if active else (BUTTON_HOVER if hovered else BUTTON)
    if not enabled:
        color = (39, 43, 51)
    pygame.draw.rect(surface, color, rect, border_radius=7)
    pygame.draw.rect(surface, LINE if enabled else (47, 51, 60), rect, 1, border_radius=7)
    text_color = (21, 24, 29) if active else (TEXT if enabled else (102, 109, 121))
    draw_text(surface, font, label, text_color, rect.center, center=True)


@dataclass(frozen=True, slots=True)
class Command:
    tx: int
    ty: int
    mode: str


@dataclass(slots=True)
class CommandSnapshot:
    commands: dict[tuple[int, int], list[Command]]
    command_order: list[tuple[int, int]]
    route_stalls: dict[tuple[int, int], int]
    route_hill_waits: dict[tuple[int, int], Command]
    pending_moves: list[PendingMove]
    selected: tuple[int, int] | None


class GameApp:
    def __init__(self, args: argparse.Namespace, *, headless: bool = False) -> None:
        pygame.init()
        pygame.display.set_caption("Generals - Auto Turn Multiplayer")
        self.headless = headless
        self.fullscreen = bool(getattr(args, "fullscreen", False))
        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H),
            self._display_flags(),
        )
        self.clock = pygame.time.Clock()

        self.font_tiny = load_font(12)
        self.font_small = load_font(14)
        self.font_body = load_font(17)
        self.font_bold = load_font(19, bold=True)
        self.font_heading = load_font(27, bold=True)
        self.font_title = load_font(54, bold=True)
        self.font_cache: dict[tuple[str, tuple[int, int, int], int], pygame.font.Font] = {}
        self.text_cache: dict[
            tuple[
                str,
                tuple[int, int, int],
                tuple[int, int, int],
                int,
                bool,
            ],
            pygame.Surface,
        ] = {}

        self.running = True
        self.state = "MENU"
        self.previous_state = "MENU"
        self.difficulty_key = args.difficulty
        self.board_size = args.size
        self.player_count = args.players
        self.theme_key = args.theme
        apply_theme(self.theme_key)
        self.turn_seconds = clamp(
            float(args.turn_seconds),
            MIN_TURN_SECONDS,
            MAX_TURN_SECONDS,
        )
        self.growth_interval = int(clamp(float(args.growth_interval), 10, 50))
        self.vision_radius = int(
            clamp(float(getattr(args, "vision_radius", 2)), 0, 5)
        )
        self.fog_enabled = args.fog != "off"
        self.seed = args.seed
        self.training = TrainingEngine(
            seed=(self.seed or 0) + 424_243,
        )
        self.board = Board(
            self.board_size,
            self.board_size,
            self.player_count,
            self.seed,
            self.growth_interval,
            self.vision_radius,
        )
        self.ais = self._create_ais()
        self.ai_takeover = False
        self.spectating = False
        self.live_result_recorded = False
        self.human_ai = GeneralsAI(
            HUMAN,
            self.difficulty_key,
            (self.seed or 0) + 104729,
            genome=(
                self.training.champion
                if self.difficulty_key == "hard"
                else None
            ),
        )

        self.human_commands: dict[tuple[int, int], list[Command]] = {}
        self.command_order: list[tuple[int, int]] = []
        self.route_stalls: dict[tuple[int, int], int] = {}
        self.route_hill_waits: dict[tuple[int, int], Command] = {}
        self.command_history: list[CommandSnapshot] = []
        self.selected: tuple[int, int] | None = None
        self.hover: tuple[int, int] | None = None
        self.pan_last: tuple[int, int] | None = None
        self.speed_dragging = False
        self.growth_dragging = False
        self.vision_dragging = False
        self.ai_count_dragging = False
        self.turn_deadline = time.monotonic() + self.turn_seconds
        self.recent_captures: list[tuple[int, int, float]] = []
        self.logs: list[str] = []
        self.visible_human: set[tuple[int, int]] = set()
        self.player_stats: dict[int, dict[str, int]] = {}
        self.minimap_surface: pygame.Surface | None = None
        self.minimap_surface_size = (0, 0)
        self.fog_mask: pygame.Surface | None = None
        self.fog_scaled_surface: pygame.Surface | None = None
        self.fog_scaled_size = (0, 0)
        self.board_render_revision = 0
        self.board_cache_surface: pygame.Surface | None = None
        self.board_cache_key: tuple[int, str, int] | None = None
        self.board_chunk_key: tuple[int, str, int] | None = None
        self.board_chunk_cache: dict[tuple[int, int], pygame.Surface] = {}
        self.arrow_overlay: pygame.Surface | None = None
        self.training_minimap_surface: pygame.Surface | None = None
        self.training_minimap_key: tuple[object, ...] | None = None
        self.training_minimap_scaled_surface: pygame.Surface | None = None
        self.training_minimap_scaled_key: tuple[object, ...] | None = None
        self.camera_velocity = [0.0, 0.0]
        self.pending_right_click: tuple[
            tuple[int, int],
            tuple[int, int],
            float,
        ] | None = None
        self.toast_message = ""
        self.toast_until = 0.0
        self.toast_success = True

        self.camera_center = [0.0, 0.0]
        self.zoom = 1.0
        self.menu_difficulty_rects: dict[str, pygame.Rect] = {}
        self.menu_size_rects: dict[int, pygame.Rect] = {}
        self.menu_ai_count_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_speed_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_theme_rects: dict[str, pygame.Rect] = {}
        self.menu_fog_rects: dict[str, pygame.Rect] = {}
        self.menu_vision_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_fullscreen_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_start_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_settings_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_help_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_training_rect = pygame.Rect(0, 0, 0, 0)
        self.settings_back_rect = pygame.Rect(0, 0, 0, 0)
        self.help_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_growth_rect = pygame.Rect(0, 0, 0, 0)
        self.pause_rect = pygame.Rect(0, 0, 0, 0)
        self.takeover_rect = pygame.Rect(0, 0, 0, 0)
        self.restart_rect = pygame.Rect(0, 0, 0, 0)
        self.menu_rect = pygame.Rect(0, 0, 0, 0)
        self.minimap_rect = pygame.Rect(0, 0, 0, 0)
        self.game_over_new_rect = pygame.Rect(0, 0, 0, 0)
        self.game_over_menu_rect = pygame.Rect(0, 0, 0, 0)
        self.training_pause_rect = pygame.Rect(0, 0, 0, 0)
        self.training_back_rect = pygame.Rect(0, 0, 0, 0)
        self.training_preview_rect = pygame.Rect(0, 0, 0, 0)

        self.refresh_view_data()
        if args.quick_start:
            self.new_game()

    def _create_ais(self) -> dict[int, GeneralsAI]:
        trained_genome = (
            self.training.champion
            if self.difficulty_key == "hard"
            else None
        )
        ai_count = max(0, self.player_count - 1)
        archetypes = assign_ai_archetypes(
            ai_count,
            (self.seed or 0) + 271_828,
        )
        return {
            player: GeneralsAI(
                player,
                self.difficulty_key,
                (self.seed or 0) + player * 7919,
                genome=trained_genome,
                archetype=archetypes[player - 1],
            )
            for player in range(1, self.player_count)
        }

    def record_live_result(self) -> None:
        winner = self.board.winner
        if winner is None or self.live_result_recorded:
            return
        active = set(self.board.active_players)
        survivors = []
        if winner in active:
            survivors.append(winner)
            active.remove(winner)
        survivors.extend(
            sorted(
                active,
                key=lambda player: (
                    self.board.stats(player)["armies"],
                    self.board.stats(player)["tiles"],
                ),
                reverse=True,
            )
        )
        eliminated = sorted(
            self.board.eliminated_players,
            key=lambda player: (
                self.board.stats(player)["armies"],
                self.board.stats(player)["tiles"],
            ),
            reverse=True,
        )
        ranking = survivors + [
            player
            for player in eliminated
            if player not in survivors
        ]
        results = []
        for rank, player in enumerate(ranking, 1):
            if player == HUMAN and not self.ai_takeover:
                continue
            ai = self.human_ai if player == HUMAN else self.ais.get(player)
            if ai is None:
                continue
            stats = self.board.stats(player)
            results.append(
                {
                    "player": player,
                    "rank": rank,
                    "winner": player == winner,
                    "tiles": int(stats["tiles"]),
                    "armies": int(stats["armies"]),
                    "genome": ai.genome.to_dict(),
                }
            )
        self.training.record_live_match(
            results,
            turn=max(0, self.board.turn - 1),
            player_count=self.player_count,
            seed=self.seed,
        )
        self.live_result_recorded = True

    @property
    def ai_count(self) -> int:
        return self.player_count - 1

    def new_game(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        elif self.seed is None:
            self.seed = random.randrange(1, 1_000_000_000)
        self.board = Board(
            self.board_size,
            self.board_size,
            self.player_count,
            self.seed,
            self.growth_interval,
            self.vision_radius,
        )
        self.ais = self._create_ais()
        self.human_ai = GeneralsAI(
            HUMAN,
            self.difficulty_key,
            (self.seed or 0) + 104729,
            genome=(
                self.training.champion
                if self.difficulty_key == "hard"
                else None
            ),
        )
        self.ai_takeover = False
        self.spectating = False
        self.live_result_recorded = False
        self.human_commands.clear()
        self.command_order.clear()
        self.route_stalls.clear()
        self.route_hill_waits.clear()
        self.command_history.clear()
        self.selected = None
        self.pending_right_click = None
        self.toast_message = ""
        self.toast_until = 0.0
        self.recent_captures.clear()
        self.logs = [
            f"{self.player_count} 名玩家进入战场，你面对 {self.player_count - 1} 个独立 AI",
            "自动回合已开启，命令会持续执行直到取消或失效",
        ]
        archetype_counts: dict[str, int] = {}
        for ai in self.ais.values():
            archetype_counts[ai.archetype_label] = (
                archetype_counts.get(ai.archetype_label, 0) + 1
            )
        if archetype_counts:
            self.logs.append(
                "AI 头衔："
                + "，".join(
                    f"{label} {count}"
                    for label, count in archetype_counts.items()
                )
            )
        self.zoom = self.min_zoom()
        human_general = self.board.general_position(HUMAN) or (self.board.width // 2, self.board.height // 2)
        self.camera_center = [human_general[0] + 0.5, human_general[1] + 0.5]
        self.refresh_view_data()
        self.turn_deadline = time.monotonic() + self.turn_seconds
        self.state = "PLAYING"

    def refresh_view_data(self) -> None:
        self.board_render_revision += 1
        self.board_cache_surface = None
        self.board_cache_key = None
        self.board_chunk_key = None
        self.board_chunk_cache.clear()
        self.minimap_surface = None
        self.minimap_surface_size = (0, 0)
        self.fog_mask = None
        self.fog_scaled_surface = None
        self.fog_scaled_size = (0, 0)
        if self.spectating or not self.fog_enabled:
            self.visible_human = {
                (x, y)
                for y in range(self.board.height)
                for x in range(self.board.width)
            }
        elif self.fog_enabled:
            self.visible_human = self.board.visibility(HUMAN)
        self.player_stats = {
            player: {"tiles": 0, "armies": 0, "cities": 0, "generals": 0}
            for player in range(self.player_count)
        }
        for row in self.board.grid:
            for tile in row:
                if tile.owner < 0 or tile.owner >= self.player_count:
                    continue
                stats = self.player_stats[tile.owner]
                stats["tiles"] += 1
                stats["armies"] += tile.army
                if tile.terrain == CITY:
                    stats["cities"] += 1
                elif tile.terrain == GENERAL:
                    stats["generals"] += 1
        if self.fog_enabled:
            self.fog_mask = pygame.Surface(
                (self.board.width, self.board.height),
                pygame.SRCALPHA,
            )
            self.fog_mask.fill(FOG)
            for x, y in self.visible_human:
                self.fog_mask.set_at((x, y), (0, 0, 0, 0))

    def _display_flags(self) -> int:
        if self.fullscreen and not self.headless:
            return pygame.SCALED | pygame.FULLSCREEN
        return 0

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H),
            self._display_flags(),
        )
        self.refresh_view_data()
        self.logs.insert(
            0,
            "已开启全屏（F11 切换）"
            if self.fullscreen
            else "已返回窗口模式（F11 切换）",
        )
        self.logs = self.logs[:8]

    def run(self) -> None:
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.update(dt)
            self.draw()
            pygame.display.flip()
        self.training.pause()
        pygame.quit()

    def update(self, dt: float) -> None:
        self.handle_keyboard_held(dt)
        self._flush_pending_right_click()
        self.recent_captures = [
            item for item in self.recent_captures if time.monotonic() - item[2] < 0.75
        ]
        if self.state == "PLAYING" and time.monotonic() >= self.turn_deadline:
            self.advance_turn()

    def show_toast(self, message: str, *, success: bool = True) -> None:
        self.toast_message = message
        self.toast_until = time.monotonic() + 2.2
        self.toast_success = success

    def record_action_animations(
        self,
        captures: list[tuple[int, int, int]],
    ) -> None:
        """Only expose feedback for actions initiated by the human side."""
        now = time.monotonic()
        for x, y, owner in captures:
            if owner == HUMAN:
                self.recent_captures.append((x, y, now))

    def try_create_human_army(self, cell: tuple[int, int]) -> bool:
        if self.spectating or self.ai_takeover:
            return False
        created, message = self.human_ai.army_controller.create_at(
            self.board,
            cell,
            enforce_army_limit=True,
        )
        self.show_toast(message, success=created)
        if not created:
            return False
        self.board.cancel_pending_moves(HUMAN, cell)
        self.human_commands.pop(cell, None)
        self.route_stalls.pop(cell, None)
        self.route_hill_waits.pop(cell, None)
        self.command_order = [source for source in self.command_order if source != cell]
        self.selected = cell
        self.logs.insert(0, message)
        self.logs = self.logs[:8]
        return True

    def mark_human_main_army(self, cell: tuple[int, int]) -> None:
        if self.spectating or self.ai_takeover:
            return
        controller = getattr(self.human_ai, "army_controller", None)
        if controller is not None:
            controller.mark_main_army(self.board, cell)

    def _flush_pending_right_click(self) -> None:
        pending = self.pending_right_click
        if pending is None:
            return
        cell, pos, started = pending
        if time.monotonic() - started < 0.30:
            return
        self.pending_right_click = None
        self.handle_game_click(pos, 3)

    def handle_right_click(self, pos: tuple[int, int]) -> None:
        cell = self.cell_at(pos)
        if cell is None:
            self.pending_right_click = None
            self.handle_game_click(pos, 3)
            return
        tile = self.board.tile(*cell)
        controller = getattr(self.human_ai, "army_controller", None)
        if (
            tile.owner == HUMAN
            and controller is not None
            and controller.release_at(cell)
        ):
            self.pending_right_click = None
            self.selected = cell
            self.show_toast("已取消军队头衔，恢复手动控制")
            self.logs.insert(0, "右键单击已接管该军队，恢复普通地块指挥")
            self.logs = self.logs[:8]
            return
        now = time.monotonic()
        pending = self.pending_right_click
        if pending is not None:
            previous_cell, previous_pos, started = pending
            if previous_cell == cell and now - started <= 0.36:
                self.pending_right_click = None
                self.try_create_human_army(cell)
                return
            self.pending_right_click = None
            self.handle_game_click(previous_pos, 3)
        self.pending_right_click = (cell, pos, now)

    def handle_keyboard_held(self, dt: float) -> None:
        if self.state not in ("PLAYING", "PAUSED"):
            self.camera_velocity[0] = 0.0
            self.camera_velocity[1] = 0.0
            return
        keys = pygame.key.get_pressed()
        dx = int(keys[pygame.K_d] or keys[pygame.K_RIGHT]) - int(
            keys[pygame.K_a] or keys[pygame.K_LEFT]
        )
        dy = int(keys[pygame.K_s] or keys[pygame.K_DOWN]) - int(
            keys[pygame.K_w] or keys[pygame.K_UP]
        )
        if dx and dy:
            diagonal = math.sqrt(2.0)
            target_x = dx / diagonal
            target_y = dy / diagonal
        else:
            target_x = float(dx)
            target_y = float(dy)
        sprint = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
        pan_speed = VIEW_PAN_PIXELS_PER_SECOND * (
            VIEW_PAN_SPRINT_MULTIPLIER if sprint else 1.0
        )
        target_x *= pan_speed
        target_y *= pan_speed
        if target_x or target_y:
            blend = min(1.0, VIEW_PAN_RESPONSE * dt)
            self.camera_velocity[0] += (
                target_x - self.camera_velocity[0]
            ) * blend
            self.camera_velocity[1] += (
                target_y - self.camera_velocity[1]
            ) * blend
        else:
            damping = max(0.0, 1.0 - VIEW_PAN_DAMPING * dt)
            self.camera_velocity[0] *= damping
            self.camera_velocity[1] *= damping
            if abs(self.camera_velocity[0]) < 1.0:
                self.camera_velocity[0] = 0.0
            if abs(self.camera_velocity[1]) < 1.0:
                self.camera_velocity[1] = 0.0
        if not self.camera_velocity[0] and not self.camera_velocity[1]:
            return
        _, cell, _ = self.board_geometry()
        old_x, old_y = self.camera_center
        self.camera_center[0] += self.camera_velocity[0] * dt / max(1.0, cell)
        self.camera_center[1] += self.camera_velocity[1] * dt / max(1.0, cell)
        self.clamp_camera()
        if self.camera_center[0] == old_x:
            self.camera_velocity[0] = 0.0
        if self.camera_center[1] == old_y:
            self.camera_velocity[1] = 0.0

    def human_moves(self) -> list[Move]:
        army_positions = self.human_ai.army_controller.positions
        scheduled_sources: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for source in self.command_order:
            if source in self.human_commands and source not in seen:
                scheduled_sources.append(source)
                seen.add(source)
        for source in self.human_commands:
            if source not in seen:
                scheduled_sources.append(source)
                seen.add(source)

        for sx, sy in scheduled_sources:
            route = self.human_commands.get((sx, sy))
            if not route:
                continue
            if (sx, sy) in army_positions:
                continue
            if not route or (sx, sy) in self.route_hill_waits:
                continue
            command = route[0]
            if (command.tx, command.ty) in army_positions:
                continue
            source = self.board.tile(sx, sy)
            if source.owner != HUMAN or source.army < 2:
                continue
            amount = source.army - 1 if command.mode == "all" else max(1, source.army // 2)
            amount = min(amount, source.army - 1)
            move = Move(sx, sy, command.tx, command.ty, amount)
            if self.board.move_legal(move, HUMAN):
                return [move]
        return []

    def advance_turn(self) -> None:
        if self.state not in ("PLAYING", "PAUSED") or self.board.winner is not None:
            return
        started = time.perf_counter()
        report = TurnReport()
        human_armies = getattr(self.human_ai, "army_controller", None)
        if human_armies is not None:
            human_armies.begin_turn(self.board, self.human_ai)
        for ai in self.ais.values():
            controller = getattr(ai, "army_controller", None)
            if controller is not None:
                controller.begin_turn(self.board, ai)
                if ai.player in self.board.active_players:
                    controller.review_ai_armies(self.board, ai)
        if (
            human_armies is not None
            and self.ai_takeover
            and HUMAN in self.board.active_players
        ):
            human_armies.review_ai_armies(self.board, self.human_ai)
        if (
            self.ai_takeover
            and HUMAN in self.board.active_players
            and human_armies is not None
        ):
            human_armies.maybe_create_auto(
                self.board,
                self.human_ai,
            )
        for player, ai in self.ais.items():
            controller = getattr(ai, "army_controller", None)
            if player in self.board.active_players and controller is not None:
                controller.maybe_create_auto(self.board, ai)
        for phase in range(2):
            observed_moves: list[Move] = []
            planned_human_actions = (
                self.human_ai.plan_moves(
                    self.board,
                    phase=phase,
                    observed_moves=tuple(observed_moves),
                )
                if self.ai_takeover
                else self.human_moves()
            )
            human_army_positions = (
                human_armies.positions
                if human_armies is not None
                else set()
            )
            planned_human_moves = [
                move
                for move in planned_human_actions
                if move.source not in human_army_positions
            ]
            planned_moves = list(planned_human_actions)
            observed_moves.extend(planned_moves)

            if not self.ai_takeover and human_armies is not None:
                human_army_moves = human_armies.plan_moves(
                    self.board,
                    self.human_ai,
                    phase,
                    tuple(planned_human_moves),
                    observed_moves=tuple(observed_moves),
                )
                planned_moves.extend(human_army_moves)
                observed_moves.extend(human_army_moves)

            for player, ai in self.ais.items():
                if player in self.board.active_players:
                    ai_moves = ai.plan_moves(
                        self.board,
                        phase=phase,
                        observed_moves=tuple(observed_moves),
                    )
                    planned_moves.extend(ai_moves)
                    observed_moves.extend(ai_moves)

            human_moves = planned_human_moves
            executed_sources = {move.source for move in human_moves}
            delayed_sources = {
                move.source
                for move in human_moves
                if self.board.movement_turns(
                    move.sx,
                    move.sy,
                    move.tx,
                    move.ty,
                    HUMAN,
                )
                > 1
            }
            waiting_to_finish = set(self.route_hill_waits) if phase == 0 else set()
            moves = planned_moves
            report.merge(self.board.resolve_movement(moves))
            if human_armies is not None:
                human_armies.record_main_army_moves(self.board, moves)
            for ai in self.ais.values():
                controller = getattr(ai, "army_controller", None)
                if controller is not None:
                    controller.record_main_army_moves(self.board, moves)
            self.advance_routes(
                executed_sources,
                delayed_sources,
                waiting_to_finish,
                count_stall=phase == 1,
            )
            if self.board.winner is not None:
                break
        self.board.finish_turn(report)
        self.prune_commands()
        self.command_history.clear()
        self.record_action_animations(report.captures)

        if report.eliminated:
            names = "、".join(player_label(player) for player in report.eliminated)
            self.logs.insert(0, f"{names} 的将军失守，已被淘汰")
        if report.captures:
            self.logs.insert(0, f"本回合 {len(report.captures)} 个地块易手")
        if report.land_growth:
            total_growth = sum(report.land_growth.values())
            self.logs.insert(0, f"扩兵轮触发：普通空地共 +{total_growth} 兵")

        if HUMAN in self.board.eliminated_players and not self.spectating:
            self.spectating = True
            self.ai_takeover = False
            self.board.cancel_pending_moves(HUMAN)
            self.human_commands.clear()
            self.command_order.clear()
            self.route_stalls.clear()
            self.route_hill_waits.clear()
            self.command_history.clear()
            self.selected = None
            self.logs.insert(0, "你的将军已被占领，现以观众模式继续观战")
        if human_armies is not None:
            human_armies.sync(self.board, self.human_ai)
        for ai in self.ais.values():
            controller = getattr(ai, "army_controller", None)
            if controller is not None:
                controller.sync(self.board, ai)
        self.refresh_view_data()

        if self.board.winner is not None:
            if not self.headless:
                self.record_live_result()
            if self.board.winner == HUMAN:
                self.logs.insert(0, "所有对手均被淘汰，你赢得了战争")
            else:
                self.logs.insert(0, f"{player_label(self.board.winner)} 统一了战场")
            self.state = "GAME_OVER"
        else:
            self.state = "PLAYING"
            self.turn_deadline = time.monotonic() + self.turn_seconds
        self.logs = self.logs[:8]
        if time.perf_counter() - started > self.turn_seconds:
            self.logs.insert(0, "本回合计算较慢，已自动延长决策时间")

    def _remove_route(self, source: tuple[int, int]) -> None:
        self.board.cancel_pending_moves(HUMAN, source)
        self.human_commands.pop(source, None)
        self.route_stalls.pop(source, None)
        self.route_hill_waits.pop(source, None)
        self.command_order = [item for item in self.command_order if item != source]
        if self.selected == source:
            self.selected = None

    def _rotate_scheduled_command(self, source: tuple[int, int]) -> None:
        if source not in self.command_order:
            return
        self.command_order = [
            item for item in self.command_order if item != source
        ]
        self.command_order.append(source)

    def advance_routes(
        self,
        executed_sources: set[tuple[int, int]],
        delayed_sources: set[tuple[int, int]],
        waiting_to_finish: set[tuple[int, int]],
        *,
        count_stall: bool = True,
    ) -> None:
        for source in waiting_to_finish:
            wait_command = self.route_hill_waits.pop(source, None)
            route = self.human_commands.get(source)
            if route is None or wait_command is None:
                continue
            next_source = (wait_command.tx, wait_command.ty)
            remaining_route = route[1:] if route and route[0] == wait_command else []
            if (
                remaining_route
                and self.board.in_bounds(*next_source)
                and self.board.tile(*next_source).owner == HUMAN
            ):
                self.human_commands.pop(source, None)
                self.human_commands[next_source] = remaining_route
                self.route_stalls[next_source] = 0
                self.command_order = [
                    next_source if item == source else item for item in self.command_order
                ]
                self._rotate_scheduled_command(next_source)
                if self.selected == source:
                    self.selected = next_source
            else:
                self._remove_route(source)

        for source in delayed_sources:
            route = self.human_commands.get(source)
            if not route:
                continue
            self.route_hill_waits[source] = route[0]
            self.route_stalls.pop(source, None)
            self._rotate_scheduled_command(source)

        for source in executed_sources - delayed_sources - waiting_to_finish:
            route = self.human_commands.get(source)
            if not route:
                continue
            first_step = route.pop(0)
            next_source = (first_step.tx, first_step.ty)
            self.route_stalls.pop(source, None)
            if (
                route
                and self.board.in_bounds(*next_source)
                and self.board.tile(*next_source).owner == HUMAN
            ):
                self.human_commands.pop(source, None)
                self.human_commands[next_source] = route
                self.route_stalls[next_source] = 0
                self.command_order = [
                    next_source if item == source else item for item in self.command_order
                ]
                self._rotate_scheduled_command(next_source)
                if self.selected == source:
                    self.selected = next_source
            else:
                self._remove_route(source)

        if count_stall:
            for source in list(self.human_commands):
                if (
                    source in executed_sources
                    or source in waiting_to_finish
                    or source in self.route_hill_waits
                ):
                    continue
                stalled = self.route_stalls.get(source, 0) + 1
                self.route_stalls[source] = stalled
                if stalled > MAX_STALLED_TURNS:
                    self._remove_route(source)
                    self.logs.insert(
                        0,
                        f"({source[0] + 1},{source[1] + 1}) 指令连续 "
                        f"{MAX_STALLED_TURNS} 回合未执行，已自动删除",
                    )

    def prune_commands(self) -> None:
        kept: dict[tuple[int, int], list[Command]] = {}
        kept_stalls: dict[tuple[int, int], int] = {}
        kept_waits: dict[tuple[int, int], Command] = {}
        removed: list[tuple[int, int]] = []
        for source, route in self.human_commands.items():
            sx, sy = source
            if not self.board.in_bounds(sx, sy):
                continue
            if self.board.tile(sx, sy).owner != HUMAN:
                continue
            cleaned: list[Command] = []
            previous = source
            for step in route:
                if not self.board.in_bounds(step.tx, step.ty):
                    break
                if self.board.tile(step.tx, step.ty).terrain == MOUNTAIN:
                    break
                if abs(previous[0] - step.tx) + abs(previous[1] - step.ty) != 1:
                    break
                cleaned.append(step)
                previous = (step.tx, step.ty)
            if cleaned:
                kept[source] = cleaned
                kept_stalls[source] = self.route_stalls.get(source, 0)
                if source in self.route_hill_waits:
                    kept_waits[source] = self.route_hill_waits[source]
            else:
                removed.append(source)
                if self.selected == source:
                    self.selected = None
        self.human_commands = kept
        self.route_stalls = kept_stalls
        self.route_hill_waits = kept_waits
        self.command_order = [source for source in self.command_order if source in kept]
        for source in removed:
            self.board.cancel_pending_moves(HUMAN, source)

    def board_viewport(self) -> pygame.Rect:
        return pygame.Rect(
            PADDING,
            HEADER_H,
            WINDOW_W - PANEL_W - PADDING * 3,
            WINDOW_H - HEADER_H - PADDING,
        )

    def board_geometry(self) -> tuple[tuple[float, float], float, pygame.Rect]:
        area = self.board_viewport()
        fit = min(area.width / self.board.width, area.height / self.board.height)
        cell = fit * self.zoom
        board_w = cell * self.board.width
        board_h = cell * self.board.height
        origin_x = area.centerx - self.camera_center[0] * cell
        origin_y = area.centery - self.camera_center[1] * cell

        if board_w <= area.width:
            origin_x = area.centerx - board_w / 2
        else:
            origin_x = clamp(origin_x, area.right - board_w, area.left)
        if board_h <= area.height:
            origin_y = area.centery - board_h / 2
        else:
            origin_y = clamp(origin_y, area.bottom - board_h, area.top)
        return (origin_x, origin_y), cell, area

    def min_zoom(self) -> float:
        area = self.board_viewport()
        fit = min(area.width / self.board.width, area.height / self.board.height)
        reference_cell = min(area.width, area.height) / MIN_VIEW_CELLS
        return clamp(reference_cell / fit, 1.0, MAX_ZOOM)

    def clamp_camera(self) -> None:
        self.camera_center[0] = clamp(self.camera_center[0], 0.0, float(self.board.width))
        self.camera_center[1] = clamp(self.camera_center[1], 0.0, float(self.board.height))

    def cell_at(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        (origin_x, origin_y), cell, area = self.board_geometry()
        if not area.collidepoint(pos):
            return None
        x = int((pos[0] - origin_x) // cell)
        y = int((pos[1] - origin_y) // cell)
        if self.board.in_bounds(x, y):
            return x, y
        return None

    def zoom_at(self, pos: tuple[int, int], factor: float) -> None:
        area = self.board_viewport()
        if not area.collidepoint(pos):
            return
        (origin_x, origin_y), old_cell, _ = self.board_geometry()
        logical_x = (pos[0] - origin_x) / old_cell
        logical_y = (pos[1] - origin_y) / old_cell
        new_zoom = clamp(self.zoom * factor, self.min_zoom(), MAX_ZOOM)
        if abs(new_zoom - self.zoom) < 0.001:
            return
        self.zoom = new_zoom
        fit = min(area.width / self.board.width, area.height / self.board.height)
        new_cell = fit * self.zoom
        self.camera_center[0] = logical_x - (pos[0] - area.centerx) / new_cell
        self.camera_center[1] = logical_y - (pos[1] - area.centery) / new_cell
        self.clamp_camera()

    def route_end(self, source: tuple[int, int]) -> tuple[int, int]:
        route = self.human_commands.get(source, [])
        if route:
            return route[-1].tx, route[-1].ty
        return source

    def valid_next_steps(self, source: tuple[int, int]) -> list[tuple[int, int]]:
        route = self.human_commands.get(source, [])
        if len(route) >= MAX_ROUTE_STEPS:
            return []
        x, y = self.route_end(source)
        steps: list[tuple[int, int]] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = x + dx, y + dy
            if self.board.in_bounds(tx, ty) and self.board.tile(tx, ty).terrain != MOUNTAIN:
                steps.append((tx, ty))
        return steps

    def record_command_state(self) -> None:
        self.command_history.append(
            CommandSnapshot(
                {
                    source: [Command(step.tx, step.ty, step.mode) for step in route]
                    for source, route in self.human_commands.items()
                },
                list(self.command_order),
                dict(self.route_stalls),
                dict(self.route_hill_waits),
                [
                    pending.clone()
                    for pending in self.board.pending_moves
                    if pending.owner == HUMAN
                ],
                self.selected,
            )
        )
        self.command_history = self.command_history[-MAX_ROUTE_STEPS:]

    def undo_last_command(self) -> None:
        if not self.command_history:
            self.logs.insert(0, "Z：没有可撤回的指挥")
            self.logs = self.logs[:8]
            return
        snapshot = self.command_history.pop()
        self.human_commands = snapshot.commands
        self.command_order = snapshot.command_order
        self.route_stalls = snapshot.route_stalls
        self.route_hill_waits = snapshot.route_hill_waits
        self.board.pending_moves = [
            pending
            for pending in self.board.pending_moves
            if pending.owner != HUMAN
        ] + [pending.clone() for pending in snapshot.pending_moves]
        self.selected = snapshot.selected
        self.logs.insert(0, "Z：已撤回上一次指挥")
        self.logs = self.logs[:8]

    def queue_command(
        self,
        source: tuple[int, int],
        target: tuple[int, int],
        mode: str,
        *,
        record_history: bool = True,
    ) -> bool:
        if (
            self.state not in ("PLAYING", "PAUSED")
            or self.ai_takeover
            or self.spectating
            or source in self.route_hill_waits
        ):
            return False
        sx, sy = source
        tile = self.board.tile(sx, sy)
        if tile.owner != HUMAN:
            return False
        if source in self.human_ai.army_controller.positions:
            return False
        route = self.human_commands.get(source, [])
        if len(route) >= MAX_ROUTE_STEPS:
            self.logs.insert(0, f"路线最多 {MAX_ROUTE_STEPS} 步")
            self.logs = self.logs[:8]
            return False
        if target in self.human_ai.army_controller.positions:
            return False
        previous = self.route_end(source)
        if abs(previous[0] - target[0]) + abs(previous[1] - target[1]) != 1:
            return False
        if not self.board.in_bounds(*target) or self.board.tile(*target).terrain == MOUNTAIN:
            return False
        if (target[0], target[1]) == previous:
            return False
        self.mark_human_main_army(source)
        command = Command(target[0], target[1], mode)
        if record_history:
            self.record_command_state()
        if not route:
            self.command_order.append(source)
            route = [command]
            self.human_commands[source] = route
            self.route_stalls[source] = 0
        else:
            route.append(command)
        mode_label = "全军" if mode == "all" else "半军"
        self.logs.insert(0, f"({sx + 1},{sy + 1}) 路线第 {len(route)} 步：{mode_label} → ({target[0] + 1},{target[1] + 1})")
        self.logs = self.logs[:8]
        return True

    def cancel_command(self, source: tuple[int, int]) -> None:
        if source in self.human_commands:
            self.record_command_state()
            self.board.cancel_pending_moves(HUMAN, source)
            self.human_commands.pop(source)
            self.route_stalls.pop(source, None)
            self.route_hill_waits.pop(source, None)
            self.command_order = [item for item in self.command_order if item != source]
            if self.selected == source:
                self.selected = None
            self.logs.insert(0, f"({source[0] + 1},{source[1] + 1}) 命令已取消")
            self.logs = self.logs[:8]

    def command_source_at(self, cell: tuple[int, int] | None) -> tuple[int, int] | None:
        if cell is None:
            return None
        if cell in self.human_commands:
            return cell
        for source, route in self.human_commands.items():
            if route and self.route_end(source) == cell:
                return source
        return None

    def handle_menu_click(self, pos: tuple[int, int]) -> None:
        if self.menu_help_rect.collidepoint(pos):
            self.previous_state = "MENU"
            self.state = "HELP"
            return
        if self.menu_training_rect.collidepoint(pos):
            self.training.start()
            self.state = "TRAINING"
            return
        if self.menu_settings_rect.collidepoint(pos):
            self.state = "SETTINGS"
            return
        if self.menu_start_rect.collidepoint(pos):
            self.seed = random.randrange(1, 1_000_000_000)
            self.new_game()

    def handle_training_click(self, pos: tuple[int, int]) -> None:
        if self.training_preview_rect.collidepoint(pos):
            self.training.cycle_live_slot()
            self.training_minimap_surface = None
            self.training_minimap_scaled_surface = None
            return
        if self.training_pause_rect.collidepoint(pos):
            self.training.toggle()
            return
        if self.training_back_rect.collidepoint(pos):
            self.training.pause()
            self.state = "MENU"

    def handle_settings_click(self, pos: tuple[int, int]) -> None:
        for key, rect in self.menu_difficulty_rects.items():
            if rect.collidepoint(pos):
                self.difficulty_key = key
                return
        for size, rect in self.menu_size_rects.items():
            if rect.collidepoint(pos):
                self.board_size = size
                return
        if self.menu_ai_count_rect.inflate(0, 16).collidepoint(pos):
            self.ai_count_dragging = True
            self.set_ai_count_from_pos(pos[0])
            return
        if self.menu_speed_rect.inflate(0, 16).collidepoint(pos):
            self.speed_dragging = True
            self.set_speed_from_pos(pos[0])
            return
        if self.menu_growth_rect.inflate(0, 16).collidepoint(pos):
            self.growth_dragging = True
            self.set_growth_from_pos(pos[0])
            return
        if self.menu_vision_rect.inflate(0, 16).collidepoint(pos):
            self.vision_dragging = True
            self.set_vision_from_pos(pos[0])
            return
        if self.menu_fullscreen_rect.collidepoint(pos):
            self.toggle_fullscreen()
            return
        for key, rect in self.menu_theme_rects.items():
            if rect.collidepoint(pos):
                self.theme_key = key
                apply_theme(key)
                self.minimap_surface = None
                self.fog_mask = None
                self.fog_scaled_surface = None
                return
        for key, rect in self.menu_fog_rects.items():
            if rect.collidepoint(pos):
                self.fog_enabled = key == "on"
                self.refresh_view_data()
                return
        if self.settings_back_rect.collidepoint(pos):
            self.state = "MENU"

    def set_speed_from_pos(self, x: int) -> None:
        rect = self.menu_speed_rect
        ratio = clamp((x - rect.x) / max(1, rect.width), 0.0, 1.0)
        self.turn_seconds = round(
            MIN_TURN_SECONDS
            + ratio * (MAX_TURN_SECONDS - MIN_TURN_SECONDS),
            1,
        )

    def set_growth_from_pos(self, x: int) -> None:
        rect = self.menu_growth_rect
        ratio = clamp((x - rect.x) / max(1, rect.width), 0.0, 1.0)
        self.growth_interval = int(round(10 + ratio * 40))

    def set_vision_from_pos(self, x: int) -> None:
        rect = self.menu_vision_rect
        ratio = clamp((x - rect.x) / max(1, rect.width), 0.0, 1.0)
        self.vision_radius = int(round(ratio * 5))

    def set_ai_count_from_pos(self, x: int) -> None:
        rect = self.menu_ai_count_rect
        ratio = clamp((x - rect.x) / max(1, rect.width), 0.0, 1.0)
        span = AI_COUNT_MAX - AI_COUNT_MIN
        self.player_count = AI_COUNT_MIN + int(round(ratio * span)) + 1

    def handle_game_click(self, pos: tuple[int, int], button: int) -> None:
        if self.help_rect.collidepoint(pos):
            self.previous_state = self.state
            self.state = "HELP"
            return
        if self.pause_rect.collidepoint(pos) and self.state in ("PLAYING", "PAUSED"):
            self.toggle_pause()
            return
        if self.takeover_rect.collidepoint(pos) and self.state in ("PLAYING", "PAUSED"):
            self.toggle_ai_takeover()
            return
        if self.restart_rect.collidepoint(pos):
            self.new_game(random.randrange(1, 1_000_000_000))
            return
        if self.menu_rect.collidepoint(pos):
            self.state = "MENU"
            return
        if self.minimap_rect.collidepoint(pos):
            self.jump_from_minimap(pos)
            return
        if self.state == "GAME_OVER":
            if self.game_over_new_rect.collidepoint(pos):
                self.new_game(random.randrange(1, 1_000_000_000))
            elif self.game_over_menu_rect.collidepoint(pos):
                self.state = "MENU"
            return
        if self.state not in ("PLAYING", "PAUSED"):
            return

        cell = self.cell_at(pos)
        if cell is None:
            self.selected = None
            return
        x, y = cell
        tile = self.board.tile(x, y)

        if button not in (1, 3):
            return
        mode = "all" if button == 1 else "half"
        if self.spectating or self.ai_takeover:
            return
        controller = getattr(self.human_ai, "army_controller", None)
        army_positions = controller.positions if controller is not None else set()
        if cell in army_positions:
            return
        if self.selected in army_positions:
            self.selected = None
        if self.selected is None:
            if tile.owner == HUMAN:
                self.selected = cell
                self.mark_human_main_army(cell)
            return

        cursor = self.selected
        source = self.command_source_at(cursor) or cursor
        source_valid = (
            self.board.in_bounds(*source)
            and self.board.tile(*source).owner == HUMAN
        )
        if cell == source and source_valid:
            self.selected = source
            self.mark_human_main_army(source)
            return

        adjacent = (
            source_valid
            and abs(cursor[0] - cell[0]) + abs(cursor[1] - cell[1]) == 1
            and tile.terrain != MOUNTAIN
        )
        if adjacent and self.queue_command(source, cell, mode):
            self.mark_human_main_army(source)
            self.selected = cell
            return

        if tile.terrain != MOUNTAIN:
            self.selected = cell
            if tile.owner == HUMAN:
                self.mark_human_main_army(cell)

    def toggle_pause(self) -> None:
        if self.state == "PLAYING":
            self.state = "PAUSED"
            self.logs.insert(0, "自动回合已暂停")
        elif self.state == "PAUSED":
            self.state = "PLAYING"
            self.turn_deadline = time.monotonic() + self.turn_seconds
            self.logs.insert(0, "自动回合继续")
        self.logs = self.logs[:8]

    def toggle_ai_takeover(self) -> None:
        if self.state not in ("PLAYING", "PAUSED") or self.spectating:
            return
        self.ai_takeover = not self.ai_takeover
        if self.ai_takeover:
            self.board.cancel_pending_moves(HUMAN)
            self.human_commands.clear()
            self.command_order.clear()
            self.route_stalls.clear()
            self.route_hill_waits.clear()
            self.command_history.clear()
            self.selected = None
            self.human_ai.army_controller.release_main_army_tag()
            self.logs.insert(0, "AI 已接管你的棋盘")
        else:
            self.logs.insert(0, "已取消 AI 接管，恢复手动操作")
        self.logs = self.logs[:8]

    def jump_from_minimap(self, pos: tuple[int, int]) -> None:
        rect = self.minimap_rect
        if rect.width <= 0 or rect.height <= 0:
            return
        x = (pos[0] - rect.x) / rect.width * self.board.width
        y = (pos[1] - rect.y) / rect.height * self.board.height
        self.camera_center = [clamp(x, 0, self.board.width), clamp(y, 0, self.board.height)]
        self.zoom = max(self.zoom, self.min_zoom() * 1.35)
        self.clamp_camera()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_ESCAPE:
                    if self.state == "HELP":
                        self.state = self.previous_state
                    elif self.state == "SETTINGS":
                        self.state = "MENU"
                    elif self.state == "TRAINING":
                        self.training.pause()
                        self.state = "MENU"
                    elif self.state in ("PLAYING", "PAUSED"):
                        if self.human_commands:
                            self.board.cancel_pending_moves(HUMAN)
                            self.human_commands.clear()
                            self.command_order.clear()
                            self.route_stalls.clear()
                            self.route_hill_waits.clear()
                            self.command_history.clear()
                            self.selected = None
                            self.logs.insert(0, "Esc：已删除场上全部指挥")
                        else:
                            self.selected = None
                    else:
                        self.running = False
                elif event.key == pygame.K_h:
                    if self.state == "HELP":
                        self.state = self.previous_state
                    elif self.state in (
                        "MENU",
                        "SETTINGS",
                        "TRAINING",
                        "PLAYING",
                        "PAUSED",
                        "GAME_OVER",
                    ):
                        self.previous_state = self.state
                        self.state = "HELP"
                elif event.key == pygame.K_SPACE and self.state in ("PLAYING", "PAUSED"):
                    self.toggle_pause()
                elif event.key == pygame.K_r:
                    self.new_game(random.randrange(1, 1_000_000_000))
                elif event.key == pygame.K_z and self.state in ("PLAYING", "PAUSED"):
                    self.undo_last_command()
                elif event.key == pygame.K_u and self.state in ("PLAYING", "PAUSED"):
                    if self.command_order:
                        self.record_command_state()
                        source = self.command_order[-1]
                        route = self.human_commands.get(source, [])
                        if route:
                            route.pop()
                        if not route:
                            self.board.cancel_pending_moves(HUMAN, source)
                            self.human_commands.pop(source, None)
                            self.route_stalls.pop(source, None)
                            self.route_hill_waits.pop(source, None)
                            self.command_order.pop()
                        self.logs.insert(0, "已撤销路线中的最后一步")
                elif event.key == pygame.K_c and self.state in ("PLAYING", "PAUSED"):
                    source = self.command_source_at(self.selected)
                    if source is not None:
                        self.cancel_command(source)
                elif event.key == pygame.K_BACKSPACE and self.state in ("PLAYING", "PAUSED"):
                    self.board.cancel_pending_moves(HUMAN)
                    self.human_commands.clear()
                    self.command_order.clear()
                    self.route_stalls.clear()
                    self.route_hill_waits.clear()
                    self.command_history.clear()
                    self.selected = None
                    self.logs.insert(0, "已清空全部持续命令")
                elif event.key == pygame.K_0:
                    self.zoom = self.min_zoom()
                    human_general = self.board.general_position(HUMAN)
                    if human_general is not None:
                        self.camera_center = [human_general[0] + 0.5, human_general[1] + 0.5]
                    self.clamp_camera()
                elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.state == "MENU":
                    self.new_game(random.randrange(1, 1_000_000_000))
            elif event.type == pygame.MOUSEWHEEL:
                if event.y:
                    self.zoom_at(pygame.mouse.get_pos(), 1.15 ** event.y)
            elif event.type == pygame.MOUSEMOTION:
                self.hover = self.cell_at(event.pos)
                if (
                    self.speed_dragging
                    or self.growth_dragging
                    or self.vision_dragging
                    or self.ai_count_dragging
                ) and self.state == "SETTINGS":
                    if self.speed_dragging:
                        self.set_speed_from_pos(event.pos[0])
                    if self.growth_dragging:
                        self.set_growth_from_pos(event.pos[0])
                    if self.vision_dragging:
                        self.set_vision_from_pos(event.pos[0])
                    if self.ai_count_dragging:
                        self.set_ai_count_from_pos(event.pos[0])
                elif self.pan_last is not None:
                    _, cell, _ = self.board_geometry()
                    dx = event.pos[0] - self.pan_last[0]
                    dy = event.pos[1] - self.pan_last[1]
                    self.camera_center[0] -= dx / cell
                    self.camera_center[1] -= dy / cell
                    self.pan_last = event.pos
                    self.clamp_camera()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2 and self.state in ("PLAYING", "PAUSED"):
                    self.pan_last = event.pos
                elif event.button == 3 and self.state in ("PLAYING", "PAUSED"):
                    self.handle_right_click(event.pos)
                elif self.state == "MENU":
                    self.handle_menu_click(event.pos)
                elif self.state == "SETTINGS":
                    self.handle_settings_click(event.pos)
                elif self.state == "TRAINING":
                    self.handle_training_click(event.pos)
                elif self.state == "HELP":
                    self.state = self.previous_state
                else:
                    self.handle_game_click(event.pos, event.button)
            elif event.type == pygame.MOUSEBUTTONUP:
                if (
                    self.speed_dragging
                    or self.growth_dragging
                    or self.vision_dragging
                    or self.ai_count_dragging
                ):
                    self.speed_dragging = False
                    self.growth_dragging = False
                    self.vision_dragging = False
                    self.ai_count_dragging = False
                elif event.button == 2:
                    self.pan_last = None

    def draw(self) -> None:
        self.handle_events()
        self.screen.fill(BG)
        if self.state == "MENU":
            self.draw_menu()
        elif self.state == "SETTINGS":
            self.draw_settings()
        elif self.state == "TRAINING":
            self.draw_training()
        elif self.state == "HELP":
            if self.previous_state == "MENU":
                self.draw_menu()
            elif self.previous_state == "SETTINGS":
                self.draw_settings()
            elif self.previous_state == "TRAINING":
                self.draw_training()
            else:
                self.draw_game()
            self.draw_help()
        else:
            self.draw_game()
        if self.state == "GAME_OVER":
            self.draw_game_over()
        elif self.state == "PAUSED":
            self.draw_pause_overlay()

    def draw_menu(self) -> None:
        training_stats = self.training.snapshot()
        self.screen.fill(BG)
        for x in range(0, WINDOW_W, 42):
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, 42):
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW_W, y))
        pygame.draw.rect(self.screen, player_color(0), pygame.Rect(0, 0, 8, WINDOW_H))
        pygame.draw.rect(self.screen, player_color(1), pygame.Rect(8, 0, 5, WINDOW_H))

        draw_text(self.screen, self.font_title, "GENERALS", TEXT, (62, 74))
        draw_text(self.screen, self.font_heading, "将军棋 · 自动回合多人战争", player_light(0), (66, 140))
        draw_text(
            self.screen,
            self.font_body,
            "多步路线指挥 · 自由混战 · 标准扩兵轮",
            MUTED,
            (66, 184),
        )

        summary = [
            (
                "AI 难度",
                (
                    "困难 · 训练冠军"
                    if self.difficulty_key == "hard"
                    else DIFFICULTIES[self.difficulty_key].label
                ),
            ),
            ("地图规模", f"{self.board_size} x {self.board_size}"),
            ("AI 数量", f"{self.ai_count} 个（总阵营 {self.player_count}）"),
            ("自动回合", f"{self.turn_seconds:.1f} 秒 / 回合"),
            ("扩兵轮", f"每 {self.growth_interval} 回合空地 +1"),
            (
                "界面与迷雾",
                ("白天" if self.theme_key == "day" else "夜间")
                + " · "
                + ("迷雾开" if self.fog_enabled else "迷雾关"),
            ),
            (
                "进化训练",
                f"{training_stats['completed_matches']} 局 · 第 "
                f"{int(training_stats['generation']) + 1} 代",
            ),
        ]
        for index, (key, value) in enumerate(summary):
            y = 232 + index * 31
            draw_text(self.screen, self.font_small, key, MUTED, (68, y))
            draw_text(self.screen, self.font_body, value, TEXT, (232, y - 2))

        self.menu_start_rect = pygame.Rect(62, 458, 234, 52)
        self.menu_training_rect = pygame.Rect(62, 522, 234, 50)
        self.menu_settings_rect = pygame.Rect(62, 584, 234, 46)
        self.menu_help_rect = pygame.Rect(62, 642, 234, 46)
        draw_button(
            self.screen,
            self.menu_start_rect,
            "开始自动对战",
            self.font_bold,
            hovered=self.menu_start_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            self.screen,
            self.menu_training_rect,
            f"AI 自对弈训练  ({training_stats['completed_matches']})",
            self.font_body,
            hovered=self.menu_training_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            self.screen,
            self.menu_settings_rect,
            "游戏设置",
            self.font_body,
            hovered=self.menu_settings_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            self.screen,
            self.menu_help_rect,
            "操作说明",
            self.font_body,
            hovered=self.menu_help_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_text(
            self.screen,
            self.font_small,
            "Enter 开始 · H 操作说明 · 训练自动保存",
            MUTED,
            (68, 712),
        )

        self.draw_board_preview(
            pygame.Rect(480, 78, 730, 690),
            f"{self.board_size} x {self.board_size} THEATER · {self.player_count}P FREE FOR ALL",
        )

    def draw_settings(self) -> None:
        self.screen.fill(BG)
        for x in range(0, WINDOW_W, 42):
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, 42):
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW_W, y))
        pygame.draw.rect(self.screen, player_color(0), pygame.Rect(0, 0, 8, WINDOW_H))
        pygame.draw.rect(self.screen, player_color(1), pygame.Rect(8, 0, 5, WINDOW_H))

        draw_text(self.screen, self.font_heading, "游戏设置", TEXT, (62, 40))
        draw_text(
            self.screen,
            self.font_small,
            "所有游戏参数集中在设置页，主界面保持简洁。",
            MUTED,
            (62, 82),
        )

        draw_text(self.screen, self.font_bold, "AI 难度", TEXT, (62, 138))
        self.menu_difficulty_rects.clear()
        labels = {"easy": "轻松", "normal": "标准", "hard": "困难"}
        x = 62
        for key in ("easy", "normal", "hard"):
            rect = pygame.Rect(x, 166, 116, 42)
            self.menu_difficulty_rects[key] = rect
            draw_button(
                self.screen,
                rect,
                labels[key],
                self.font_body,
                active=key == self.difficulty_key,
                hovered=rect.collidepoint(pygame.mouse.get_pos()),
            )
            x += 128

        draw_text(self.screen, self.font_bold, "地图规模", TEXT, (62, 226))
        self.menu_size_rects.clear()
        x = 62
        for size in MAP_SIZES:
            rect = pygame.Rect(x, 254, 70, 42)
            self.menu_size_rects[size] = rect
            draw_button(
                self.screen,
                rect,
                str(size),
                self.font_small,
                active=size == self.board_size,
                hovered=rect.collidepoint(pygame.mouse.get_pos()),
            )
            x += 78

        draw_text(self.screen, self.font_bold, "AI 数量", TEXT, (62, 314))
        self.menu_ai_count_rect = pygame.Rect(62, 366, 210, 8)
        pygame.draw.rect(
            self.screen,
            TRACK,
            self.menu_ai_count_rect,
            border_radius=4,
        )
        ai_ratio = (self.ai_count - AI_COUNT_MIN) / (AI_COUNT_MAX - AI_COUNT_MIN)
        ai_fill = self.menu_ai_count_rect.copy()
        ai_fill.width = int(self.menu_ai_count_rect.width * ai_ratio)
        pygame.draw.rect(
            self.screen,
            player_color(1),
            ai_fill,
            border_radius=4,
        )
        ai_knob = (
            self.menu_ai_count_rect.x
            + int(self.menu_ai_count_rect.width * ai_ratio)
        )
        pygame.draw.circle(
            self.screen,
            player_light(1),
            (ai_knob, self.menu_ai_count_rect.centery),
            9,
        )
        pygame.draw.circle(
            self.screen,
            TEXT,
            (ai_knob, self.menu_ai_count_rect.centery),
            9,
            2,
        )
        draw_text(
            self.screen,
            self.font_body,
            f"{self.ai_count} 个 AI",
            TEXT,
            (292, 348),
        )
        draw_text(self.screen, self.font_tiny, "1", MUTED, (62, 380))
        draw_text(self.screen, self.font_tiny, "36", MUTED, (254, 380))
        draw_text(
            self.screen,
            self.font_small,
            f"总阵营 {self.player_count}：你 + {self.ai_count} 个独立 AI",
            MUTED,
            (62, 394),
        )

        draw_text(self.screen, self.font_bold, "自动回合速度", TEXT, (62, 428))
        self.menu_speed_rect = pygame.Rect(62, 460, 210, 8)
        pygame.draw.rect(self.screen, TRACK, self.menu_speed_rect, border_radius=4)
        speed_span = MAX_TURN_SECONDS - MIN_TURN_SECONDS
        speed_ratio = (self.turn_seconds - MIN_TURN_SECONDS) / speed_span
        fill = self.menu_speed_rect.copy()
        fill.width = int(self.menu_speed_rect.width * speed_ratio)
        pygame.draw.rect(self.screen, player_color(HUMAN), fill, border_radius=4)
        knob_x = self.menu_speed_rect.x + int(self.menu_speed_rect.width * speed_ratio)
        pygame.draw.circle(self.screen, player_light(HUMAN), (knob_x, self.menu_speed_rect.centery), 9)
        pygame.draw.circle(self.screen, TEXT, (knob_x, self.menu_speed_rect.centery), 9, 2)
        draw_text(
            self.screen,
            self.font_body,
            f"{self.turn_seconds:.1f} 秒 / 回合",
            TEXT,
            (292, 450),
        )
        draw_text(self.screen, self.font_tiny, "极速 0.0s", MUTED, (62, 474))
        draw_text(self.screen, self.font_tiny, "慢 3.0s", MUTED, (234, 474))

        draw_text(self.screen, self.font_bold, "平地扩兵轮间隔（丘陵为 2 倍）", TEXT, (62, 502))
        self.menu_growth_rect = pygame.Rect(62, 534, 210, 8)
        pygame.draw.rect(self.screen, TRACK, self.menu_growth_rect, border_radius=4)
        growth_ratio = (self.growth_interval - 10) / 40
        growth_fill = self.menu_growth_rect.copy()
        growth_fill.width = int(self.menu_growth_rect.width * growth_ratio)
        pygame.draw.rect(self.screen, player_color(2), growth_fill, border_radius=4)
        growth_knob = self.menu_growth_rect.x + int(self.menu_growth_rect.width * growth_ratio)
        pygame.draw.circle(self.screen, player_light(2), (growth_knob, self.menu_growth_rect.centery), 9)
        pygame.draw.circle(self.screen, TEXT, (growth_knob, self.menu_growth_rect.centery), 9, 2)
        draw_text(
            self.screen,
            self.font_body,
            f"每 {self.growth_interval} 回合空地 +1",
            TEXT,
            (292, 524),
        )
        draw_text(self.screen, self.font_tiny, "10", MUTED, (62, 548))
        draw_text(self.screen, self.font_tiny, "50", MUTED, (258, 548))

        draw_text(self.screen, self.font_bold, "显示模式", TEXT, (330, 576))
        self.menu_fullscreen_rect = pygame.Rect(330, 604, 142, 42)
        draw_button(
            self.screen,
            self.menu_fullscreen_rect,
            "全屏：开" if self.fullscreen else "全屏：关",
            self.font_small,
            active=self.fullscreen,
            hovered=self.menu_fullscreen_rect.collidepoint(
                pygame.mouse.get_pos()
            ),
        )

        draw_text(self.screen, self.font_bold, "界面主题", TEXT, (62, 576))
        self.menu_theme_rects = {
            "day": pygame.Rect(62, 604, 112, 42),
            "night": pygame.Rect(184, 604, 112, 42),
        }
        for key, label in (("day", "白天模式"), ("night", "夜间模式")):
            rect = self.menu_theme_rects[key]
            draw_button(
                self.screen,
                rect,
                label,
                self.font_small,
                active=self.theme_key == key,
                hovered=rect.collidepoint(pygame.mouse.get_pos()),
            )

        draw_text(self.screen, self.font_bold, "战场迷雾", TEXT, (62, 646))
        self.menu_fog_rects = {
            "off": pygame.Rect(62, 674, 112, 42),
            "on": pygame.Rect(184, 674, 112, 42),
        }
        for key, label in (("off", "关闭"), ("on", "开启")):
            rect = self.menu_fog_rects[key]
            draw_button(
                self.screen,
                rect,
                label,
                self.font_small,
                active=(self.fog_enabled == (key == "on")),
                hovered=rect.collidepoint(pygame.mouse.get_pos()),
            )

        draw_text(self.screen, self.font_bold, "视野半径", TEXT, (330, 646))
        draw_text(
            self.screen,
            self.font_body,
            f"{self.vision_radius} 格",
            TEXT,
            (420, 646),
        )
        self.menu_vision_rect = pygame.Rect(330, 690, 130, 8)
        pygame.draw.rect(
            self.screen,
            TRACK,
            self.menu_vision_rect,
            border_radius=4,
        )
        vision_ratio = self.vision_radius / 5
        vision_fill = self.menu_vision_rect.copy()
        vision_fill.width = int(self.menu_vision_rect.width * vision_ratio)
        pygame.draw.rect(
            self.screen,
            player_color(3),
            vision_fill,
            border_radius=4,
        )
        vision_knob = (
            self.menu_vision_rect.x
            + int(self.menu_vision_rect.width * vision_ratio)
        )
        pygame.draw.circle(
            self.screen,
            player_light(3),
            (vision_knob, self.menu_vision_rect.centery),
            9,
        )
        pygame.draw.circle(
            self.screen,
            TEXT,
            (vision_knob, self.menu_vision_rect.centery),
            9,
            2,
        )
        draw_text(self.screen, self.font_tiny, "0", MUTED, (330, 704))
        draw_text(self.screen, self.font_tiny, "5", MUTED, (452, 704))

        self.settings_back_rect = pygame.Rect(62, 730, 234, 46)
        draw_button(
            self.screen,
            self.settings_back_rect,
            "返回主菜单",
            self.font_bold,
            hovered=self.settings_back_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_text(self.screen, self.font_small, "Esc 返回主菜单 · 参数会在新对局中生效", MUTED, (62, 786))

        self.draw_board_preview(
            pygame.Rect(480, 78, 730, 690),
            f"{self.board_size} x {self.board_size} THEATER · AUTO TURN · FREE FOR ALL",
        )

    def draw_training(self) -> None:
        live = self.training.live_view()
        training_stats = self.training.snapshot(live_view=live)
        self.screen.fill(BG)
        for x in range(0, WINDOW_W, 42):
            pygame.draw.line(self.screen, GRID_LINE, (x, 0), (x, WINDOW_H))
        for y in range(0, WINDOW_H, 42):
            pygame.draw.line(self.screen, GRID_LINE, (0, y), (WINDOW_W, y))
        pygame.draw.rect(self.screen, player_color(1), pygame.Rect(0, 0, 8, WINDOW_H))
        pygame.draw.rect(self.screen, player_color(2), pygame.Rect(8, 0, 5, WINDOW_H))

        generation = int(training_stats["generation"])
        completed = int(training_stats["completed_matches"])
        best_fitness = float(training_stats["best_fitness"])
        runtime = "运行中" if training_stats["running"] else "已暂停"
        draw_text(self.screen, self.font_heading, "AI 自对弈进化训练", TEXT, (62, 40))
        draw_text(
            self.screen,
            self.font_small,
            "动态 4/12/20 AI 混战 + 1v1 · 每 5 局更新 · 近 100 局冠军",
            MUTED,
            (62, 82),
        )

        draw_text(self.screen, self.font_bold, "训练状态", TEXT, (62, 130))
        status_lines = [
            ("状态", runtime),
            ("完成对局", f"{completed} 局"),
            ("当前世代", f"第 {generation + 1} 代"),
            (
                "当前对局",
                (
                        f"第 {training_stats['current_turn']} 回合 · "
                        f"存活 {training_stats['current_active']} 人"
                    if training_stats["current_turn"]
                    else (
                        f"并行训练中 · {self.training.parallel_workers} 进程"
                        if self.training.parallel_workers > 1
                        else "准备下一局"
                    )
                ),
            ),
            ("训练配比", "混战 70% · 1v1 专项 30%"),
            (
                "回灌 / 待汇总",
                f"{int(training_stats.get('online_matches', 0))} 局 · "
                f"{int(training_stats.get('pending_matches', 0))}/"
                f"{int(training_stats.get('update_interval_matches', 5))} 局",
            ),
            (
                "计算速度",
                f"{float(training_stats['turns_per_second']):.1f} 回合/秒",
            ),
            (
                "对局吞吐",
                f"{float(training_stats['matches_per_hour']):.1f} 局/小时",
            ),
            (
                "近100局冠军适应度",
                (
                    f"{best_fitness:,.0f}"
                    if best_fitness != float("-inf")
                    else "尚无"
                ),
            ),
        ]
        for index, (label, value) in enumerate(status_lines):
            y = 168 + index * 25
            draw_text(self.screen, self.font_small, label, MUTED, (64, y))
            draw_text(self.screen, self.font_body, value, TEXT, (226, y - 2))

        gene_labels = {
            "aggression": "攻击",
            "defense": "防守",
            "expansion": "扩地",
            "exploration": "探索",
            "search": "搜索",
            "coordination": "协同",
            "home_guard": "守城",
            "consolidation": "集结",
            "risk_tolerance": "冒险",
            "all_in": "梭哈",
            "development": "发育",
            "task_commitment": "任务承诺",
            "local_advantage": "局部优势",
        }
        draw_text(self.screen, self.font_bold, "当前冠军策略", TEXT, (62, 402))
        champion = AIGenome.from_mapping(training_stats["champion"])
        for index, name in enumerate(AIGenome.GENE_NAMES):
            column = index % 2
            row = index // 2
            x = 64 + column * 238
            y = 438 + row * 27
            value = float(getattr(champion, name))
            color = SUCCESS if value >= 1.05 else (
                SELECT_COLOR if value < 0.95 else TEXT
            )
            draw_text(
                self.screen,
                self.font_small,
                f"{gene_labels[name]}  {value:.2f}",
                color,
                (x, y),
            )

        draw_text(self.screen, self.font_bold, "最近完成对局", TEXT, (62, 624))
        recent = list(training_stats["history"])[-4:]
        if not recent:
            draw_text(
                self.screen,
                self.font_small,
                "开始训练后会在这里显示胜者、回合、适应度和种子。",
                MUTED,
                (64, 662),
            )
        else:
            for index, item in enumerate(reversed(recent)):
                winner = item.get("winner")
                winner_text = (
                    f"AI {int(winner) + 1:02d}"
                    if isinstance(winner, int)
                    else "全局排名"
                )
                mode_text = {
                    "duel": "1v1",
                    "online": "实战",
                }.get(str(item.get("mode")), "混战")
                match_count = max(1, int(item.get("match_count", 1)))
                draw_text(
                    self.screen,
                    self.font_small,
                    f"#{item.get('match', '?')}  {mode_text}"
                    f" x{match_count}  {winner_text}  "
                    f"{item.get('turn', '?')} 回合 · 适应度 "
                    f"{float(item.get('best_fitness', 0)):,.0f} · "
                    f"种子 {item.get('seed', '?')}",
                    TEXT if index == 0 else MUTED,
                    (64, 660 + index * 27),
                )

        self.training_pause_rect = pygame.Rect(62, 766, 178, 44)
        self.training_back_rect = pygame.Rect(252, 766, 178, 44)
        draw_button(
            self.screen,
            self.training_pause_rect,
            "暂停训练" if training_stats["running"] else "继续训练",
            self.font_body,
            hovered=self.training_pause_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            self.screen,
            self.training_back_rect,
            "返回主菜单",
            self.font_body,
            hovered=self.training_back_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_text(
            self.screen,
            self.font_tiny,
            "每局自动保存；冠军仅在困难难度和困难档 AI 接管中生效。",
            MUTED,
            (64, 818),
        )

        preview = pygame.Rect(620, 118, 600, 650)
        self.training_preview_rect = preview
        pygame.draw.rect(self.screen, BOARD_BG, preview, border_radius=8)
        pygame.draw.rect(self.screen, LINE, preview, 1, border_radius=8)
        slot = int(training_stats.get("live_slot", 0))
        slot_count = max(1, int(training_stats.get("live_slot_count", 1)))
        live_caption = (
            f"LIVE SELF-PLAY · {live['width']} x {live['height']} · "
            f"{live['active']} AI · 对局 {slot + 1}/{slot_count}"
            if live is not None
            else (
                f"LIVE SELF-PLAY · 对局 {slot + 1}/{slot_count} · "
                "等待快照"
            )
        )
        draw_text(
            self.screen,
            self.font_small,
            live_caption,
            MUTED,
            (preview.x + 20, preview.y + 18),
        )
        self._draw_training_board(
            pygame.Rect(
                preview.x + 42,
                preview.y + 66,
                preview.width - 84,
                preview.height - 106,
            ),
            view=live,
        )

    def _draw_training_board(
        self,
        rect: pygame.Rect,
        *,
        view: dict[str, object] | None = None,
    ) -> None:
        if view is None:
            view = self.training.live_view()
        if view is None:
            draw_text(
                self.screen,
                self.font_body,
                (
                    "多个训练进程正在并行对局..."
                    if self.training.parallel_workers > 1
                    else "正在生成下一张地图..."
                ),
                MUTED,
                rect.center,
                center=True,
            )
            return
        width = int(view["width"])
        height = int(view["height"])
        tiles = view["tiles"]
        if (
            width <= 0
            or height <= 0
            or not isinstance(tiles, (list, tuple))
            or len(tiles) != width * height
        ):
            draw_text(
                self.screen,
                self.font_body,
                "正在等待完整训练快照...",
                MUTED,
                rect.center,
                center=True,
            )
            return
        cell = max(
            1,
            min(rect.width // width, rect.height // height),
        )
        cache_key = (
            view.get("stamp"),
            self.theme_key,
            width,
            height,
            cell,
        )
        if (
            self.training_minimap_surface is None
            or self.training_minimap_key != cache_key
        ):
            minimap = pygame.Surface((width, height))
            for index, (terrain, owner) in enumerate(tiles):
                x = index % width
                y = index // width
                if terrain == MOUNTAIN:
                    color = MOUNTAIN_COLOR
                elif terrain == HILL:
                    base = LAND_ALT if (x + y) % 2 else LAND
                    color = (
                        mix(base, player_color(owner), 0.68)
                        if owner >= 0
                        else base
                    )
                elif terrain == CITY and owner == NEUTRAL:
                    color = CITY_COLOR
                elif owner >= 0:
                    color = mix(LAND, player_color(owner), 0.68)
                else:
                    color = LAND_ALT if (x + y) % 2 else LAND
                minimap.set_at((x, y), color)
            self.training_minimap_surface = minimap
            self.training_minimap_key = cache_key

        board_width = cell * width
        board_height = cell * height
        start_x = rect.centerx - board_width // 2
        start_y = rect.centery - board_height // 2
        if self.training_minimap_scaled_key != cache_key:
            self.training_minimap_scaled_surface = pygame.transform.scale(
                self.training_minimap_surface,
                (board_width, board_height),
            )
            self.training_minimap_scaled_key = cache_key
        self.screen.blit(
            self.training_minimap_scaled_surface,
            (start_x, start_y),
        )
        pygame.draw.rect(
            self.screen,
            LINE,
            (
                start_x - 1,
                start_y - 1,
                board_width + 2,
                board_height + 2,
            ),
            1,
        )

    def draw_board_preview(self, preview: pygame.Rect, caption: str) -> None:
        pygame.draw.rect(self.screen, BOARD_BG, preview, border_radius=8)
        pygame.draw.rect(self.screen, LINE, preview, 1, border_radius=8)
        preview_board = pygame.Rect(preview.x + 20, preview.y + 20, preview.width - 40, 620)
        cell = max(4, preview_board.width // 51)
        board_px = cell * 51
        start_x = preview_board.centerx - board_px // 2
        start_y = preview_board.centery - board_px // 2
        mountains: set[tuple[int, int]] = set()
        terrain_rng = random.Random(
            self.board_size * 7919 + self.player_count * 104729
        )
        for _ in range(10):
            gx = terrain_rng.randrange(4, 47)
            gy = terrain_rng.randrange(4, 47)
            dx, dy = terrain_rng.choice(
                ((1, 0), (-1, 0), (0, 1), (0, -1))
            )
            for _step in range(terrain_rng.randint(7, 20)):
                if 0 <= gx < 51 and 0 <= gy < 51:
                    mountains.add((gx, gy))
                if terrain_rng.random() < 0.18:
                    dx, dy = terrain_rng.choice(
                        ((1, 0), (-1, 0), (0, 1), (0, -1))
                    )
                gx += dx
                gy += dy
        for gy in range(51):
            for gx in range(51):
                rect = pygame.Rect(start_x + gx * cell + 1, start_y + gy * cell + 1, cell - 2, cell - 2)
                color = LAND_ALT if (gx + gy) % 2 else LAND
                if (gx, gy) in mountains:
                    color = MOUNTAIN_COLOR
                if (gx * 5 + gy * 11) % 181 == 0:
                    color = CITY_COLOR
                pygame.draw.rect(self.screen, color, rect)
        preview_rng = random.Random(self.board_size * 1009 + self.player_count * 9176)
        for index in range(self.player_count):
            angle = preview_rng.uniform(0.0, math.tau)
            radius = preview_rng.uniform(4.0, 22.0)
            px = int(start_x + (25.5 + math.cos(angle) * radius) * cell)
            py = int(start_y + (25.5 + math.sin(angle) * radius) * cell)
            pygame.draw.circle(self.screen, player_color(index), (px, py), max(3, cell // 2 + 1))
        draw_text(
            self.screen,
            self.font_small,
            caption,
            MUTED,
            (preview.x + 22, preview.bottom - 30),
        )

    def draw_game(self) -> None:
        self.screen.fill(BG)
        draw_text(self.screen, self.font_heading, "GENERALS", TEXT, (PADDING, 13))
        if self.spectating:
            status = "观众模式"
        elif self.ai_takeover:
            status = "AI 接管中"
        else:
            status = "自动回合进行中" if self.state == "PLAYING" else "已暂停"
        draw_text(
            self.screen,
            self.font_small,
            f"第 {self.board.turn} 回合 · {status} · 存活 {len(self.board.active_players)}/{self.player_count}",
            MUTED,
            (PADDING + 176, 25),
        )
        _, cell, _ = self.board_geometry()
        draw_text(
            self.screen,
            self.font_small,
            f"{self.board.width}x{self.board.height} · {self.player_count}P · {self.turn_seconds:.1f}s · 缩放 {round(self.zoom / self.min_zoom() * 100)}%",
            MUTED,
            (WINDOW_W - PADDING - 48, 22),
            right=True,
        )
        self.help_rect = pygame.Rect(WINDOW_W - PADDING - 36, 14, 36, 36)
        draw_button(
            self.screen,
            self.help_rect,
            "?",
            self.font_bold,
            hovered=self.help_rect.collidepoint(pygame.mouse.get_pos()),
        )
        self.draw_board()
        self.draw_panel()
        if self.toast_message and time.monotonic() < self.toast_until:
            color = SUCCESS if self.toast_success else player_color(1)
            toast = pygame.Rect(
                PADDING + 245,
                HEADER_H + 16,
                430,
                42,
            )
            pygame.draw.rect(self.screen, PANEL, toast, border_radius=7)
            pygame.draw.rect(self.screen, color, toast, 2, border_radius=7)
            draw_text(
                self.screen,
                self.font_small,
                self.toast_message,
                TEXT,
                toast.center,
                center=True,
            )

    def draw_board(self) -> None:
        (origin_x, origin_y), cell, area = self.board_geometry()
        draw_origin_x = int(origin_x)
        draw_origin_y = int(origin_y)
        old_clip = self.screen.get_clip()
        self.screen.set_clip(area)
        pygame.draw.rect(self.screen, BOARD_BG, area)
        board_rect = pygame.Rect(
            draw_origin_x,
            draw_origin_y,
            int(cell * self.board.width),
            int(cell * self.board.height),
        )
        pygame.draw.rect(self.screen, BOARD_BG, board_rect)

        x_start = max(0, int(math.floor(-origin_x / cell)))
        y_start = max(0, int(math.floor(-origin_y / cell)))
        x_end = min(self.board.width, int(math.ceil((area.right - origin_x) / cell)) + 1)
        y_end = min(self.board.height, int(math.ceil((area.bottom - origin_y) / cell)) + 1)

        visible = self.visible_human
        cache_pixels = self.board.width * self.board.height * cell * cell
        board_cache_key = (
            self.board_render_revision,
            self.theme_key,
            int(round(cell * 1000)),
        )
        use_board_cache = cache_pixels <= BOARD_CACHE_MAX_PIXELS
        if use_board_cache:
            if (
                self.board_cache_surface is None
                or self.board_cache_key != board_cache_key
            ):
                self.board_cache_surface = self._build_board_cache(cell, visible)
                self.board_cache_key = board_cache_key
            self.screen.blit(self.board_cache_surface, (draw_origin_x, draw_origin_y))
        else:
            chunk_prefix = (
                self.board_render_revision,
                self.theme_key,
                int(round(cell * 1000)),
            )
            if self.board_chunk_key != chunk_prefix:
                self.board_chunk_key = chunk_prefix
                self.board_chunk_cache.clear()
            self._draw_board_chunks(
                draw_origin_x,
                draw_origin_y,
                cell,
                visible,
                x_start,
                y_start,
                x_end,
                y_end,
            )

        if self.fog_mask is not None:
            if (
                self.fog_scaled_surface is None
                or self.fog_scaled_size != board_rect.size
            ):
                self.fog_scaled_surface = pygame.transform.scale(
                    self.fog_mask,
                    board_rect.size,
                )
                self.fog_scaled_size = board_rect.size
            self.screen.blit(self.fog_scaled_surface, board_rect.topleft)

        if cell >= 10:
            for y in range(y_start, y_end):
                for x in range(x_start, x_end):
                    tile = self.board.tile(x, y)
                    is_visible = (x, y) in visible
                    show_army = is_visible and tile.army > 0
                    if tile.terrain == CITY and tile.owner == NEUTRAL:
                        show_army = True
                    if show_army:
                        rect = pygame.Rect(
                            draw_origin_x + int(x * cell),
                            draw_origin_y + int(y * cell),
                            max(1, int(math.ceil(cell))),
                            max(1, int(math.ceil(cell))),
                        )
                        self.draw_army(tile.army, tile.terrain, rect, cell)

        for ai in (self.human_ai, *self.ais.values()):
            controller = getattr(ai, "army_controller", None)
            if controller is None:
                continue
            for army in controller.armies:
                lx, ly = army.position
                if not self.board.in_bounds(lx, ly) or (lx, ly) not in visible:
                    continue
                army_tile = self.board.tile(lx, ly)
                if (
                    army_tile.owner != army.player
                    or army_tile.terrain == GENERAL
                    or army_tile.army <= 2
                ):
                    continue
                rect = pygame.Rect(
                    draw_origin_x + int(lx * cell),
                    draw_origin_y + int(ly * cell),
                    max(1, int(math.ceil(cell))),
                    max(1, int(math.ceil(cell))),
                )
                self.draw_independent_army_label(
                    rect,
                    cell,
                    text=f"军{army.unit_id}",
                )

        if self.selected is not None:
            sx, sy = self.selected
            rect = pygame.Rect(draw_origin_x + int(sx * cell), draw_origin_y + int(sy * cell), int(cell) + 1, int(cell) + 1)
            pygame.draw.rect(self.screen, SELECT_COLOR, rect.inflate(-2, -2), 3)
            route_end = self.route_end(self.selected)
            if route_end != self.selected:
                end_rect = pygame.Rect(
                    draw_origin_x + int(route_end[0] * cell),
                    draw_origin_y + int(route_end[1] * cell),
                    int(cell) + 1,
                    int(cell) + 1,
                )
                pygame.draw.rect(self.screen, SELECT_COLOR, end_rect.inflate(-5, -5), 2)
            for tx, ty in self.valid_next_steps(self.selected):
                target = pygame.Rect(draw_origin_x + int(tx * cell), draw_origin_y + int(ty * cell), int(cell) + 1, int(cell) + 1)
                pygame.draw.rect(self.screen, player_light(HUMAN), target.inflate(-2, -2), 2)
                pygame.draw.circle(self.screen, player_light(HUMAN), target.center, max(2, int(cell * 0.07)))

        if self.hover is not None and area.collidepoint(pygame.mouse.get_pos()):
            hx, hy = self.hover
            rect = pygame.Rect(draw_origin_x + int(hx * cell), draw_origin_y + int(hy * cell), int(cell) + 1, int(cell) + 1)
            pygame.draw.rect(self.screen, (235, 240, 247), rect.inflate(-1, -1), 2)

        self.draw_move_arrows(draw_origin_x, draw_origin_y, cell, area)

        for x, y, started in self.recent_captures:
            age = (time.monotonic() - started) / 0.75
            if age >= 1:
                continue
            rect = pygame.Rect(draw_origin_x + int(x * cell), draw_origin_y + int(y * cell), int(cell), int(cell))
            color = mix(SELECT_COLOR, BG, age)
            pygame.draw.circle(self.screen, color, rect.center, int(cell * (0.22 + age * 0.55)), max(1, 3 - int(age * 3)))

        self.screen.set_clip(old_clip)
        pygame.draw.rect(self.screen, LINE, area, 1)

    def _draw_board_chunks(
        self,
        draw_origin_x: int,
        draw_origin_y: int,
        cell: float,
        visible: set[tuple[int, int]],
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
    ) -> None:
        chunk_x_start = (x_start // BOARD_CHUNK_TILES) * BOARD_CHUNK_TILES
        chunk_y_start = (y_start // BOARD_CHUNK_TILES) * BOARD_CHUNK_TILES
        for chunk_y in range(chunk_y_start, y_end, BOARD_CHUNK_TILES):
            for chunk_x in range(chunk_x_start, x_end, BOARD_CHUNK_TILES):
                key = (chunk_x, chunk_y)
                surface = self.board_chunk_cache.get(key)
                if surface is None:
                    surface = self._build_board_chunk(
                        chunk_x,
                        chunk_y,
                        cell,
                        visible,
                    )
                    self.board_chunk_cache[key] = surface
                    if len(self.board_chunk_cache) > BOARD_CHUNK_CACHE_LIMIT:
                        self.board_chunk_cache.pop(
                            next(iter(self.board_chunk_cache))
                        )
                self.screen.blit(
                    surface,
                    (
                        draw_origin_x + int(chunk_x * cell),
                        draw_origin_y + int(chunk_y * cell),
                    ),
                )

    def _build_board_chunk(
        self,
        start_x: int,
        start_y: int,
        cell: float,
        visible: set[tuple[int, int]],
    ) -> pygame.Surface:
        width = min(BOARD_CHUNK_TILES, self.board.width - start_x)
        height = min(BOARD_CHUNK_TILES, self.board.height - start_y)
        surface = pygame.Surface(
            (
                max(1, int(math.ceil(cell * width)) + 1),
                max(1, int(math.ceil(cell * height)) + 1),
            )
        ).convert()
        surface.fill(BOARD_BG)
        tile_size = max(1, int(math.ceil(cell)) + 1)
        for local_y in range(height):
            y = start_y + local_y
            for local_x in range(width):
                x = start_x + local_x
                tile = self.board.tile(x, y)
                rect = pygame.Rect(
                    int(local_x * cell),
                    int(local_y * cell),
                    tile_size,
                    tile_size,
                )
                self.draw_tile_base(
                    tile,
                    rect,
                    (x, y) in visible,
                    x,
                    y,
                    cell,
                    surface=surface,
                )
        return surface

    def _build_board_cache(
        self,
        cell: float,
        visible: set[tuple[int, int]],
    ) -> pygame.Surface:
        surface = pygame.Surface(
            (
                max(1, int(math.ceil(cell * self.board.width))),
                max(1, int(math.ceil(cell * self.board.height))),
            )
        ).convert()
        surface.fill(BOARD_BG)
        tile_size = max(1, int(math.ceil(cell)) + 1)
        for y, row in enumerate(self.board.grid):
            for x, tile in enumerate(row):
                rect = pygame.Rect(
                    int(x * cell),
                    int(y * cell),
                    tile_size,
                    tile_size,
                )
                self.draw_tile_base(
                    tile,
                    rect,
                    (x, y) in visible,
                    x,
                    y,
                    cell,
                    surface=surface,
                )
        return surface

    def draw_tile_base(
        self,
        tile: object,
        rect: pygame.Rect,
        visible: bool,
        x: int,
        y: int,
        cell: float,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target_surface = self.screen if surface is None else surface
        if tile.terrain == MOUNTAIN:
            pygame.draw.rect(target_surface, MOUNTAIN_COLOR, rect.inflate(-2, -2))
            if cell >= 13:
                self.draw_mountain(rect, surface=target_surface)
        else:
            if visible and tile.owner >= 0:
                fill = mix(LAND, player_color(tile.owner), 0.60)
            elif tile.terrain == HILL:
                fill = LAND_ALT if (x + y) % 2 else LAND
            elif tile.terrain == CITY:
                fill = CITY_DARK
            else:
                fill = LAND_ALT if (x + y) % 2 else LAND
            pygame.draw.rect(target_surface, fill, rect.inflate(-2, -2))
            if tile.terrain == HILL and cell >= 10:
                self.draw_hill(rect, surface=target_surface)
            elif tile.terrain == CITY and cell >= 10:
                self.draw_city(rect, surface=target_surface)
            elif tile.terrain == GENERAL and visible and tile.owner >= 0 and cell >= 13:
                self.draw_general(rect, tile.owner, surface=target_surface)

        if visible and tile.owner >= 0:
            border = player_light(tile.owner)
            pygame.draw.rect(target_surface, border, rect.inflate(-2, -2), 2)
        elif tile.terrain == CITY:
            pygame.draw.rect(target_surface, CITY_LIGHT, rect.inflate(-2, -2), 2)

    def draw_army(self, army: int, terrain: int, rect: pygame.Rect, cell: float) -> None:
        size = max(9, min(17, int(cell * 0.42)))
        font = self.cached_font(size, bold=army >= 10)
        center = rect.center
        if terrain == GENERAL:
            center = (rect.centerx, rect.bottom - max(7, rect.height // 7))
        text = str(army)
        if cell < 15 and army >= 1_000:
            text = f"{army / 1_000:.1f}k"
        bold = army >= 10
        key = (text, TEXT, TEXT_SHADOW, size, bold)
        label = self.text_cache.get(key)
        if label is None:
            label = font.render(text, True, TEXT)
            shadow = font.render(text, True, TEXT_SHADOW)
            combined = pygame.Surface(
                (shadow.get_width() + 1, shadow.get_height() + 1),
                pygame.SRCALPHA,
            )
            combined.blit(shadow, (1, 1))
            combined.blit(
                label,
                (
                    max(0, (combined.get_width() - label.get_width()) // 2),
                    max(0, (combined.get_height() - label.get_height()) // 2),
                ),
            )
            label = combined
            self.text_cache[key] = label
        self.screen.blit(label, label.get_rect(center=center))

    def draw_independent_army_label(
        self,
        rect: pygame.Rect,
        cell: float,
        *,
        text: str,
    ) -> None:
        pygame.draw.rect(
            self.screen,
            INDEPENDENT_ARMY_LABEL_COLOR,
            rect.inflate(-2, -2),
            3,
        )
        if cell < 11:
            return
        size = max(10, min(15, int(cell * 0.42)))
        font = self.cached_font(size, bold=True)
        label = font.render(text, True, (255, 255, 255))
        badge = pygame.Rect(
            rect.x + 2,
            rect.y + 2,
            max(label.get_width() + 6, 10),
            max(label.get_height() + 3, 10),
        )
        pygame.draw.rect(self.screen, INDEPENDENT_ARMY_LABEL_COLOR, badge)
        self.screen.blit(label, label.get_rect(center=badge.center))

    def cached_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (str(bold), (0, 0, 0), size)
        font = self.font_cache.get(key)
        if font is None:
            font = load_font(size, bold=bold)
            self.font_cache[key] = font
        return font

    def draw_mountain(
        self,
        rect: pygame.Rect,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target_surface = self.screen if surface is None else surface
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        peak = (x + w // 2, y + int(h * 0.22))
        left = (x + int(w * 0.20), y + int(h * 0.78))
        right = (x + int(w * 0.82), y + int(h * 0.78))
        pygame.draw.polygon(target_surface, MOUNTAIN_LIGHT, (peak, left, right))
        pygame.draw.lines(target_surface, MOUNTAIN_EDGE, False, (left, peak, right), 1)

    def draw_hill(
        self,
        rect: pygame.Rect,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target_surface = self.screen if surface is None else surface
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        first = (
            (x + int(w * 0.08), y + int(h * 0.72)),
            (x + int(w * 0.34), y + int(h * 0.34)),
            (x + int(w * 0.60), y + int(h * 0.72)),
        )
        second = (
            (x + int(w * 0.40), y + int(h * 0.78)),
            (x + int(w * 0.68), y + int(h * 0.46)),
            (x + int(w * 0.94), y + int(h * 0.78)),
        )
        pygame.draw.lines(target_surface, HILL_LIGHT, False, first, 2)
        pygame.draw.lines(target_surface, HILL_EDGE, False, second, 1)

    def draw_city(
        self,
        rect: pygame.Rect,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target_surface = self.screen if surface is None else surface
        center = rect.center
        radius = max(4, rect.width // 7)
        pygame.draw.circle(target_surface, CITY_COLOR, center, radius + 1)
        pygame.draw.circle(target_surface, CITY_LIGHT, center, radius, 1)
        for angle in (0, 90, 180, 270):
            radians = math.radians(angle)
            start = (center[0] + int(math.cos(radians) * (radius + 2)), center[1] + int(math.sin(radians) * (radius + 2)))
            end = (center[0] + int(math.cos(radians) * (radius + 6)), center[1] + int(math.sin(radians) * (radius + 6)))
            pygame.draw.line(target_surface, CITY_LIGHT, start, end, 1)

    def draw_general(
        self,
        rect: pygame.Rect,
        owner: int,
        *,
        surface: pygame.Surface | None = None,
    ) -> None:
        target_surface = self.screen if surface is None else surface
        color = player_light(owner)
        x, y, w, h = rect.x, rect.y, rect.width, rect.height
        points = [
            (x + int(w * 0.24), y + int(h * 0.56)),
            (x + int(w * 0.29), y + int(h * 0.24)),
            (x + int(w * 0.44), y + int(h * 0.40)),
            (x + int(w * 0.50), y + int(h * 0.18)),
            (x + int(w * 0.57), y + int(h * 0.40)),
            (x + int(w * 0.72), y + int(h * 0.24)),
            (x + int(w * 0.77), y + int(h * 0.56)),
        ]
        pygame.draw.polygon(target_surface, ICON_DARK, points)
        pygame.draw.lines(target_surface, color, True, points, 2)

    def draw_move_arrows(
        self,
        origin_x: float,
        origin_y: float,
        cell: float,
        area: pygame.Rect,
    ) -> None:
        if not self.human_commands:
            return
        if self.arrow_overlay is None or self.arrow_overlay.get_size() != area.size:
            self.arrow_overlay = pygame.Surface(area.size, pygame.SRCALPHA)
        overlay = self.arrow_overlay
        overlay.fill((0, 0, 0, 0))
        for source, route in self.human_commands.items():
            sx, sy = source
            if self.board.tile(sx, sy).owner != HUMAN:
                continue
            segments: list[Command] = []
            wait_target = self.route_hill_waits.get(source)
            if wait_target is not None:
                segments.append(wait_target)
            segments.extend(route)
            points = [(sx, sy)] + [(step.tx, step.ty) for step in segments]
            for index in range(len(points) - 1):
                start_tile = points[index]
                end_tile = points[index + 1]
                start = (
                    int(origin_x + start_tile[0] * cell + cell / 2 - area.x),
                    int(origin_y + start_tile[1] * cell + cell / 2 - area.y),
                )
                end = (
                    int(origin_x + end_tile[0] * cell + cell / 2 - area.x),
                    int(origin_y + end_tile[1] * cell + cell / 2 - area.y),
                )
                if (
                    start[0] < -30
                    or start[1] < -30
                    or start[0] > area.width + 30
                    or start[1] > area.height + 30
                ) and (
                    end[0] < -30
                    or end[1] < -30
                    or end[0] > area.width + 30
                    or end[1] > area.height + 30
                ):
                    continue
                step = segments[index]
                width = (
                    max(5, min(12, int(cell * 0.30)))
                    if step.mode == "all"
                    else max(4, min(10, int(cell * 0.24)))
                )
                color = player_light(HUMAN) if index == 0 else mix(player_light(HUMAN), BG, min(0.55, index * 0.12))
                stall = self.route_stalls.get(source, 0)
                if stall:
                    color = mix(color, MUTED, min(0.65, stall * 0.09))
                self.draw_arrow(overlay, start, end, color, width)
                if cell >= 16 and index > 0:
                    step_center = (
                        int(origin_x + end_tile[0] * cell + cell * 0.22 - area.x),
                        int(origin_y + end_tile[1] * cell + cell * 0.22 - area.y),
                    )
                    radius = max(8, int(cell * 0.20))
                    pygame.draw.circle(overlay, BG, step_center, radius)
                    pygame.draw.circle(overlay, color, step_center, radius, 2)
                    step_font = self.cached_font(
                        max(10, int(cell * 0.28)),
                        bold=True,
                    )
                    step_label = step_font.render(str(index + 1), True, TEXT)
                    overlay.blit(step_label, step_label.get_rect(center=step_center))
        self.screen.blit(overlay, area.topleft)

    @staticmethod
    def draw_arrow(
        surface: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
        width: int,
    ) -> None:
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, math.hypot(dx, dy))
        ux, uy = dx / length, dy / length
        line_start = (int(start[0] + ux * length * 0.24), int(start[1] + uy * length * 0.24))
        line_end = (int(start[0] + ux * length * 0.68), int(start[1] + uy * length * 0.68))
        pygame.draw.line(surface, color, line_start, line_end, width)
        head = max(10, min(24, int(width * 2.6)))
        left = (
            end[0] - int(ux * head) - int(uy * head * 0.55),
            end[1] - int(uy * head) + int(ux * head * 0.55),
        )
        right = (
            end[0] - int(ux * head) + int(uy * head * 0.55),
            end[1] - int(uy * head) - int(ux * head * 0.55),
        )
        tip = (
            int(start[0] + ux * length * 0.78),
            int(start[1] + uy * length * 0.78),
        )
        pygame.draw.polygon(surface, color, (tip, left, right))

    def draw_panel(self) -> None:
        panel = pygame.Rect(WINDOW_W - PANEL_W - PADDING, HEADER_H, PANEL_W, WINDOW_H - HEADER_H - PADDING)
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=8)
        pygame.draw.rect(self.screen, LINE, panel, 1, border_radius=8)
        x = panel.x + 18
        y = panel.y + 16
        panel_title = "战局观战" if self.spectating else "战局指挥"
        draw_text(self.screen, self.font_bold, panel_title, TEXT, (x, y))
        y += 34

        self.minimap_rect = pygame.Rect(x, y, panel.width - 36, 176)
        self.draw_minimap(self.minimap_rect)
        y += 190

        remaining = max(0.0, self.turn_deadline - time.monotonic())
        if self.state == "PAUSED":
            label = "自动回合已暂停"
            ratio = 0.0
        elif self.turn_seconds <= 0.0:
            label = "自动回合：极速模式"
            ratio = 1.0
        else:
            label = f"下一次自动结算  {remaining:0.1f}s"
            ratio = clamp(remaining / self.turn_seconds, 0.0, 1.0)
        draw_text(self.screen, self.font_small, label, MUTED, (x, y))
        bar = pygame.Rect(x, y + 22, panel.width - 36, 6)
        pygame.draw.rect(self.screen, TRACK, bar, border_radius=3)
        fill = bar.copy()
        fill.width = int(bar.width * ratio)
        pygame.draw.rect(self.screen, player_color(HUMAN), fill, border_radius=3)
        y += 42

        half = (panel.width - 46) // 2
        self.pause_rect = pygame.Rect(x, y, half, 38)
        self.restart_rect = pygame.Rect(x + half + 10, y, half, 38)
        pause_label = "继续" if self.state == "PAUSED" else "暂停 SPACE"
        draw_button(
            self.screen,
            self.pause_rect,
            pause_label,
            self.font_small,
            hovered=self.pause_rect.collidepoint(pygame.mouse.get_pos()),
            enabled=self.state in ("PLAYING", "PAUSED"),
        )
        draw_button(
            self.screen,
            self.restart_rect,
            "重开 R",
            self.font_small,
            hovered=self.restart_rect.collidepoint(pygame.mouse.get_pos()),
        )
        y += 48

        self.takeover_rect = pygame.Rect(x, y, panel.width - 36, 36)
        if self.spectating:
            takeover_label = "观众模式"
        else:
            takeover_label = "取消 AI 接管" if self.ai_takeover else "AI 接管我的棋盘"
        draw_button(
            self.screen,
            self.takeover_rect,
            takeover_label,
            self.font_small,
            active=self.ai_takeover,
            hovered=self.takeover_rect.collidepoint(pygame.mouse.get_pos()),
            enabled=self.state in ("PLAYING", "PAUSED") and not self.spectating,
        )
        y += 44

        draw_text(self.screen, self.font_bold, "玩家排名（前 8）", TEXT, (x, y))
        y += 28
        ranking = sorted(
            range(self.player_count),
            key=lambda player: (
                player not in self.board.active_players,
                -self.player_stats[player]["tiles"],
                -self.player_stats[player]["armies"],
                player,
            ),
        )
        row_h = 21
        for player in ranking[:8]:
            stats = self.player_stats[player]
            active = player in self.board.active_players
            row = pygame.Rect(x, y, panel.width - 36, row_h - 2)
            if player == HUMAN:
                pygame.draw.rect(self.screen, HIGHLIGHT_ROW, row, border_radius=4)
            pygame.draw.circle(self.screen, player_color(player), (row.x + 8, row.centery), 5)
            name = player_label(player)
            color = TEXT if active else (104, 111, 123)
            draw_text(self.screen, self.font_tiny, name, color, (row.x + 19, row.y + 3))
            detail = f"{stats['tiles']}格  {stats['armies']}兵"
            if not active:
                detail = "已淘汰"
            draw_text(self.screen, self.font_tiny, detail, color, (row.right - 4, row.y + 3), right=True)
            y += row_h

        y += 4
        self.menu_rect = pygame.Rect(x, y, panel.width - 36, 34)
        draw_button(
            self.screen,
            self.menu_rect,
            "返回主菜单",
            self.font_small,
            hovered=self.menu_rect.collidepoint(pygame.mouse.get_pos()),
        )
        y += 42
        draw_text(self.screen, self.font_tiny, "相邻单击：左键全军 / 右键半军", MUTED, (x, y))
        draw_text(self.screen, self.font_tiny, "双击右键己方格：创建独立军队", MUTED, (x, y + 18))
        draw_text(self.screen, self.font_tiny, "右键一次军队：取消头衔并接管", MUTED, (x, y + 36))
        draw_text(self.screen, self.font_tiny, "C取消 · Z撤回 · Space暂停 · Esc清空", MUTED, (x, y + 54))

    def draw_minimap(self, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, BOARD_SURFACE, rect, border_radius=6)
        pygame.draw.rect(self.screen, LINE, rect, 1, border_radius=6)
        inner = rect.inflate(-8, -8)
        cell = max(1, min(inner.width // self.board.width, inner.height // self.board.height))
        board_w = cell * self.board.width
        board_h = cell * self.board.height
        start_x = inner.centerx - board_w // 2
        start_y = inner.centery - board_h // 2
        if self.minimap_surface is None or self.minimap_surface_size != inner.size:
            surface = pygame.Surface(inner.size).convert()
            surface.fill(BOARD_SURFACE)
            local_start_x = start_x - inner.x
            local_start_y = start_y - inner.y
            for y, row in enumerate(self.board.grid):
                for x, tile in enumerate(row):
                    visible = (x, y) in self.visible_human
                    if tile.terrain == MOUNTAIN:
                        color = MOUNTAIN_COLOR
                    elif tile.terrain == HILL:
                        color = LAND_ALT if (x + y) % 2 else LAND
                    elif tile.terrain == CITY and tile.owner == NEUTRAL:
                        color = CITY_COLOR
                    elif visible and tile.owner >= 0:
                        color = player_color(tile.owner)
                    elif tile.owner == HUMAN:
                        color = player_color(HUMAN)
                    else:
                        color = MINIMAP_FOG
                    pygame.draw.rect(
                        surface,
                        color,
                        pygame.Rect(
                            local_start_x + x * cell,
                            local_start_y + y * cell,
                            cell,
                            cell,
                        ),
                    )
            self.minimap_surface = surface
            self.minimap_surface_size = inner.size
        self.screen.blit(self.minimap_surface, inner.topleft)

        (origin_x, origin_y), board_cell, area = self.board_geometry()
        visible_left = clamp((area.left - origin_x) / board_cell, 0, self.board.width)
        visible_top = clamp((area.top - origin_y) / board_cell, 0, self.board.height)
        visible_right = clamp((area.right - origin_x) / board_cell, 0, self.board.width)
        visible_bottom = clamp((area.bottom - origin_y) / board_cell, 0, self.board.height)
        view = pygame.Rect(
            start_x + int(visible_left * cell),
            start_y + int(visible_top * cell),
            max(2, int((visible_right - visible_left) * cell)),
            max(2, int((visible_bottom - visible_top) * cell)),
        )
        pygame.draw.rect(self.screen, TEXT, view, 1)

    def draw_pause_overlay(self) -> None:
        banner = pygame.Rect(WINDOW_W // 2 - 170, HEADER_H + 15, 340, 46)
        pygame.draw.rect(self.screen, PANEL, banner, border_radius=7)
        pygame.draw.rect(self.screen, player_color(HUMAN), banner, 1, border_radius=7)
        draw_text(self.screen, self.font_body, "已暂停 · 可继续指挥 · Space 继续", TEXT, banner.center, center=True)

    def draw_game_over(self) -> None:
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((7, 9, 13, 190))
        self.screen.blit(veil, (0, 0))
        modal = pygame.Rect(WINDOW_W // 2 - 275, WINDOW_H // 2 - 145, 550, 290)
        pygame.draw.rect(self.screen, PANEL, modal, border_radius=8)
        pygame.draw.rect(self.screen, LINE, modal, 1, border_radius=8)
        won = self.board.winner == HUMAN
        title = "战争胜利" if won else "战争失败"
        subtitle = (
            "你淘汰了全部对手"
            if won
            else (
                f"你已被淘汰，{player_label(self.board.winner)} 赢得最终胜利"
                if self.board.winner is not None
                else "你的将军已被占领"
            )
        )
        color = SUCCESS if won else player_color(1)
        draw_text(self.screen, self.font_title, title, color, (modal.centerx, modal.y + 62), center=True)
        draw_text(self.screen, self.font_body, subtitle, TEXT, (modal.centerx, modal.y + 116), center=True)
        alive = len(self.board.active_players)
        stats = self.player_stats[HUMAN]
        summary = f"第 {self.board.turn} 回合 · 存活 {alive} 人 · 你拥有 {stats['tiles']} 格 / {stats['armies']} 兵"
        draw_text(self.screen, self.font_small, summary, MUTED, (modal.centerx, modal.y + 154), center=True)
        self.game_over_new_rect = pygame.Rect(modal.x + 55, modal.bottom - 78, 200, 48)
        self.game_over_menu_rect = pygame.Rect(modal.right - 255, modal.bottom - 78, 200, 48)
        draw_button(
            self.screen,
            self.game_over_new_rect,
            "再来一局",
            self.font_body,
            hovered=self.game_over_new_rect.collidepoint(pygame.mouse.get_pos()),
        )
        draw_button(
            self.screen,
            self.game_over_menu_rect,
            "返回主菜单",
            self.font_body,
            hovered=self.game_over_menu_rect.collidepoint(pygame.mouse.get_pos()),
        )

    def draw_help(self) -> None:
        veil = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        veil.fill((7, 9, 13, 215))
        self.screen.blit(veil, (0, 0))
        modal = pygame.Rect(135, 85, 1010, 670)
        pygame.draw.rect(self.screen, PANEL, modal, border_radius=8)
        pygame.draw.rect(self.screen, LINE, modal, 1, border_radius=8)
        draw_text(self.screen, self.font_heading, "规则与自动回合操作", TEXT, (modal.x + 34, modal.y + 26))
        draw_text(self.screen, self.font_small, "按 H 或 Esc 返回", MUTED, (modal.right - 34, modal.y + 36), right=True)

        sections = [
            (
                "自动回合",
                [
                    "倒计时结束自动结算，无需手动过回合。",
                    "设置页统一调整速度、地图、玩家数、主题、迷雾和扩兵轮。",
                    "Space 暂停或继续；暂停期间仍可查看地图和编辑指挥。",
                ],
            ),
            (
                "多人自由混战与辅助",
                [
                    "可选 2 至 37 名玩家；你控制蓝色，其他 AI 互不结盟。",
                    "普通 AI 每阶段仅一个地块行动；独立军队拥有单独行动槽。",
                    "右侧按钮可让 AI 接管你的棋盘，再次点击恢复手动。",
                    "你的将军失守后进入观众模式，战局继续直到最终胜者产生。",
                    "AI 始终遵守自己的视野；关闭迷雾只影响你的画面。",
                ],
            ),
            (
                "本地自对弈训练",
                [
                    "12 个 AI 按混战与 1v1 混合模式连续自对弈。",
                    "按胜负、领土、兵力、歼敌和生存能力做精英进化。",
                    "训练数据自动保存到 training_data，关闭程序后仍保留。",
                    "训练冠军仅应用于困难难度和困难档 AI 接管。",
                ],
            ),
            (
                "地图、丘陵与视野",
                [
                    "方向键/WASD 移动视角，滚轮缩放，0 返回最小比例。",
                    "丘陵移动与扩兵更慢；己方丘陵不受额外延迟。",
                    "丘陵阻挡直线视野；0 至 5 格视野可在设置页调整。",
                    "关闭迷雾时你可见全图，AI 仍读取自己的视野。",
                ],
            ),
            (
                "命令",
                [
                    "左键全军、右键半军；单击相邻格追加一步路径。",
                    "单击非相邻格只更换选中；点击战场外取消选中。",
                    "最多连续指挥 64 步，两兵地块也可开始路线。",
                    "选中主 AI 或真人指挥的部队会建立隐藏主军队标签，随该部队移动并在 10 回合未操作后撤销。",
                    "双击右键己方格创建一支由独立 AI 控制的军队。",
                    "右键单击一支己方军队会取消头衔，立即恢复手动控制。",
                    "C 取消路线；Z 撤回；U 撤最后一步；Backspace/Esc 清空。",
                    "F11 随时切换窗口/全屏；设置页也可切换并看到当前状态。",
                ],
            ),
            (
                "独立军队",
                [
                    "军队上限为 4 + 向下取整(总兵力 / 8000)；真人冷却 10 回合，AI 35 回合。",
                    "主城不能创建；军队格不能被主棋盘选中或承接地块路线。",
                    "独立军队会识别隐藏主军队标签并把兵力补到该主军队；主军队可进入主城。",
                    "A 为全国最高非军队地块兵力的两倍，军队会尝试补到 A。",
                    "达到 A 后自动进攻；敌国部队进入本土时优先回防。",
                    "相邻敌方主城且兵力占优时直接斩首；长期小范围绕圈会强制向外突破。",
                    "军队不会补移动中的部队，也不会进入另一支军队格。",
                    "右键点击己方军队可取消头衔并立即恢复手动控制。",
                    "军队行动不消耗主 AI 行动；编队被消灭后自动撤销头衔。",
                ],
            ),
        ]
        columns = (sections[:3], sections[3:])
        for column_index, column_sections in enumerate(columns):
            x = modal.x + 34 + column_index * 500
            y = modal.y + 88
            for title, lines in column_sections:
                draw_text(
                    self.screen,
                    self.font_bold,
                    title,
                    player_light(HUMAN),
                    (x, y),
                )
                y += 31
                for line in lines:
                    draw_text(
                        self.screen,
                        self.font_tiny,
                        "· " + line,
                        MUTED,
                        (x + 16, y),
                    )
                    y += 22
                y += 8

    def smoke_test(self) -> None:
        for _ in range(6):
            if self.state == "GAME_OVER":
                break
            self.advance_turn()
            self.draw()
        self.draw()
        pygame.display.flip()

    def screenshot(self, path: str) -> None:
        self.draw()
        pygame.display.flip()
        pygame.image.save(self.screen, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generals: automatic multiplayer vs AI")
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="normal")
    parser.add_argument("--size", type=int, choices=MAP_SIZES, default=45)
    parser.add_argument("--players", type=int, choices=range(2, MAX_PLAYERS + 1), default=2)
    parser.add_argument("--theme", choices=tuple(THEMES), default="day")
    parser.add_argument("--fog", choices=("on", "off"), default="on")
    parser.add_argument("--turn-seconds", type=float, default=DEFAULT_TURN_SECONDS)
    parser.add_argument("--growth-interval", type=int, choices=range(10, 51), default=20)
    parser.add_argument("--vision-radius", type=int, choices=range(0, 6), default=2)
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quick-start", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--screenshot", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    headless = bool(args.smoke_test or args.screenshot)
    if headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        args.quick_start = True
    app = GameApp(args, headless=headless)
    if args.smoke_test:
        app.smoke_test()
        pygame.quit()
        return 0
    if args.screenshot:
        app.screenshot(args.screenshot)
        pygame.quit()
        return 0
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
