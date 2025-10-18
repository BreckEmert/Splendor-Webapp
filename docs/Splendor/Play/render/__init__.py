# Splendor/Play/render/__init__.py

from pathlib import Path
FONT_PATH = Path(__file__).resolve().parent / "Resources" / "arialbd.ttf"

from .board_geometry import BoardGeometry, Rect, Coord
from .board_renderer import BoardRenderer
from .overlay_renderer import OverlayRenderer
from .static_renderer import (
    draw_game_state, 
    take_3_indices, 
    take_2_diff_indices
)
