"""Tiled/progressive render scheduling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True, order=True)
class RenderTile:
    level: int
    y0: int
    x0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def pixels(self) -> int:
        return self.width * self.height


def generate_tiles(width: int, height: int, tile_size: int = 64, level: int = 0) -> list[RenderTile]:
    tiles: list[RenderTile] = []
    for y in range(0, int(height), int(tile_size)):
        for x in range(0, int(width), int(tile_size)):
            tiles.append(RenderTile(level, x0=x, y0=y, x1=min(x + tile_size, width), y1=min(y + tile_size, height)))
    return tiles


def progressive_levels(width: int, height: int, min_width: int = 480) -> list[tuple[int, int, int]]:
    """Return (level, width, height), coarse-to-full."""
    out: list[tuple[int, int, int]] = []
    w, h = int(width), int(height)
    scale = 1
    while w // (scale * 2) >= min_width:
        scale *= 2
    level = 0
    while scale >= 1:
        out.append((level, max(1, width // scale), max(1, height // scale)))
        level += 1
        scale //= 2
    return out


def generate_progressive_tiles(width: int, height: int, tile_size: int = 64, min_width: int = 480) -> Iterator[RenderTile]:
    for level, w, h in progressive_levels(width, height, min_width):
        yield from generate_tiles(w, h, tile_size, level)
