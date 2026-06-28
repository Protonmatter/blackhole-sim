from blackhole_sim.tiles import generate_tiles, progressive_levels, generate_progressive_tiles


def test_generate_tiles_covers_image():
    tiles = generate_tiles(130, 70, tile_size=64)
    assert len(tiles) == 6
    assert sum(t.pixels for t in tiles) == 130 * 70
    assert tiles[-1].x1 == 130
    assert tiles[-1].y1 == 70


def test_progressive_levels_reaches_full_resolution():
    levels = progressive_levels(1920, 1080, min_width=480)
    assert levels[0][1] <= 960
    assert levels[-1] == (len(levels)-1, 1920, 1080)


def test_progressive_tiles_have_levels():
    tiles = list(generate_progressive_tiles(1024, 512, tile_size=128, min_width=256))
    assert tiles
    assert max(t.level for t in tiles) >= 1
