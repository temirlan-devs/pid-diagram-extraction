from typing import List, Tuple

import numpy as np


def split_image(image: np.ndarray, tile_size: int = 1024) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split an image into tiles and return a list of:
    (tile_image, x_offset, y_offset)
    """
    height, width = image.shape[:2]
    tiles = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = image[y:y + tile_size, x:x + tile_size]
            tiles.append((tile, x, y))

    return tiles