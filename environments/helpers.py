from collections import defaultdict
from typing import Tuple

import numpy as np
from pathlib import Path

# Constants
WALL = '#'
DOOR = 'D'
DANGER_ZONE = 'x'
LEVELS_DIR = 'levels'
LEVEL = 'level'
AGENT = 'agent'
IS_FREE_CELL = 0
IS_OCCUPIED_CELL = 1

DOORS = 'doors'
IS_CLOSED_DOOR = IS_OCCUPIED_CELL
IS_OPEN_DOOR = -1

LEVEL_IDX = 0

TO_BE_AVERAGED = ['dirt_amount', 'dirty_tiles']
IGNORED_DF_COLUMNS = ['Episode', 'Run', 'train_step', 'step', 'index', 'dirt_amount', 'dirty_tile_count']

ACTIONMAP = defaultdict(lambda: (0, 0), dict(north=(-1, 0), east=(0, 1),
                                             south=(1, 0), west=(0, -1),
                                             north_east=(-1, +1), south_east=(1, 1),
                                             south_west=(+1, -1), north_west=(-1, -1)
                                             )
                        )

HORIZONTAL_DOOR_MAP = np.asarray([[0, 0, 0], [1, 0, 1], [0, 0, 0]])
VERTICAL_DOOR_MAP = np.asarray([[0, 1, 0], [0, 0, 0], [0, 1, 0]])

HORIZONTAL_DOOR_ZONE_1 = np.asarray([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
HORIZONTAL_DOOR_ZONE_2 = np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
VERTICAL_DOOR_ZONE_1 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
VERTICAL_DOOR_ZONE_2 = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, 1]])

NOT_VALID = False
VALID = True


# Utility functions
def parse_level(path):
    with path.open('r') as lvl:
        level = list(map(lambda x: list(x.strip()), lvl.readlines()))
    if len(set([len(line) for line in level])) > 1:
        raise AssertionError('Every row of the level string must be of equal length.')
    return level


def one_hot_level(level, wall_char=WALL):
    grid = np.array(level)
    binary_grid = np.zeros(grid.shape, dtype=np.int8)
    binary_grid[grid == wall_char] = 1
    return binary_grid


def check_position(slice_to_check_against: np.ndarray, position_to_check: Tuple[int, int]):
    x_pos, y_pos = position_to_check

    # Check if agent colides with grid boundrys
    valid = not (
            x_pos < 0 or y_pos < 0
            or x_pos >= slice_to_check_against.shape[0]
            or y_pos >= slice_to_check_against.shape[0]
    )

    # Check for collision with level walls
    valid = valid and not slice_to_check_against[x_pos, y_pos]
    return valid


if __name__ == '__main__':
    parsed_level = parse_level(Path(__file__).parent / 'factory' / 'levels' / 'simple.txt')
    y = one_hot_level(parsed_level)
    print(np.argwhere(y == 0))
