# https://www.hackerrank.com/challenges/dip-image-segmentation-2/problem
# tag(s): image-processing, dfs
from functools import partial

image = [
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
]


def get_num_8_pixel_connectivity_groups() -> int:
    n = len(image)
    m = len(image[0])

    explore_partial = partial(explore, n, m)
    num_groups = 0
    
    for y in range(n):
        for x in range(m):
            if image[y][x] == 1:
                explore_partial(x, y)
                num_groups += 1

    return num_groups


def explore(n: int, m: int, x: int , y: int):
    if x >= m or x < 0:
        return
    if y >= n or y < 0:
        return
    if image[y][x] == 0:
        return

    image[y][x] = 0

    # up
    explore(n, m, x, y - 1)
    # down
    explore(n, m, x, y + 1)
    # left
    explore(n, m, x - 1, y)
    # right
    explore(n, m, x + 1, y)

    # up-left
    explore(n, m, x - 1, y - 1)
    # up-right
    explore(n, m, x + 1, y - 1)
    # down-left
    explore(n, m, x - 1, y + 1)
    # down-right
    explore(n, m, x + 1, y + 1)


if __name__ == "__main__":
    print(get_num_8_pixel_connectivity_groups())
