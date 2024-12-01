from utils import scrape

import numpy as np

def aoc1():
    inp = scrape()
    left_col, right_col = zip(*[(int(left), int(r)) for left, r in (line.split() for line in inp)])

    np_left = np.array(left_col)
    np_right = np.array(right_col)

    np_left.sort()
    np_right.sort()
    diff = np.abs(np_left - np_right)
    print("part 1: ", np.sum(diff))

    unique_right, count_right = np.unique(np_right, return_counts=True)
    count_dict_right = dict(zip(unique_right, count_right))

    total_score = 0
    for l_num in np_left:
        total_score += l_num * count_dict_right.get(l_num, 0)
    print("part 2: ", total_score)
