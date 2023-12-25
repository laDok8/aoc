#!/usr/bin/python3
import datetime
import inspect
import math
import os
import pickle
import re
import time
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce, cache
from heapq import heappush, heappop
from itertools import dropwhile, takewhile
from random import sample

import networkx as nx
import numpy as np
import requests as r
import z3
from dotenv import dotenv_values

ENV_VARS = dotenv_values('.env')
cookies = {"session": ENV_VARS['SESSION']}


def print_timing(func):
    """create a timing decorator function"""

    def wrapper(*arg):
        start = time.perf_counter()
        result = func(*arg)
        end = time.perf_counter()
        fs = '{} took {:.2f} ms'
        print(fs.format(func.__name__, (end - start) * 1e3))
        return result

    return wrapper


def scrape(year: int = 0, day: int = 0, separator: str = '\n'):
    # determine day from callee function name
    day = int(str.replace(inspect.stack()[1].function, 'aoc', '')) if day == 0 else day
    year = datetime.date.today().year if year == 0 else year
    # substitute year and day
    url = f'https://adventofcode.com/{year}/day/{day}/input'
    file = f'inp{day}.txt'
    if os.path.exists(file):
        return open(file, encoding='utf-8').read().split(separator)
    inp = r.get(url, cookies=cookies, timeout=1).text.rstrip()
    with open(file, 'w', encoding='UTF-8') as f:
        f.write(inp)
    return inp.split(separator)


def aoc1():
    inp = scrape()
    sum_part1, sum_part2 = 0, 0
    for line in inp:
        # remove all non digits
        line = re.sub(r'\D', '', line)
        # get first and last number
        if len(line) == 0:
            continue
        sum_part1 += int(line[0]) * 10 + int(line[-1])

    map_replace = {'zero': 'z0o', 'one': 'o1e', 'two': 't2o', 'three': 't3e', 'four': 'f4r', 'five': 'f5e',
                   'six': 's6x', 'seven': 's7n', 'eight': 'e8t', 'nine': 'n9e'}
    for line in inp:
        for key, val in map_replace.items():
            line = line.replace(key, val)
        # remove all non digits
        line = re.sub(r'\D', '', line)
        # get first and last number
        sum_part2 += int(line[0]) * 10 + int(line[-1])
    print('part 1:', sum_part1, '\npart 2:', sum_part2)


def aoc2():
    def separate_game(_game):
        game_split = _game.split(':')
        _id = int(game_split[0][4:])
        sets = game_split[1].split(';')
        maxes = {'red': 0, 'green': 0, 'blue': 0}

        for take in sets:
            for color in maxes.keys():
                takes_in_color = [s for s in take.split(',') if color in s]
                takes_in_color = [int(s[:-len(color)]) for s in takes_in_color]
                maxes[color] = sum(takes_in_color) if sum(takes_in_color) > maxes[color] else maxes[color]
        return _id, maxes

    possible_ids_sum, fewest_cubes_sum, inp = 0, 0, scrape()
    part1_limits = {'red': 12, 'green': 13, 'blue': 14}
    for game in inp:
        game_id, cur_lims = separate_game(game)
        if all(cur_lims[color] <= part1_limits[color] for color in cur_lims.keys()):
            possible_ids_sum += game_id
        fewest_cubes_sum += reduce((lambda x, y: x * y), cur_lims.values())

    print("part 1:", possible_ids_sum, "\npart 2:", fewest_cubes_sum)


class Rectangle:
    x, y, width = 0, 0, 0

    def __init__(self, x, y, width):
        self.x, self.y, self.width = x, y, width

    def is_adjacent(self, other):
        return (abs(self.y - other.y) <= 1 and (
                self.x - 1 <= other.x <= self.x + self.width or other.x - 1 <= self.x <= other.x + other.width))

    def __str__(self):
        return f'x:{self.x}, y:{self.y}, width:{self.width}'


def aoc3():
    all_parts, all_nums, inp = [], [], scrape()
    for (_y, line) in enumerate(inp):
        num_iter, part_iter = re.finditer(r'\d+', line), re.finditer(r'[^.\d]', line)
        for num in num_iter:
            all_nums.append((Rectangle(x=num.span()[0], y=_y, width=num.span()[1] - num.span()[0]), int(num.group())))
        for part in part_iter:
            all_parts.append(Rectangle(part.span()[0], _y, 1))

    sum_part1 = sum(
        map(lambda _x: _x[1], filter(lambda _num: any(_num[0].is_adjacent(_part) for _part in all_parts), all_nums)))
    sum_part2 = sum(reduce(lambda _x, y, p=part: _x * y,
                           map(lambda x, p=part: x[1], filter(lambda _num, p=part: p.is_adjacent(_num[0]), all_nums)),
                           1) for part in all_parts if
                    len(list(filter(lambda _num, p=part: p.is_adjacent(_num[0]), all_nums))) == 2)
    print("part 1:", sum_part1, "\npart 2:", sum_part2)


def aoc4():
    sum_points_part1, card_str, card_storage = 0, "Card ", {}
    for card_num, nums in enumerate(scrape(), start=1):
        winning, mine = [list(map(int, part.split())) for part in nums.split(':')[1].split('|')]
        conjunction = set(winning) & set(mine)
        sum_points_part1 += 2 ** (len(conjunction) - 1) if len(conjunction) > 0 else 0

        # part 2
        count_current = card_storage.setdefault(card_str + str(card_num), 1)
        # add new instances
        for i in range(card_num + 1, len(conjunction) + card_num + 1):
            card_storage[card_str + str(i)] = card_storage.get(card_str + str(i), 1) + count_current
    print("part 1:", sum_points_part1, "\npart 2:", sum(i for i in card_storage.values()))


def get_min_location_from_ranges(cur_range, all_steps):
    for f in all_steps:
        modified_r = []
        for dest, src, ln in f:
            src_end, next_r = src + ln, []
            mapping = lambda x, _dest=dest, _src=src: x + _dest - _src
            for seed_rng_start, seed_rng_end in cur_range:
                before = (seed_rng_start, min(seed_rng_end, src))
                inter = (max(seed_rng_start, src), min(seed_rng_end, src_end))
                after = (max(seed_rng_start, src_end), seed_rng_end)

                if before[1] > before[0]:
                    next_r.append(before)
                if inter[1] > inter[0]:
                    modified_r.append((mapping(inter[0]), mapping(inter[1])))
                if after[1] > after[0]:
                    next_r.append(after)
            cur_range = next_r
        cur_range += modified_r
    return min(cur_range)[0]


def aoc5():
    inp, step_maps = scrape(separator='\n\n'), []
    # parse input -> each inner list is X->Y map (stored as separate ranges)
    for maps in inp[1:]:
        nums = [int(i) for i in maps.split()[2:]]
        cur_step_map = []
        for dest, src, ln in zip(*(iter(nums),) * 3):
            cur_step_map.append((dest, src, ln))
        step_maps.append(cur_step_map)

    # create ranges !! ALL intervals [start, end)
    seeds, seed_ranges_part1 = [int(i) for i in inp[0][6:].split()], []
    for start in seeds:
        seed_ranges_part1.append((start, start + 1))
    seed_ranges_part2 = [(start, start + rng) for start, rng in zip(seeds[::2], seeds[1::2])]

    print("part 1:", get_min_location_from_ranges(seed_ranges_part1, step_maps))
    print("part 2:", get_min_location_from_ranges(seed_ranges_part2, step_maps))


def aoc6():
    inp, acc_prod = scrape(), 1
    race_time, record_dist = [int(i) for i in inp[0][5:].split()], [int(i) for i in inp[1][9:].split()]
    for cur_time, cur_record in zip(race_time, record_dist):
        acc = 0
        for hold_time in range(1, cur_time):
            run_time = cur_time - hold_time
            acc += 1 if (run_time * hold_time) > cur_record else 0
        acc_prod *= acc

    acc, merged_time, merged_dist = 0, int(inp[0][5:].replace(' ', '')), int(inp[1][9:].replace(' ', ''))
    for hold_time in range(1, merged_time):
        run_time = merged_time - hold_time
        acc += 1 if (run_time * hold_time) > merged_dist else 0
    print("part 1:", acc_prod, "\npart 2:", acc)


def get_poker_hand_type(occurrences: dict):
    if m := max(occurrences.values()) >= 4:
        return str(14 - m)  # five/four of a kind
    if sorted(occurrences.values()) == [2, 3]:
        return '7'  # full house
    if sorted(occurrences.values()) == [1, 1, 3]:
        return '6'  # three of a kind
    if sorted(occurrences.values()) == [1, 2, 2]:
        return '5'  # two pairs
    if sorted(occurrences.values()) == [1, 1, 1, 2]:
        return '4'  # one pair
    return '3'  # high card


def aoc7():
    Hand = namedtuple('Hand', ['raw', 'bid', 'strength'])  # strength = TYPE + raw
    # for easier string sort
    raw_strength_map_p1 = {'T': 'a', 'J': 'b', 'Q': 'c', 'K': 'd', 'A': 'e'}
    raw_strength_map_p2 = raw_strength_map_p1.copy()
    raw_strength_map_p2['J'] = '0'

    all_hands_p1, all_hands_p2 = [], []
    for raw, bid in (line.split() for line in scrape()):

        occurrences = {card: raw.count(card) for card in raw}
        hand_type = get_poker_hand_type(occurrences)
        all_hands_p1.append(Hand(raw, bid, hand_type + ''.join([raw_strength_map_p1.get(c, c) for c in raw])))

        # remove jokes and add to most common for part2
        if occurrences.get('J', 0) != 5:
            count = occurrences.get('J', 0)
            occurrences.pop('J', None)
            most_common = max(occurrences, key=occurrences.get)
            occurrences[most_common] += count

        hand_type = get_poker_hand_type(occurrences)
        all_hands_p2.append(Hand(raw, bid, hand_type + ''.join([raw_strength_map_p2.get(c, c) for c in raw])))

    all_hands_p1.sort(key=lambda x: x.strength)
    all_hands_p2.sort(key=lambda x: x.strength)

    acc_p1, acc_p2 = 0, 0
    for rank, h in enumerate(all_hands_p1, 1):
        acc_p1 += int(h.bid) * rank
    for rank, h in enumerate(all_hands_p2, 1):
        acc_p2 += int(h.bid) * rank
    print("part 1:", acc_p1, "\npart 2:", acc_p2)


def calculate_path_d8(starts, nodes, steps) -> int:
    period = [0] * len(starts)
    for i in range(len(starts)):
        # start_steps can be ignored
        cur_n, step = starts[i], 0
        while cur_n[-1] != 'Z':
            cur_n = nodes[cur_n][0] if steps[step] == 'L' else nodes[cur_n][1]
            period[i] += 1
            step = (step + 1) % len(steps)
    return math.lcm(*period)


def aoc8():
    inp = scrape(separator='\n\n')
    steps, nodes = inp[0], {}

    for line in inp[1].split('\n'):
        node_start, node_opts = map(str.strip, line.split('='))
        _l, _r = re.sub(r'[(),]', '', node_opts).split()
        nodes[node_start] = (_l, _r)

    steps_p1 = calculate_path_d8(['AAA'], nodes, steps)
    steps_p2 = calculate_path_d8([k for k in nodes.keys() if k[-1] == 'A'], nodes, steps)

    print("part 2:", steps_p1, "\npart 2:", steps_p2)


def aoc9():
    acc_p1, acc_p2 = 0, 0
    for nums in ([int(i) for i in line.split()] for line in scrape()):
        last, first, diffs = [], [], nums
        while len(set(diffs)) != 1:
            diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
            last, first = last + diffs[-1:], first + diffs[:1]
        acc_p1 += sum(last + nums[-1:])
        acc_p2 += reduce(lambda _acc, x: x - _acc, reversed(nums[:1] + first))
    print("part 1:", acc_p1, "\npart 2:", acc_p2)


@dataclass(slots=True)
class Pos:
    x: int
    y: int
    dir: int = 0  # 0 - up, 1 - right, 2 - down, 3 - left

    def __add__(self, other):
        return Pos(self.x + other.x, self.y + other.y, self.dir)

    def __mul__(self, other):
        return Pos(self.x * other, self.y * other, self.dir)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.dir == other.dir

    def __hash__(self):
        return hash((self.x, self.y, self.dir))

    def distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)

    # rotation
    def cw(self):
        return Pos(self.x, self.y, (self.dir + 1) % 4)

    def ccw(self):
        return Pos(self.x, self.y, (self.dir - 1) % 4)

    # move
    def step(self):
        direction_mapping = {0: lambda: Pos(self.x, self.y - 1, self.dir), 1: lambda: Pos(self.x + 1, self.y, self.dir),
                             2: lambda: Pos(self.x, self.y + 1, self.dir),
                             3: lambda: Pos(self.x - 1, self.y, self.dir), }

        return direction_mapping.get(self.dir, lambda: None)()


# 4 boundary fill
def try_fill(cur_loc: Pos, flooded: list, dst: list):
    if cur_loc in flooded or dst[cur_loc.x][cur_loc.y] != 0:
        return
    flooded.append(cur_loc)
    try_fill(cur_loc + Pos(0, -1), flooded, dst)
    try_fill(cur_loc + Pos(0, 1), flooded, dst)
    try_fill(cur_loc + Pos(-1, 0), flooded, dst)
    try_fill(cur_loc + Pos(1, 0), flooded, dst)


def aoc10():
    # switch X and Y for sanity
    grid = np.transpose(np.array([list(line) for line in scrape()]))
    dst = np.zeros(grid.shape, dtype=int)

    start = np.where(grid == 'S')
    start = Pos(start[0][0], start[1][0])
    # WARN: manual step
    grid[start.x, start.y] = '7'

    # discover by queue
    discovered = [start]
    while discovered:
        cur = discovered.pop(0)
        if cur == start and dst[cur.x, cur.y] != 0:
            dst[cur.x, cur.y] = 0
            continue

        p1 = Pos(cur.x, cur.y - 1) if grid[cur.x, cur.y] in ['|', 'L', 'J'] else None
        p2 = Pos(cur.x, cur.y + 1) if grid[cur.x, cur.y] in ['|', 'F', '7'] else None
        p3 = Pos(cur.x - 1, cur.y) if grid[cur.x, cur.y] in ['-', 'J', '7'] else None
        p4 = Pos(cur.x + 1, cur.y) if grid[cur.x, cur.y] in ['-', 'L', 'F'] else None

        if p1 and dst[p1.x, p1.y] == 0:
            discovered.append(p1)
            dst[p1.x, p1.y] = dst[cur.x, cur.y] + 1
        if p2 and dst[p2.x, p2.y] == 0:
            discovered.append(p2)
            dst[p2.x, p2.y] = dst[cur.x, cur.y] + 1
        if p3 and dst[p3.x, p3.y] == 0:
            discovered.append(p3)
            dst[p3.x, p3.y] = dst[cur.x, cur.y] + 1
        if p4 and dst[p4.x, p4.y] == 0:
            discovered.append(p4)
            dst[p4.x, p4.y] = dst[cur.x, cur.y] + 1

    print('part 1:', max(max(line) for line in dst))

    dst[start.x, start.y] = 1  # mark as wall
    directions = {'up': Pos(0, -1), 'right': Pos(1, 0), 'down': Pos(0, 1), 'left': Pos(-1, 0)}
    dirs_clockwise = list(directions.keys())
    flooded = []
    # -1 is CCW, 1 is CW
    direction_mapping = {'down': {'L': -1, 'J': 1}, 'up': {'F': 1, '7': -1}, 'left': {'L': 1, 'F': -1},
                         'right': {'J': -1, '7': 1}}

    # tuple cur_direction, current_inside like left right hand
    # walk the maze and flood fill any empty space, works weirdly only if walked both ways
    for cur_directives in [('left', 'down'), ('down', 'left')]:
        # make first step
        cur_step = start + directions[cur_directives[0]]

        while cur_step != start:
            try_fill(cur_step + directions[cur_directives[1]], flooded, dst)
            if grid[cur_step.x, cur_step.y] not in ['L', 'J', '7', 'F']:
                cur_step += directions[cur_directives[0]]
                continue

            # change direction
            cur_directives = (dirs_clockwise[(dirs_clockwise.index(cur_directives[0]) +
                                              direction_mapping[cur_directives[0]][grid[cur_step.x, cur_step.y]]) % len(
                dirs_clockwise)], dirs_clockwise[(dirs_clockwise.index(cur_directives[1]) +
                                                  direction_mapping[cur_directives[0]][
                                                      grid[cur_step.x, cur_step.y]]) % len(dirs_clockwise)])

            cur_step += directions[cur_directives[0]]

    print('part 2:', len(flooded))


def aoc11():
    for part, multi in enumerate([2, 1e6]):
        grid = np.array([list(line) for line in scrape()])
        stars = np.where(grid == '#')
        stars = [Pos(x, y) for x, y in zip(stars[0], stars[1])]

        def adjust_coordinates(stars: list[Pos], axis: str, _multi: int):
            max_coord = max(stars, key=lambda x: getattr(x, axis)).__getattribute__(axis)

            for i in range(max_coord, 0, -1):
                if any(getattr(s, axis) == i for s in stars):
                    continue
                for s in stars:
                    if getattr(s, axis) > i:
                        setattr(s, axis, getattr(s, axis) + (_multi - 1))

        # simulate aging
        adjust_coordinates(stars, 'x', multi)
        adjust_coordinates(stars, 'y', multi)

        acc, visited = 0, []
        for s1 in stars:
            visited.append(s1)
            for s2 in stars:
                if s2 in visited:
                    continue
                acc += s1.distance(s2)

        print(f'part {part + 1}: {int(acc)}')


# get amount of continuous ? in the prefix
def start_symb(s: str, symbs: list[str]):
    return len(list(takewhile(lambda x: x in symbs, s)))


@dataclass(slots=True)
class Problem:
    rest_record: str  # type = # - Damaged, ? - Unknown, . - Empty
    groups: list[int]

    def __hash__(self):
        return hash((self.rest_record, tuple(self.groups)))


def solve_hash(start_hash: int, work_r: str, work_g: list[int]):
    # not enough #s or no groups
    if len(work_g) == 0 or (to_fill := work_g[0] - start_hash) < 0:
        return 0
    work_r = work_r[start_hash:]
    # remove rest of Qs ( q also may be dot)
    if start_symb(work_r, ['?', '#']) < to_fill:
        return 0
    work_r = work_r[to_fill:]
    # edge case
    if len(work_r) == 0:
        return 1 if len(work_g) == 1 else 0  # return solve_cur_prob(Problem(work_r, work_g[1:]))

    # drop first q
    if work_r[0] not in ['?', '.']:
        return 0
    work_r = work_r[1:]
    # create new problem
    return solve_cur_prob(Problem(work_r, work_g[1:]))


@cache
def solve_cur_prob(cur_prob: Problem):
    work_r, work_g = cur_prob.rest_record, cur_prob.groups
    work_r = ''.join(dropwhile(lambda x: x == '.', work_r))

    # it's okay
    if len(work_r) == 0 and len(work_g) == 0:
        return 1
    # not okay
    if len(work_r) == 0 and len(work_g) != 0:
        return 0

    # deterministic situation ##?. 3 -> ### or ##?? 3 -> ###
    if (start_hash := start_symb(work_r, ['#'])) > 0:
        return solve_hash(start_hash, work_r, work_g)
    # non-deterministic ? situation -> create both # and . problems
    if work_r[0] == '?':
        return solve_cur_prob(Problem('#' + work_r[1:], work_g)) + solve_cur_prob(Problem(work_r[1:], work_g))
    return 0


def aoc12():
    for part, folds in enumerate([1, 5]):
        acc = 0

        for record, groups in (line.split() for line in scrape()):
            groups = [int(i) for i in groups.split(',')]  # list of really continuous damaged

            # unfold problem
            groups = [it for sublist in [groups] * folds for it in sublist]
            record = ((record + '?') * folds)[:-1] if folds > 1 else record

            acc += solve_cur_prob(Problem(record, groups))
        print(f'part {part + 1}: {int(acc)}')


# returns number of diffs between 2 strs
def diff_str(s1: str, s2: str) -> int:
    return sum(1 for a, b in zip(s1, s2) if a != b)


def aoc13():
    inp = scrape(separator='\n\n')

    for part in range(2):
        acc = 0
        for puzzle in inp:
            grid = np.array([list(line) for line in puzzle.split('\n')], dtype=str)
            line_str, column_str = [], []
            for line in grid:
                line_str.append(''.join(line))
            for line in np.transpose(grid):
                column_str.append(''.join(line))

            row_match, col_match = -1, -1

            # find mirroring in rows
            for i in range(1, len(line_str)):
                diffs, shortest_end = 0, min(len(line_str[:i]), len(line_str[i:]))
                for j in range(0, shortest_end):
                    diffs += diff_str(line_str[i - 1 - j], line_str[i + j])

                if diffs == part:
                    row_match = i
                    break
            if row_match != -1:
                acc += (row_match * 100)
                continue

            # find mirroring in cols
            for i in range(1, len(column_str)):
                diffs, shortest_end = 0, min(len(column_str[:i]), len(column_str[i:]))
                for j in range(0, shortest_end):
                    diffs += diff_str(column_str[i - 1 - j], column_str[i + j])

                if diffs == part:
                    col_match = i
                    break
            # there's always one
            acc += col_match
        print(f'part {part + 1}: {int(acc)}')


def d14_supp_strength(grid) -> int:
    acc = 0
    for y in range(grid.shape[0]):
        row_val = grid.shape[0] - y
        occs = ''.join(grid[y, :]).count('O')
        acc += occs * row_val
    return acc


def aoc14():
    grid = np.array([list(line) for line in scrape()], dtype=str)

    cached, limit, itr = {}, int(1e9), 0
    while True:
        if itr >= limit:
            break

        for rot in range(4):
            for x, y in np.ndindex(grid.shape):
                if grid[y, x] == 'O' and grid[y - 1, x] == '.':
                    above = np.flip(grid[:y, x])
                    taken = len(list(takewhile(lambda c: c == '.', above)))
                    new_pos = y - taken
                    grid[y, x], grid[new_pos, x] = '.', 'O'

            if itr == 0 and rot == 0:
                print("part 1:", d14_supp_strength(grid), end='')
            grid = np.rot90(grid, 3)

        if prev_seen := cached.get(hash(grid.tobytes())):
            diff = itr - prev_seen
            while itr + diff < limit:
                itr += diff
        else:
            cached[hash(grid.tobytes())] = itr
        itr += 1

    print(' part 2:', d14_supp_strength(grid))


def d15_hash(inp: str) -> int:
    return reduce(lambda _acc, x: (_acc + ord(x)) * 17 % 256, inp, 0)


def aoc15():
    acc_p1, inp = 0, scrape(separator=',')

    boxes_lbls = [[] for _ in range(256)]
    boxes_focals = [[] for _ in range(256)]
    for puz in inp:
        acc_p1 += d15_hash(puz)

        idx = puz.find('=') if '=' in puz else puz.find('-')
        lbl = puz[:idx]
        box_id, focal = d15_hash(lbl), puz[idx + 1:]

        if puz[idx] == '=':
            idx = boxes_lbls[box_id].index(lbl) if lbl in boxes_lbls[box_id] else -1
            if idx == -1:
                boxes_lbls[box_id].append(lbl)
                boxes_focals[box_id].append(int(focal))
            else:
                boxes_focals[box_id][idx] = int(focal)
        else:
            idx = boxes_lbls[box_id].index(lbl) if lbl in boxes_lbls[box_id] else -1
            if idx != -1:
                boxes_lbls[box_id].pop(idx)
                boxes_focals[box_id].pop(idx)

    acc_p2 = 0
    for lbls, focals in zip(boxes_lbls, boxes_focals):
        if len(lbls) == 0:
            continue
        box_id = d15_hash(lbls[0]) + 1
        for i, f in enumerate(focals, 1):
            acc_p2 += box_id * i * f

    print('part 1:', acc_p1, 'part 1:', acc_p2)


def _pretty_print_grid(grid):
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            print(grid[x, y], end='')
        print()
    print()


def aoc16():
    # very slow due to Pos abstraction but hash helps
    # switch X and Y for sanity
    grid = np.array([list(line) for line in scrape()]).T

    possible_starts = []
    max_x, max_y = grid.shape[0], grid.shape[1]
    for y in range(max_y):
        possible_starts.append(Pos(0, y, 1))
        possible_starts.append(Pos(grid.shape[0] - 1, y, 3))
    for x in range(max_x):
        possible_starts.append(Pos(x, 0, 2))
        possible_starts.append(Pos(x, grid.shape[1] - 1, 0))

    accs = []
    for start in possible_starts:
        passed = np.zeros(grid.shape, dtype=int)
        visited = []

        work_stack = [start]
        while work_stack:
            cur = work_stack.pop(0)

            # out of borders / already passed -> continue
            if cur.x < 0 or cur.y < 0 or cur.x >= max_x or cur.y >= max_y or hash(cur) in visited:
                continue
            passed[cur.x, cur.y] = 1

            if (grid[cur.x, cur.y] not in ['|', '\\', '/', '-'] or grid[cur.x, cur.y] == '|' and cur.dir in [0, 2] or
                    grid[cur.x, cur.y] == '-' and cur.dir in [1, 3]):
                nxt = cur.step()
                work_stack.append(nxt)
            elif grid[cur.x, cur.y] == '|' and cur.dir in [1, 3] or grid[cur.x, cur.y] == '-' and cur.dir in [0, 2]:
                nxt = cur.cw().step()
                nxt2 = cur.ccw().step()
                work_stack.append(nxt)
                work_stack.append(nxt2)
            elif grid[cur.x, cur.y] == '/' and cur.dir in [0, 2] or grid[cur.x, cur.y] == '\\' and cur.dir in [1, 3]:
                nxt = cur.cw().step()
                work_stack.append(nxt)
            else:
                nxt = cur.ccw().step()
                work_stack.append(nxt)
        accs.append(passed.sum())
    print('part 1:', accs[0], '\npart 2:', max(accs))


def astar(maze: list[list[int]], low_straight: int = 0, high_straight: int = 3) -> int:
    """Returns min heat loss path from start to end in maze"""
    directions = {(0, -1), (1, 0), (0, 1), (-1, 0)}
    pq = [(0, 0, 0, 0, 0, 0)]
    seen = set()

    while pq:
        g, x, y, dx, dy, walked = heappop(pq)
        if (x, y, dx, dy, walked) in seen:
            continue
        seen.add((x, y, dx, dy, walked))

        if x == len(maze) - 1 and y == len(maze[0]) - 1 and walked > low_straight:
            return g

        for ndf in directions:
            # block long straights and 180 degree turns
            if walked >= high_straight and ndf == (dx, dy) or ndf == (-dx, -dy):
                continue
            # block short straights
            if ndf != (dx, dy) and walked <= low_straight and (dx, dy) != (0, 0):
                continue

            new_pos = (x + ndf[0], y + ndf[1])
            if not (0 <= new_pos[0] < len(maze) and 0 <= new_pos[1] < len(maze[0])):
                continue
            _walked = walked + 1 if ndf == (dx, dy) else 1
            heappush(pq, (g + maze[new_pos[0]][new_pos[1]], new_pos[0], new_pos[1], ndf[0], ndf[1], _walked))
    return 0


def aoc17():
    # switch X and Y for sanity
    grid = np.array([list(line) for line in scrape()], dtype=int).T

    print("part 1:", astar(grid, 0, 3), "\npart 2:", astar(grid, 3, 10))


def shoelace_area(points: list[tuple[int, int]]) -> int:
    acc = 0
    for i in range(len(points) - 1):
        acc += points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
    return abs(acc) // 2


@print_timing
def aoc18():
    directions = {'R': (1, 0), 'D': (0, 1), 'L': (-1, 0), 'U': (0, -1)}
    cur_p1, cur_p2, bound_p1, bound_p2, inp = (0, 0), (0, 0), 0, 0, scrape()
    points_p1, points_p2 = [cur_p1], [cur_p2]
    dir_vals = list(directions.values())

    for _dir, num, paint in [i.split() for i in inp]:
        bound_p1 += int(num)
        cur_p1 = (cur_p1[0] + directions[_dir][0] * int(num), cur_p1[1] + directions[_dir][1] * int(num))
        points_p1.append(cur_p1)

        # part 2
        num2 = int(paint[2:-2], 16)
        dir_int = int(paint[-2])
        bound_p2 += int(num2)
        cur_p2 = cur_p2[0] + dir_vals[dir_int][0] * int(num2), cur_p2[1] + dir_vals[dir_int][1] * int(num2)
        points_p2.append(cur_p2)

    # picks theorem
    print("part 1:", shoelace_area(points_p1) + bound_p1 // 2 + 1)
    print("part 2:", shoelace_area(points_p2) + bound_p2 // 2 + 1)


def solve_flow(interval: tuple, flow_dict: dict, cur_flow: list) -> int:
    nm, letter, op, val = cur_flow[0]
    indice_l = "xmas".index(letter) * 2
    _new_prob, _rest_prob = list(interval), list(interval)
    if op == '>':
        _new_prob[indice_l] = max(_new_prob[indice_l], val + 1)
        _rest_prob[indice_l + 1] = _new_prob[indice_l] - 1
    else:
        _new_prob[indice_l + 1] = min(_new_prob[indice_l + 1], val - 1)
        _rest_prob[indice_l] = _new_prob[indice_l + 1] + 1
    new_prob, rest_prob = tuple(_new_prob), tuple(_rest_prob)

    if nm == 'R':
        acc = 0
    elif nm == 'A':
        acc = math.prod(new_prob[i * 2 + 1] - new_prob[i * 2] + 1 for i in range(4))
    else:
        acc = solve_flow(new_prob, flow_dict, flow_dict[nm])

    acc += solve_flow(rest_prob, flow_dict, cur_flow[1:]) if len(cur_flow) > 1 else 0
    return acc


def solve_flow_p1(all_vals: list[tuple], flow_dict: dict) -> int:
    acc_p1 = 0
    for cur_val in all_vals:
        cur_state = 'in'
        while cur_state not in ['R', 'A']:
            for cond in flow_dict[cur_state]:
                nm, letter, op, val = cond
                if op == '>':
                    if cur_val["xmas".index(letter)] > val:
                        cur_state = nm
                        break
                else:
                    if cur_val["xmas".index(letter)] < val:
                        cur_state = nm
                        break

        if cur_state == 'A':
            acc_p1 += sum(list(cur_val))
    return acc_p1


def aoc19():
    flows, inp = scrape(separator='\n\n')
    flow_dict = {'A': 'foo', 'R': 'foo'}
    for line in flows.split('\n'):
        name, cmds = line[:-1].split('{')
        cmds = cmds.split(',')
        conds = []
        for cmd in cmds:
            cmd = cmd.split(':')
            if len(cmd) == 1:
                conds.append((cmd[0], 'x', '>', 0))
                continue
            nm, cmd = cmd[1], cmd[0]
            op = '>' if '>' in cmd else '<'
            letter, val = cmd.split(op)
            conds.append((nm, letter, op, int(val)))
        flow_dict[name] = conds

    # parse input to json
    all_vals = []
    for line in inp.split('\n'):
        groups = [int(g) for g in re.findall(r'(\d+)', line)]
        all_vals.append(tuple(g for g in groups))

    print("part 1:", solve_flow_p1(all_vals, flow_dict))
    print("part 2:", solve_flow((1, 4000, 1, 4000, 1, 4000, 1, 4000), flow_dict, flow_dict['in']))


def aoc20():
    # module: name<->type, output and separate (default) state
    modules = {}
    modules_state = {}  # High = True, Low = False
    button_press = ('broadcaster', False, 'button')
    for line in scrape():
        nm, out = line.split('->')
        typ, nm = nm.strip()[0], nm.strip()[1:]
        assert typ in ['%', '&', 'b']
        assert (nm if typ != 'b' else typ + nm) not in modules
        modules[nm if typ != 'b' else typ + nm] = (typ, [o.strip() for o in out.strip().split(',')])
        if typ == '%':
            modules_state[nm] = False

    # make default connections to conjunct
    for nm, (typ, outputs) in modules.items():
        for o in outputs:
            if o not in modules:
                continue
            t, _ = modules[o]
            if t == '&':
                if o not in modules_state:
                    modules_state[o] = {}
                modules_state[o][nm] = False

    # presses, iter
    low_p, hi_p, press = 0, 0, 1
    while press <= int(1e3):
        message_q = [button_press]
        low_p += 1
        while len(message_q) != 0:
            mdl_nm, msg, sender = message_q.pop(0)
            if mdl_nm not in modules:
                continue

            if mdl_nm == 'broadcaster':
                for rcvs in modules[mdl_nm][1]:
                    message_q.append((rcvs, False, mdl_nm))
                low_p += len(modules[mdl_nm][1])
                continue
            typ, outputs = modules[mdl_nm]

            state = modules_state[mdl_nm]
            if typ == '%':
                if msg:
                    continue
                _st = not state
                modules_state[mdl_nm] = _st
                for rcvs in outputs:
                    message_q.append((rcvs, _st, mdl_nm))
                if _st:
                    hi_p += len(outputs)
                else:
                    low_p += len(outputs)
            elif typ == '&':
                # update
                state[sender] = msg
                _msg = not all(state.values())

                for rcvs in outputs:
                    message_q.append((rcvs, _msg, mdl_nm))
                if _msg:
                    hi_p += len(outputs)
                else:
                    low_p += len(outputs)
            else:
                pass
        press += 1
    print("part 1:", low_p * hi_p)

    cycle_len = []
    for ot in modules['broadcaster'][1]:
        press = 1
        state_cche = {}
        while True:
            message_q = [(ot, False, 'broadcaster')]
            low_p += 1
            while len(message_q) != 0:
                mdl_nm, msg, sender = message_q.pop(0)
                if mdl_nm not in modules:
                    continue

                typ, outputs = modules[mdl_nm]

                state = modules_state[mdl_nm]
                if typ == '%':
                    if msg:
                        continue
                    _st = not state
                    modules_state[mdl_nm] = _st
                    for rcvs in outputs:
                        message_q.append((rcvs, _st, mdl_nm))

                elif typ == '&':
                    # update
                    state[sender] = msg
                    _msg = not all(state.values())

                    for rcvs in outputs:
                        message_q.append((rcvs, _msg, mdl_nm))
                else:
                    pass

            hsh = hash(str(modules_state))
            if hsh in state_cche:
                cycle_len.append(press - 1)
                break
            state_cche[hash(str(modules_state))] = (low_p, hi_p, press)
            press += 1
    print("part 2:", math.lcm(*cycle_len))


def bfs_d21(grid: np.ndarray, cur_pos: tuple, steps: int) -> list[int]:
    directions = {(0, -1), (1, 0), (0, 1), (-1, 0)}
    reachable = [0, 0]  # [even, odd]

    q = [cur_pos]
    visited = set()
    for stp in range(1, steps + 1):
        for i in range(len(q)):  # once iterator is created it's immutable
            cur = q.pop(0)
            for ndf in directions:
                new_pos = (cur[0] + ndf[0], cur[1] + ndf[1])
                inf_new_pos = (new_pos[0] % grid.shape[0], new_pos[1] % grid.shape[1])

                if grid[inf_new_pos[0], inf_new_pos[1]] == '#' or new_pos in visited:
                    continue
                visited.add(new_pos)
                q.append((new_pos))
                reachable[stp % 2] += 1
    return reachable


def day21_regression(grid: np.ndarray, start: tuple) -> int:
    """ stolen polynomial idea"""
    assert (w := len(grid)) // 2 == len(grid[0]) // 2 == start[0] == start[
        1], "The grid needs to be square with S exactly in the middle"

    # After having crossed the border of the first grid, all further border crossings are seperated by n steps (length/width of grid)
    # Therefore, the total number of grids to traverse in any direction is 26_501_365 // n = x_final
    # Assumption: at step 26_501_365 another border crossing is taking place
    # If so, then it follows that the first crossing takes place at 26_501_365 % n = remainder => start[0]
    x_final = (65 + 131 * 202300) // w
    border_crossings = [start[0], start[0] + w, start[0] + 2 * w]

    Y = []
    for b in border_crossings:
        Y.append(bfs_d21(grid, start, b)[b % 2])

    coefficients = np.polyfit([0, 1, 2], Y, deg=2)  # get coefficients for quadratic equation y = a*x^2 + bx + c
    y_final = np.polyval(coefficients, x_final)  # using coefficients, get y value at x_final
    return y_final.round().astype(int)


def aoc21():
    grid = np.array([list(line) for line in scrape()], dtype=str).T
    start = np.where(grid == 'S')
    start = (start[0][0], start[1][0])
    reachable = bfs_d21(grid, start, 64)
    print("part 1:", reachable[0])

    reachable = day21_regression(grid, start)
    print("part 2:", reachable)


@print_timing
def aoc21():
    grid = np.array([list(line) for line in scrape()], dtype=str).T
    start = np.where(grid == 'S')
    start = (start[0][0], start[1][0])
    print("part 1:", bfs_d21(grid, start, 64)[0])
    print("part 2:", day21_regression(grid, start))


class Brick_sparse:
    """to optimize lookup time store dicts based on ez"""
    bricks = {}
    bricks_end = {}

    def add(self, brick):
        self.bricks[brick.z] = self.bricks.get(brick.z, []) + [brick]
        self.bricks_end[brick.ez] = self.bricks_end.get(brick.ez, []) + [brick]

    def get_start(self, z):
        return self.bricks.get(z, [])

    def get_end(self, z):
        return self.bricks_end.get(z, [])

    def remove(self, brick):
        self.bricks[brick.z].remove(brick)
        self.bricks_end[brick.ez].remove(brick)


@dataclass(slots=True, unsafe_hash=True)
class Brick:
    my_id: int
    x: int
    y: int
    z: int
    ex: int  # end cord instead of widths
    ey: int
    ez: int

    def is_supported_by(self, other) -> bool:
        return self.z - 1 == other.ez and (other.x <= self.x <= other.ex or self.x <= other.x <= self.ex) and (
                other.y <= self.y <= other.ey or self.y <= other.y <= self.ey)


def pretty_print(bricks: list[Brick], project: str = 'x'):
    # print as projected stack
    max_z = max(bricks, key=lambda x: x.z).z
    if project == 'x':
        max_x = max(bricks, key=lambda x: x.x).x
    else:
        max_x = max(bricks, key=lambda x: x.y).y

    for z in range(max_z + 1, 0, -1):
        for y in range(max_x + 1):
            if project == 'x':
                # get brick ID to print at this position
                _brick = '.'
                for b in bricks:
                    if b.x <= y <= b.ex and b.z <= z <= b.ez:
                        _brick = b.my_id
                        break
                print(_brick, end='')
            else:
                _brick = '.'
                for b in bricks:
                    if b.y <= y <= b.ey and b.z <= z <= b.ez:
                        _brick = b.my_id
                        break
                print(_brick, end='')
        print(f' {z}')


def cached(cachefile):
    """
    A function that creates a decorator which will use "cachefile" for caching the results of the decorated function "fn".
    """

    def decorator(fn):  # define a decorator for a function "fn"
        def wrapped(*args, **kwargs):  # define a wrapper that will finally call "fn" with all arguments
            # if cache exists -> load it and return its content
            if os.path.exists(cachefile):
                with open(cachefile, 'rb') as cachehandle:
                    print("using cached result from '%s'" % cachefile)
                    return pickle.load(cachehandle)

            # execute the function with all arguments passed
            res = fn(*args, **kwargs)

            # write to cache file
            with open(cachefile, 'wb') as cachehandle:
                pickle.dump(res, cachehandle)

            return res

        return wrapped

    return decorator  # return this "customized" decorator that uses "cachefile"


@cached('sift.pickle')
def sift(bricks: list[Brick]):
    supports = {b.my_id: set() for b in bricks}
    is_supported_count = {b.my_id: 0 for b in bricks}
    max_z = max(bricks, key=lambda x: x.z).z
    for z in range(0, max_z + 1):
        # print(f'z={z}')
        level_bricks = [b for b in bricks if b.z == z]
        # ground
        if z == 1:
            continue
        # try to sift
        for br in level_bricks:
            bricks.remove(br)
            under = [b for b in bricks if b.ez == br.z - 1]
            while not any(br.is_supported_by(s) for s in under):
                if br.z == 1:
                    break
                # print(len(under), br.z)
                br.z -= 1
                br.ez -= 1
                under = [b for b in bricks if b.ez == br.z - 1]
            bricks.append(br)
            for i in under:
                if br.is_supported_by(i):
                    supports[i.my_id].add(br.my_id)
                    is_supported_count[br.my_id] += 1

    return bricks, supports, is_supported_count


def my_d22_cache(func):
    """cache BFS results"""
    cche = {}

    def wrapper(brick: Brick, supports: dict[int, set[int]], is_supported_count: dict[int, int],
                bricks: list[Brick]) -> int:
        # id and is_supported_count are only things changing
        arg = hash((brick.my_id, str(is_supported_count)))
        if arg in cche:
            return cche[arg]
        result = func(brick, supports, is_supported_count, bricks)
        cche[arg] = result
        # print('cache hit', cur_pos)
        return result

    return wrapper


# @my_d22_cache
def countFalls(brick: Brick, supports: dict[int, set[int]], is_supported_count: dict[int, int],
               bricks: list[Brick]) -> int:
    supporting = supports[brick.my_id]
    if len(supporting) == 0:
        return 0
    accum = 0
    for s in supporting:
        cur_b = [b for b in bricks if b.my_id == s][0]
        is_supported_count[s] -= 1

        if is_supported_count[s] == 0:
            accum += 1
            # get brick
            accum += countFalls(cur_b, supports, is_supported_count, bricks)
    return accum


@print_timing
def aoc22():
    """ lord forgive me for I have sinned"""
    inp = scrape(separator='\n')
    cords = [[[int(x) for x in r.split(',')] for r in i.split('~')] for i in inp]
    _id, bricks = 0, []
    brick_storage = Brick_sparse()
    for c in cords:
        bricks.append(Brick(_id, c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2]))
        brick_storage.add(Brick(_id, c[0][0], c[0][1], c[0][2], c[1][0], c[1][1], c[1][2]))
        _id += 1

    assert all(b.z <= b.ez for all_brick_list in brick_storage.bricks.values() for b in all_brick_list)

    # pretty_print(bricks, 'x')
    # print('\n\n')
    # pretty_print(bricks, 'y')

    bricks, supports, is_supported_count = sift(bricks)

    can_be_removed = 0
    for b in bricks:
        supporting = supports[b.my_id]
        if all(is_supported_count[s] > 1 for s in supporting):
            can_be_removed += 1
    print("part 1:", can_be_removed)

    caused_fail_acc = 0
    for b in bricks:
        caused_fail = countFalls(b, supports, is_supported_count.copy(), bricks)
        caused_fail_acc += caused_fail

    print("part 2:", caused_fail_acc)


class Graph:
    @dataclass(slots=True)
    class Adjacency:
        # typehint as int list
        v: tuple[int, int]
        w: int
        against_grain: bool = False  # for undirected in part 2

    adj: dict[tuple[int, int], list[Adjacency]] = {}
    start: tuple[int, int]
    end: tuple[int, int]

    def __init__(self, grid: np.ndarray):
        start, end = (1, 0), (grid.shape[0] - 2, grid.shape[1] - 1)
        self.start, self.end = start, end

        directions = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        slopes = ['^', '<', 'v', '>']

        q = [(start, directions[slopes.index('v')], False)]
        self.adj = {end: []}

        while q:
            cur_node, cur_dir, against_grain = q.pop(0)
            weight = 1
            cur_pos = (cur_node[0] + cur_dir[0], cur_node[1] + cur_dir[1])
            visited = {cur_node}
            while True:
                visited.add(cur_pos)

                # already visited -> just add edge
                if cur_pos in self.adj.keys():
                    if not self.adj.get(cur_node):
                        self.adj[cur_node] = []
                    if Graph.Adjacency(cur_pos, weight, against_grain) not in self.adj[cur_node]:
                        self.adj[cur_node].append(Graph.Adjacency(cur_pos, weight, against_grain))
                    break

                allowed_dir = []
                for ndf in directions:
                    new_pos = (cur_pos[0] + ndf[0], cur_pos[1] + ndf[1])

                    if grid[new_pos] == '#' or new_pos in visited:
                        continue

                    allowed_dir.append((cur_pos, ndf))

                # more than 1 allowed dir -> split otherwise do step
                if len(allowed_dir) > 1:
                    if not self.adj.get(cur_node):
                        self.adj[cur_node] = []
                    if Graph.Adjacency(cur_pos, weight, against_grain) not in self.adj[cur_node]:
                        self.adj[cur_node].append(Graph.Adjacency(cur_pos, weight, against_grain))
                    for ad in allowed_dir:
                        if ad not in q:
                            c_pos, c_dir = ad
                            next_pos = (c_pos[0] + c_dir[0], c_pos[1] + c_dir[1])
                            _against_grain = slopes.index(grid[next_pos]) == (directions.index(c_dir) + 2) % 4
                            q.append((c_pos, c_dir, _against_grain))
                    break
                # 1 allowed dir
                cur_pos, cur_dir = allowed_dir[0]
                weight += 1
                cur_pos = (cur_pos[0] + cur_dir[0], cur_pos[1] + cur_dir[1])

    def longest_path(self):
        # find longest path from start to end in directed graph

        dist = {self.start: 0}
        q = [self.start]
        while q:
            cur = q.pop(0)
            for adj in self.adj[cur]:
                if adj.against_grain:
                    continue

                if adj.v not in dist:
                    dist[adj.v] = dist[cur] + adj.w
                    q.append(adj.v)
                else:
                    if dist[cur] + adj.w > dist[adj.v]:
                        dist[adj.v] = dist[cur] + adj.w
                        q.append(adj.v)
                    else:
                        continue
        return dist[self.end]

    def longest_path_undirected(self, unvisited: set[tuple[int, int]], cur: tuple[int, int]) -> int:
        # find the longest path from start to end without visiting same node twice

        new_unvis = unvisited - {cur}
        paths = self.adj[cur]
        _max = 0

        for path in [path for path in paths if path.v in new_unvis]:
            _max = max(self.longest_path_undirected(new_unvis, path.v) + path.w, _max)

        return _max


def aoc23():
    grid = np.array([list(line) for line in scrape()], dtype=str).T
    g = Graph(grid)
    print("part 1:", g.longest_path())
    print("part 2:", g.longest_path_undirected(set(g.adj.keys()), g.start))


def aoc24():
    acc_p1, lines, test_area = 0, [], (int(2e14), int(4e14))
    for line in scrape():
        p, v = (list(map(int, part.split(','))) for part in line.split('@'))
        lines.append((np.array(p[:2]).T, np.array(v[:2]).T))

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            c1, v1 = lines[i]
            c2, v2 = lines[j]

            x, _, rank, _ = np.linalg.lstsq(np.array([v1, -v2]).T, c2 - c1, rcond=None)
            if rank != 2 or any(x < 0):
                continue

            # check if in future both ways
            inter = c1 + v1 * x[0]
            x, _, _, _ = np.linalg.lstsq(np.array([v2, -v1]).T, c1 - c2, rcond=None)
            if not all(test_area[0] <= val <= test_area[1] for val in inter) or any(x < 0):
                continue

            acc_p1 += 1
    print("part 1:", acc_p1)

    z3_solver = z3.Solver()
    rock = z3.RealVector('rock', 6)
    t = z3.RealVector('t', 3)

    num = 0
    for line in scrape():
        if num == 3:  # 3 are enough
            break
        p, v = (list(map(int, part.split(','))) for part in line.split('@'))
        z3_solver.add([(rock[dim] + rock[dim + 3] * t[num] == p[dim] + v[dim] * t[num]) for dim in range(3)])
        num += 1

    z3_solver.check()
    print("part 2:", z3_solver.model().eval(sum(rock[:3])))


def aoc25():
    g = nx.Graph()
    for k, vals in (line.split(':') for line in scrape()):
        vals = vals.split()
        for v in vals:
            g.add_edge(k, v, capacity=1)
    # import matplotlib.pyplot as plt
    # nx.draw(g, with_labels=True)
    # plt.savefig('graph.svg')
    cut_val, partition = 0, (set(), set())
    while cut_val != 3:
        s, e = sample(list(g.nodes()), 2)
        cut_val, partition = nx.minimum_cut(g, s, e)
    print("part 1:", len(partition[0]) * len(partition[1]))


if __name__ == '__main__':
    # start aoc for given calendar day
    today = datetime.date.today().day
    aocs = [aoc1, aoc2, aoc3, aoc4, aoc5, aoc6, aoc7, aoc8, aoc9, aoc10, aoc11, aoc12, aoc13, aoc14, aoc15, aoc16,
            aoc17, aoc18, aoc19, aoc20, aoc21, aoc22, aoc23, aoc24, aoc25]
    aocs[today - 1]()
