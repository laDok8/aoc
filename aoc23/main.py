#!/usr/bin/python3

import datetime
import inspect
import os
import re
import time
from collections import namedtuple
from functools import reduce

import requests as r
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
        return open(file).read().split(separator)
    inp = r.get(url, cookies=cookies).text.rstrip()
    with open(file, 'w') as f:
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
    print(sum_part1)

    map_replace = {'zero': 'z0o', 'one': 'o1e', 'two': 't2o', 'three': 't3e', 'four': 'f4r', 'five': 'f5e',
                   'six': 's6x', 'seven': 's7n', 'eight': 'e8t', 'nine': 'n9e'}
    for line in inp:
        for key, val in map_replace.items():
            line = line.replace(key, val)
        # remove all non digits
        line = re.sub(r'\D', '', line)
        # get first and last number
        sum_part2 += int(line[0]) * 10 + int(line[-1])
    print(sum_part2)


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
        if all([cur_lims[color] <= part1_limits[color] for color in cur_lims.keys()]):
            possible_ids_sum += game_id
        fewest_cubes_sum += reduce((lambda x, y: x * y), cur_lims.values())

    print("part 1: ", possible_ids_sum, "\npart 2: ", fewest_cubes_sum)


class Rectangle:
    x, y, width = 0, 0, 0

    def __init__(self, x, y, width):
        self.x, self.y, self.width = x, y, width

    def is_adjacent(self, other):
        return (abs(self.y - other.y) <= 1 and (
                self.x - 1 <= other.x <= self.x + self.width or other.x - 1 <= self.x <= other.x + other.width))


def aoc3():
    all_parts, all_nums, inp = [], [], scrape()
    for (_y, line) in enumerate(inp):
        num_iter, part_iter = re.finditer(r'\d+', line), re.finditer(r'[^.\d]', line)
        for num in num_iter:
            all_nums.append((Rectangle(num.span()[0], _y, num.span()[1] - num.span()[0]), int(num.group())))
        for part in part_iter:
            all_parts.append(Rectangle(part.span()[0], _y, 1))

    sum_part1 = sum(
        map(lambda _x: _x[1], filter(lambda _num: any(_num[0].is_adjacent(_part) for _part in all_parts), all_nums)))
    sum_part2 = sum(
        reduce(lambda _x, y: _x * y, map(lambda x: x[1], filter(lambda _num: part.is_adjacent(_num[0]), all_nums)), 1)
        for part in all_parts if len(list(filter(lambda _num: part.is_adjacent(_num[0]), all_nums))) == 2)
    print("part 1: ", sum_part1, "\npart 2: ", sum_part2)


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
    print("part 1 :", sum_points_part1, "\npart 2 :", sum([i for i in card_storage.values()]))


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

    print("part 1 :", get_min_location_from_ranges(seed_ranges_part1, step_maps))
    print("part 2 :", get_min_location_from_ranges(seed_ranges_part2, step_maps))


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
    print("part 1 :", acc_prod, "\npart 2 :", acc)


def get_poker_hand_type(occurrences: dict):
    if max(occurrences.values()) == 5:
        return '9'  # five of a kind
    elif max(occurrences.values()) == 4:
        return '8'  # four of a kind
    elif sorted(occurrences.values()) == [2, 3]:
        return '7'  # full house
    elif sorted(occurrences.values()) == [1, 1, 3]:
        return '6'  # three of a kind
    elif sorted(occurrences.values()) == [1, 2, 2]:
        return '5'  # two pairs
    elif sorted(occurrences.values()) == [1, 1, 1, 2]:
        return '4'  # one pair
    else:
        return '3'  # high card


def aoc7():
    inp = scrape()
    Hand = namedtuple('Hand', ['raw', 'bid', 'strength'])  # strength = TYPE + raw
    # for easier string sort
    raw_strength_map_p1 = {'T': 'a', 'J': 'b', 'Q': 'c', 'K': 'd', 'A': 'e'}
    raw_strength_map_p2 = raw_strength_map_p1.copy()
    raw_strength_map_p2['J'] = '0'

    all_hands_p1, all_hands_p2 = [], []
    for line in inp:
        raw, bid = line.split()

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
    print("part1 :", acc_p1, "part2 :", acc_p2)


def aoc8():
    pass


if __name__ == '__main__':
    # start aoc for given calendar day
    today = datetime.date.today().day
    today_f_name = "aoc" + str(today)
    eval(today_f_name)()
