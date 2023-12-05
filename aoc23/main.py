#!/usr/bin/python3

import datetime
import inspect
import multiprocessing
import os
import re
import time
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


def get_min_location_from_ranges(start2, end2, step_maps, cur_min, inp):
    my_min = 999999999999
    for s in range(start2, end2+1):
        for maps in step_maps:
            for dest, src, ln in maps:
                if src <= s <= src + ln:
                    s += dest - src
                    break
        my_min = min(my_min, s)
    cur_min.value = min(cur_min.value, my_min)
    print("min:", my_min)


@print_timing
def aoc5():
    inp, step_maps = scrape(separator='\n\n'), []
    # parse input -> each inner list is X->Y map (stored as separate ranges)
    for maps in inp[1:]:
        nums = [int(i) for i in maps.split()[2:]]
        cur_step_map = []
        for dest, src, ln in zip(*(iter(nums),) * 3):
            cur_step_map.append((dest, src, ln))
        step_maps.append(cur_step_map)

    seeds = [int(i) for i in inp[0][6:].split()]
    seed_ranges_part1, seed_ranges_part2 = [], []
    # create ranges
    for start in seeds:
        seed_ranges_part1.append((start, start))
    for start, rng in zip(*(iter(seeds),) * 2):
        seed_ranges_part2.append((start, start + rng - 1))

    # part 2 no workey
    #print("part 1 :", get_min_location_from_ranges(seed_ranges_part1, step_maps))
    #print("part 2 :", get_min_location_from_ranges(seed_ranges_part2, step_maps))

    cur_min = multiprocessing.Value('l', 999999999999)  # 'i' represents a signed int
    num_processes = 5  # Adjust the number of threads as needed
    process_list = []

    for start, interval in zip(*(iter(seeds),) * 2):
        # split each range into num_processes subranges
        for i in range(num_processes):
            start2, end2 = start + i * interval // num_processes, start + (i + 1) * interval // num_processes - 1
            process = multiprocessing.Process(
                target=get_min_location_from_ranges,
                args=(start2, end2, step_maps, cur_min, inp)
            )
            process_list.append(process)
            process.start()

    for process in process_list:
        process.join()

    print("part 2:", cur_min.value)
#part 2: 37806487
#aoc5 took 2797553.95 ms


def aoc6():
    pass


if __name__ == '__main__':
    # start aoc for given calendar day
    today = datetime.date.today().day
    today_f_name = "aoc" + str(today)
    eval(today_f_name)()
