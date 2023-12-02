import datetime
import inspect
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
        fs = '{} took {:.2f} microseconds'
        print(fs.format(func.__name__, (end - start) * 1e6))
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


if __name__ == '__main__':
    # start aoc for give calendar day
    today = datetime.date.today().day
    today_f_name = "aoc" + str(today)
    eval(today_f_name)()
