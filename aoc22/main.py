import copy
import os
import time

import numpy as np
import requests as r

cookies = {"session": "COOKIE_VALUE"}


def print_timing(func):
    """create a timing decorator function"""

    def wrapper(*arg):
        start = time.perf_counter()
        result = func(*arg)
        end = time.perf_counter()
        fs = '{} took {:.3f} microseconds'
        print(fs.format(func.__name__, (end - start) * 1000000))
        return result

    return wrapper


def scrape(url_generic: str = 'https://adventofcode.com/{year}/day/{day}/input', file_generic: str = 'inp{day}.txt',
           year: int = 2022, day: int = 1, separator: str = '\n'):
    # substitute year and day
    url = url_generic.format(year=year, day=day)
    file = file_generic.format(day=day)
    if os.path.exists(file):
        return open(file, encoding='UTF-8').read().split(separator)
    inp = r.get(url, cookies=cookies, timeout=1).text.rstrip()
    with open(file, 'w', encoding='UTF-8') as fp:
        fp.write(inp)
    return inp.split(separator)


def aoc1():
    elf_sum, inp = [], scrape(day=1, separator='\n\n')
    for elf in inp:
        elf_sum.append(sum(int(i) for i in elf.strip().split('\n')))
    elf_sum = -np.sort(-np.array(elf_sum))
    print(elf_sum[0], np.sum(elf_sum[:3]))


def aoc2():
    score1, score2, inp = 0, 0, scrape(day=2)
    # win function
    wf = {'AX': 3, 'AY': 6, 'AZ': 0, 'BX': 0, 'BY': 3, 'BZ': 6, 'CX': 6, 'CY': 0, 'CZ': 3}
    # map part 2 to previous function
    response = {'AX': 'Z', 'AY': 'X', 'AZ': 'Y', 'BX': 'X', 'BY': 'Y', 'BZ': 'Z', 'CX': 'Y', 'CY': 'Z', 'CZ': 'X'}

    for i, j in map(str.split, inp):
        response_2 = response[i + j]

        # win function & draw function
        score1 += wf[i + j] + int(ord(j) - ord('X')) + 1
        score2 += wf[i + response_2] + int(ord(response_2) - ord('X')) + 1
    print(score1, score2)


def aoc3():
    sum1, sum2, inp, inp2 = 0, 0, scrape(day=3), scrape(day=3)
    for (first, second) in map(lambda x: (x[:len(x) // 2], x[len(x) // 2:]), inp):
        same = (set(first) & set(second)).pop()
        sum1 += ord(same) - ord('a') + 1 if same.islower() else ord(same) - ord('A') + 27

    for (i, j, k) in zip(inp2[0::3], inp2[1::3], inp2[2::3]):
        same = (set(i) & set(j) & set(k)).pop()
        sum2 += ord(same) - ord('a') + 1 if same.islower() else ord(same) - ord('A') + 27
    print(sum1, sum2)


def aoc4():
    pairs, pairs2, inp = 0, 0, scrape(day=4)
    for (first, second) in map(lambda x: x.split(','), inp):
        fl, fh = map(int, first.split('-'))
        sl, sh = map(int, second.split('-'))

        # full overlap
        pairs += 1 if fl <= sl and fh >= sh or sl <= fl and sh >= fh else 0
        # partial overlap
        pairs2 += 1 if fl <= sl <= fh or sl <= fl <= sh else 0
    print(pairs, pairs2)


def aoc5():
    inp = scrape(day=5)
    stack, instructions = inp[:inp.index('') - 1:], inp[inp.index('') + 1:]

    c, s = [], []
    for inp in stack[::-1]:
        for idx, i in enumerate(range(1, len(inp), 4)):
            ix = inp[i].strip()
            if len(s) < idx + 1:
                s.append([])
            if ix:
                s[idx].append(ix)

    c2, s2 = [], copy.deepcopy(s)
    for ins in instructions:
        foo = ins.split()
        count, from_, to_ = int(foo[1]), int(foo[3]) - 1, int(foo[5]) - 1
        for i in range(count):
            c.append(s[from_].pop())
            c2.append(s2[from_].pop())
        # apend in inverse order for part1
        c = c[::-1]
        for i in range(count):
            s[to_].append(c.pop())
            s2[to_].append(c2.pop())

    print("".join([s[i][-1] for i in range(len(s))]), "".join([s2[i][-1] for i in range(len(s2))]))


def aoc6():
    s, s2, inp = 0, 0, scrape(day=6)[0]
    for i in range(len(inp)):
        st = inp[i:i + 4]
        st2 = inp[i:i + 14]
        if len(set(st)) == 4 and s == 0:
            s = i + 4
        if len(set(st2)) == 14 and s2 == 0:
            s2 = i + 14
        if s and s2:
            break
    print(s, s2)


def aoc7():
    inp, sums, cur_dir = scrape(day=7), {}, ()
    inp = list(filter(lambda x: not x.startswith("$ ls") and not x.startswith("dir"), inp))

    for i in inp:
        if i.startswith('$ cd'):
            _dir = i.split()[2]
            cur_dir = cur_dir[:-1] if _dir == '..' else cur_dir + (_dir,)
            continue

        num = int(i.split()[0])
        for j in range(1, len(cur_dir) + 1):
            sums[cur_dir[:j]] = sums.get(cur_dir[:j], 0) + num

    needed_sum = int(7e7) - sums[('/',)]
    s, s2 = sum(filter(lambda x: x < 1e5, sums.values())), min(x for x in sums.values() if x + needed_sum > int(3e7))
    print(s, s2)


aoc7()
