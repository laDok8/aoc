import os

import numpy as np
import pandas as pd
import requests as r

cookies = {"session": "COOKIE_VALUE"}


def scrape(url, file):
    if os.path.exists(file):
        return
    inp = r.get(url, cookies=cookies, timeout=1)
    with open(file, 'w', encoding='UTF-8') as fp:
        fp.write(inp.text)


def aoc1():
    scrape('https://adventofcode.com/2021/day/1/input', 'inp1.txt')
    # part 1
    with open('inp1.txt', 'r', encoding='UTF-8') as fp:
        ar = np.array([int(x) for x in fp.readlines()])
    print('decrease:', np.sum(ar[:-1] < ar[1:]))

    # part 2
    arr = ar[:-2] + ar[1:-1] + ar[2:]
    print('sliding decrease:', np.sum(arr[:-1] < arr[1:]))


def aoc2():
    scrape('https://adventofcode.com/2021/day/2/input', 'inp2.txt')
    # part 1
    df = pd.read_csv('inp2.txt', sep=' ', names=['inst', 'move'])
    df1 = df.groupby('inst').sum()
    y = df1.loc['down', 'move'] - df1.loc['up', 'move']
    x = df1.loc['forward', 'move']
    print("final depth:", x * y)

    # part 2
    df['aim'] = df.apply(lambda x: x['move'] if x['inst'] == 'down' else -x['move'] if x['inst'] == 'up' else 0,
                         axis=1).cumsum()
    df = df[df['inst'] == 'forward']
    print('final depth:', x * (df['aim'] * df['move']).sum())


def aoc3():
    scrape('https://adventofcode.com/2021/day/3/input', 'inp3.txt')
    # part 1
    inp = []
    with open('inp3.txt', 'r', encoding='UTF-8') as fp:
        for line in fp.readlines():
            row = []
            for c in line.replace('\n', ''):
                row += [int(c)]
            inp.append([row])
    inp = np.squeeze(np.array(inp, dtype=int), axis=1)
    res = [[], []]
    for col in inp.T:
        res[0].append(np.argmax(np.bincount(col)))
        res[1].append(np.argmin(np.bincount(col)))

    _r = [''.join(str(x) for x in res[0]), ''.join(str(x) for x in res[1])]
    print('power consumption:', int(_r[0], 2) * int(_r[1], 2))
    # part 2
    res1, res2, i = inp[:], inp[:], 0
    while np.shape(res1)[0] != 1 or np.shape(res2)[0] != 1:
        _min = np.argmin(np.bincount(res2.T[i]))
        _max = 1 if np.bincount(res1.T[i])[0] == np.bincount(res1.T[i])[1] else np.argmax(np.bincount(res1.T[i]))
        res1, res2, i = res1[res1[:, i] == _max], res2[res2[:, i] == _min], i + 1
    _r = [''.join(str(x) for x in res1[0]), ''.join(str(x) for x in res2[0])]
    print('life support rating:', int(_r[0], 2) * int(_r[1], 2))


aoc3()
