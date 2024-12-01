import datetime
import inspect
import os
import time

import requests as r
from dotenv import dotenv_values


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
    cookies = dotenv_values(".env")

    # determine day from callee function name
    caller_function_name = inspect.stack()[1].function
    day = int(str.replace(caller_function_name, 'aoc', '')) if day == 0 else day
    year = datetime.date.today().year if year == 0 else year
    # substitute year and day
    url = f'https://adventofcode.com/{year}/day/{day}/input'
    file = f'inp{day}.txt'
    if os.path.exists(file):
        lines = open(file, encoding='utf-8').read().split(separator)
        if lines and lines[-1] == '':
            lines.pop()
        return lines
    inp = r.get(url, cookies=cookies, timeout=1).text.rstrip()
    with open(file, 'w', encoding='UTF-8') as f:
        f.write(inp)
    return inp.split(separator)
