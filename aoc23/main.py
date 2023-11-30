import datetime
import os
import time
import requests as r
import inspect
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
    pass


if __name__ == '__main__':
    # start aoc for give calendar day
    today = datetime.date.today().day
    today_f_name = "aoc" + str(today)
    eval(today_f_name)()
